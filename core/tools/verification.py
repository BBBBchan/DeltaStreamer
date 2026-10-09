"""Look-Back Verification & Correction for DeltaStreamer (Pillar 3).

Provides: :class:`ConsistencyChecker` (cross-layer claim extraction +
consistency scoring), :class:`ReplayScheduler` (decision-theoretic
expected-information-gain scheduling), and :func:`run_verification_pipeline`
(the full verify → replay → correct closed loop).

Design: verification is *opt-in* (gated by ``--enable_verification``)
and only activates during the POST-QUESTION phase when the model is
about to emit ``</Response>``.  It reuses the existing replay
infrastructure (``ReplayTool`` + ``_pending_tool_text``) for look-back.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Soft dependency — the caller is responsible for importing these at the
# call site (inside agent.py) where the full Swift / vLLM stack is
# already available.
try:
    from .memory_graph import EntityIndex, MemoryGraph, EvidencePath, op_correct_node  # noqa: F401
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class Claim:
    """A single atomic fact extracted from memory evidence."""
    subject: str          # entity name or description
    property: str         # "location" | "color" | "state" | "count" | ...
    value: str            # the asserted value
    time: int             # timestamp from the source
    source_layer: str     # "working" | "episodic" | "long_term"
    source_node_id: int   # the event / episode / pin id

    def key(self) -> Tuple[str, str]:
        return (self.subject.lower(), self.property.lower())


@dataclass
class ClaimGroup:
    """Claims that describe the same (subject, property) pair, possibly
    from different layers / times."""
    subject: str
    property: str
    claims: List[Claim] = field(default_factory=list)
    status: str = "unverified"  # verified | conflict | uncertain
    conflict_score: float = 0.0
    suggested_replay_target: Optional[int] = None


@dataclass
class VerificationResult:
    """Output of the consistency checker for one claim group."""
    status: str           # "verified" | "conflict" | "uncertain"
    conflict_score: float
    suggested_replay_target: Optional[int]
    description: str
    claims: List[Claim] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Consistency Checker
# ---------------------------------------------------------------------------
# Simple state-transition validity table for common entities.
# A transition (s_old, s_new) is INVALID if it implies teleportation
# or disappearance without an intermediate action.
_INVALID_TRANSITIONS: Set[Tuple[str, str]] = {
    # Object location teleportation.
    ("on_counter", "on_table"),
    ("on_table", "on_counter"),
    ("on_floor", "on_shelf"),
    ("on_shelf", "on_floor"),
    # Object state change without intermediate.
    ("closed", "broken"),
    ("full", "empty"),
    ("intact", "broken"),
    # Disappearance.
    ("present", "absent"),
    ("visible", "absent"),
}

# Valid transitions that can bridge an invalid one.
_VALID_BRIDGES: Dict[Tuple[str, str], str] = {
    ("on_counter", "on_table"): "being_held",
    ("on_table", "on_counter"): "being_held",
    ("closed", "broken"): "being_opened",
    ("full", "empty"): "being_poured",
    ("present", "absent"): "exiting",
}


class ConsistencyChecker:
    """Cross-layer claim extraction and consistency verification."""

    # Simple claim extraction patterns.
    _STATE_PATTERNS = [
        # "[FINISHED]" / "[IN-PROGRESS]" tags
        (re.compile(r'(\w+(?:\s+\w+){0,3})\s*'
                    r'\[(FINISHED|IN-PROGRESS|STARTING|ABORTED)\]'),
         "state"),
        # "X is on Y" / "X moved to Y"
        (re.compile(r'(\w+(?:\s+\w+){0,3})\s+(?:is|was)\s+(on|under|behind|'
                    r'next\s+to|inside|in front of)\s+(\w+(?:\s+\w+){0,3})'),
         "location"),
        # "X is ADJ"  (color / state)
        (re.compile(r'(\w+(?:\s+\w+){0,3})\s+(?:is|was|appears?)\s+'
                    r'(blue|red|green|yellow|white|black|open|closed|'
                    r'full|empty|broken|intact|moving|static|visible|hidden)'),
         "property"),
        # Counts: "N Xs"  (with a numeral)
        (re.compile(r'(\d+)\s+(\w+(?:\s+\w+){0,2})s?\b'),
         "count"),
    ]

    _TOK_RE = re.compile(r"[A-Za-z0-9_]{2,}")

    def __init__(
        self,
        min_agreement: float = 0.6,
        conflict_threshold: float = 0.3,
    ) -> None:
        self.min_agreement = float(min_agreement)
        self.conflict_threshold = float(conflict_threshold)

    # ---- Public API ---------------------------------------------------
    def extract_claims(
        self,
        evidence_paths: List[EvidencePath],
    ) -> List[Claim]:
        """Extract atomic claims from ranked evidence paths."""
        claims: List[Claim] = []
        seen: Set[str] = set()
        for path in evidence_paths:
            for node in path.nodes:
                for pattern, prop in self._STATE_PATTERNS:
                    for m in pattern.finditer(node.text or ""):
                        if prop == "state":
                            subj = m.group(1).strip().lower()
                            val = m.group(2).strip()
                        elif prop == "location":
                            subj = m.group(1).strip().lower()
                            val = f"{m.group(2).strip()} {m.group(3).strip()}"
                        elif prop == "property":
                            subj = m.group(1).strip().lower()
                            val = m.group(2).strip()
                        elif prop == "count":
                            subj = m.group(2).strip().lower()
                            val = m.group(1).strip()
                        else:
                            continue
                        ck = f"{subj}|{prop}|{val}|{node.layer}|{node.node_id}"
                        if ck in seen:
                            continue
                        seen.add(ck)
                        claims.append(Claim(
                            subject=subj,
                            property=prop,
                            value=val,
                            time=node.t or 0,
                            source_layer=node.layer,
                            source_node_id=node.node_id,
                        ))
        return claims

    def group_by_identity(
        self,
        claims: List[Claim],
    ) -> List[ClaimGroup]:
        """Group claims by (subject, property)."""
        groups: Dict[Tuple[str, str], ClaimGroup] = {}
        for c in claims:
            k = c.key()
            if k not in groups:
                groups[k] = ClaimGroup(subject=c.subject, property=c.property)
            groups[k].claims.append(c)
        return list(groups.values())

    def quick_resolve(self, group: ClaimGroup) -> Optional[str]:
        """Stage 2a: zero-cost rule-based resolution (no model required).

        Resolves simple conflicts without triggering an expensive replay:
          1. working + episodic agree, only long_term differs
             → long_term is likely a stale/superseded pin → trust working+episodic
          2. Temporal gap ≤ 2 s between consecutive claims
             → normal frame-rate jitter, not a real gap
          3. conflict_score < 0.15
             → low severity, mark uncertain but don't replay

        Returns "verified", "uncertain", or None (cannot resolve, needs replay).
        """
        claims = group.claims
        if len(claims) < 2:
            return "uncertain"

        # Rule 1: working + episodic agree, pin layer disagrees.
        vals_by_layer: Dict[str, Set[str]] = {}
        for c in claims:
            vals_by_layer.setdefault(c.source_layer, set()).add(c.value.lower())
        wk_vals = vals_by_layer.get("working", set())
        ep_vals = vals_by_layer.get("episodic", set())
        lt_vals = vals_by_layer.get("long_term", set())
        if wk_vals and ep_vals and wk_vals == ep_vals:
            if lt_vals and lt_vals != wk_vals:
                # working + episodic agree — pin is stale.
                return "verified"

        # Rule 2: temporal gap check.
        sorted_claims = sorted(claims, key=lambda c: c.time)
        gaps = [
            sorted_claims[i + 1].time - sorted_claims[i].time
            for i in range(len(sorted_claims) - 1)
        ]
        if all(g <= 2 for g in gaps):
            return "verified"  # small gaps are normal

        # Rule 3: low conflict score.
        n_unique_vals = len({c.value.lower() for c in claims})
        if n_unique_vals == 1:
            return "verified"
        agreement = max(
            sum(1 for c in claims if c.value.lower() == v)
            for v in {c.value.lower() for c in claims}
        ) / len(claims)
        if agreement >= 0.85:
            return "uncertain"

        # Cannot resolve — needs deep verification (replay).
        return None

    def check_cross_layer(
        self,
        group: ClaimGroup,
    ) -> VerificationResult:
        """Check whether the same fact is consistent across memory layers."""
        n = len(group.claims)
        if n <= 1:
            return VerificationResult(
                status="uncertain",
                conflict_score=0.0,
                suggested_replay_target=None,
                description="single source, cannot verify",
                claims=list(group.claims),
            )

        # Extract unique (layer, value) pairs.
        layer_values: Dict[str, Set[str]] = {}
        for c in group.claims:
            layer_values.setdefault(c.source_layer, set()).add(c.value.lower())

        # Agreement: do all layers agree on the value?
        all_values: Set[str] = set()
        for vs in layer_values.values():
            all_values |= vs

        if len(all_values) == 1:
            # Perfect agreement across all sources.
            return VerificationResult(
                status="verified",
                conflict_score=0.0,
                suggested_replay_target=None,
                description="all layers agree",
                claims=list(group.claims),
            )

        # Disagreement: score by weighted source reliability.
        # working > episodic > long_term  (working is closest to pixels)
        weights = {"working": 1.0, "episodic": 0.8, "long_term": 0.6}
        value_scores: Dict[str, float] = {}
        for c in group.claims:
            w = weights.get(c.source_layer, 0.5)
            value_scores[c.value.lower()] = value_scores.get(
                c.value.lower(), 0.0) + w

        best_value = max(value_scores, key=lambda k: value_scores[k])
        total_weight = sum(value_scores.values()) or 1.0
        agreement_rate = value_scores[best_value] / total_weight

        if agreement_rate >= self.min_agreement:
            return VerificationResult(
                status="verified",
                conflict_score=1.0 - agreement_rate,
                suggested_replay_target=None,
                description=f"majority agreement ({agreement_rate:.2f})",
                claims=list(group.claims),
            )

        # Conflict: suggest replaying the earliest working-layer frame.
        replay_t = None
        for c in sorted(group.claims, key=lambda x: x.time):
            if c.source_layer == "working":
                replay_t = c.time
                break

        return VerificationResult(
            status="conflict",
            conflict_score=1.0 - agreement_rate,
            suggested_replay_target=replay_t,
            description=(
                f"disagreement across layers (agreement={agreement_rate:.2f}); "
                f"best value='{best_value}' from {set(layer_values.keys())}"
            ),
            claims=list(group.claims),
        )

    def check_temporal_coherence(
        self,
        group: ClaimGroup,
        entity_index: Optional[EntityIndex] = None,
    ) -> Optional[VerificationResult]:
        """Check whether the entity's state chain is temporally coherent.

        Returns a ``VerificationResult`` if a gap or invalid transition
        is detected, otherwise ``None``.
        """
        if len(group.claims) < 2:
            return None
        sorted_claims = sorted(group.claims, key=lambda c: c.time)

        gaps: List[Tuple[int, int, str, str]] = []
        for i in range(len(sorted_claims) - 1):
            a, b = sorted_claims[i], sorted_claims[i + 1]
            transition = (a.value.lower(), b.value.lower())
            if transition in _INVALID_TRANSITIONS:
                gaps.append((a.time, b.time, a.value, b.value))

        if not gaps:
            return None

        # Compute a conflict score from the number and severity of gaps.
        score = min(1.0, 0.3 * len(gaps))
        replay_t = gaps[0][0]  # replay the earliest gap source
        desc = (
            f"temporal coherence violation(s): "
            + "; ".join(f"t={t0}→t={t1} '{v0}'→'{v1}'" for t0, t1, v0, v1 in gaps)
        )
        return VerificationResult(
            status="conflict",
            conflict_score=score,
            suggested_replay_target=replay_t,
            description=desc,
            claims=list(group.claims),
        )


# ---------------------------------------------------------------------------
# Decision-Theoretic Replay Scheduler
# ---------------------------------------------------------------------------
class ReplayScheduler:
    """Schedule replay targets by expected information gain.

    E[gain | replay] = P(correct | replay) - P(correct | no_replay)

    where P(correct | replay) ≈ 0.95 (visual verification),
    and P(correct | no_replay) ≈ current agreement rate.
    """

    def __init__(
        self,
        max_replays: int = 3,
        cost_threshold: float = 0.15,
        p_correct_if_replay: float = 0.95,
    ) -> None:
        self.max_replays = int(max_replays)
        self.cost_threshold = float(cost_threshold)
        self.p_correct_replay = float(p_correct_if_replay)

    def schedule(
        self,
        conflicts: List[VerificationResult],
        current_time: int,
        question_time: int,
    ) -> List[int]:
        """Return the ordered list of recommended replay target frames.

        Only returns targets whose expected gain exceeds the cost
        threshold and caps at ``max_replays``.
        """
        if not conflicts:
            return []

        # Filter: only real conflicts with a valid replay target.
        candidates = [
            c for c in conflicts
            if c.status == "conflict" and c.suggested_replay_target is not None
        ]
        if not candidates:
            return []

        # Compute expected gain per candidate.
        scored: List[Tuple[float, int, VerificationResult]] = []
        for c in candidates:
            p_no_replay = max(0.0, 1.0 - c.conflict_score)
            expected_gain = self.p_correct_replay - p_no_replay
            # Time-decay bonus: prefer conflicts that are temporally closer
            # to question_time.
            if c.suggested_replay_target is not None:
                time_dist = abs(c.suggested_replay_target - question_time)
                time_factor = max(0.5, 1.0 - 0.02 * time_dist)
                expected_gain *= time_factor
            if expected_gain > self.cost_threshold:
                scored.append((expected_gain, c.suggested_replay_target, c))

        # Sort by gain descending.
        scored.sort(key=lambda x: -x[0])

        # Deduplicate targets (same frame → keep highest gain).
        seen_targets: Set[int] = set()
        targets: List[int] = []
        for gain, target, _ in scored:
            if target not in seen_targets:
                seen_targets.add(target)
                targets.append(target)
            if len(targets) >= self.max_replays:
                break

        return targets


# ---------------------------------------------------------------------------
# Verification Pipeline (top-level entry point)
# ---------------------------------------------------------------------------
def run_verification_pipeline(
    answer: str,
    graph: Any,            # MemoryGraph
    checker: ConsistencyChecker,
    scheduler: ReplayScheduler,
    current_time: int,
    question_time: int,
    retrieval_query: Optional[str] = None,
) -> Tuple[bool, List[int], List[VerificationResult]]:
    """Execute the full verify → schedule → correct pipeline.

    Returns:
      (needs_replay, replay_targets, all_results)

    When ``needs_replay`` is True, the caller should inject the
    replay targets into ``_pending_tool_text`` and re-prompt the model
    on the same frame.  After the replay round(s), the caller should
    invoke :func:`apply_corrections` with the verified facts.
    """
    # 1) Extract evidence from current memory.
    try:
        evidence = graph.retrieve_backtrack(
            query=retrieval_query or answer,
            top_k=16,
            current_t=current_time,
        )
    except Exception:
        return False, [], []

    # 2) Extract claims + group + check.
    claims = checker.extract_claims(evidence)
    groups = checker.group_by_identity(claims)

    results: List[VerificationResult] = []
    conflicts: List[VerificationResult] = []
    for g in groups:
        # ── Stage 2a: Quick rule-based resolution ──────────────────
        quick = checker.quick_resolve(g)
        if quick == "verified":
            results.append(VerificationResult(
                status="verified", conflict_score=0.0,
                suggested_replay_target=None,
                description="quick-resolved: " + (
                    "working+episodic agree" if len(g.claims) >= 2
                    else "single source"),
                claims=list(g.claims)))
            continue
        elif quick == "uncertain":
            results.append(VerificationResult(
                status="uncertain", conflict_score=0.1,
                suggested_replay_target=None,
                description="quick-resolved: low confidence",
                claims=list(g.claims)))
            continue  # Don't replay for uncertain.

        # ── Stage 2b: Full cross-layer + temporal check ────────────
        cl_result = checker.check_cross_layer(g)
        temporal_result = checker.check_temporal_coherence(g)
        if temporal_result is not None and temporal_result.conflict_score > 0.0:
            cl_result.conflict_score = max(
                cl_result.conflict_score, temporal_result.conflict_score)
            if cl_result.status != "conflict":
                cl_result.status = "conflict"
            cl_result.description += " | " + temporal_result.description
            if cl_result.suggested_replay_target is None:
                cl_result.suggested_replay_target = temporal_result.suggested_replay_target
        results.append(cl_result)
        if cl_result.status == "conflict":
            conflicts.append(cl_result)

    # 3) Schedule replays.
    targets = scheduler.schedule(conflicts, current_time, question_time)
    needs_replay = len(targets) > 0

    return needs_replay, targets, results


def apply_corrections(
    store: Any,      # MemoryStore
    graph: Any,      # MemoryGraph
    verified_fact: str,
    source_node_id: int,
    source_layer: str,
    current_t: int,
) -> int:
    """Apply a correction to the memory DAG after visual verification.

    Returns the correction event ID.
    """
    from .memory_graph import op_correct_node
    return op_correct_node(
        store=store,
        graph=graph,
        node_id=source_node_id,
        layer=source_layer,
        corrected_text=verified_fact,
        current_t=current_t,
    )
