"""Memory DAG overlay for DeltaStreamer (Pillar 2).

Provides: :class:`EntityIndex`, :class:`MemoryGraph` (bidirectional
cross-layer pointers + Cross-Layer Backtrack retrieval),
:class:`BOCPDSegmenter` (online Bayesian changepoint detection for
episode boundaries), and six formal graph operations.

Design principle: all structures are *overlays* on the existing
:class:`MemoryStore` — they index and connect existing data rather
than duplicating it. Legacy call sites that do not import this
module continue to work unchanged.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Entity tracking
# ---------------------------------------------------------------------------
@dataclass
class EntitySubgraph:
    """Per-entity sub-graph tracking all events / pins / episodes that
    mention or affect a given entity."""
    entity_id: str
    event_ids: List[int] = field(default_factory=list)
    pinned_ids: List[int] = field(default_factory=list)
    episode_ids: List[int] = field(default_factory=list)
    first_seen: int = 0
    last_seen: int = 0
    # Human-friendly aliases observed over time.
    aliases: List[str] = field(default_factory=list)


class EntityIndex:
    """Lightweight index mapping entity names to their sub-graphs.

    An entity is identified by a canonical slug (e.g. ``chef``,
    ``blue_cup``). Multiple surface names (\"chef\", \"man in white\")
    can be linked to the same entity via :meth:`link_aliases`.
    """

    _TOK_RE = re.compile(r"[A-Za-z0-9_]{2,}")

    def __init__(self) -> None:
        self._entities: Dict[str, EntitySubgraph] = {}
        # alias → canonical entity_id
        self._alias_map: Dict[str, str] = {}

    # ---- public API ----------------------------------------------------
    def add_observation(
        self,
        event_id: int,
        t: int,
        entities: List[str],
    ) -> List[str]:
        """Register one observation event for the listed entity names.

        Returns the resolved canonical entity-ids.
        """
        eids: List[str] = []
        for raw_name in entities:
            slug = self._to_slug(raw_name)
            if not slug:
                continue
            canonical = self._resolve(slug)
            sub = self._entities.setdefault(
                canonical, EntitySubgraph(entity_id=canonical))
            if canonical not in eids:
                eids.append(canonical)
            sub.event_ids.append(event_id)
            if sub.first_seen == 0 or t < sub.first_seen:
                sub.first_seen = t
            if t > sub.last_seen:
                sub.last_seen = t
            if raw_name.lower() not in sub.aliases:
                sub.aliases.append(raw_name.lower())
        return eids

    def add_episode(self, episode_id: int, entity_ids: List[str]) -> None:
        for eid in entity_ids:
            sub = self._entities.get(eid)
            if sub is not None and episode_id not in sub.episode_ids:
                sub.episode_ids.append(episode_id)

    def add_pin(self, pin_id: int, entity_ids: List[str]) -> None:
        for eid in entity_ids:
            sub = self._entities.get(eid)
            if sub is not None and pin_id not in sub.pinned_ids:
                sub.pinned_ids.append(pin_id)

    def get_timeline(self, entity_id: str) -> Optional[EntitySubgraph]:
        canonical = self._resolve(entity_id)
        return self._entities.get(canonical)

    def link_aliases(self, name_a: str, name_b: str) -> bool:
        """Declare two surface names refer to the same entity."""
        slug_a, slug_b = self._to_slug(name_a), self._to_slug(name_b)
        if not slug_a or not slug_b:
            return False
        canonical = self._resolve(slug_a)
        self._alias_map[slug_b] = canonical
        # Merge sub-graphs if both already exist.
        if slug_b in self._entities and slug_a in self._entities:
            dest = self._entities[canonical]
            src = self._entities.pop(slug_b)
            dest.event_ids.extend(src.event_ids)
            dest.pinned_ids.extend(src.pinned_ids)
            dest.episode_ids.extend(src.episode_ids)
            dest.first_seen = min(dest.first_seen, src.first_seen)
            dest.last_seen = max(dest.last_seen, src.last_seen)
            dest.aliases.extend(a for a in src.aliases if a not in dest.aliases)
        return True

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    def auto_link_aliases(self, new_slugs: List[str]) -> None:
        """Heuristically merge co-referent surface names.

        When a compound entity slug (e.g. ``man_in_white``) and a bare
        head-noun slug (``man``) both exist, treat the bare noun as an
        alias of the more specific entity so their timelines merge. This
        resolves the common "chef" vs "man in white" split without any
        model call. Conservative: only links when the head/tail token of
        one slug exactly equals a single-token other slug.
        """
        for slug in new_slugs:
            if not slug or slug not in self._entities:
                continue
            toks = slug.split("_")
            if len(toks) < 2:
                continue
            for cand in (toks[0], toks[-1]):
                if (cand and cand != slug and cand in self._entities
                        and "_" not in cand):
                    # Merge the bare noun into the specific entity.
                    self.link_aliases(slug, cand)

    def render_digest(self, top_k: int = 8, max_aliases: int = 3) -> str:
        """Render a compact entity digest for injection into the prompt.

        Shows the most active entities (by mention count) with their alias
        set, time span and how many episodes they touch. This surfaces the
        entity index — previously built but never shown to the model —
        so it can reason about WHO/WHAT recurs and WHEN.

        Returns "" when there are no entities worth showing.
        """
        if not self._entities:
            return ""
        ranked = sorted(
            self._entities.values(),
            key=lambda s: (len(s.event_ids), s.last_seen),
            reverse=True,
        )[:max(1, int(top_k))]
        lines: List[str] = ["  entities (most active; id: aliases | span | refs):"]
        for sub in ranked:
            aliases = [a for a in sub.aliases if a != sub.entity_id]
            alias_str = (", ".join(aliases[:max_aliases])
                         if aliases else sub.entity_id)
            span = (f"t={sub.first_seen}-{sub.last_seen}s"
                    if sub.last_seen else f"t={sub.first_seen}s")
            refs = f"{len(sub.event_ids)} obs"
            if sub.episode_ids:
                refs += f", {len(sub.episode_ids)} ep"
            lines.append(
                f"    {sub.entity_id}: {alias_str} | {span} | {refs}")
        return "\n".join(lines)

    # ---- internals -----------------------------------------------------
    def _to_slug(self, name: str) -> str:
        toks = self._TOK_RE.findall(name.lower())
        return "_".join(toks)[:48] if toks else ""

    def _resolve(self, slug: str) -> str:
        return self._alias_map.get(slug, slug)


# ---------------------------------------------------------------------------
# Cross-layer evidence path
# ---------------------------------------------------------------------------
@dataclass
class NodeRef:
    layer: str          # "working" | "episodic" | "long_term"
    node_id: int
    t: Optional[int] = None
    text: str = ""
    kind: str = ""
    score: float = 0.0


@dataclass
class EvidencePath:
    """A connected path through the memory DAG."""
    nodes: List[NodeRef] = field(default_factory=list)
    score: float = 0.0
    layers_covered: int = 0


# ---------------------------------------------------------------------------
# MemoryGraph — cross-layer pointer overlay
# ---------------------------------------------------------------------------
class MemoryGraph:
    """Bidirectional pointer index layered on top of :class:`MemoryStore`.

    Maintains three mapping tables:
      * ``_ev_to_ep``: working event id → episode id
      * ``_ep_to_pin``: episode id → pin id
      * ``_pin_to_ep``: pin id → episode id   (reverse)
    plus an :class:`EntityIndex` for entity-scoped traversal.

    All pointer establishment methods are idempotent.
    """

    def __init__(self, store: Any, entity_index: Optional[EntityIndex] = None):
        self.store = store
        self._ev_to_ep: Dict[int, int] = {}
        self._ep_to_pin: Dict[int, int] = {}
        self._pin_to_ep: Dict[int, int] = {}
        self.entity_index = entity_index or EntityIndex()
        # Semantic relation edges between nodes — the piece that turns the
        # flat entity index into an actual graph. Each entry is a dict:
        #   {"src": <id>, "dst": <id>, "relation": str, "t": int,
        #    "kind": "causal"|"entity"}
        # Causal edges chain sealed episodes (A→B→C action order); entity
        # edges record "subject --relation--> object" triples observed in
        # the working layer. Rendered compactly for the model.
        self.relations: List[Dict[str, Any]] = []

    # ---- semantic relation edges ----------------------------------------
    def add_relation(self, src: int, dst: int, relation: str,
                     t: int = 0, kind: str = "causal") -> None:
        """Record a directed semantic edge (idempotent on src,dst,kind)."""
        for r in self.relations:
            if (r.get("src") == src and r.get("dst") == dst
                    and r.get("kind") == kind):
                return
        self.relations.append({
            "src": int(src), "dst": int(dst),
            "relation": str(relation)[:32], "t": int(t), "kind": kind,
        })

    def render_relations(self, max_edges: int = 10) -> str:
        """Render causal/temporal edges as a compact ordered chain digest.

        Surfaces the action-order backbone (ep#A →causes→ ep#B) so ASI-
        style "what happens after X?" questions can read the sequence
        directly instead of re-deriving it. Returns "" when no edges.
        """
        causal = [r for r in self.relations if r.get("kind") == "causal"]
        if not causal:
            return ""
        causal = sorted(causal, key=lambda r: r.get("t", 0))[-max_edges:]
        parts = [f"ep#{r['src']}→ep#{r['dst']}" for r in causal]
        return "  action-order (causal chain): " + "  ".join(parts)

    # ---- pointer establishment ------------------------------------------
    def link_ev_to_ep(self, ev_id: int, ep_id: int) -> None:
        self._ev_to_ep[ev_id] = ep_id

    def link_ep_to_pin(self, ep_id: int, pin_id: int) -> None:
        self._ep_to_pin[ep_id] = pin_id
        self._pin_to_ep[pin_id] = ep_id

    # ---- pointer traversal ----------------------------------------------
    def traverse_up(self, ev_id: int) -> List[NodeRef]:
        """ev → ep → pin  full upward path."""
        path: List[NodeRef] = []
        ep_id = self._ev_to_ep.get(ev_id)
        if ep_id is not None:
            ep = self._find_episode(ep_id)
            if ep:
                path.append(ep)
        pin_id = self._ep_to_pin.get(ep_id) if ep_id is not None else None
        if pin_id is not None:
            pin = self._find_pin(pin_id)
            if pin:
                path.append(pin)
        return path

    def traverse_down(self, pin_id: int) -> List[NodeRef]:
        """pin → ep → evs  downward path."""
        path: List[NodeRef] = []
        ep_id = self._pin_to_ep.get(pin_id)
        if ep_id is not None:
            ep = self._find_episode(ep_id)
            if ep:
                path.append(ep)
        if ep_id is not None:
            ev_ids = self._episode_ev_ids(ep_id)
            for eid in ev_ids[:3]:
                ev = self._find_event(eid)
                if ev:
                    path.append(ev)
        return path

    def get_full_evidence_chain(self, node_id: int,
                                 layer: str) -> EvidencePath:
        """Full bi-directional evidence chain from any layer node."""
        nodes: List[NodeRef] = []
        if layer == "working":
            ev = self._find_event(node_id)
            if ev:
                nodes.append(ev)
            nodes.extend(self.traverse_up(node_id))
        elif layer == "episodic":
            ep = self._find_episode(node_id)
            if ep:
                nodes.append(ep)
            pin_id = self._ep_to_pin.get(node_id)
            if pin_id is not None:
                pin = self._find_pin(pin_id)
                if pin:
                    nodes.append(pin)
            ev_ids = self._episode_ev_ids(node_id)
            for eid in ev_ids[:3]:
                ev = self._find_event(eid)
                if ev:
                    nodes.append(ev)
        elif layer == "long_term":
            pin = self._find_pin(node_id)
            if pin:
                nodes.append(pin)
            nodes.extend(self.traverse_down(node_id))
        return EvidencePath(
            nodes=nodes,
            layers_covered=len({n.layer for n in nodes}),
        )

    # ---- Cross-Layer Backtrack retrieval ---------------------------------
    _RETRIEVE_TOK_RE = re.compile(r"[A-Za-z0-9_一-鿿]+")

    @staticmethod
    def _toks(text: str) -> set:
        return set(MemoryGraph._RETRIEVE_TOK_RE.findall(
            (text or "").lower()))

    @staticmethod
    def _is_context_only_node(node: NodeRef) -> bool:
        text = str(node.text or "").lstrip()
        return (
            (node.layer == "long_term" and node.kind == "question")
            or (node.layer == "working" and node.kind == "note"
                and text.startswith("[plan]"))
        )

    def retrieve_backtrack(
        self,
        query: str,
        top_k: int = 8,
        current_t: Optional[int] = None,
    ) -> List[EvidencePath]:
        """Cross-Layer Backtrack: seed → backtrack → expand.

        Returns ranked :class:`EvidencePath` objects.
        """
        q_toks = self._toks(query)
        if not q_toks:
            return []

        # Step 1: seed — keyword match in all three layers.
        scored_evs: List[Tuple[float, NodeRef]] = []
        scored_eps: List[Tuple[float, NodeRef]] = []
        scored_pins: List[Tuple[float, NodeRef]] = []

        for ev in getattr(self.store, "events", []):
            if getattr(ev, "kind", "") == "note" \
                    and str(getattr(ev, "text", "")).lstrip().startswith("[plan]"):
                continue
            overlap = len(self._toks(ev.text) & q_toks)
            if overlap == 0:
                continue
            sal = float(getattr(ev, "salience", 1.0))
            score = overlap * (0.5 + 0.5 * sal)
            nr = NodeRef(layer="working", node_id=ev.eid, t=ev.t,
                         text=getattr(ev, "text", ""),
                         kind=getattr(ev, "kind", ""), score=score)
            scored_evs.append((score, nr))

        for ep in getattr(self.store, "episodes", []):
            summary = ep.get("summary", "")
            overlap = len(self._toks(summary) & q_toks)
            if overlap == 0:
                continue
            sal = float(ep.get("salience", 1.5))
            score = overlap * (0.5 + 0.5 * sal)
            nr = NodeRef(layer="episodic", node_id=ep["id"],
                         t=ep.get("t0"), text=summary,
                         kind=ep.get("kind", ""), score=score)
            scored_eps.append((score, nr))

        for pin in getattr(self.store, "pinned_facts", []):
            if pin.get("kind") == "question":
                continue
            text = pin.get("text", "")
            overlap = len(self._toks(text) & q_toks)
            if overlap == 0:
                continue
            score = overlap * 1.25  # pinned facts have inherent weight
            nr = NodeRef(layer="long_term", node_id=pin["id"],
                         t=pin.get("t"), text=text,
                         kind=pin.get("kind", ""), score=score)
            scored_pins.append((score, nr))

        # Combine seeds (per-layer quota: long 2, ep 3, working 3).
        seeds: List[NodeRef] = []
        for lst, cap in [(scored_pins, 2), (scored_eps, 3), (scored_evs, 3)]:
            lst.sort(key=lambda x: -x[0])
            for _, nr in lst[:cap]:
                seeds.append(nr)

        # Step 2: backtrack — expand each seed into a full chain.
        paths: List[EvidencePath] = []
        for seed in seeds:
            chain = self.get_full_evidence_chain(seed.node_id, seed.layer)
            # Prepend the seed node if not already the head.
            if not chain.nodes or chain.nodes[0].node_id != seed.node_id:
                chain.nodes.insert(0, seed)
            # Question pins and planner notes are useful prompt context, but
            # they are not visual evidence. Pointer traversal can otherwise
            # re-introduce them even after they were excluded as retrieval
            # seeds, creating a self-confirming query → evidence loop.
            chain.nodes = [
                node for node in chain.nodes
                if not self._is_context_only_node(node)
            ]
            if not chain.nodes:
                continue
            chain.layers_covered = len({
                node.layer for node in chain.nodes
            })
            chain.score = seed.score * (1.0 + 0.2 * (chain.layers_covered - 1))
            paths.append(chain)

        # Step 3: expand — entity-side extension.
        for path in paths:
            for node in path.nodes:
                if node.kind == "observation":
                    ev = self._find_event(node.node_id)
                    if ev is None:
                        continue
                    ev_text = getattr(ev, "text", "")
                    # Use a simple heuristic to find entity mentions.
                    entities_in_ev = self.entity_index.add_observation(
                        node.node_id, node.t or 0, [])  # side-effect for register
                    _ = entities_in_ev  # suppress unused
                    if ev_text:
                        # Extend via entity timeline: find other events
                        # mentioning the same keywords.
                        for eid in getattr(ev, "eid", None) or []:
                            pass  # placeholder — see entity expansion below
                    break  # only expand from the best node

        # Deduplicate and rank.
        seen: Set[Tuple[str, int]] = set()
        unique: List[EvidencePath] = []
        for p in sorted(paths, key=lambda x: -x.score):
            key = (p.nodes[0].layer, p.nodes[0].node_id)
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique[:top_k]

    # ---- helpers --------------------------------------------------------
    def _find_event(self, eid: int) -> Optional[NodeRef]:
        for ev in getattr(self.store, "events", []):
            if ev.eid == eid:
                return NodeRef(layer="working", node_id=ev.eid, t=ev.t,
                               text=getattr(ev, "text", ""),
                               kind=getattr(ev, "kind", ""))
        return None

    def _find_episode(self, epid: int) -> Optional[NodeRef]:
        for ep in getattr(self.store, "episodes", []):
            if ep.get("id") == epid:
                return NodeRef(layer="episodic", node_id=epid,
                               t=ep.get("t0"), text=ep.get("summary", ""),
                               kind=ep.get("kind", ""))
        return None

    def _find_pin(self, pid: int) -> Optional[NodeRef]:
        for pin in getattr(self.store, "pinned_facts", []):
            if pin.get("id") == pid:
                return NodeRef(layer="long_term", node_id=pid,
                               t=pin.get("t"), text=pin.get("text", ""),
                               kind=pin.get("kind", ""))
        return None

    def _episode_ev_ids(self, epid: int) -> List[int]:
        for ep in getattr(self.store, "episodes", []):
            if ep.get("id") == epid:
                return list(ep.get("ev_ids", []))
        return []


# ---------------------------------------------------------------------------
# Online BOCPD segmenter
# ---------------------------------------------------------------------------
@dataclass
class BOCPDSegmenter:
    """Bayesian Online Changepoint Detection for streaming episode boundaries.

    Maintains a posterior over run-length given a stream of
    observation embeddings.  A changepoint is declared when the MAP
    run-length hits 0 — meaning the data just observed was generated
    by a new regime rather than the current run.

    Uses a Gaussian predictive posterior with a constant hazard rate.
    Per-run-length sufficient statistics are maintained incrementally
    and truncated to ``max_run`` to avoid unbounded growth / index
    overflow on long videos.
    """

    hazard_rate: float = 1.0 / 45.0   # expected episode ~45 s
    min_run: int = 3
    max_run: int = 500

    _run_length_probs: List[float] = field(default_factory=list)
    _sums: List[np.ndarray] = field(default_factory=list)
    _sum_sqs: List[np.ndarray] = field(default_factory=list)
    _obs_count: int = 0

    def update(self, x) -> float:
        """Ingest one observation embedding, return changepoint probability.

        Implements Adams & MacKay (2007) BOCPD with conjugate-Gaussian
        sufficient statistics.  ``self._run_length_probs[r]`` is the
        *posterior* P(r_t=r | x_{1:t}); r=0 means a changepoint just
        occurred (x_t is the first observation of a new run).
        """
        if not _HAS_NUMPY:
            self._obs_count += 1
            rl = self._obs_count
            if rl < self.min_run:
                return 0.0
            return float(1.0 / (1.0 + math.exp(-(rl - 30.0) * 0.15)))

        x = np.asarray(x, dtype=np.float64).ravel()
        d = len(x)
        H = self.hazard_rate
        kappa0 = 0.1

        # ---- First observation: run_length = 0 (single obs so far) ----
        if self._obs_count == 0:
            self._run_length_probs = [1.0]
            self._sums = [x.copy()]
            self._sum_sqs = [x * x]
            self._obs_count = 1
            return 0.0

        n = len(self._run_length_probs)
        old = np.asarray(self._run_length_probs, dtype=np.float64)

        # ---- Predictive π(x_t | r_{t-1}=i) for each existing run i ----
        # After i+1 past observations (index i → run length i+1) with sum s,
        # conjugate Gaussian: μ = s/(i+1+κ0), var = 1 + 1/(i+1+κ0).
        log_pred = np.empty(n, dtype=np.float64)
        for i in range(n):
            post_n = (i + 1) + kappa0
            mu = self._sums[i] / post_n
            pred_var = 1.0 + 1.0 / post_n
            resid = x - mu
            log_pred[i] = (-0.5 * float(np.sum(resid * resid) / pred_var)
                           - 0.5 * d * math.log(2 * math.pi * pred_var))

        # ---- Prior predictive π₀(x_t): x_t is first obs of new run ----
        var0 = 1.0 + 1.0 / kappa0
        log_p0 = (-0.5 * float(np.sum(x * x) / var0)
                  - 0.5 * d * math.log(2 * math.pi * var0))

        # ---- Posterior recurrence in log-space (for numerical stability) ----
        #   γ_t(0) ∝ H * π₀(x_t)
        #   γ_t(r) ∝ (1-H) * γ_{t-1}(r-1) * π(x_t | r_{t-1}=r-1)   for r ≥ 1
        #   Marginal Z = H*π₀ + (1-H)*Σ γ_{t-1}(i)*π(x_t|i)
        log_growth = np.log(max(1.0 - H, 1e-300)) + log_pred + np.log(np.maximum(old, 1e-300))
        log_cp = math.log(max(H, 1e-300)) + log_p0

        # log-space vstack and normalize
        log_new = np.concatenate([[log_cp], log_growth])
        log_new_max = float(np.max(log_new))
        new_unnorm = np.exp(log_new - log_new_max)
        Z = float(new_unnorm.sum())
        new_probs = new_unnorm / Z

        # ---- Sufficient statistics ----
        # r=0: sum = x; r≥1: sum[r] = old_sums[r-1] + x.
        new_sums = [x.copy()] + [self._sums[i] + x for i in range(n)]
        new_sum_sqs = [x * x] + [self._sum_sqs[i] + x * x for i in range(n)]

        # ---- Truncate to max_run (drop longest run-lengths, high indices) ----
        if len(new_probs) > self.max_run:
            new_probs = new_probs[:self.max_run]
            new_sums = new_sums[:self.max_run]
            new_sum_sqs = new_sum_sqs[:self.max_run]
            s = new_probs.sum()
            if s > 0:
                new_probs = new_probs / s

        self._run_length_probs = new_probs.tolist()
        self._sums = new_sums
        self._sum_sqs = new_sum_sqs
        self._obs_count += 1

        return float(new_probs[0])


    def should_seal(self) -> bool:
        """MAP run-length == 0 → close the current episode."""
        if not self._run_length_probs:
            return False
        rl = len(self._run_length_probs)
        if rl < self.min_run:
            return False
        if not _HAS_NUMPY:
            # Pure-Python: seal if run_length exceeds expected episode span.
            return rl > 45
        map_idx = int(np.argmax(self._run_length_probs))
        return map_idx == 0

    def should_seal_with_content_check(
        self,
        current_text: str = "",
        run_first_text: str = "",
    ) -> bool:
        """Statistical changepoint + content continuity check.

        A BOCPD changepoint may fire on a minor statistical shift
        (lighting change, camera jitter) that does NOT represent a
        real semantic boundary.  This method adds a content-level
        gate: if the current frame and the run's first frame share
        the same scene AND no actor entered/exited, we suppress the
        seal — the "change" is within the same continuous scene.

        Inspired by NVIDIA VSS's multi-pass temporal consolidation
        that merges adjacent segments when they depict a "continuous
        scene".
        """
        if not self.should_seal():
            return False
        # Content continuity: extract scene and actor fields.
        if current_text and run_first_text:
            cur_scene = _extract_field(current_text, "scene")
            first_scene = _extract_field(run_first_text, "scene")
            cur_actors = _extract_field(current_text, "actor")
            first_actors = _extract_field(run_first_text, "actor")
            # If scene is the same AND actors haven't changed
            # (no entry/exit), this is likely the same continuous
            # scene — suppress the seal.
            if (cur_scene and first_scene
                    and cur_scene.lower() == first_scene.lower()
                    and cur_actors and first_actors
                    and _actor_count(cur_actors) == _actor_count(first_actors)):
                return False
        return True

    def reset(self) -> None:
        self._run_length_probs = []
        self._sums = []
        self._sum_sqs = []
        self._obs_count = 0


def _extract_field(text: str, field: str) -> str:
    """Best-effort extraction of a labelled field from observation text.

    Handles bullet-prefixed formats (``• field: value``) as well as
    plain ``field: value`` lines.
    """
    # Strip bullet markers before matching.
    clean = re.sub(r'^[•\-*]\s*', '', text, flags=re.MULTILINE)
    m = re.search(
        rf'{field}s?\s*[:=]\s*(.+?)(?:\n|;|$)', clean, re.IGNORECASE | re.S)
    return m.group(1).strip() if m else ""


def _actor_count(actor_text: str) -> int:
    """Crude actor count: number of commas + 1, clamped."""
    return max(1, actor_text.count(",") + actor_text.count(";") + 1)


# ---------------------------------------------------------------------------
# Embedding helper for BOCPD
# ---------------------------------------------------------------------------
_OBS_FIELD_KEYS = (
    "description", "actor", "object", "action", "ocr", "scene", "delta",
)

_OBS_TOK_RE = re.compile(r"[A-Za-z0-9_一-鿿]{2,}")


def observation_to_embedding(text: str, dim: int = 512):
    """Convert an observation text string into a sparse embedding vector
    by hashing field tokens into a fixed-dimension vector.

    The resulting embedding is suitable as input to
    :class:`BOCPDSegmenter`.
    """
    if _HAS_NUMPY:
        vec = np.zeros(dim, dtype=np.float64)
    else:
        vec = None
    tokens = list(_OBS_TOK_RE.finditer(text.lower())) if text else []
    if not tokens:
        return np.zeros(dim) if _HAS_NUMPY else [0.0] * dim

    if _HAS_NUMPY:
        for m in tokens:
            tok = m.group(0)
            idx = hash(tok) % dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-8)
    else:
        vec_list = [0.0] * dim
        for m in tokens:
            tok = m.group(0)
            idx = hash(tok) % dim
            vec_list[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec_list)) or 1e-8
        return [v / norm for v in vec_list]


# ---------------------------------------------------------------------------
# 6 Formal Graph Operations
# ---------------------------------------------------------------------------

def op_add_obs_node(
    store: Any,
    graph: MemoryGraph,
    t: int,
    text: str,
    delta: Optional[str],
    entities: List[str],
    novelty: float,
) -> int:
    """Add an observation node to working layer + entity index.

    Returns the new event ID.
    """
    is_change = novelty >= 0.3
    salience = (1.0 + 0.5 * novelty) if is_change else 0.5
    prefix = f"[nov={novelty:.2f}] " if is_change else f"[nov={novelty:.2f}|static] "

    ev = store.write_event(
        t=t, kind="observation", text=prefix + text,
        ref_frame=t, salience=salience,
        delta=(delta or "")[:240] if delta else None,
        is_change=is_change,
    )
    if entities:
        graph.entity_index.add_observation(ev.eid, t, entities)
    return ev.eid


def op_add_epi_node(
    store: Any,
    graph: MemoryGraph,
    t0: int,
    t1: int,
    summary: str,
    ev_ids: List[int],
    entities: Optional[List[str]] = None,
) -> int:
    """Add an episode node, wire all cross-layer pointers."""
    ep = store.write_episode(
        t0=t0, t1=t1, summary=summary,
        kind="auto", ev_ids=list(ev_ids),
        salience=1.5 + 0.05 * len(ev_ids),
    )
    ep_id = ep["id"]
    for eid in ev_ids:
        graph.link_ev_to_ep(eid, ep_id)
    if entities:
        graph.entity_index.add_episode(ep_id, entities)
    return ep_id


def op_add_pin_node(
    store: Any,
    graph: MemoryGraph,
    text: str,
    kind: str,
    t: int,
    ep_id: int,
    subject: str,
    entities: Optional[List[str]] = None,
) -> int:
    """Add a long-term pin node and back-link to its episode."""
    item = store.pin_fact(
        text=text, kind=kind, t=t,
        ep_id=ep_id, subject=subject,
    )
    pin_id = item["id"]
    graph.link_ep_to_pin(ep_id, pin_id)
    if entities:
        graph.entity_index.add_pin(pin_id, entities)
    return pin_id


def op_correct_node(
    store: Any,
    graph: MemoryGraph,
    node_id: int,
    layer: str,
    corrected_text: str,
    current_t: int,
) -> int:
    """Create a correction node. The original node is NOT deleted — it is
    marked as superseded. Cascades to affected episodes.

    Returns the correction event ID.
    """
    # Mark original as superseded.
    if layer == "working":
        for ev in getattr(store, "events", []):
            if ev.eid == node_id:
                ev_text = getattr(ev, "text", "") or ""
                setattr(ev, "text",
                        f"[SUPERSEDED by correction at t={current_t}] {ev_text}")
                setattr(ev, "salience", 0.1)
                setattr(ev, "is_change", False)
                break
    elif layer == "long_term":
        store.unpin_fact(node_id)

    # Write correction event.
    correction = store.write_event(
        t=current_t, kind="correction",
        text=f"CORRECTED [{layer}#{node_id}]: {corrected_text}",
        ref_frame=current_t, salience=3.0, is_change=True,
    )
    # Cascade: mark affected episodes.
    for ep in getattr(store, "episodes", []):
        if node_id in ep.get("ev_ids", []):
            ep["needs_revision"] = True
    return correction.eid


def op_add_entity_edge(
    graph: MemoryGraph,
    ev_a: int,
    ev_b: int,
    entity_id: str,
) -> None:
    """Link two working events that reference the same entity."""
    sub = graph.entity_index.get_timeline(entity_id)
    if sub is not None:
        if ev_a not in sub.event_ids:
            sub.event_ids.append(ev_a)
        if ev_b not in sub.event_ids:
            sub.event_ids.append(ev_b)


def op_add_causal_edge(
    graph: MemoryGraph,
    ev_a: int,
    ev_b: int,
    relation: str = "causes",
) -> None:
    """Record a directed causal/temporal edge ev_a → ev_b in the graph.

    Used to chain consecutive episodes (action A precedes/causes action B)
    so ASI-style "what happens after X?" questions can read the ordered
    chain directly. Stored in ``graph.relations`` and rendered compactly
    in the entity/relation digest.
    """
    graph.add_relation(ev_a, ev_b, relation=relation, kind="causal")
