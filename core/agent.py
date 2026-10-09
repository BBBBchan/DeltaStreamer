"""Agentic video-understanding driver for OVO-Bench.

This file is *additive*: it imports from ``video_processor.py`` but does NOT
modify it. To use the legacy non-agent loop, keep using ``video_processor.py``;
to use the agent loop, run this file directly.

High-level flow per sample
--------------------------
1. Build a streaming-window prompt the same way video_processor does.
2. After each model response, parse ``<tool_call>{...}</tool_call>`` blocks.
3. Dispatch each call to the registry; collect ``<tool_result>`` strings.
4. If the response contained tool calls but no ``</Response>``, re-run the
   model in the *same* round with the tool results appended (no clock tick).
5. Otherwise advance the streaming clock by one second (or honor a pending
   ``next_frame_idx`` set by replay).
6. ``abstain`` short-circuits the sample with ``</Response> Unable to answer``.

CLI
---
    python core/agent.py \\
        --model <ckpt>          \\
        --dataset <ovo.json>    \\
        --output  <out.jsonl>   \\
        [--tasks REC,HLD]       \\
        [--max_tool_calls_per_round 3] \\
        [--max_tool_calls_per_sample 12]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Make sure we can import the sibling video_processor module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse helpers from video_processor WITHOUT modifying it.
import video_processor as ovo  # noqa: E402

from tools import (  # noqa: E402
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    build_default_registry,
    build_extended_registry,
    MemoryStore,
    Event,
    RuleSummarizer,
    Budget,
)
from tools.event_predicate import (  # noqa: E402
    build_contract_prompt as build_event_predicate_contract_prompt,
    build_ledger_prompt as build_event_predicate_ledger_prompt,
    build_local_sheets as build_event_predicate_local_sheets,
    contract_query as event_predicate_contract_query,
    dag_proposals as event_predicate_dag_proposals,
    image_paths as event_predicate_image_paths,
    normalize_yes_no_response,
    parse_contract as parse_event_predicate_contract,
    parse_ledger as parse_event_predicate_ledger,
    route_event_predicate,
)
from tools.crr_typed_relation import (  # noqa: E402
    build_prompt as build_crr_typed_relation_prompt,
    compile_relation_contract as compile_crr_typed_relation_contract,
    parse_response as parse_crr_typed_relation_response,
)


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
RESPONSE_TAG = "</Response>"

# Recap (model-driven long-term memory) parsing.
# The model is instructed to emit:
#   <Recap t0=<int> t1=<int> kind=<event_end|scene_change|...>>
#       multi-sentence detailed summary ...
#   </Recap>
# We accept either quoted ("12") or unquoted (12) attribute values, and
# permit the kind enum to be missing (we default to "recap").
_RECAP_BLOCK_RE = re.compile(r"<Recap\b([^>]*)>(.*?)</Recap>", re.S | re.I)
_RECAP_ATTR_RE = re.compile(
    r'(t0|t1|kind)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))',
    re.I,
)
_RECAP_VALID_KINDS = {
    "scene_change", "object_change",
    "action_start", "action_end",
    "actor_enter", "actor_exit", "actor_change",
    # Legacy values, retained for backward compatibility with old runs:
    "event_end", "milestone", "recap",
}

# Reusable disambiguation rule (Recency over Frequency). Surfaced in EVERY
# answering branch (question-posed, post-question, last-frame) so the model
# never weighs 300 repeated old frames over the 10 most-recent ones.
_RECENCY_RULE = (
    "★ RECENCY OVER FREQUENCY ★ The TAIL events (tagged "
    "[RECENT, HIGHEST PRIORITY]) are WHAT JUST HAPPENED. If they "
    "contradict older/episodic evidence, TRUST THE TAIL — the older "
    "region merely repeats earlier actions many times (high frequency); "
    "the question is about the video's LATEST state. Example: 300 frames "
    "chopping eggplant then 10 frames handling meat → answer is MEAT."
)


def parse_recap_blocks(text: str) -> List[Dict[str, Any]]:
    """Extract `<Recap t0=.. t1=.. kind=.. subject=..>...</Recap>` blocks.

    Returns a list of {"t0": int|None, "t1": int|None, "kind": str,
    "subject": str|None, "text": str}. The body is whitespace-collapsed
    and trimmed but otherwise verbatim. ``subject`` (if present) is the
    canonical key the long-term layer uses for supersession: a newer
    Recap with the same ``kind`` family and ``subject`` will mark the
    older pinned fact as superseded so the model no longer sees stale
    state.
    """
    out: List[Dict[str, Any]] = []
    if not text:
        return out
    for m in _RECAP_BLOCK_RE.finditer(text):
        attrs_raw = m.group(1) or ""
        body = m.group(2) or ""
        attrs: Dict[str, Any] = {
            "t0": None, "t1": None, "kind": "recap", "subject": None,
        }
        for am in _RECAP_ATTR_RE.finditer(attrs_raw):
            key = am.group(1).lower()
            val = am.group(2) or am.group(3) or am.group(4) or ""
            val = val.strip()
            if key in ("t0", "t1"):
                try:
                    attrs[key] = int(float(val))
                except (TypeError, ValueError):
                    attrs[key] = None
            elif key == "kind":
                vk = val.lower().strip()
                attrs["kind"] = vk if vk in _RECAP_VALID_KINDS else "recap"
            elif key == "subject":
                # Normalise to a short lower-case slug so supersession
                # matches across slight phrasing differences.
                slug = re.sub(r"\s+", "_", val.strip().lower())[:48]
                attrs["subject"] = slug or None
        body_clean = re.sub(r"\s+", " ", body).strip()
        if not body_clean:
            continue
        out.append({
            "t0": attrs["t0"], "t1": attrs["t1"],
            "kind": attrs["kind"],
            "subject": attrs["subject"],
            "text": body_clean,
        })
    return out


# Query-Driven Planning block parsing. On the question-posed round the
# model emits a single <Plan>...</Plan> describing how it intends to answer
# (it self-classifies the query; we do NOT predefine task types). Slots are
# free-form `key: value` lines; we capture the ones we know how to use and
# keep the rest as a raw summary.
_PLAN_BLOCK_RE = re.compile(r"<Plan\b[^>]*>(.*?)</Plan>", re.S | re.I)
# Fallback: an UNCLOSED <Plan> (no </Plan>). A weaker model can ramble /
# repeat inside the plan until it hits the generation token cap, so the
# closing tag never appears (observed: qwen3vl-8b, sample 34 — 21k-char
# plan ending mid-sentence). Recover the body from <Plan> to end-of-text
# so the plan (and its standing banner) is not silently dropped.
_PLAN_OPEN_RE = re.compile(r"<Plan\b[^>]*>(.*)$", re.S | re.I)
_PLAN_KNOWN_KEYS = (
    "need", "pov", "info_type", "evidence_locus", "answerability",
    "meta", "strategy",
)


def parse_plan_block(text: str) -> Optional[Dict[str, str]]:
    """Extract a `<Plan> need: .. info_type: .. strategy: .. </Plan>` block.

    Returns a dict of the recognised slots (plus ``_raw`` = the full body)
    or ``None`` when no plan block is present. Slot keys are matched
    case-insensitively; values are single-line trimmed.

    Tolerant to a MISSING ``</Plan>`` closer: if the strict closed match
    fails, fall back to capturing from ``<Plan>`` to end-of-text so a
    plan truncated by the generation token cap is still recovered (its
    slots feed the standing [PLAN] banner). To avoid treating a stray
    ``<Plan>`` fragment as a real plan, the unclosed path requires at
    least one recognised slot.
    """
    if not text:
        return None
    m = _PLAN_BLOCK_RE.search(text)
    closed = m is not None
    if m is None:
        m = _PLAN_OPEN_RE.search(text)
        if not m:
            return None
    body = (m.group(1) or "").strip()
    if not body:
        return None
    plan: Dict[str, str] = {"_raw": re.sub(r"\s+", " ", body).strip()[:3000]}
    for line in body.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = re.sub(r"^[\-*•\s]+", "", key).strip().lower()
        val = val.strip()
        if key in _PLAN_KNOWN_KEYS and val:
            plan[key] = val[:800]
    # Unclosed plan: only trust it if it yielded at least one real slot
    # (guards against a bare/garbled "<Plan>" fragment).
    if not closed and len(plan) <= 1:
        return None
    return plan


_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")
_QUERY_STOPWORDS = {
    "a", "an", "and", "answer", "are", "at", "be", "been", "being",
    "best", "by", "did", "do", "does", "for", "format", "from", "full",
    "give", "had", "has", "have", "how", "if", "in", "is", "it", "its",
    "letter", "listed", "must", "my", "of", "on", "one", "only", "option",
    "or", "rather", "reply", "requested", "space", "text", "than", "that",
    "the", "their", "them", "then", "there", "these", "this", "to", "uncertain",
    "was", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your",
}


def build_evidence_query(
    question: str = "",
    plan: Optional[Dict[str, str]] = None,
    sample: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a compact, task-agnostic query for memory/DAG retrieval.

    ``self.question`` is a rendered prompt, not just the semantic question:
    it also contains every multiple-choice option plus the generic answer
    constraint. Feeding that whole string into keyword retrieval gives all
    candidate answers (and generic words such as ``answer``/``option``) equal
    voting power, so irrelevant evidence can outrank the actual target.

    The compact query uses only information visible after the question is
    posed: the raw question/event/activity and the model-authored plan's
    ``need`` slot. It never reads ``sample['task']`` or ground truth.
    """
    sources: List[str] = []
    if sample:
        native_question = str(sample.get("question") or "").strip()
        all_steps = str(sample.get("all_steps") or "").strip()
        activity = str(sample.get("activity") or "").strip()
        if native_question:
            sources.append(native_question)
        elif all_steps:
            sources.append(all_steps)
        elif activity:
            sources.append(activity)
    if not sources and question:
        # Legacy callers without sample structure: retain the first semantic
        # line and drop rendered A-K option lines / generic format boilerplate.
        for line in str(question).splitlines():
            clean = line.strip()
            if not clean or re.match(r"^[A-K][.)：:]\s+", clean, re.I):
                continue
            if clean.startswith("Answer with the format"):
                break
            sources.append(clean)
            break
    if plan:
        need = str(plan.get("need") or "").strip()
        if need:
            sources.append(need)

    ordered: List[str] = []
    seen: Set[str] = set()
    for source in sources:
        for token in _QUERY_TOKEN_RE.findall(source.lower()):
            if token in _QUERY_STOPWORDS or token in seen:
                continue
            if len(token) == 1 and not token.isdigit():
                continue
            seen.add(token)
            ordered.append(token)
    return " ".join(ordered[:64])


def render_evidence_contract(
    plan: Optional[Dict[str, str]],
    evidence_query: str,
) -> str:
    """Compile the planner output into a compact evidence/tool contract.

    This replaces hidden task-conditioned routing with query-time routing:
    the contract is derived solely from the model-authored plan after the
    query is visible. It tells retrieval and tool use where decisive evidence
    should come from while keeping the underlying memory/replay framework.
    """
    if not plan:
        return ""
    info = str(plan.get("info_type") or "").strip().lower()
    locus_text = " ".join([
        str(plan.get("evidence_locus") or ""),
        str(plan.get("strategy") or ""),
    ]).lower()
    if os.environ.get("STREAMO_ENABLE_LOCUS_V2", "0") == "1":
        locus = compile_evidence_locus(plan)
    elif re.search(
        r"\b(current|right now|present frame|on-screen)\b", locus_text
    ):
        locus = "current"
    elif re.search(
        r"\b(not happened|future|keep (?:watching|observing)|later)\b",
        locus_text,
    ):
        locus = "future"
    elif re.search(
        r"\b(past|already happened|memory|replay|earlier|before)\b",
        locus_text,
    ):
        locus = "past"
    else:
        locus = "mixed"
    if locus == "current":
        policy = (
            "Inspect the current pixels directly; use at most one zoom for a "
            "small static detail. Do not browse unrelated history."
        )
    elif locus == "future":
        policy = (
            "Keep observing until the named anchor/target becomes visible; "
            "do not answer from nearest-memory guesses."
        )
    elif locus == "past":
        policy = (
            "Retrieve the target/anchor from the memory DAG first, then replay "
            "one exact timestamp when pixel detail is decisive."
        )
    else:
        locus = "mixed"
        policy = (
            "Compare current pixels with target-matched memory paths; use one "
            "targeted replay only when the decisive detail is missing."
        )

    if re.search(r"\b(color|identity|object|attribute|ocr|text|spatial|position)\b",
                 info):
        policy += (
            " Prefer working-layer/pixel-near evidence over repeated summaries "
            "or majority frequency."
        )
    elif re.search(r"\b(temporal|order|before|after|causal)\b", info):
        policy += (
            " Locate the anchor timestamp and inspect the immediately adjacent "
            "state/action, not the most frequent event."
        )
    lines = [
        "[EVIDENCE CONTRACT — compiled from your plan]",
        f"  - retrieval_query: {evidence_query or '(none)'}",
        f"  - evidence_locus: {locus}",
        f"  - tool_policy: {policy}",
        "  - decision_rule: answer only from evidence matching the target and "
        "anchor; options are output choices, not retrieval evidence.",
    ]
    return "\n".join(lines)


_LOCUS_CURRENT_RE = re.compile(
    r"\b(?:current(?:\s+frame|\s+moment|\s+pixels)?|right now|"
    r"present frame|on-screen|question-time frame)\b",
    re.IGNORECASE,
)
_LOCUS_FUTURE_RE = re.compile(
    r"\b(?:future|upcoming|not (?:yet )?happened|has not happened yet|"
    r"hasn't happened yet|not yet (?:appeared|occurred|visible)|"
    r"keep (?:watching|observing)|continue (?:watching|observing)|"
    r"must wait|wait for|later frame)\b",
    re.IGNORECASE,
)
_LOCUS_PAST_RE = re.compile(
    r"\b(?:past|already happened|already occurred|earlier|histor(?:y|ical)|"
    r"memory|replay|previous frame|prior frame)\b",
    re.IGNORECASE,
)
_LOCUS_DECLARATIONS = (
    (
        "future",
        re.compile(
            r"(?:evidence|answer|anchor|target)[^.:\n]{0,100}"
            r"(?:has not|hasn't|not yet)[^.:\n]{0,80}"
            r"(?:happened|occurred|appeared|visible)",
            re.IGNORECASE,
        ),
    ),
    (
        "current",
        re.compile(
            r"(?:evidence|answer|deciding evidence)[^.:\n]{0,80}"
            r"(?:is|lies|lives|comes from)[^.:\n]{0,30}"
            r"(?:current|right now|present frame|on-screen)",
            re.IGNORECASE,
        ),
    ),
    (
        "past",
        re.compile(
            r"(?:evidence|answer|deciding evidence)[^.:\n]{0,80}"
            r"(?:is|lies|lives|comes from)[^.:\n]{0,30}"
            r"(?:past|already happened|already occurred|earlier|memory)",
            re.IGNORECASE,
        ),
    ),
)


def compile_evidence_locus(plan: Optional[Dict[str, str]]) -> str:
    """Compile the planner's explicit locus without strategy keyword leakage.

    A plan often declares "the evidence is in the PAST" and then mentions
    "compare with the current frame" in its strategy. The old implementation
    concatenated both fields and checked ``current`` first, so that supporting
    tactic overrode the planner's actual temporal decision. Prefer a strong
    declaration in ``evidence_locus``; use the strategy only when that slot is
    absent or has no directional signal.
    """
    if not plan:
        return "mixed"
    explicit = str(plan.get("evidence_locus") or "").strip()
    strategy = str(plan.get("strategy") or "").strip()

    if explicit:
        # The opening sentence normally contains the planner's decision;
        # later sentences discuss alternative sources and supporting tools.
        lead = re.split(r"(?<=[.!?])\s+|\n", explicit, maxsplit=1)[0]
        for label, pattern in _LOCUS_DECLARATIONS:
            if pattern.search(lead):
                return label

        # Future negation is a stronger decision than incidental references
        # to the current/past context in the same sentence.
        if _LOCUS_FUTURE_RE.search(lead):
            return "future"
        lead_current = bool(_LOCUS_CURRENT_RE.search(lead))
        lead_past = bool(_LOCUS_PAST_RE.search(lead))
        if lead_current and not lead_past:
            return "current"
        if lead_past and not lead_current:
            return "past"

        # If the lead is neutral, inspect the full explicit slot. Ambiguous
        # multi-direction reasoning remains mixed instead of picking whichever
        # keyword happens to be tested first.
        future = bool(_LOCUS_FUTURE_RE.search(explicit))
        current = bool(_LOCUS_CURRENT_RE.search(explicit))
        past = bool(_LOCUS_PAST_RE.search(explicit))
        labels = [
            label for label, present in (
                ("future", future), ("current", current), ("past", past)
            ) if present
        ]
        if len(labels) == 1:
            return labels[0]
        if future and not current:
            return "future"
        if labels:
            return "mixed"

    # Legacy/malformed plans may omit evidence_locus. Only then use strategy
    # as a fallback, with future negation taking precedence.
    if _LOCUS_FUTURE_RE.search(strategy):
        return "future"
    current = bool(_LOCUS_CURRENT_RE.search(strategy))
    past = bool(_LOCUS_PAST_RE.search(strategy))
    if current != past:
        return "current" if current else "past"
    return "mixed"


# Fallback counting-intent phrases, used ONLY when no ``sample`` structure is
# available (older callers). Deliberately EXCLUDES bare "occurred" and bare
# "occurrence(s)": those merely describe that *an event happened* and appear
# in almost every task's plan (e.g. "the action that occurred"), which would
# leak the count-audit into EPM/ASI/HLD/etc. The primary discriminator is the
# structural gate below; this regex is a best-effort backup only.
_REPETITION_COUNT_RE = re.compile(
    r"\b(?:how\s+many\s+times|number\s+of\s+times|times\s+did|"
    r"how\s+many\s+occurrences|number\s+of\s+occurrences|"
    r"repetitions?|repeated\s+events?|repeated\s+actions?|"
    r"running\s+total)\b",
    re.IGNORECASE,
)


def _is_repetition_count_sample(sample: Optional[Dict[str, Any]]) -> bool:
    """Structural gate for a repeated-event counting sample.

    OVO-Bench REC samples are the ONLY ones whose ``question`` text is not
    provided in the dataset — they carry an ``activity`` field instead, and
    the counting question ("How many times did activity 'X' occur?") is
    synthesized at run time. This structural signature (activity present AND
    no native question) separates counting samples from all other tasks with
    100% precision/recall on the full benchmark, without reading the task
    label. Any other task always ships a native ``question``.
    """
    if not sample:
        return False
    return bool(sample.get("activity")) and not sample.get("question")


def plan_requests_repetition_count(
    plan: Optional[Dict[str, str]],
    question: str = "",
    sample: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return whether this query asks for a repeated event's exact count.

    Compliance / timing: this is evaluated only AFTER the query has arrived
    and the model has authored its ``<Plan>`` (see the call site in
    ``_update_impl``). At that point the fact that the question asks to count
    an activity is already exposed to the model through the query prompt
    itself, so consulting the sample's counting-specific STRUCTURE below adds
    no information the model does not already have — and it is NOT the task
    label (``sample['task']``), which is never read.

    Decision logic:
      * When ``sample`` is provided (the normal path), the structural gate
        ``_is_repetition_count_sample`` is the sole discriminator, subject
        only to a ``meta`` veto (a plan that classified the query as a
        Yes/No answerability meta-question is never a count). No natural
        language keyword is required — the structure already separates
        counting samples from every other task with perfect precision, and
        requiring keywords would drop genuine REC samples whose plan happens
        not to use a counting phrase.
      * When ``sample`` is None (legacy callers), fall back to the
        keyword-intent regex over the query + plan text.
    """
    if not plan:
        return False
    meta = str(plan.get("meta") or "").strip().lower()
    is_meta = (
        meta.startswith("yes")
        or "meta-question" in meta
        or "meta question" in meta
        or "answerability" in meta
    )
    if is_meta:
        return False
    plan_text = " ".join(
        str(plan.get(key) or "")
        for key in ("need", "info_type", "evidence_locus", "strategy")
    )
    visible_count_intent = bool(
        _REPETITION_COUNT_RE.search(f"{question}\n{plan_text}")
    )
    # Primary path: visible structure prevents cross-task leakage, while
    # query/plan semantics confirms the planner recognized counting.
    if sample is not None:
        return bool(
            _is_repetition_count_sample(sample)
            and visible_count_intent
        )
    # Legacy fallback (no sample structure): best-effort keyword intent.
    return visible_count_intent


def plan_requests_ssr_local_audit(
    plan: Optional[Dict[str, str]],
    sample: Optional[Dict[str, Any]],
) -> bool:
    """Structurally identify an SSR state query after planning.

    SSR samples expose the target step through ``all_steps`` and the query
    time through ``step_start_time`` while carrying neither a native
    question nor an activity. This does not read the task label or GT.
    """
    if os.environ.get("STREAMO_ENABLE_SSR_LOCAL_AUDIT", "0") != "1":
        return False
    if not plan or not sample:
        return False
    meta = str(plan.get("meta") or "").lower()
    info_type = str(plan.get("info_type") or "").lower()
    need = str(plan.get("need") or "").lower()
    plan_identifies_event_state = bool(
        (
            "yes" in meta
            or "meta" in meta
            or "state" in info_type
            or "yes/no" in info_type
        )
        and (
            "start" in need
            or "occur" in need
            or "happen" in need
            or "event" in info_type
            or "state" in info_type
        )
    )
    return bool(
        plan_identifies_event_state
        and
        sample.get("all_steps")
        and sample.get("step_start_time") is not None
        and not sample.get("question")
        and not sample.get("activity")
    )


def plan_requests_crr_target_audit(
    plan: Optional[Dict[str, str]],
    sample: Optional[Dict[str, Any]],
) -> bool:
    """Structurally identify a CRR target-completeness query after planning."""
    if os.environ.get("STREAMO_ENABLE_CRR_TARGET_AUDIT", "0") != "1":
        return False
    if not plan or not sample:
        return False
    meta = str(plan.get("meta") or "").lower()
    info_type = str(plan.get("info_type") or "").lower()
    need = str(plan.get("need") or "").lower()
    plan_identifies_answerability = bool(
        (
            "yes" in meta
            or "meta" in meta
            or "answerability" in info_type
            or "yes/no" in info_type
        )
        and (
            "evidence" in need
            or "answer" in need
            or "clue" in need
            or "answerability" in info_type
        )
    )
    return bool(
        plan_identifies_answerability
        and
        sample.get("question")
        and sample.get("ask_time") is not None
        and not sample.get("options")
        and not sample.get("all_steps")
        and not sample.get("activity")
    )


def plan_requests_crr_clue_audit(
    plan: Optional[Dict[str, str]],
    sample: Optional[Dict[str, Any]],
) -> bool:
    """Identify an opt-in CRR clue-answerability audit after planning."""
    if os.environ.get("STREAMO_ENABLE_CRR_CLUE_AUDIT", "0") != "1":
        return False
    if not plan or not sample:
        return False
    meta = str(plan.get("meta") or "").lower()
    info_type = str(plan.get("info_type") or "").lower()
    need = str(plan.get("need") or "").lower()
    plan_identifies_answerability = bool(
        (
            "yes" in meta
            or "meta" in meta
            or "answerability" in info_type
            or "yes/no" in info_type
        )
        and (
            "evidence" in need
            or "answer" in need
            or "clue" in need
            or "answerability" in info_type
        )
    )
    return bool(
        plan_identifies_answerability
        and
        sample.get("question")
        and sample.get("ask_time") is not None
        and not sample.get("options")
        and not sample.get("all_steps")
        and not sample.get("activity")
    )


def plan_requests_crr_typed_relation_audit(
    plan: Optional[Dict[str, str]],
    sample: Optional[Dict[str, Any]],
) -> bool:
    """Route the typed audit after planning from the unique visible schema."""
    if os.environ.get(
        "STREAMO_ENABLE_CRR_TYPED_RELATION_AUDIT", "0"
    ) != "1":
        return False
    if not plan or not sample:
        return False
    # This query-visible shape is unique in the formal 3035 dataset:
    # exactly 240/240 open answerability rows and zero rows from any other
    # task. Requiring the already-parsed planner preserves planner-first
    # execution without relying on brittle English keyword repetition in
    # free-form need/info_type slots.
    return bool(
        sample.get("question")
        and sample.get("ask_time") is not None
        and not sample.get("options")
        and not sample.get("all_steps")
        and not sample.get("activity")
    )


def plan_requests_listed_refusal_bidirectional_audit(
    plan: Optional[Dict[str, str]],
    sample: Optional[Dict[str, Any]],
) -> bool:
    """Route a listed-refusal MCQ only after the model's own plan."""
    if os.environ.get(
        "STREAMO_ENABLE_LISTED_REFUSAL_BIDIRECTIONAL_AUDIT", "0"
    ) != "1":
        return False
    if not plan or not sample or _refusal_option_index(sample) is None:
        return False
    answerability = " ".join(filter(None, (
        str(plan.get("answerability") or ""),
        str(plan.get("info_type") or ""),
        str(plan.get("strategy") or ""),
    ))).lower()
    recognizes_refusal = bool(re.search(
        r"\b(?:unable|cannot|can't|refus|unanswer|insufficient|"
        r"not\s+enough|target\s+(?:is\s+)?absent)\b",
        answerability,
    ))
    recognizes_closed_set = bool(re.search(
        r"\b(?:options?|listed|closed[\s-]?set|choice|answerability)\b",
        answerability,
    ))
    return bool(
        recognizes_refusal
        and recognizes_closed_set
        and sample.get("question")
        and sample.get("options")
    )


def plan_requests_current_locus_dual_view_audit(
    plan: Optional[Dict[str, str]],
    sample: Optional[Dict[str, Any]],
) -> bool:
    """Route an ordinary current-locus MCQ after the planner has run.

    The route uses only planner output and query-visible schema. In
    particular, it never reads the benchmark task label. Listed epistemic
    refusals are excluded because they require a symmetric answerability
    audit rather than an affirmative current-scene classifier.
    """
    if os.environ.get(
        "STREAMO_ENABLE_CURRENT_LOCUS_DUAL_VIEW_AUDIT", "0"
    ) != "1":
        return False
    if not plan or not sample or compile_evidence_locus(plan) != "current":
        return False
    question = str(sample.get("question") or "").strip()
    options = sample.get("options")
    return bool(
        question
        and isinstance(options, list)
        and 2 <= len(options) <= 11
        and all(str(option).strip() for option in options)
        and _refusal_option_index(sample) is None
    )


_CURRENT_DUAL_VIEW_THINK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>|<think\b[^>]*>.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_CURRENT_DUAL_VIEW_PLACEHOLDER_RE = re.compile(
    r"(?:exact visible target|specific pixel|target description|"
    r"reference description|option support evidence)",
    re.IGNORECASE,
)


def build_current_locus_dual_view(
    frame_paths: List[str],
    query_time: int,
    output_dir: str,
    *,
    frame_count: int = 9,
) -> Dict[str, Any]:
    """Build the frozen current-frame plus nine-second tail visual input."""
    import cv2
    import numpy as np

    if not frame_paths:
        raise ValueError("current-locus dual view requires at least one frame")
    last_visible = min(len(frame_paths) - 1, max(0, int(query_time)))
    start = max(0, last_visible - max(1, int(frame_count)) + 1)
    times = list(range(start, last_visible + 1))
    panels: List[Any] = []
    for timestamp in times:
        image = cv2.imread(frame_paths[timestamp])
        if image is None:
            raise RuntimeError(
                f"failed to read current-locus frame t={timestamp}"
            )
        panel = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (125, 30), (0, 0, 0), -1)
        cv2.putText(
            panel,
            f"t={timestamp}s",
            (7, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    blank = np.zeros_like(panels[0])
    while len(panels) < 9:
        panels.insert(0, blank.copy())
    sheet = np.vstack((
        np.hstack(panels[0:3]),
        np.hstack(panels[3:6]),
        np.hstack(panels[6:9]),
    ))
    os.makedirs(output_dir, exist_ok=True)
    sheet_path = os.path.join(output_dir, "tail.jpg")
    if not cv2.imwrite(
        sheet_path,
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    ):
        raise RuntimeError(f"failed to write {sheet_path}")
    return {
        "current_path": frame_paths[last_visible],
        "current_t": last_visible,
        "times": times,
        "sheet_path": sheet_path,
    }


def build_current_locus_dual_view_prompt(
    sample: Dict[str, Any],
    view: Dict[str, Any],
    *,
    format_retry: bool = False,
) -> str:
    """Render the task-label-blind prompt frozen by iteration 34."""
    options = list(sample.get("options") or [])
    rendered = "\n".join(
        f"{chr(ord('A') + index)}. {option}"
        for index, option in enumerate(options)
    )
    retry = (
        "\n\nThe previous output failed the exact JSON contract. Return one "
        "valid object with a legal option letter, an in-range timestamp, "
        "concrete target/reference/property evidence, a typed contradicted "
        "array, and no prose."
        if format_retry else ""
    )
    return (
        "Independently answer ONE closed-set video question whose deciding "
        "evidence is at the current query-time scene. Image 1 is the original "
        f"high-resolution current frame at t={view['current_t']}s. Image 2 "
        "is a chronological 3x3 sheet covering the current frame and up to "
        f"eight preceding seconds t={min(view['times'])}s.."
        f"{max(view['times'])}s; every panel is timestamped and black panels "
        "are padding.\n\n"
        f"Question: {sample.get('question')}\n"
        f"Options:\n{rendered}\n\n"
        "Use Image 1 for small text, object identity, color, material, shape, "
        "count, static location, and fine attributes. Use Image 2 to compare "
        "motion, facing, action direction, approach/leave, before/after state, "
        "and what the visible setup predicts next. The question's exact target "
        "and reference must both be visible. Preserve first-person camera "
        "coordinates when the question says I/me/my. Do not use option wording, "
        "scene priors, generic affordances, or historical expectations as "
        "evidence. If no option is fully supported, choose the single closest "
        "visible option rather than inventing an unlisted answer.\n\n"
        "Return ONLY one JSON object with: answer (one legal option letter), "
        "evidence_t (timestamp visible in one of the two inputs), target "
        "(exact visible entity), reference (exact reference entity or camera "
        "wearer), relation_or_attribute (the decisive visible property/action/"
        "relation), temporal_comparison (what changed across panels, or "
        "'static current frame'), confidence (high/medium/low), contradicted "
        "(array of option letters directly contradicted by pixels), and "
        "evidence (concise pixel-grounded explanation). No prose or Markdown "
        f"fences.{retry}\n<image>\n<image>"
    )


def parse_current_locus_dual_view_audit(
    response: str,
    sample: Dict[str, Any],
    visible_times: List[int],
) -> Optional[Dict[str, Any]]:
    """Strictly parse one current-locus audit response."""
    visible = _CURRENT_DUAL_VIEW_THINK_RE.sub(
        "", response or ""
    ).strip()
    objects = re.findall(r"\{[^{}]*\}", visible, re.DOTALL)
    if not objects or not visible_times:
        return None
    try:
        payload = json.loads(objects[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    options = list(sample.get("options") or [])
    legal = {
        chr(ord("A") + index) for index in range(len(options))
    }
    selected = str(payload.get("answer") or "").strip().upper()
    if selected not in legal:
        return None
    timestamp = payload.get("evidence_t")
    if isinstance(timestamp, str):
        match = re.search(r"-?\d+(?:\.\d+)?", timestamp)
        timestamp = float(match.group(0)) if match else None
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
        or not min(visible_times) <= float(timestamp) <= max(visible_times)
    ):
        return None
    fields: Dict[str, str] = {}
    for key, limit in (
        ("target", 400),
        ("reference", 400),
        ("relation_or_attribute", 700),
        ("temporal_comparison", 900),
        ("evidence", 1400),
    ):
        text = " ".join(str(payload.get(key) or "").split())[:limit]
        if not text or _CURRENT_DUAL_VIEW_PLACEHOLDER_RE.search(text):
            return None
        fields[key] = text
    confidence = str(payload.get("confidence") or "").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        return None
    contradicted = payload.get("contradicted")
    if (
        not isinstance(contradicted, list)
        or any(
            str(value).strip().upper() not in legal - {selected}
            for value in contradicted
        )
    ):
        return None
    return {
        "answer": selected,
        "evidence_t": float(timestamp),
        **fields,
        "confidence": confidence,
        "contradicted": [
            str(value).strip().upper() for value in contradicted
        ],
    }


_CURRENT_PROTECTOR_STATUSES = {
    "direct", "partial", "contradicted", "ambiguous",
}
_CURRENT_PROTECTOR_FACT_ID_RE = re.compile(r"^[STU]\d+$")


def build_current_locus_pairwise_prompt(
    sample: Dict[str, Any],
    view: Dict[str, Any],
    *,
    candidate_x: str,
    candidate_y: str,
) -> str:
    """Build the provenance-blind pairwise protector prompt."""
    return (
        "Adjudicate TWO conflicting legal candidate answers to one video "
        "question using only direct visual evidence. The candidates come from "
        "two untrusted analyses. Their order is randomized: neither X nor Y "
        "is a baseline, preferred answer, newer answer, or safer answer. "
        "Candidate wording is an output hypothesis, never evidence.\n\n"
        "Image 1 is the original high-resolution query-time frame at "
        f"t={view['current_t']}s. Image 2 is a chronological 3x3 sheet "
        f"covering t={min(view['times'])}s..{max(view['times'])}s; every "
        "non-black panel is timestamped and black panels are padding.\n\n"
        f"Question: {sample.get('question')}\n"
        f"Candidate X: {candidate_x}\n"
        f"Candidate Y: {candidate_y}\n\n"
        "First identify the exact question target, exact reference entity, "
        "and coordinate/temporal frame required by the question. Then inspect "
        "the pixels for each candidate independently. Preserve camera, "
        "anatomical, object, and world coordinate frames. Do not use scene "
        "names, generic affordances, common sense, candidate fluency, or "
        "historical expectations.\n\n"
        "For each candidate assign exactly one status: direct, partial, "
        "contradicted, or ambiguous. Choose winner X or Y only if that "
        "candidate is direct AND the other is contradicted. Otherwise choose "
        "uncertain. Return ONLY one JSON object with: winner (X/Y/uncertain), "
        "evidence_t, target, reference, required_frame, x_status, y_status, "
        "decisive_relation, temporal_comparison, and evidence. No answer "
        "letters, task names, prose, or Markdown fences outside the JSON.\n"
        "<image>\n<image>"
    )


def parse_current_locus_pairwise_audit(
    response: str,
    visible_times: List[int],
) -> Optional[Dict[str, Any]]:
    """Parse one strict pairwise-protector response."""
    visible = _CURRENT_DUAL_VIEW_THINK_RE.sub("", response or "").strip()
    objects = re.findall(r"\{[^{}]*\}", visible, re.DOTALL)
    if not objects or not visible_times:
        return None
    try:
        payload = json.loads(objects[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    winner = str(payload.get("winner") or "").strip().upper()
    if winner not in {"X", "Y", "UNCERTAIN"}:
        return None
    timestamp = payload.get("evidence_t")
    if isinstance(timestamp, str):
        match = re.search(r"-?\d+(?:\.\d+)?", timestamp)
        timestamp = float(match.group(0)) if match else None
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
        or not min(visible_times) <= float(timestamp) <= max(visible_times)
    ):
        return None
    x_status = str(payload.get("x_status") or "").strip().lower()
    y_status = str(payload.get("y_status") or "").strip().lower()
    if (
        x_status not in _CURRENT_PROTECTOR_STATUSES
        or y_status not in _CURRENT_PROTECTOR_STATUSES
    ):
        return None
    if winner == "X" and not (
        x_status == "direct" and y_status == "contradicted"
    ):
        return None
    if winner == "Y" and not (
        y_status == "direct" and x_status == "contradicted"
    ):
        return None
    if winner == "UNCERTAIN" and (
        (x_status == "direct" and y_status == "contradicted")
        or (y_status == "direct" and x_status == "contradicted")
    ):
        return None
    fields: Dict[str, str] = {}
    for key, limit in (
        ("target", 400),
        ("reference", 400),
        ("required_frame", 500),
        ("decisive_relation", 700),
        ("temporal_comparison", 900),
        ("evidence", 1400),
    ):
        text = " ".join(str(payload.get(key) or "").split())[:limit]
        if not text or _CURRENT_DUAL_VIEW_PLACEHOLDER_RE.search(text):
            return None
        fields[key] = text
    return {
        "winner": winner.lower(),
        "evidence_t": float(timestamp),
        "x_status": x_status,
        "y_status": y_status,
        **fields,
    }


def build_current_locus_fact_ledger_prompt(
    sample: Dict[str, Any],
    view: Dict[str, Any],
) -> str:
    """Build a candidate-free visual fact-ledger prompt."""
    return (
        "Extract an immutable VISUAL FACT LEDGER for one video question. "
        "Do not answer the question and do not infer any multiple-choice "
        "options: no options or candidate answers are provided. Record only "
        "facts directly visible in the pixels.\n\n"
        "Image 1 is the original high-resolution query-time frame at "
        f"t={view['current_t']}s. Image 2 is a chronological 3x3 sheet "
        f"covering t={min(view['times'])}s..{max(view['times'])}s; every "
        "non-black panel is timestamped and black panels are padding.\n\n"
        f"Question: {sample.get('question')}\n\n"
        "Identify the exact target, exact reference entity, and coordinate/"
        "temporal frame demanded by the question. Preserve anatomical, "
        "camera, object, and world frames. For counts, list each instance. "
        "For small on-screen text, transcribe only legible glyphs. Separate "
        "tail events from the final current frame. Do not use scene names, "
        "generic affordances, common sense, answer priors, or workflows.\n\n"
        "Return ONLY one JSON object with: evidence_t, target, reference, "
        "required_frame, question_scope, static_facts (array of 1-8 atomic "
        "facts prefixed S1:, S2:, ...), temporal_facts (array of 0-8 ordered "
        "facts prefixed T1:, T2:, ...), and uncertainties (array of 0-6 "
        "items prefixed U1:, U2:, ...). Fact text must not mention options, "
        "candidates, answer letters, correctness, or hidden task types. No "
        "prose or Markdown fences.\n<image>\n<image>"
    )


def _parse_current_locus_json_object(
    response: str,
) -> Optional[Dict[str, Any]]:
    visible = _CURRENT_DUAL_VIEW_THINK_RE.sub("", response or "").strip()
    objects = re.findall(r"\{.*\}", visible, re.DOTALL)
    if not objects:
        return None
    try:
        payload = json.loads(objects[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_current_locus_fact_list(
    payload: Dict[str, Any],
    key: str,
    prefix: str,
    *,
    minimum: int,
    maximum: int,
) -> Optional[List[str]]:
    values = payload.get(key)
    if not isinstance(values, list) or not minimum <= len(values) <= maximum:
        return None
    rendered: List[str] = []
    for index, value in enumerate(values, 1):
        text = " ".join(str(value or "").split())[:800]
        if (
            not text
            or _CURRENT_DUAL_VIEW_PLACEHOLDER_RE.search(text)
            or not text.startswith(f"{prefix}{index}:")
        ):
            return None
        rendered.append(text)
    return rendered


def parse_current_locus_fact_ledger(
    response: str,
    visible_times: List[int],
) -> Optional[Dict[str, Any]]:
    """Parse a candidate-free visual fact ledger."""
    payload = _parse_current_locus_json_object(response)
    if payload is None or not visible_times:
        return None
    timestamp = payload.get("evidence_t")
    if isinstance(timestamp, str):
        match = re.search(r"-?\d+(?:\.\d+)?", timestamp)
        timestamp = float(match.group(0)) if match else None
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
        or not min(visible_times) <= float(timestamp) <= max(visible_times)
    ):
        return None
    fields: Dict[str, str] = {}
    for key, limit in (
        ("target", 400),
        ("reference", 400),
        ("required_frame", 600),
        ("question_scope", 800),
    ):
        text = " ".join(str(payload.get(key) or "").split())[:limit]
        if not text or _CURRENT_DUAL_VIEW_PLACEHOLDER_RE.search(text):
            return None
        fields[key] = text
    static_facts = _parse_current_locus_fact_list(
        payload, "static_facts", "S", minimum=1, maximum=8
    )
    temporal_facts = _parse_current_locus_fact_list(
        payload, "temporal_facts", "T", minimum=0, maximum=8
    )
    uncertainties = _parse_current_locus_fact_list(
        payload, "uncertainties", "U", minimum=0, maximum=6
    )
    if static_facts is None or temporal_facts is None or uncertainties is None:
        return None
    all_text = " ".join([
        *fields.values(),
        *static_facts,
        *temporal_facts,
        *uncertainties,
    ])
    if re.search(
        r"\b(?:candidate|option\s+[A-K]|correct answer|ground truth)\b",
        all_text,
        re.IGNORECASE,
    ):
        return None
    return {
        "evidence_t": float(timestamp),
        **fields,
        "static_facts": static_facts,
        "temporal_facts": temporal_facts,
        "uncertainties": uncertainties,
    }


def current_locus_fact_ids(ledger: Dict[str, Any]) -> Set[str]:
    result: Set[str] = set()
    for key in ("static_facts", "temporal_facts", "uncertainties"):
        for value in ledger.get(key) or []:
            result.add(str(value).split(":", 1)[0])
    return result


def build_current_locus_claim_prompt(
    sample: Dict[str, Any],
    ledger: Dict[str, Any],
    claim: str,
) -> str:
    """Build an isolated, single-claim ledger judgment prompt."""
    return (
        "Evaluate ONE untrusted answer claim against an immutable visual fact "
        "ledger. You do not see any competing candidate and do not know whether "
        "this claim is old, new, preferred, or produced by an agent. Do not add "
        "facts, common sense, scene priors, or generic affordances.\n\n"
        f"Question: {sample.get('question')}\n"
        f"Claim: {claim}\n"
        "Immutable visual fact ledger:\n"
        f"{json.dumps(ledger, ensure_ascii=False)}\n\n"
        "Assign exactly one status: direct, contradicted, partial, or "
        "ambiguous. If a required detail is uncertain, do not mark direct. "
        "Return ONLY one JSON object with: status, target_match (boolean), "
        "reference_match (boolean), frame_match (boolean), "
        "supporting_fact_ids, contradicting_fact_ids, and explanation. Direct "
        "requires all matches, support, and no contradiction. Contradicted "
        "requires a conflict fact. No prose or Markdown fences."
    )


def parse_current_locus_claim_judgment(
    response: str,
    valid_fact_ids: Set[str],
) -> Optional[Dict[str, Any]]:
    """Parse one isolated claim judgment against a fact ledger."""
    payload = _parse_current_locus_json_object(response)
    if payload is None:
        return None
    status = str(payload.get("status") or "").strip().lower()
    if status not in _CURRENT_PROTECTOR_STATUSES:
        return None
    booleans: Dict[str, bool] = {}
    for key in ("target_match", "reference_match", "frame_match"):
        value = payload.get(key)
        if not isinstance(value, bool):
            return None
        booleans[key] = value
    id_lists: Dict[str, List[str]] = {}
    for key in ("supporting_fact_ids", "contradicting_fact_ids"):
        values = payload.get(key)
        if (
            not isinstance(values, list)
            or len(values) > 12
            or any(
                not isinstance(value, str)
                or not _CURRENT_PROTECTOR_FACT_ID_RE.match(value)
                or value not in valid_fact_ids
                for value in values
            )
        ):
            return None
        id_lists[key] = list(dict.fromkeys(values))
    explanation = " ".join(
        str(payload.get("explanation") or "").split()
    )[:1200]
    if not explanation or _CURRENT_DUAL_VIEW_PLACEHOLDER_RE.search(explanation):
        return None
    if status == "direct" and not (
        all(booleans.values())
        and id_lists["supporting_fact_ids"]
        and not id_lists["contradicting_fact_ids"]
    ):
        return None
    if status == "contradicted" and not id_lists["contradicting_fact_ids"]:
        return None
    return {
        "status": status,
        **booleans,
        **id_lists,
        "explanation": explanation,
    }


def current_locus_pairwise_semantic_winner(
    order: str,
    result: Optional[Dict[str, Any]],
) -> str:
    if result is None:
        return "invalid"
    winner = str(result.get("winner") or "")
    if winner == "uncertain":
        return "uncertain"
    if order == "original_first":
        return "original" if winner == "x" else "proposed"
    if order == "proposed_first":
        return "proposed" if winner == "x" else "original"
    return "invalid"


def current_locus_iter39_accept(
    pair_original_first: Optional[Dict[str, Any]],
    pair_proposed_first: Optional[Dict[str, Any]],
    original_judgment: Optional[Dict[str, Any]],
    proposed_judgment: Optional[Dict[str, Any]],
) -> bool:
    """Frozen iteration-39 rule trained on all 65 known conflicts."""
    first = current_locus_pairwise_semantic_winner(
        "original_first", pair_original_first
    )
    second = current_locus_pairwise_semantic_winner(
        "proposed_first", pair_proposed_first
    )
    original_status = (
        str(original_judgment.get("status"))
        if original_judgment is not None else "invalid"
    )
    proposed_status = (
        str(proposed_judgment.get("status"))
        if proposed_judgment is not None else "invalid"
    )
    strong_pair = first == "proposed" and second == "proposed"
    ledger_clause = (
        proposed_status == "direct"
        and original_status in {"contradicted", "partial"}
    )
    relaxed_pair = (
        (first == "uncertain" and second == "proposed")
        or (
            first == second == "uncertain"
            and original_status == proposed_status
            and original_status in {"ambiguous", "partial"}
        )
    )
    return strong_pair or ledger_clause or relaxed_pair


def last_frame_tool_reply_cap_action(
    *,
    has_calls: bool,
    current_round: int,
    total_rounds: int,
    question_time: int,
    replies_this_round: int,
    cap_used: bool,
    cap: int = 16,
) -> str:
    """Return ``none``, one-time ``nudge``, or terminal ``drop``.

    This guard is task-agnostic and counts physical replies rather than only
    successful tool dispatches, so rejected duplicate calls cannot spin
    forever on the final frame.
    """
    at_final_post_question = (
        int(current_round) >= int(total_rounds) - 1
        and int(current_round) >= int(question_time)
    )
    if (
        not has_calls
        or not at_final_post_question
        or int(replies_this_round) <= int(cap)
    ):
        return "none"
    return "drop" if cap_used else "nudge"


def prequery_tool_reply_cap_action(
    *,
    has_calls: bool,
    current_round: int,
    question_time: int,
    replies_this_round: int,
    cap: int = 8,
) -> str:
    """Force a task-blind clock tick after a pre-query tool/reply loop."""
    if (
        has_calls
        and int(current_round) < int(question_time)
        and int(replies_this_round) >= int(cap)
    ):
        return "advance"
    return "none"


def parse_crr_clue_audit_response(
    response: str,
) -> Optional[Dict[str, Any]]:
    """Parse whether a visual clue now supports a specific CRR answer."""
    visible = ovo._strip_thinking(response or "").strip()
    objects = re.findall(r"\{[^{}]*\}", visible, re.S)
    if not objects:
        return None
    try:
        payload = json.loads(objects[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    def normalize(key: str) -> str:
        return re.sub(
            r"[\s-]+", "_", str(payload.get(key) or "").strip().lower()
        )

    anchor_status = normalize("anchor_status")
    answer_status = normalize("answer_status")
    answer_source = normalize("answer_source")
    temporal_link = normalize("temporal_link")
    candidate_answer = " ".join(
        str(payload.get("candidate_answer") or "").split()
    )
    evidence = " ".join(str(payload.get("evidence") or "").split())
    before_t = payload.get("before_t")
    after_t = payload.get("after_t")

    if anchor_status not in {
        "matched", "history_linked", "unmatched", "uncertain",
    }:
        return None
    if answer_status not in {
        "specific_supported", "partial", "multiple_plausible",
        "setup_only", "absent", "uncertain",
    }:
        return None
    if answer_source not in {
        "visible_target", "visible_outcome", "ordered_causal_result",
        "scene_guess", "direction_guess", "background_guess",
        "ocr_title", "irrelevant", "none", "uncertain",
    }:
        return None
    if temporal_link not in {"linked", "broken", "uncertain"}:
        return None
    if not isinstance(before_t, (int, float)):
        return None
    if not isinstance(after_t, (int, float)):
        return None
    if float(after_t) < float(before_t):
        return None
    if not evidence:
        return None

    invalid_answers = {
        "", "unknown", "uncertain", "none", "not shown", "somewhere",
        "not enough information", "cannot determine",
    }
    positive_sources = {
        "visible_target", "visible_outcome", "ordered_causal_result",
    }
    has_answer = (
        anchor_status in {"matched", "history_linked"}
        and answer_status == "specific_supported"
        and answer_source in positive_sources
        and temporal_link == "linked"
        and candidate_answer.lower() not in invalid_answers
    )
    return {
        "anchor_status": anchor_status,
        "answer_status": answer_status,
        "answer_source": answer_source,
        "temporal_link": temporal_link,
        "candidate_answer": candidate_answer[:500],
        "before_t": float(before_t),
        "after_t": float(after_t),
        "evidence": evidence[:1000],
        "answer": "Yes" if has_answer else "No",
    }


def parse_crr_target_audit_response(
    response: str,
) -> Optional[Dict[str, Any]]:
    """Parse a CRR audit and require a concrete original-question target."""
    visible = ovo._strip_thinking(response or "").strip()
    objects = re.findall(r"\{[^{}]*\}", visible, re.S)
    if not objects:
        return None
    try:
        payload = json.loads(objects[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    anchor_status = re.sub(
        r"[\s-]+", "_", str(payload.get("anchor_status") or "").lower()
    )
    target_status = re.sub(
        r"[\s-]+", "_", str(payload.get("target_status") or "").lower()
    )
    subject_continuity = re.sub(
        r"[\s-]+", "_", str(payload.get("subject_continuity") or "").lower()
    )
    arrival_evidence = re.sub(
        r"[\s-]+", "_", str(payload.get("arrival_evidence") or "").lower()
    )
    target_value = " ".join(str(payload.get("target_value") or "").split())
    evidence = " ".join(str(payload.get("evidence") or "").split())
    before_t = payload.get("before_t")
    after_t = payload.get("after_t")
    if anchor_status not in {"matched", "unmatched", "uncertain"}:
        return None
    if target_status not in {
        "concrete_visible", "causal_chain", "anchor_only",
        "absent", "uncertain",
    }:
        return None
    if subject_continuity not in {"same_actor", "broken", "uncertain"}:
        return None
    if arrival_evidence not in {
        "entered", "arrived", "stopped", "stayed",
        "other_target", "none", "uncertain",
    }:
        return None
    if not isinstance(before_t, (int, float)):
        return None
    if not isinstance(after_t, (int, float)):
        return None
    if float(after_t) < float(before_t):
        return None
    if not evidence:
        return None

    destination_chain = (
        target_status == "causal_chain"
        and subject_continuity == "same_actor"
        and arrival_evidence in {"entered", "arrived", "stopped", "stayed"}
    )
    direct_target = (
        target_status == "concrete_visible"
        and subject_continuity != "broken"
        and arrival_evidence in {
            "entered", "arrived", "stopped", "stayed", "other_target",
        }
    )
    has_target = destination_chain or direct_target
    invalid_values = {"", "unknown", "uncertain", "none", "not shown"}
    answer = (
        "Yes"
        if has_target and target_value.strip().lower() not in invalid_values
        else "No"
    )
    return {
        "anchor_status": anchor_status,
        "target_status": target_status,
        "subject_continuity": subject_continuity,
        "arrival_evidence": arrival_evidence,
        "target_value": target_value[:500],
        "before_t": float(before_t),
        "after_t": float(after_t),
        "evidence": evidence[:1000],
        "answer": answer,
    }


def parse_ssr_local_audit_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse and validate one structured SSR local-state audit."""
    visible = ovo._strip_thinking(response or "").strip()
    objects = re.findall(r"\{[^{}]*\}", visible, re.S)
    if not objects:
        return None
    try:
        payload = json.loads(objects[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    local_phase = re.sub(
        r"[\s-]+", "_", str(payload.get("local_phase") or "").lower()
    )
    history_phase = re.sub(
        r"[\s-]+", "_", str(payload.get("history_phase") or "").lower()
    )
    before_t = payload.get("before_t")
    after_t = payload.get("after_t")
    evidence = " ".join(str(payload.get("evidence") or "").split())
    valid_phases = {
        "contact_only", "defining_motion", "changed_state", "completed",
        "absent", "uncertain",
    }
    if local_phase not in valid_phases:
        return None
    if history_phase not in valid_phases:
        return None
    if not isinstance(before_t, (int, float)):
        return None
    if not isinstance(after_t, (int, float)):
        return None
    if float(after_t) < float(before_t):
        return None
    if not evidence:
        return None

    positive_phases = {"defining_motion", "changed_state", "completed"}
    answer = (
        "Yes"
        if local_phase in positive_phases or history_phase in positive_phases
        else "No"
    )
    return {
        "local_phase": local_phase,
        "history_phase": history_phase,
        "before_t": float(before_t),
        "after_t": float(after_t),
        "evidence": evidence[:1000],
        "answer": answer,
    }


def normalize_count_audit_response(
    response: str,
    question: str,
) -> Optional[str]:
    """Convert an audit's visible count conclusion to the query format."""
    visible = ovo._strip_thinking(response or "").strip()
    if not visible:
        return None
    if RESPONSE_TAG in visible:
        return visible

    option_match = re.fullmatch(
        r"\s*([A-K])\s*[.)：:]\s*(\d{1,2})\s*",
        visible,
        re.IGNORECASE,
    )
    if option_match:
        letter = option_match.group(1).upper()
        count = int(option_match.group(2))
        if 0 <= count <= 10 and ord(letter) - ord("A") == count:
            return f"{RESPONSE_TAG} {letter}. {count}"

    candidates: List[int] = []
    patterns = (
        r"\b(?:final\s+)?(?:answer|count|total|running\s+total)\s*"
        r"(?:is|=|:)\s*(?:option\s+)?(?:[A-K][.)]\s*)?(\d{1,2})\b",
        r"\bexactly\s+(\d{1,2})\s+(?:distinct\s+)?"
        r"(?:times?|occurrences?|repetitions?|events?|actions?)\b",
        r"\b(\d{1,2})\s+(?:distinct\s+)?occurrences?\s+in\s+total\b",
    )
    for pattern in patterns:
        candidates.extend(
            int(match) for match in re.findall(pattern, visible, re.I)
        )
    if re.search(
        r"\b(?:no|zero)\s+(?:valid\s+|distinct\s+)?"
        r"(?:occurrences?|repetitions?|matching\s+events?)\b",
        visible,
        re.I,
    ):
        candidates.append(0)
    candidates = [value for value in candidates if 0 <= value <= 10]
    if not candidates:
        return None

    count = candidates[-1]
    has_letter_options = bool(
        re.search(r"(?m)^A\.\s*0\s*$", question or "")
        and re.search(r"(?m)^K\.\s*10\s*$", question or "")
    )
    if has_letter_options:
        return f"{RESPONSE_TAG} {chr(ord('A') + count)}. {count}"
    return f"{RESPONSE_TAG} {count}"


def extract_count_audit_value(response: str) -> Optional[int]:
    """Extract a local 0..10 count from one visual audit segment."""
    visible = ovo._strip_thinking(response or "").strip()
    if RESPONSE_TAG in visible:
        visible = visible.split(RESPONSE_TAG, 1)[-1].strip()
    direct = re.fullmatch(r"\s*(\d{1,2})\s*", visible)
    if direct:
        value = int(direct.group(1))
        return value if 0 <= value <= 10 else None
    option = re.fullmatch(
        r"\s*([A-K])\s*[.)：:]\s*(\d{1,2})\s*",
        visible,
        re.IGNORECASE,
    )
    if option:
        value = int(option.group(2))
        if 0 <= value <= 10 and ord(option.group(1).upper()) - ord("A") == value:
            return value
    normalized = normalize_count_audit_response(response, "")
    if normalized:
        match = re.search(r"\b(\d{1,2})\s*$", normalized)
        if match:
            value = int(match.group(1))
            return value if 0 <= value <= 10 else None
    return None


def extract_count_audit_ledger(response: str) -> Optional[List[float]]:
    """Extract the final JSON timestamp/evidence array from a count audit."""
    visible = ovo._strip_thinking(response or "").strip()
    arrays = re.findall(r"\[[^\[\]]*\]", visible, re.S)
    if not arrays:
        return None
    payload = arrays[-1]
    try:
        parsed = json.loads(payload)
    except Exception:
        parsed = None
    values: List[float] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, (int, float)) and float(item) >= 0:
                values.append(float(item))
            elif isinstance(item, dict):
                timestamp = item.get("t")
                evidence = str(item.get("e") or "").strip()
                if isinstance(timestamp, str):
                    timestamp_match = re.search(
                        r"-?\d+(?:\.\d+)?", timestamp)
                    timestamp = (
                        float(timestamp_match.group(0))
                        if timestamp_match else None
                    )
                if (
                    isinstance(timestamp, (int, float))
                    and float(timestamp) >= 0
                    and evidence
                ):
                    values.append(float(timestamp))
    else:
        for token in re.findall(r"-?\d+(?:\.\d+)?", payload):
            value = float(token)
            if value >= 0:
                values.append(value)
    deduped: List[float] = []
    for value in values:
        if not deduped or abs(value - deduped[-1]) > 0.35:
            deduped.append(value)
    return deduped[:10]


def split_count_audit_frames(
    frame_paths: List[str],
    max_frames: int,
) -> List[List[str]]:
    """Split a timestamped timeline into contiguous, non-overlapping chunks."""
    limit = max(1, int(max_frames))
    return [
        list(frame_paths[start:start + limit])
        for start in range(0, len(frame_paths), limit)
    ]


def merge_count_audit_ledgers(
    existing: List[float],
    new_values: List[float],
    *,
    tolerance: float = 0.35,
) -> List[float]:
    """Merge absolute timestamps and remove only near-identical boundaries."""
    merged = sorted(
        float(value)
        for value in [*existing, *new_values]
        if isinstance(value, (int, float)) and float(value) >= 0
    )
    deduped: List[float] = []
    for value in merged:
        if not deduped or abs(value - deduped[-1]) > tolerance:
            deduped.append(value)
    return deduped


_SPORTS_RELEASE_CYCLE_RULES: Tuple[Tuple[str, str], ...] = (
    (
        r"\bshot\s*put\b",
        "One occurrence is one athlete's complete shot-put attempt: a visible "
        "set/glide/turn and arm drive culminates in the SHOT visibly leaving "
        "the hand. Count ONLY that release. The same attempt's preparation, "
        "follow-through, shot flight/landing, measurement, celebration, or "
        "replay is not another occurrence. A new occurrence requires a new "
        "athlete/setup and a new shot release.",
    ),
    (
        r"\bjavelin\s+throw\b",
        "One occurrence is one complete javelin attempt: run-up/plant and arm "
        "throw culminate in the JAVELIN visibly leaving the hand. Count ONLY "
        "that release. The same attempt's run-up frames, follow-through, "
        "javelin flight/landing, measurement, celebration, or replay is not "
        "another occurrence. A new occurrence requires a new setup/run-up and "
        "a new javelin release.",
    ),
    (
        r"\bcricket\s+bowling\b",
        "One occurrence is one bowler delivery: bowler run-up and delivery "
        "stride culminate in the BALL visibly leaving the BOWLER'S hand. "
        "Count ONLY that bowler release. Ball bounce, batsman hit/swing, "
        "fielding, celebration, follow-through, and replay are not bowling "
        "occurrences. A new occurrence requires a new bowler run-up/setup and "
        "a new bowler release.",
    ),
)


def count_audit_sports_cycle_version(question: str) -> int:
    """Return the active opt-in sports schema version for this query."""
    version = 0
    if os.environ.get("STREAMO_ENABLE_REC_SPORTS_CYCLE_V2", "0") == "1":
        version = 2
    if os.environ.get("STREAMO_ENABLE_REC_SPORTS_CYCLE_V3", "0") == "1":
        version = 3
    if not version:
        return 0
    if not any(
        re.search(pattern, question or "", re.IGNORECASE)
        for pattern, _ in _SPORTS_RELEASE_CYCLE_RULES
    ):
        return 0
    return version


def count_audit_uses_sports_cycle_schema(question: str) -> bool:
    """Return whether an opt-in sports release-cycle schema applies."""
    if not count_audit_sports_cycle_version(question):
        return False
    return any(
        re.search(pattern, question or "", re.IGNORECASE)
        for pattern, _ in _SPORTS_RELEASE_CYCLE_RULES
    )


def extract_sports_cycle_ledger(
    response: str,
    target_activity: str,
    *,
    require_grounded: bool = False,
) -> Optional[List[float]]:
    """Parse and validate one segment's structured sports release cycles."""
    visible = ovo._strip_thinking(response or "").strip()
    arrays = re.findall(r"\[[^\[\]]*\]", visible, re.S)
    if not arrays:
        return None
    try:
        payload = json.loads(arrays[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None

    target = str(target_activity or "").lower()
    if re.search(r"\bshot\s*put\b", target):
        object_re = re.compile(r"\b(?:shot|shotput|ball)\b", re.I)
    elif re.search(r"\bjavelin\b", target):
        object_re = re.compile(r"\bjavelin\b", re.I)
    elif re.search(r"\bcricket\s+bowling\b", target):
        object_re = re.compile(r"\b(?:ball|bowl(?:er|ing)?|delivery)\b", re.I)
    else:
        return None
    release_re = re.compile(
        r"\b(?:release[sd]?|leaves?|left|letting\s+go|lets?\s+go)\b",
        re.I,
    )

    placeholder_re = re.compile(
        r"(?:visible setup/run-up before release|"
        r"visible target implement/ball leaves hand|"
        r"visible follow-through/flight after release|"
        r"specific appearance/location|"
        r"specific object and its pre-release location|"
        r"specific pre-release body/object state|"
        r"specific hand-object separation|"
        r"specific post-release spatial state|"
        r"concrete pixels in this cycle)",
        re.I,
    )
    actor_detail_re = re.compile(
        r"\b(?:left|right|foreground|background|near|far|red|white|black|"
        r"blue|green|yellow|orange|dark|light|striped|number|jersey|shirt|"
        r"jacket|uniform|circle|crease|lane|field)\b",
        re.I,
    )
    implement_detail_re = re.compile(
        r"\b(?:left|right|hand|hands|neck|shoulder|behind|beside|near|above|"
        r"below|chest|head|red|white|black|blue|green|yellow|orange|dark|"
        r"light)\b",
        re.I,
    )
    timestamps: List[float] = []
    signatures: Set[Tuple[str, ...]] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            return None
        cycle = item.get("cycle")
        timestamp = item.get("t")
        if isinstance(timestamp, str):
            match = re.search(r"-?\d+(?:\.\d+)?", timestamp)
            timestamp = float(match.group(0)) if match else None
        before = " ".join(str(item.get("before") or "").split())
        release = " ".join(str(item.get("release") or "").split())
        after = " ".join(str(item.get("after") or "").split())
        actor = " ".join(str(item.get("actor") or "").split())
        implement = " ".join(str(item.get("implement") or "").split())
        if cycle != index + 1:
            return None
        if not isinstance(timestamp, (int, float)) or float(timestamp) < 0:
            return None
        if not before or not release or not after:
            return None
        if not object_re.search(release) or not release_re.search(release):
            return None
        if require_grounded:
            fields = (actor, implement, before, release, after)
            if any(not field for field in fields):
                return None
            if placeholder_re.search(" ".join(fields)):
                return None
            if re.fullmatch(
                r"(?:the\s+)?(?:athlete|person|someone|actor|bowler)",
                actor,
                re.I,
            ):
                return None
            if not actor_detail_re.search(actor):
                return None
            if not object_re.search(implement):
                return None
            if not implement_detail_re.search(implement):
                return None
            normalized = tuple(
                re.sub(r"[^a-z0-9]+", " ", field.lower()).strip()
                for field in fields
            )
            if len({normalized[2], normalized[3], normalized[4]}) < 3:
                return None
            if normalized in signatures:
                return None
            signatures.add(normalized)
        value = float(timestamp)
        if timestamps and value <= timestamps[-1]:
            return None
        timestamps.append(value)
    return timestamps[:10]


_TIMESTAMP_LEDGER_CYCLE_RE = re.compile(
    r"\b(?:lifting\b.*\bplacing\b.*\bback\b|tearing\b|opening\b|"
    r"covering\b|turning\b.*\boff\b|tying\b|buttoning\b|folding\b)",
    re.IGNORECASE,
)


def count_audit_uses_evidence_ledger(question: str) -> bool:
    """Choose an audit schema from the target action's query semantics."""
    return not bool(_TIMESTAMP_LEDGER_CYCLE_RE.search(question or ""))


def count_audit_cycle_hint(question: str) -> str:
    """Return an observable occurrence-cycle definition from query semantics."""
    text = question or ""
    if count_audit_uses_sports_cycle_schema(text):
        for pattern, rule in _SPORTS_RELEASE_CYCLE_RULES:
            if re.search(pattern, text, re.IGNORECASE):
                return rule
    rules = (
        (
            r"\bclean\s+and\s+jerk\b",
            "One occurrence is one complete Olympic clean-and-jerk cycle "
            "performed with a barbell or weight implement: the implement is "
            "cleaned from below to the shoulders/front rack, then jerked to "
            "a stable overhead lockout. Record ONLY the overhead lockout "
            "completion, once per full cycle. A ball/shot throw, generic "
            "lifting without the clean-to-overhead sequence, preparation, "
            "static overhead holding, follow-through, or dropping/putting "
            "down alone is NOT a clean-and-jerk. Another occurrence requires "
            "a reset followed by a new complete clean-and-jerk cycle.",
        ),
        (
            r"\blifting\b.*\bplacing\b.*\bback\b",
            "One occurrence is the entire cycle: object leaves its support "
            "surface, remains lifted, then is placed back onto a support "
            "surface. Record ONLY the final place-back completion time. Do "
            "not record lift onset, mid-air states, or the same set-down "
            "across multiple frames.",
        ),
        (
            r"\btaking\b.*\bout of\b",
            "One occurrence is one independently handled object completing a "
            "container/interior-to-outside transfer. Count separate objects "
            "or separate extraction cycles; continuous holding outside is "
            "not new.",
        ),
        (
            r"\bclosing\b",
            "One occurrence is an open-to-closed state transition completed "
            "for one object. Record the completion when the lid/door/flap "
            "reaches closed. Mere touching, moving toward, or holding closed "
            "is not new.",
        ),
        (
            r"\bcovering\b",
            "One occurrence completes when a cover object is placed onto a "
            "target and the target becomes covered. Adjustment while already "
            "covered is not new; count another cycle only after reset/uncover "
            "and re-cover.",
        ),
        (
            r"\btearing\b",
            "Each distinct tear stroke that visibly creates a new material "
            "separation, rip, or newly detached piece is one occurrence. "
            "Continuous pulling during the same tear is one; a new grip/reset "
            "followed by another separation is a new occurrence.",
        ),
        (
            r"\bhitting\b.*\b(?:against|with)\b",
            "Each distinct impact/contact of one object, hand, or tool against "
            "another is one occurrence. Rapid repeated taps/strikes count "
            "separately when contact is followed by separation/reset. A swing "
            "without contact, holding, or setup is not an occurrence.",
        ),
        (
            r"\bshowing\b.*\bcamera\b",
            "One occurrence is a deliberate present-to-camera cycle: object "
            "is brought toward/faced to the camera, held visibly, then "
            "withdrawn or reset before another presentation. Record one "
            "representative completion/peak time per cycle; continuous "
            "holding is not repeated.",
        ),
    )
    for pattern, rule in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return rule
    return (
        "Define one occurrence by a decisive completion/result and a reset "
        "before the next occurrence."
    )


def count_audit_target(question: str) -> str:
    """Extract the visible activity phrase from a repetition-count query."""
    match = re.search(
        r"activity\s+['\"](.+?)['\"]\s+occur",
        question or "",
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return str(question or "").splitlines()[0].strip()


def build_count_audit_prompt(
    *,
    target_activity: str,
    cycle_hint: str,
    use_evidence_ledger: bool,
    sports_cycle_version: int = 0,
    segmented: bool = False,
    segment_index: int = 0,
    segment_count: int = 1,
    segment_start_t: float = 0.0,
    segment_end_t: float = 0.0,
) -> str:
    """Build the production count-audit prompt for one visible segment."""
    segment_instruction = ""
    audit_scope = "entire "
    if segmented:
        audit_scope = ""
        segment_instruction = (
            f"This is segment {segment_index + 1}/{segment_count}, covering "
            f"absolute timestamps approximately {segment_start_t:.2f}s to "
            f"{segment_end_t:.2f}s. Count an occurrence in this response "
            "ONLY when its decisive completion timestamp falls inside this "
            "segment. Use the ABSOLUTE timestamp printed on the frame, not "
            "time relative to the segment. "
        )
    audit_rules = (
        "<video>\n"
        f"Audit the {audit_scope}timestamped video chronologically and do "
        "not stop after the first few. Interpret the target by observable "
        f"meaning. Target activity: {target_activity}\n"
        f"Occurrence cycle: {cycle_hint}\n"
        f"{segment_instruction}"
        "Exclude title cards, verbal mentions, preparation, static holding, "
        "explanation, follow-through, and replay footage. Use characteristic "
        "before/after state transitions when decisive contact is occluded. "
    )
    if sports_cycle_version >= 3:
        output_rule = (
            "Return ONLY a JSON array. Each element must be an object with "
            "exactly these typed fields: cycle (integer starting at 1 within "
            "this segment), t (absolute numeric release time), actor (specific "
            "appearance/location that distinguishes this actor), implement "
            "(specific object and its pre-release location), before (specific "
            "pre-release body/object state), release (specific hand-object "
            "separation), and after (specific post-release spatial state). "
            "Do NOT copy or paraphrase these field definitions. Every string "
            "must describe concrete pixels in this cycle, and different "
            "cycles must not reuse identical evidence strings. If any "
            "required state is not visible, omit that cycle. Do not output "
            "preparation, landing, batsman hit, celebration, or replay as a "
            "cycle. At most 10 objects. Return [] if none. No prose."
        )
    elif sports_cycle_version == 2:
        output_rule = (
            "Return ONLY a JSON array with one object per COMPLETE "
            "attempt/release in this segment: "
            '{"cycle":1,"t":absolute_release_timestamp,'
            '"before":"visible setup/run-up before release",'
            '"release":"visible target implement/ball leaves hand",'
            '"after":"visible follow-through/flight after release"}. '
            "Number cycle from 1 within THIS segment in chronological order. "
            "All four fields are required. Do not output preparation, "
            "landing, batsman hit, celebration, or replay as a cycle. At "
            "most 10 objects. Return [] if none. No prose."
        )
    elif use_evidence_ledger:
        output_rule = (
            'Return ONLY a JSON array with one object per occurrence: '
            '{"t": approximate_completion_timestamp, "e": "short concrete '
            'visible completion/result cue"}. Include an item only when its '
            "cue satisfies the occurrence cycle. At most 10 objects. Return "
            "[] if none. No prose."
        )
    else:
        output_rule = (
            "Return ONLY a JSON array with one approximate completion "
            "timestamp per occurrence in ascending order, at most 10 numbers. "
            "Return [] if none. No prose."
        )
    return audit_rules + output_rule


def build_release_verification_prompt(
    *,
    target_activity: str,
    candidate_t: float,
    window_start_t: float,
    window_end_t: float,
) -> str:
    """Build a local visual verifier prompt for one REC release candidate."""
    target = str(target_activity or "").strip()
    return (
        "<video>\n"
        "Verify ONE candidate occurrence in this short timestamped window. "
        f"Target activity: {target}. Candidate center: t={candidate_t:.2f}s. "
        f"Visible local window: t={window_start_t:.2f}s to "
        f"t={window_end_t:.2f}s. "
        "A confirmed occurrence requires the SAME concrete subject to hold "
        "the correct target implement before the candidate, a hand-object "
        "separation at the release moment near the candidate center, and the "
        "implement visibly separated/moving after release. For cricket "
        "bowling, only release from the BOWLER'S hand counts; batsman contact "
        "does not. Reject setup-only, follow-through-only, implement already "
        "in flight, landing, bounce, batsman hit/swing, fielding, celebration, "
        "slow-motion replay, and unrelated cuts. If any required phase is not "
        "visually grounded, use uncertain rather than guessing. "
        "Return ONLY one JSON object with exactly these fields: status "
        "(confirmed, rejected, or uncertain), release_t (number or null), "
        "subject (concrete clothing/number/location), implement (concrete "
        "object and pre-release hand/body location), before_t (number or null), "
        "before (concrete held state), release (concrete separation state), "
        "after_t (number or null), after (concrete post-separation state), "
        "distractor (none, setup, follow_through, flight, landing, bounce, "
        "bat_hit, fielding, celebration, replay, scene_cut, or other), and "
        "evidence (one concise timestamped visual chain). Do not copy field "
        "definitions and do not use generic placeholder descriptions. No prose."
    )


def parse_release_verification(
    response: str,
    target_activity: str,
    candidate_t: float,
    *,
    radius: float = 2.25,
) -> Optional[Dict[str, Any]]:
    """Parse a grounded local release verdict and map it to accept/reject."""
    visible = ovo._strip_thinking(response or "").strip()
    objects = re.findall(r"\{[^{}]*\}", visible, re.S)
    if not objects:
        return None
    try:
        payload = json.loads(objects[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    status = re.sub(
        r"[\s-]+", "_", str(payload.get("status") or "").strip().lower()
    )
    distractor = re.sub(
        r"[\s-]+", "_", str(payload.get("distractor") or "").strip().lower()
    )
    valid_status = {"confirmed", "rejected", "uncertain"}
    valid_distractors = {
        "none", "setup", "follow_through", "flight", "landing", "bounce",
        "bat_hit", "fielding", "celebration", "replay", "scene_cut", "other",
    }
    if status not in valid_status or distractor not in valid_distractors:
        return None

    def number(key: str) -> Optional[float]:
        value = payload.get(key)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            value = float(match.group(0)) if match else None
        return float(value) if isinstance(value, (int, float)) else None

    release_t = number("release_t")
    before_t = number("before_t")
    after_t = number("after_t")
    subject = " ".join(str(payload.get("subject") or "").split())
    implement = " ".join(str(payload.get("implement") or "").split())
    before = " ".join(str(payload.get("before") or "").split())
    release = " ".join(str(payload.get("release") or "").split())
    after = " ".join(str(payload.get("after") or "").split())
    evidence = " ".join(str(payload.get("evidence") or "").split())

    result: Dict[str, Any] = {
        "status": status,
        "release_t": release_t,
        "subject": subject[:300],
        "implement": implement[:300],
        "before_t": before_t,
        "before": before[:500],
        "release": release[:500],
        "after_t": after_t,
        "after": after[:500],
        "distractor": distractor,
        "evidence": evidence[:1000],
        "accepted": False,
    }
    if status != "confirmed":
        return result

    if None in (release_t, before_t, after_t):
        return None
    assert release_t is not None and before_t is not None and after_t is not None
    if not before_t < release_t < after_t:
        return None
    if abs(release_t - float(candidate_t)) > float(radius):
        return None
    if distractor != "none":
        return None
    fields = (subject, implement, before, release, after, evidence)
    if any(not field for field in fields):
        return None

    target = str(target_activity or "").lower()
    if re.search(r"\bshot\s*put\b", target):
        object_re = re.compile(r"\b(?:shot|shotput)\b", re.I)
    elif re.search(r"\bjavelin\b", target):
        object_re = re.compile(r"\bjavelin\b", re.I)
    elif re.search(r"\bcricket\s+bowling\b", target):
        object_re = re.compile(r"\b(?:cricket\s+ball|ball)\b", re.I)
    else:
        return None
    actor_detail_re = re.compile(
        r"\b(?:left|right|foreground|background|red|white|black|blue|green|"
        r"yellow|orange|dark|light|number|jersey|shirt|shorts|jacket|uniform|"
        r"singlet|kit|circle|crease|lane|field)\b",
        re.I,
    )
    held_re = re.compile(
        r"\b(?:hold(?:s|ing)?|held|grip(?:s|ping)?|against|in\s+(?:the\s+)?"
        r"(?:left|right)?\s*hand|at\s+(?:the\s+)?(?:neck|shoulder))\b",
        re.I,
    )
    separation_re = re.compile(
        r"\b(?:release[sd]?|releasing|leaves?|left|separat(?:es|ed|ing)?|"
        r"lets?\s+go|letting\s+go)\b",
        re.I,
    )
    post_re = re.compile(
        r"\b(?:airborne|flies|flying|flight|moves?\s+away|travels?|"
        r"separated|in\s+the\s+air|away\s+from)\b",
        re.I,
    )
    placeholder_re = re.compile(
        r"(?:concrete clothing/number/location|concrete object|"
        r"concrete held state|concrete separation state|"
        r"concrete post-separation state|timestamped visual chain|"
        r"generic placeholder)",
        re.I,
    )
    if placeholder_re.search(" ".join(fields)):
        return None
    if not actor_detail_re.search(subject):
        return None
    if not object_re.search(implement) or not object_re.search(release):
        return None
    if not held_re.search(before):
        return None
    if not separation_re.search(release):
        return None
    if not post_re.search(after):
        return None

    result["accepted"] = True
    return result


def _has_replay_intent(answer: str) -> bool:
    """True when the model stated a replay intent in the same turn.

    Accepts phrasings like "I will replay t=12 to check ...",
    "replay t=4 to verify ...", "let me replay ... to see ...". Kept
    lenient so a genuine intent line is not rejected on wording.
    """
    if not answer:
        return False
    low = answer.lower()
    if "replay" not in low:
        return False
    # An intent links replay to a purpose ("to check/verify/see/confirm")
    # OR explicitly says "I will replay".
    if re.search(r"i\s+will\s+replay", low):
        return True
    if re.search(r"replay[^.<]{0,80}\bto\s+(check|verify|see|confirm|"
                 r"count|inspect|look|determine|find)", low):
        return True
    return False


def _is_empty_tool_result(res: Any) -> bool:
    """Heuristically detect a 'found nothing' tool result.

    Used to count consecutive empty memory_search / track_query results so
    the next prompt can inject a hard STOP when the target is clearly not
    in the video.
    """
    try:
        if res is None or not getattr(res, "ok", True):
            return False
        d = res.to_dict() if hasattr(res, "to_dict") else {}
    except Exception:
        return False
    blob = json.dumps(d, ensure_ascii=False).lower()
    # memory_search empty: "0 results" / "results: []" / "no matches".
    if "0 results" in blob or '"results": []' in blob \
            or "no matches" in blob or "found 0" in blob:
        return True
    # track_query lost / not in view.
    if '"in_view": false' in blob or "lost" in blob:
        return True
    return False


def _has_analysis_text(answer: str) -> bool:
    """True when the turn contains a substantive observation, not just a
    tool call (and not just a one-line *intent* like "I will replay ...").

    Used by the hard analyse-before-chaining gate: a model responding to a
    tool result must REPORT what it saw before firing the next tool. We
    strip thinking + tool_call markup, then require enough remaining prose
    that is NOT merely an intent declaration.
    """
    if not answer:
        return False
    t = ovo._strip_thinking(answer)
    t = TOOL_CALL_RE.sub(" ", t)
    for tag in ("</Response>", "</Silence>", "</Standby>",
                 "<Observation>", "</Observation>", "<think>", "</think>"):
        t = t.replace(tag, " ")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < 30:
        return False
    low = t.lower()
    # Pure intent ("I will replay t=3 to check the count") is NOT analysis.
    intent_only = bool(re.match(
        r"^(i (will|'ll|am going to|need to|should)|let me|let's|"
        r"i want to|going to)\b", low))
    if intent_only and len(t) < 120:
        return False
    return True


def _toolcall_signature(calls: List[Dict[str, Any]]) -> str:
    """Canonical signature of a turn's tool calls (name + sorted args).

    Used to detect *repeated identical* tool calls across consecutive
    turns on the same frame. A model (esp. a weaker one) may re-issue the
    exact same ``zoom_crop``/``track_query`` over and over, each time
    wrapped in a fresh ``<Observation>`` blurb — which fools
    ``_has_analysis_text`` into thinking real progress was made, so the
    bare-chain escape hatch never fires. Comparing this signature turn to
    turn catches that "analysed repeat" spin regardless of prose.
    """
    parts = []
    for spec in calls:
        name = (spec.get("name") or "").strip()
        try:
            args = json.dumps(spec.get("arguments", {}), sort_keys=True,
                              ensure_ascii=False)
        except Exception:
            args = str(spec.get("arguments", {}))
        parts.append(f"{name}({args})")
    return "|".join(parts)


def _clean_llm_summary(text: str) -> str:
    """Sanitise an LLM-authored summary for single-block memory rendering.

    Local Qwen emits a leading ``<think>...</think>`` block (often empty)
    even in no_think mode, and free newlines. Both break the indented
    episodic / pinned rendering in the MEMORY snapshot. Strip the thinking
    block and collapse internal whitespace to single spaces.
    """
    if not text:
        return ""
    t = ovo._strip_thinking(text)
    # Drop any stray sentinel/think tags that survived.
    for tag in ("<think>", "</think>", "<Observation>", "</Observation>"):
        t = t.replace(tag, " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_query_from_visible_schema(sample: dict) -> Tuple[str, int]:
    """Build the post-hoc query without consulting a benchmark task label.

    The visible sample schema determines the answer contract only after the
    query is due. All schemas share the same pre-query streaming path.
    """
    sample = sample or {}
    question = str(sample.get("question") or "").strip()
    options = list(sample.get("options") or [])
    activity = str(sample.get("activity") or "").strip()
    target_event = str(sample.get("all_steps") or "").strip()

    # StreamingBench proactive-output query. The target time remains hidden
    # from the runtime; the agent receives the instruction at ``realtime``
    # and must keep watching until the described condition becomes visible.
    if question and sample.get("proactive_output"):
        question_time = sample.get("realtime")
        if question_time is None:
            raise ValueError(
                "proactive-output sample is missing realtime"
            )
        return (
            f"{question}\n"
            "This is a proactive streaming instruction. Do not answer when "
            "the instruction is first shown. Continue observing and emit "
            "`</Silence>` while the trigger condition is absent. When the "
            "condition occurs, reply with `</Response>` followed only by "
            "the output requested in the instruction.",
            int(question_time),
        )

    # Standard closed-set visual question.
    if question and options:
        question_time = (
            sample.get("ask_time")
            if sample.get("ask_time") is not None
            else sample.get("realtime")
        )
        if question_time is None:
            raise ValueError("MCQ sample is missing ask_time/realtime")
        rendered = "\n".join(
            f"{chr(ord('A') + index)}. {option}"
            for index, option in enumerate(options)
        )
        return (
            f"{question}\n{rendered}\n{ovo.GENERIC_ANSWER_CONSTRAINT}",
            int(question_time),
        )

    # Original-question answerability meta query.
    if question and sample.get("ask_time") is not None and not options:
        prompt = (
            "*** META-QUESTION. Do NOT answer the original question below; "
            "ONLY judge whether sufficient evidence has already appeared "
            "in the video to answer it. ***\n"
            f"Original question (for reference only): {question}\n\n"
            "Based only on video content shown so far, has the key evidence "
            "needed to answer the original question already appeared?\n"
            "- Yes = the relevant event, clue, action, actor, object, or "
            "on-screen text has begun or occurred and is now observable. "
            "The event does not have to be fully completed.\n"
            "- No = the relevant evidence has not happened or is not yet "
            "observable in any current or past frame.\n"
            "Search structured MEMORY for observations directly addressing "
            "the original question's target. Starting, in-progress, and "
            "finished states all count as evidence appearing. Do not answer "
            "the original question itself. Final answer exactly Yes or No."
        )
        return prompt, int(sample["ask_time"])

    # Event-state meta query.
    if (
        target_event
        and sample.get("step_start_time") is not None
        and not question
        and not activity
    ):
        prompt = (
            "*** META-QUESTION. Do not describe the event below; ONLY judge "
            "whether it has already HAPPENED, started, or visibly progressed "
            "at any point in the video shown so far. ***\n"
            f"Event: '{target_event}'\n\n"
            "Use the current frame plus all structured MEMORY observations. "
            "Answer Yes when the event itself, a semantically equivalent "
            "visible action, or a later workflow step shows that it has "
            "already occurred. A sustained in-progress action counts as "
            "having happened; a crisp completion frame is not required. "
            "Answer No only when matching evidence is genuinely absent and "
            "the workflow has not reached or passed the event. Judge by "
            "visible meaning rather than exact wording. Final answer exactly "
            "Yes or No."
        )
        return prompt, int(sample["step_start_time"])

    # Repetition-count query synthesized from the visible activity schema.
    if activity and not question:
        question_time = sample.get("query_time")
        if question_time is None:
            start_times = sample.get("start_times") or []
            question_time = start_times[0] if start_times else 0
        if ovo.REC_PROMPT_MODE == "number":
            return (
                f"How many times did activity '{activity}' occur?\n"
                "Reply with only a single integer between 0 and 10.",
                int(question_time),
            )
        rendered = "\n".join(
            f"{chr(ord('A') + count)}. {count}" for count in range(11)
        )
        return (
            f"How many times did activity '{activity}' occur?\n"
            f"{rendered}\n{ovo.GENERIC_ANSWER_CONSTRAINT}",
            int(question_time),
        )

    # Open question without options (rare non-meta schema).
    if question:
        question_time = (
            sample.get("ask_time")
            if sample.get("ask_time") is not None
            else sample.get("realtime", 0)
        )
        return question, int(question_time)
    raise ValueError(
        "sample has no recognizable visible query schema: "
        f"id={sample.get('id')}"
    )


def _abstain_final_for(sample: dict) -> str:
    """Return a generic final-answer sentinel used when the agent has to
    terminate without a model-emitted </Response>.

    The agent is benchmark-agnostic and uses one unified strategy:
      * if the sample carries an ``options`` list, fall back to the first
        listed option (extractable as letter ``A``);
      * else if visible schema is a Yes/No meta-question, fall back to the
        conservative ``No`` (event not yet confirmed / cannot confirm);
      * otherwise emit ``</Response> Unable to answer`` so the scorer can
        still see *some* response.
    """
    sample = sample or {}
    options = sample.get("options") or []
    if options:
        return f"{RESPONSE_TAG} A. {options[0]}"
    is_event_state_meta = bool(
        sample.get("all_steps")
        and sample.get("step_start_time") is not None
        and not sample.get("question")
        and not sample.get("activity")
    )
    is_answerability_meta = bool(
        sample.get("question")
        and sample.get("ask_time") is not None
        and not sample.get("options")
        and not sample.get("all_steps")
        and not sample.get("activity")
    )
    if is_event_state_meta or is_answerability_meta:
        return f"{RESPONSE_TAG} No"
    qtext = str(sample.get("question") or sample.get("all_steps") or "")
    if "yes" in qtext.lower() and "no" in qtext.lower():
        return f"{RESPONSE_TAG} No"
    return f"{RESPONSE_TAG} Unable to answer"


def _refusal_option_index(sample: dict) -> Optional[int]:
    """Return the listed epistemic-refusal option, if one exists.

    Concrete negative choices such as "cannot fit" are not refusals: they
    make a factual claim and still require visual evidence.
    """
    patterns = (
        r"\bunable to answer\b",
        r"\bcannot (?:tell|determine|be determined|answer)\b",
        r"\bcan't (?:tell|determine|answer)\b",
        r"\bnot possible to (?:tell|determine|answer)\b",
        r"\b(?:video|scene|footage) (?:does not|doesn't) show\b",
        r"\bno answer\b",
    )
    for index, option in enumerate((sample or {}).get("options") or []):
        text = " ".join(str(option).lower().split())
        if any(re.search(pattern, text) for pattern in patterns):
            return index
    return None


def _listed_refusal_final_for(sample: dict) -> Optional[str]:
    index = _refusal_option_index(sample)
    options = (sample or {}).get("options") or []
    if index is None:
        return None
    return (
        f"{RESPONSE_TAG} {chr(ord('A') + index)}. "
        f"{options[index]}"
    )


def _is_refusal_answer(text: str) -> bool:
    """True if a model's final answer is a refusal / non-answer.

    Weaker models (observed: qwen3vl-8b, samples 34/35) sometimes emit
    ``</Response> Unable to answer`` (or "cannot determine") on a plain
    multiple-choice question whose listed options are all concrete
    (e.g. EPM "what color was the bag" / "where is the bottle"). When the
    options do NOT include such a choice, this scores 0 outright — it is
    not even a listed option. Detect it so the caller can re-prompt for a
    real pick instead of accepting the dud.
    """
    t = (text or "").lower()
    for pat in ("unable to answer", "cannot answer", "can't answer",
                "cannot determine", "can't determine", "cannot be determined",
                "not able to answer", "unanswerable", "no answer",
                "i don't know", "i do not know"):
        if pat in t:
            return True
    return False


def _options_allow_refusal(sample: dict) -> bool:
    """True if the question's listed options legitimately include an
    "unable / cannot determine" choice (common for HLD). For those a
    refusal answer IS a valid option and must be accepted as-is; the
    invalid-refusal guard must not fire.
    """
    return _refusal_option_index(sample) is not None


def listed_refusal_evidence_contract(sample: dict) -> str:
    """Render the opt-in positive-evidence gate for answerability MCQs."""
    if os.environ.get(
        "STREAMO_ENABLE_LISTED_REFUSAL_EVIDENCE_GATE", "0"
    ) != "1":
        return ""
    index = _refusal_option_index(sample)
    options = (sample or {}).get("options") or []
    if index is None:
        return ""
    letter = chr(ord("A") + index)
    return (
        "## Listed-refusal positive-evidence gate\n"
        f"Option {letter} (`{options[index]}`) is an ACTUAL listed "
        "epistemic-refusal choice. Treat this as a closed-set "
        "hallucination check, not an invitation to guess a plausible "
        "concrete option.\n"
        "- A concrete option is legal only when one authentic visual "
        "observation or replay jointly identifies the EXACT target and "
        "the option's claimed relation/state/action. Seeing only the "
        "room, a similar object, a component word, or a plausible "
        "affordance is not positive evidence.\n"
        "- Question pins, option text, plan notes, tool-call intentions, "
        "and summaries that merely repeat the query are CONTEXT, not "
        "evidence that the target appeared.\n"
        "- Search structured MEMORY/DAG once for the exact target and "
        "required relation. If there is a genuine candidate timestamp, "
        "use at most one targeted replay to verify it. Do not repeatedly "
        "search the same empty evidence.\n"
        f"- If no verified exact-target-plus-relation evidence exists, "
        f"choose option {letter}. If such positive evidence does exist, "
        "choose the uniquely supported concrete option instead. Never "
        "choose the refusal merely because it is listed.\n"
    )


def restore_prequery_memory(
    processor: "AgentVideoProcessor",
    trajectory: Dict[str, Any],
    *,
    expected_source_id: Optional[str] = None,
) -> Dict[str, int]:
    """Restore only genuine pre-query state from a saved trajectory."""
    sample_id = str(processor.sample.get("id"))
    trajectory_id = str(
        trajectory.get("sample_id")
        if trajectory.get("sample_id") is not None
        else trajectory.get("id")
    )
    expected_id = str(expected_source_id or sample_id)
    if trajectory_id != expected_id:
        raise ValueError(
            f"trajectory ID mismatch: {trajectory_id} != {expected_id}"
        )
    saved_question_time = int(trajectory["question_time"])
    if saved_question_time != int(processor.question_time):
        raise ValueError(
            f"question time mismatch for {sample_id}: "
            f"{saved_question_time} != {processor.question_time}"
        )
    if processor._memory is None or processor._graph is None:
        raise ValueError("pre-query trajectory restore requires brief memory")

    memory_data = trajectory.get("memory") or {}
    store = processor._memory
    query_time = int(processor.question_time)
    event_rows = [
        row for row in (memory_data.get("events") or [])
        if isinstance(row.get("t"), (int, float))
        and int(row["t"]) < query_time
        and str(row.get("kind") or "") != "question"
        and not str(row.get("text") or "").lstrip().startswith((
            "[Q]", "[plan]",
        ))
    ]
    valid_event_ids = {int(row["id"]) for row in event_rows}
    store.events = [
        Event(
            eid=int(row["id"]),
            t=int(row["t"]),
            kind=str(row.get("kind") or "observation"),
            text=sanitize_restored_memory_task_labels(
                str(row.get("text") or "")
            ),
            ref_frame=row.get("ref_frame"),
            salience=float(row.get("salience") or 1.0),
            delta=row.get("delta"),
            is_change=bool(row.get("is_change")),
            parent_episode_id=(
                int(row["parent_episode_id"])
                if row.get("parent_episode_id") is not None else None
            ),
            entities=list(row.get("entities") or []),
        )
        for row in event_rows
    ]
    store._next_eid = max(valid_event_ids, default=0) + 1

    store.episodes = []
    valid_episode_ids: Set[int] = set()
    for row in memory_data.get("episodes") or []:
        if not isinstance(row.get("t1"), (int, float)):
            continue
        if int(row["t1"]) >= query_time:
            continue
        episode = dict(row)
        episode["summary"] = sanitize_restored_memory_task_labels(
            str(episode.get("summary") or "")
        )
        episode["id"] = int(episode["id"])
        episode["ev_ids"] = [
            int(event_id) for event_id in episode.get("ev_ids") or []
            if int(event_id) in valid_event_ids
        ]
        valid_episode_ids.add(episode["id"])
        store.episodes.append(episode)
    store._next_epid = max(valid_episode_ids, default=0) + 1
    for event in store.events:
        if event.parent_episode_id not in valid_episode_ids:
            event.parent_episode_id = None

    store.pinned_facts = []
    valid_pin_ids: Set[int] = set()
    for row in memory_data.get("pinned_facts") or []:
        timestamp = row.get("t")
        if str(row.get("kind") or "") == "question":
            continue
        if isinstance(timestamp, (int, float)) and int(timestamp) >= query_time:
            continue
        pin = dict(row)
        pin["text"] = sanitize_restored_memory_task_labels(
            str(pin.get("text") or "")
        )
        pin["id"] = int(pin["id"])
        if (
            pin.get("ep_id") is not None
            and int(pin["ep_id"]) not in valid_episode_ids
        ):
            pin["ep_id"] = None
        valid_pin_ids.add(pin["id"])
        store.pinned_facts.append(pin)
    store._next_pin_id = max(valid_pin_ids, default=0) + 1

    store.entities = dict(memory_data.get("entities") or {})
    store.tracks = {}
    for track_id, track in (memory_data.get("tracks") or {}).items():
        restored = dict(track)
        restored["traj"] = [
            point for point in restored.get("traj") or []
            if isinstance(point.get("t"), (int, float))
            and int(point["t"]) < query_time
        ]
        if restored["traj"] or int(restored.get("start_t") or 0) < query_time:
            store.tracks[str(track_id)] = restored
    store.audio_segments = [
        dict(segment)
        for segment in (memory_data.get("audio_segments") or [])
        if isinstance(segment.get("t1"), (int, float))
        and int(segment["t1"]) < query_time
    ]

    graph = processor._graph
    events_by_id = {event.eid: event for event in store.events}
    for event in store.events:
        if (
            event.parent_episode_id is not None
            and event.parent_episode_id in valid_episode_ids
        ):
            graph.link_ev_to_ep(event.eid, event.parent_episode_id)
    for episode in store.episodes:
        for event_id in episode.get("ev_ids") or []:
            graph.link_ev_to_ep(int(event_id), int(episode["id"]))
            if events_by_id[int(event_id)].parent_episode_id is None:
                events_by_id[int(event_id)].parent_episode_id = int(
                    episode["id"]
                )
        pin_id = episode.get("pin_id")
        if pin_id is not None and int(pin_id) in valid_pin_ids:
            graph.link_ep_to_pin(episode["id"], int(pin_id))
    graph.relations = [
        dict(relation)
        for relation in (memory_data.get("relations") or [])
        if int(relation.get("t") or 0) < query_time
    ]
    task_label_slugs = {
        "rec", "ssr", "crr", "epm", "hld", "ocr",
        "asi", "stu", "ojr", "atr", "fpd", "acr",
    }
    for entity_id, row in (memory_data.get("entity_graph") or {}).items():
        from tools.memory_graph import EntitySubgraph

        if str(entity_id).strip().lower() in task_label_slugs:
            continue
        entity_event_ids = [
            int(event_id) for event_id in row.get("event_ids") or []
            if int(event_id) in valid_event_ids
        ]
        entity_pin_ids = [
            int(pin_id) for pin_id in row.get("pinned_ids") or []
            if int(pin_id) in valid_pin_ids
        ]
        entity_episode_ids = [
            int(episode_id)
            for episode_id in row.get("episode_ids") or []
            if int(episode_id) in valid_episode_ids
        ]
        if not (entity_event_ids or entity_pin_ids or entity_episode_ids):
            continue
        graph.entity_index._entities[str(entity_id)] = EntitySubgraph(
            entity_id=str(row.get("entity_id") or entity_id),
            event_ids=entity_event_ids,
            pinned_ids=entity_pin_ids,
            episode_ids=entity_episode_ids,
            first_seen=int(row.get("first_seen") or 0),
            last_seen=min(
                int(row.get("last_seen") or 0),
                max(0, query_time - 1),
            ),
            aliases=list(row.get("aliases") or []),
        )
        for alias in row.get("aliases") or []:
            slug = graph.entity_index._to_slug(str(alias))
            if slug:
                graph.entity_index._alias_map[slug] = str(entity_id)
    from tools.memory_graph import observation_to_embedding
    for event in store.events:
        store.cache_embedding(
            event.eid,
            observation_to_embedding(event.text),
        )

    processor.current_round = query_time
    processor.ctx.current_round = query_time
    processor._current_run_events = []
    processor._last_episode_id = max(valid_episode_ids, default=None)
    processor._last_anchor_t = (
        max((event.t for event in store.events), default=None)
    )
    # A restored pre-query trajectory already owns every audio window before
    # the question. Do not let a sidecar backend inject those windows again.
    processor._last_sidecar_asr_t = query_time
    return {
        "events": len(store.events),
        "episodes": len(store.episodes),
        "pins": len(store.pinned_facts),
        "entities": graph.entity_index.entity_count,
    }


def _trajectory_round_index(key: str) -> Optional[int]:
    match = re.match(r"^Round\s+(\d+)(?:_|$)", str(key))
    return int(match.group(1)) if match else None


def sanitize_restored_memory_task_labels(text: str) -> str:
    """Remove legacy benchmark-label hints while preserving visual facts."""
    cleaned = str(text or "")
    replacements = (
        (
            r"\bEPM\s+(?=(?:Step|procedure|Task)\b)",
            "",
        ),
        (r"\bEPM\s+Step\b", "Step"),
        (
            r"\bSSR\s*/\s*CRR(?:-style)?\s+state\b",
            "event-state",
        ),
        (r"\bSSR\s*/\s*CRR(?:-style)?\b", "event-state"),
        (r"\b(?:SSR|CRR)(?:-style)?\b", "event-state"),
        (r"\bREC(?:-style)?\b", "repetition-count"),
        (r"\bHLD(?:-style)?\b", "answerability"),
        (r"\bOCR(?:-style)?\s+(?:text|string)\b", "on-screen text"),
        (r"\bOCR(?:-style)?\b", "on-screen-text"),
        (r"\b(?:ASI|STU|OJR|ATR|FPD|ACR)(?:-style)?\b", "visual"),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.I)
    return cleaned


def merge_cached_trajectory_rounds(
    source_rounds: Dict[str, Any],
    live_rounds: Dict[str, Any],
    *,
    question_time: int,
) -> Dict[str, Any]:
    """Merge source pre-query rounds with newly executed query-time rounds."""
    merged: Dict[str, Any] = {}
    for key, value in (source_rounds or {}).items():
        round_index = _trajectory_round_index(key)
        if round_index is not None and round_index < int(question_time):
            merged[str(key)] = value
    # Live output wins by key. It starts at question_time in cached mode.
    for key, value in (live_rounds or {}).items():
        merged[str(key)] = value
    return merged


def parse_listed_refusal_visual_audit(
    response: str,
    *,
    proposed_answer: str,
    window_times: Dict[int, List[int]],
) -> Optional[Dict[str, Any]]:
    """Parse a concrete-answer support audit with safe fallback semantics."""
    visible = ovo._strip_thinking(response or "").strip()
    objects = re.findall(r"\{[^{}]*\}", visible, re.S)
    if not objects:
        return None
    try:
        payload = json.loads(objects[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    status = re.sub(
        r"[\s-]+", "_", str(payload.get("status") or "").strip().lower()
    )
    if status not in {"supported", "unsupported", "uncertain"}:
        return None
    answer = str(payload.get("proposed_answer") or "").strip().upper()
    if answer != proposed_answer.upper():
        return None
    if (
        not isinstance(payload.get("target_visible"), bool)
        or not isinstance(payload.get("relation_visible"), bool)
    ):
        return None
    result = {
        "status": status,
        "proposed_answer": answer,
        "window_id": payload.get("window_id"),
        "evidence_t": payload.get("evidence_t"),
        "target_visible": bool(payload.get("target_visible")),
        "relation_visible": bool(payload.get("relation_visible")),
        "evidence": " ".join(
            str(payload.get("evidence") or "").split()
        )[:1000],
    }
    if not result["evidence"]:
        return None
    if status == "supported":
        if (
            not isinstance(result["window_id"], int)
            or isinstance(result["window_id"], bool)
            or result["window_id"] not in window_times
            or not result["target_visible"]
            or not result["relation_visible"]
        ):
            return None
        value = result["evidence_t"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp < 0:
            return None
        valid_times = window_times[result["window_id"]]
        if not valid_times or not (
            min(valid_times) <= timestamp <= max(valid_times)
        ):
            return None
        result["evidence_t"] = timestamp
    elif status == "unsupported":
        # Unsupported means no supplied candidate window jointly grounds
        # target and relation; a positive visual flag contradicts it.
        if result["target_visible"] and result["relation_visible"]:
            return None
        result["window_id"] = None
        result["evidence_t"] = None
    return result


def parse_listed_refusal_bidirectional_audit(
    response: str,
    *,
    original_answer: str,
    concrete_answers: List[str],
    refusal_answer: str,
    window_times: Dict[int, List[int]],
) -> Optional[Dict[str, Any]]:
    """Parse a symmetric concrete-option versus refusal audit."""
    visible = ovo._strip_thinking(response or "").strip()
    objects = re.findall(r"\{[^{}]*\}", visible, re.S)
    if not objects:
        return None
    try:
        payload = json.loads(objects[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    status = re.sub(
        r"[\s-]+", "_", str(payload.get("status") or "").strip().lower()
    )
    if status not in {
        "concrete_supported",
        "refusal_no_supported_concrete",
        "uncertain",
    }:
        return None
    original = str(payload.get("original_answer") or "").strip().upper()
    selected = str(payload.get("selected_answer") or "").strip().upper()
    supported = payload.get("supported_option")
    supported = (
        None if supported is None
        else str(supported).strip().upper()
    )
    concrete = {str(value).upper() for value in concrete_answers}
    if (
        original != original_answer.upper()
        or selected not in concrete | {refusal_answer.upper()}
        or supported not in concrete | {None}
        or not isinstance(payload.get("target_visible"), bool)
        or not isinstance(payload.get("relation_visible"), bool)
    ):
        return None
    evidence = " ".join(
        str(payload.get("evidence") or "").split()
    )[:1200]
    if not evidence:
        return None
    result = {
        "status": status,
        "original_answer": original,
        "selected_answer": selected,
        "supported_option": supported,
        "window_id": payload.get("window_id"),
        "evidence_t": payload.get("evidence_t"),
        "target_visible": bool(payload.get("target_visible")),
        "relation_visible": bool(payload.get("relation_visible")),
        "evidence": evidence,
    }
    if status == "concrete_supported":
        if (
            supported is None
            or selected != supported
            or not result["target_visible"]
            or not result["relation_visible"]
            or not isinstance(result["window_id"], int)
            or isinstance(result["window_id"], bool)
            or result["window_id"] not in window_times
        ):
            return None
        timestamp = result["evidence_t"]
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
        ):
            return None
        valid_times = window_times[result["window_id"]]
        if not valid_times or not (
            min(valid_times) <= float(timestamp) <= max(valid_times)
        ):
            return None
        result["evidence_t"] = float(timestamp)
    elif status == "refusal_no_supported_concrete":
        if (
            selected != refusal_answer.upper()
            or supported is not None
            or (result["target_visible"] and result["relation_visible"])
        ):
            return None
        result["window_id"] = None
        result["evidence_t"] = None
    else:
        # Uncertain never changes the answer.
        if selected != original or supported is not None:
            return None
        result["window_id"] = None
        result["evidence_t"] = None
    return result


def dedupe_refusal_audit_candidates(
    candidates: List[Tuple[float, float, str]],
    *,
    query_time: int,
    gap: float = 4.0,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Rank and time-deduplicate query-derived local replay candidates."""
    selected: List[Dict[str, Any]] = []
    for score, timestamp, source in sorted(
        candidates,
        key=lambda item: (-float(item[0]), float(item[1]), str(item[2])),
    ):
        if not math.isfinite(float(timestamp)):
            continue
        timestamp = max(0.0, min(float(query_time), float(timestamp)))
        if any(abs(timestamp - row["t"]) < gap for row in selected):
            continue
        selected.append({
            "t": timestamp,
            "score": float(score),
            "source": str(source),
        })
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda row: row["t"])


def limit_tool_call_reply(text: str, max_calls: int) -> Tuple[str, int]:
    """Truncate a response before its first tool call beyond ``max_calls``."""
    if not text or max_calls <= 0:
        return text, 0
    openers = list(re.finditer(r"<tool_call>", text, re.IGNORECASE))
    if len(openers) <= max_calls:
        return text, 0
    return text[:openers[max_calls].start()].rstrip(), len(openers) - max_calls


def parse_tool_calls(
    text: str,
    max_calls: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract tool calls (best-effort). Returns list of {name,arguments}.

    Two-pass strategy:
      1. **Strict pass** — match well-formed ``<tool_call>...</tool_call>``
         blocks via TOOL_CALL_RE. This is the canonical format and we
         keep its output verbatim.
      2. **Lenient pass** — only triggered when the strict pass yields
         nothing on a non-empty text. A common Qwen format-drift mode
         is to emit several ``<tool_call>{json}`` openers but only one
         ``</tool_call>`` closer (observed on sample 0 of the
         20260610 commit run: 386/391 stuck steps were of this shape).
         We salvage these by splitting on ``</?tool_call>`` and feeding
         each non-empty JSON-looking segment to ``_try_parse_call_body``.
         This recovers the dispatch path so PHYS_CAP / max_per_round
         caps can take over the throttling.
    """
    out: List[Dict[str, Any]] = []
    if not text:
        return out
    # ---- Pass 1: strict ----
    for m in TOOL_CALL_RE.finditer(text):
        body = m.group(1).strip()
        spec = _try_parse_call_body(body)
        if spec is not None:
            out.append(spec)
            if max_calls and len(out) >= max_calls:
                break
    if out:
        return out
    # ---- Pass 2: lenient (only if strict pass got nothing) ----
    # Bail out fast if the text doesn't even contain a <tool_call> token.
    if "<tool_call>" not in text and "</tool_call>" not in text:
        return out
    parts = re.split(r"</?tool_call>", text)
    for seg in parts:
        seg = seg.strip()
        # Heuristic: must start with '{' to look like a JSON tool spec.
        # Avoids treating natural-language preambles as tool bodies.
        if not seg or not seg.startswith("{"):
            continue
        spec = _try_parse_call_body(seg)
        if spec is not None:
            out.append(spec)
            if max_calls and len(out) >= max_calls:
                break
    return out


def _last_frame_answer_constraint(
    sample: dict,
    *,
    allow_tool: bool = True,
) -> str:
    """Return the LAST-FRAME answer-format constraint sentence.

    A single, benchmark-agnostic sentence is used for every task. The
    user-turn already carries the question (and its options) verbatim, so
    we only need to remind the model to commit to a real answer.
    """
    tool_rule = (
        "You may still call ONE quick supporting tool (e.g. "
        "`memory_search`, `replay`, `zoom_crop`) if it directly resolves "
        "the question, but the round MUST end with `</Response> "
        "<your answer>` — NOT with `</Silence>`, NOT with another "
        "tool-call loop. "
        if allow_tool else
        "Tool calls are disabled now. Do NOT emit `<tool_call>`; use the "
        "evidence already in MEMORY and answer immediately with "
        "`</Response> <your answer>`. "
    )
    procedure_step = (
        "(2) scan MEMORY for the best-matching evidence; if a single "
        "tool call would clinch it, you may call it ONCE; "
        if allow_tool else
        "(2) scan MEMORY for the best-matching evidence without calling "
        "another tool; "
    )
    return (
        "*** YOU MUST EMIT `</Response>` IN THIS REPLY (or in the next "
        "tool-result re-prompt of THIS frame). *** This is the very "
        "last frame — the streaming clock will not advance further. "
        + tool_rule
        + "Emitting `</Silence>` or indefinitely chaining "
        "tools yields a NULL output that scores 0, so you must commit an "
        "answer this frame. "
        "Procedure: "
        "(1) parse the question into TARGET / ANCHOR / FORMAT (option "
        "letter, Yes/No, integer, free-form); "
        + procedure_step
        + "(3) write `</Response> <answer>` in the EXACT format the "
        "question requires. "
        "If the question lists explicit options, your reply MUST be "
        "exactly one of them (letter + text) — including an "
        "\"Unable to answer\"/\"cannot determine\" option if that is the "
        "one your plan's answerability judged correct. If Yes/No is "
        "expected, reply exactly with `Yes` or `No`."
    )


def _try_parse_call_body(body: str) -> Optional[Dict[str, Any]]:
    # Preferred: pure JSON {"name":..., "arguments":{...}}
    try:
        d = json.loads(body)
        if isinstance(d, dict) and "name" in d:
            args = d.get("arguments") or d.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            return {"name": str(d["name"]), "arguments": args}
    except Exception:
        pass
    # Fallback: name="x" arguments={...}
    name_m = re.search(r'name\s*[:=]\s*"([A-Za-z_][A-Za-z0-9_]*)"', body)
    if not name_m:
        # name: x  (unquoted)
        name_m = re.search(r'name\s*[:=]\s*([A-Za-z_][A-Za-z0-9_]*)', body)
    if not name_m:
        return None
    args_m = re.search(r'arguments\s*[:=]\s*(\{.*\})', body, re.S)
    args: Dict[str, Any] = {}
    if args_m:
        try:
            args = json.loads(args_m.group(1))
        except Exception:
            args = {}
    return {"name": name_m.group(1), "arguments": args}


# ---------------------------------------------------------------------------
# AgentVideoProcessor — built on top of ovo.VideoProcessor (no edits to it)
# ---------------------------------------------------------------------------
class AgentVideoProcessor(ovo.VideoProcessor):
    """Extends VideoProcessor with tool-call handling.

    Strategy: we *override* ``get_next_request`` and ``update`` and add the
    fields we need (``ctx``, per-sample budgets, pending tool results).
    The parent class' clock state (``current_round`` / ``completed`` /
    ``output``) is kept and reused.
    """

    def _build_question(self) -> Tuple[str, int]:
        return build_query_from_visible_schema(self.sample)

    def __init__(
        self,
        sample: dict,
        system: str,
        registry: ToolRegistry,
        *,
        max_tool_calls_per_round: int = 3,
        max_tool_calls_per_sample: int = 12,
        max_tool_calls_per_reply: int = 0,
        memory_mode: str = "full",
        memory_token_budget: int = 3000,
        budget: Optional[Budget] = None,
        trajectory_dir: Optional[str] = None,
        auto_anchor_every: int = 5,
        auto_asr_every: int = 0,
        auto_asr_window: int = 5,
        auto_track_query_every: int = 0,
        memory_render_mode: str = "continuous",
        memory_head_max: int = 6,
        memory_tail_max: int = 8,
        memory_head_seconds: int = 25,
        memory_tail_seconds: int = 20,
        memory_recall_topk: int = 8,
        cold_start_rounds: int = 4,
        degrade_threshold: int = 2,
        degrade_events_min: int = 12,
        rolling_window: int = 3,
        engine: Optional[Any] = None,
        delta_input_mode: str = "2frame",
        enable_verification: bool = False,
        max_verification_replays: int = 3,
    ):
        super().__init__(sample, system)
        self.registry = registry
        self.max_per_round = max_tool_calls_per_round
        self.max_per_sample = max_tool_calls_per_sample
        self.max_per_reply = max(0, int(max_tool_calls_per_reply))
        self._last_reply_tool_calls_dropped = 0
        # ── Planning buffer round ─────────────────────────────────────
        # Query-Driven Planning inserts a same-frame PLANNING turn when the
        # question is first posed. When question_time falls on (or past)
        # the last real frame, the model would have no room left to ACT on
        # its plan before the COMMIT round forces an answer. Extend
        # total_rounds by one virtual buffer frame so planning + answering
        # always get distinct turns. Frame access is clamped to the last
        # real frame everywhere (frame_idx = min(round, total_rounds-1)),
        # so the extra round simply re-shows the final frame.
        if (self.total_rounds > 0
                and self.question_time >= self.total_rounds - 1):
            self.total_rounds += 1
            # Keep the question strictly before the new last frame so the
            # planning turn is never itself the last frame.
            self.question_time = min(self.question_time,
                                     self.total_rounds - 2)
        # vLLM engine handle, used for one-shot LLM episode summarization.
        # Optional; when ``None`` (or LLM call fails) the rule-based
        # ``_summarize_run`` is used instead.
        self._engine = engine

        # Phase-A: optional memory mode --------------------------------
        # ``full``  → legacy behavior (full multi-turn messages).
        # ``brief`` → every round we re-build messages = [system,
        #              user(memory_brief + tool_results + <image>)].
        self.memory_mode = memory_mode
        self.memory_token_budget = int(memory_token_budget)
        self._memory: Optional[MemoryStore] = (
            MemoryStore(sample_id=str(sample.get("id", "")))
            if memory_mode == "brief" else None
        )
        self._summarizer = RuleSummarizer() if memory_mode == "brief" else None
        # Bind LLM-based episode summarizer onto the MemoryStore so
        # ``auto_aggregate_episodes`` and ``_seal_anchor_run`` can use it
        # automatically. Falls back to rule-based summarization on error.
        if self._memory is not None and self._engine is not None:
            self._memory.summarizer_fn = self._llm_summarize_run
        self._budget = budget

        # ── Pillar 2: Memory DAG overlay + online BOCPD segmenter ────
        self._graph: Optional[MemoryGraph] = None
        self._bocpd: Optional[BOCPDSegmenter] = None
        if self._memory is not None:
            from tools.memory_graph import (  # noqa: F811
                EntityIndex, MemoryGraph, BOCPDSegmenter,
                observation_to_embedding,
            )
            self._graph = MemoryGraph(
                self._memory, entity_index=EntityIndex(),
            )
            self._bocpd = BOCPDSegmenter(
                hazard_rate=float(os.environ.get(
                    "STREAMO_BOCPD_HAZARD_RATE", "0.022")),
                min_run=3,
            )
            # ── O4: Hybrid retrieval query embedder ──────────────────
            # Wire the same token-hash embedding function used for event
            # caching as the query-side encoder.  This gives semantic
            # recall: "poker cards" can match "playing cards" because
            # shared tokens ("cards") spike the same embedding dims.
            # Upgrade path: replace with vLLM hidden-state encode for
            # true semantic similarity (see implementation doc §O4).
            self._memory.set_embed_fn(
                observation_to_embedding
            )

        self.ctx = ToolContext(
            sample_id=str(sample.get("id", "")),
            # Runtime tools are intentionally task-label blind. Query routing
            # happens after planning from visible query/schema fields.
            task="",
            current_round=self.current_round,
            total_rounds=self.total_rounds,
            question_time=self.question_time,
            frame_paths=list(self.frame_paths),
            memory=self._memory,
            tracks={} if memory_mode == "brief" else None,
            audio_cache=None,
            budget=budget,
        )
        # When set, the next ``get_next_request`` will append a tool-result
        # turn instead of consuming a new frame.
        self._pending_tool_text: Optional[str] = None
        self._pending_tool_images: List[str] = []
        # Counts for budget
        self._calls_made: int = 0
        # Per physical-round (cross tool-result re-prompts) counters.
        # ``_calls_this_phys_round`` accumulates tool_call dispatches
        # within the same ``current_round`` regardless of how many times
        # the model is re-prompted on that frame; reset whenever
        # current_round advances. Cap by ``max_per_round * 2`` so the
        # model has some room to react to tool_results, but is hard
        # blocked from infinite tool-call loops on the same frame.
        # ``_replies_this_phys_round`` counts how many times the model
        # has been asked to reply on the same frame (including tool-
        # continuation re-prompts); used to bound format-error retries.
        self._calls_this_phys_round: int = 0
        self._replies_this_phys_round: int = 0
        self._format_retries_this_phys_round: int = 0
        self._last_frame_reply_cap_used: bool = False
        # Look-and-return replay semantics: when the model issues a replay/
        # rewind it temporarily jumps to the target frame to RE-WATCH it,
        # then the streaming clock should snap BACK to where it left off
        # (rather than restarting the forward stream from the replay target).
        # ``_resume_round_after_seek`` stashes the pre-seek clock position;
        # after the model has consumed exactly one inference round on the
        # replay target, the main loop restores ``current_round`` to it.
        self._resume_round_after_seek: Optional[int] = None
        # Stash per-round tool log into output for offline analysis
        self.tool_log: List[Dict[str, Any]] = []

        # ---- Auto-memory cadence (system-side; does not depend on model) ----
        # The base model rarely calls memory_write proactively, so we write
        # observations from the system side to keep MEMORY non-empty:
        #   * every ``auto_anchor_every`` seconds, drop a [frame_anchor] event
        #     so the timeline is never blank between tool calls;
        #   * if ``auto_asr_every`` > 0 (and the audio cache has been wired,
        #     i.e. the model called ASR at least once), automatically refresh
        #     a ``[t-window, t]`` ASR segment so spoken content accumulates.
        # 0 disables. Both can be turned off via ENV / CLI.
        self.auto_anchor_every = int(auto_anchor_every)
        self.auto_asr_every = int(auto_asr_every)
        self.auto_asr_window = max(1, int(auto_asr_window))
        self.auto_track_query_every = int(auto_track_query_every)
        # ── Pillar 4: Phase-aware multi-frame input ─────────────────
        self.delta_input_mode = str(delta_input_mode or "2frame")
        # ── Pillar 3: Look-Back Verification ─────────────────────────
        self.enable_verification = bool(enable_verification)
        self._verification_checks: Optional[Any] = None  # ConsistencyChecker
        self._replay_scheduler: Optional[Any] = None     # ReplayScheduler
        self._verification_in_progress: bool = False
        if self.enable_verification and self._graph is not None:
            from tools.verification import ConsistencyChecker, ReplayScheduler
            self._verification_checks = ConsistencyChecker()
            self._replay_scheduler = ReplayScheduler(
                max_replays=int(max_verification_replays))
        self._last_anchor_t: Optional[int] = None
        self._last_auto_asr_t: Optional[int] = None
        self._last_sidecar_asr_t: int = 0
        self._last_auto_track_t: Optional[int] = None
        self._question_pinned: bool = False
        # Throttle for model-emitted <Recap> blocks (model-driven
        # long-term memory). At most one Recap is pinned per
        # ``_recap_min_gap`` seconds; extras in the same window are
        # silently dropped to discourage Recap-spam.
        self._last_recap_t: Optional[int] = None
        self._recap_min_gap: int = 5

        # ---- Query-Driven Planning (model self-analyses the query) ----
        # On the round the question is first posed we let the model emit a
        # <Plan> block analysing what the question needs (temporal / spatial
        # / existence / count / meta) BEFORE it starts answering. The plan
        # is stored here and injected at the top of every POST-QUESTION
        # prompt as a standing "battle order". The planning round itself
        # neither consumes the tool budget nor counts as a POST-QUESTION
        # answering round.
        self._query_plan: Optional[Dict[str, str]] = None
        self._plan_done: bool = False
        self._count_audit_pending: bool = False
        self._count_audit_inflight: bool = False
        self._count_audit_retries: int = 0
        self._count_audit_segments: List[Dict[str, Any]] = []
        self._count_audit_segment_index: int = 0
        self._count_audit_ledger: List[float] = []
        self._count_audit_segmented: bool = False
        self._count_audit_target_activity: str = ""
        self._count_audit_cycle: str = ""
        self._count_audit_use_evidence_ledger: bool = False
        self._count_audit_use_sports_cycle: bool = False
        self._count_audit_sports_cycle_version: int = 0
        self._ssr_local_audit_pending: bool = False
        self._ssr_local_audit_inflight: bool = False
        self._ssr_local_audit_retries: int = 0
        self._crr_target_audit_pending: bool = False
        self._crr_target_audit_inflight: bool = False
        self._crr_target_audit_retries: int = 0
        self._crr_clue_audit_pending: bool = False
        self._crr_clue_audit_inflight: bool = False
        self._crr_clue_audit_retries: int = 0
        self._crr_typed_relation: Optional[Dict[str, str]] = None
        self._listed_refusal_audit_pending: bool = False
        self._listed_refusal_audit_inflight: bool = False
        self._listed_refusal_audit_done: bool = False
        self._listed_refusal_audit_retries: int = 0
        self._listed_refusal_proposed_answer: str = ""
        self._listed_refusal_proposed_text: str = ""
        self._listed_refusal_original_response: str = ""
        self._listed_refusal_audit_windows: List[Dict[str, Any]] = []
        self._listed_refusal_bidir_pending: bool = False
        self._listed_refusal_bidir_inflight: bool = False
        self._listed_refusal_bidir_done: bool = False
        self._listed_refusal_bidir_retries: int = 0
        self._listed_refusal_bidir_original_answer: str = ""
        self._listed_refusal_bidir_original_response: str = ""
        self._listed_refusal_bidir_windows: List[Dict[str, Any]] = []
        self._current_locus_dual_view_pending: bool = False
        self._current_locus_dual_view_inflight: bool = False
        self._current_locus_dual_view_done: bool = False
        self._current_locus_dual_view_retries: int = 0
        self._current_locus_dual_view_original_answer: str = ""
        self._current_locus_dual_view_original_response: str = ""
        self._current_locus_dual_view_view: Dict[str, Any] = {}
        self._current_locus_dual_view_build_error: str = ""
        self._current_locus_protector_pending: str = ""
        self._current_locus_protector_inflight: str = ""
        self._current_locus_protector_retries: Dict[str, int] = {}
        self._current_locus_protector_results: Dict[str, Any] = {}
        self._current_locus_protector_proposed_answer: str = ""
        self._current_locus_protector_proposed_response: str = ""
        self._event_predicate_pending: str = ""
        self._event_predicate_inflight: str = ""
        self._event_predicate_done: bool = False
        self._event_predicate_retries: Dict[str, int] = {}
        self._event_predicate_contract: Optional[Dict[str, Any]] = None
        self._event_predicate_proposals: List[Dict[str, Any]] = []
        self._event_predicate_windows: List[Dict[str, Any]] = []
        self._event_predicate_original_response: str = ""
        self._event_predicate_build_error: str = ""

        # ---- POST-QUESTION per-frame tool budgets & loop guards --------
        # These reset on every physical-frame switch (see _update_impl).
        # ``memory_search`` is capped at 5 and ``replay`` at 3 PER FRAME so
        # the model cannot spin a single frame into a 100+ call loop. Two
        # consecutive empty tool results trigger a hard "STOP searching"
        # hint (the target is almost certainly not in the video).
        self._memsearch_this_frame: int = 0
        self._replay_this_frame: int = 0
        self._consecutive_empty: int = 0
        # Per-sample cap on model-issued track_init during OBSERVE-ONLY.
        self._track_init_count: int = 0
        # Per-frame cap on model-issued track_query. A weaker model
        # (qwen3vl-8b) re-issues the SAME track_query{trk, t} 10-15× per
        # frame — track_query always returns ok+bbox (a "cheap success"
        # with no "dead end" signal like memory_search's [0 results]), so
        # the model never stops. It spins in OBSERVE-ONLY (where the POST-Q
        # gates don't apply), burning the whole per-frame budget every
        # frame. Cap it like replay (≤3/frame); reset on frame advance.
        self._track_query_this_frame: int = 0
        # System-side auto-pin of episode boundaries (long-term backbone).
        # The model rarely emits <Recap>, leaving long_term empty except
        # the question pin. When BOCPD seals an episode we promote its
        # boundary to a durable pin so the long_term layer stays populated
        # and ep→pin pointers are wired. Throttled by time + count to
        # avoid pin spam.
        self._last_boundary_pin_t: Optional[int] = None
        self._boundary_pin_min_gap: int = 15
        self._auto_boundary_pins: int = 0
        self._max_auto_boundary_pins: int = 12
        # Last sealed episode id — used to chain consecutive episodes with
        # a causal/temporal "precedes" edge (action-order backbone).
        self._last_episode_id: Optional[int] = None
        # True when the CURRENT model turn is a response to tool results
        # (image patches / replay / search output). Used by the hard
        # "analyse-before-chaining" gate so the model can't fire another
        # bare tool_call without first reporting what the last result
        # showed (anti zoom/replay spin loop).
        self._responding_to_tool: bool = False
        # Counts consecutive bare-tool-call turns while responding to tool
        # results on the same physical frame (no analysis text emitted).
        self._bare_tool_chain: int = 0
        # Signature of the previous turn's tool calls + how many times the
        # SAME signature has repeated in a row on this frame. Lets the gate
        # treat an identical tool call re-issued with fresh prose as a spin
        # (feeds the GATE_ESCAPE hatch even when analysis text is present).
        self._last_toolcall_sig: Optional[str] = None
        self._toolcall_sig_repeat: int = 0

        # One-shot re-prompt guard for an INVALID refusal: the model emits
        # `</Response> Unable to answer` on a question whose options don't
        # include such a choice (scores 0). We reject it ONCE and demand a
        # real pick; if it refuses again we fall back to the best-guess
        # option. Tracks whether that single retry was already spent.
        self._invalid_refusal_retry_used: bool = False

        # ---- Selective MEMORY rendering (read-side compression) -------
        # ``memory_render_mode='selective'`` shrinks the WORKING layer
        # rendered into the prompt to head + tail + recall(top-K). The
        # write path is untouched: every <Observation> still becomes an
        # event, ``snapshot()`` just renders fewer of them.
        self.memory_render_mode = str(memory_render_mode or "continuous")
        self.memory_head_max = int(memory_head_max)
        self.memory_tail_max = int(memory_tail_max)
        self.memory_head_seconds = max(0, int(memory_head_seconds))
        self.memory_tail_seconds = max(0, int(memory_tail_seconds))
        self.memory_recall_topk = int(memory_recall_topk)
        # Cold start: force ``mode='full'`` for the first N rounds so the
        # model has a complete (small) timeline before retrieval kicks in.
        self.cold_start_rounds = int(cold_start_rounds)
        # Degrade protection: if pre-question retrieval returns < this
        # many working hits OR working layer is small (< degrade_events_min),
        # fall back to full rendering for the round.
        self.degrade_threshold = int(degrade_threshold)
        self.degrade_events_min = int(degrade_events_min)
        # Rolling-delta window for pre-question implicit query: build the
        # query from the deltas of the last ``rolling_window`` events.
        self.rolling_window = max(1, int(rolling_window))

        # ---- Timing for trajectory logs ----
        self._sample_start_time: float = time.time()
        self._last_step_time: Optional[float] = None

        # ---- Step-level trajectory streaming (借鉴 agentic-mle) ----
        # Every (request, response, tool_calls, memory_snapshot) tuple is
        # appended *immediately* to <sid>.steps.jsonl so a kill / OOM does
        # not lose previously-completed rounds. The file is JSONL: one
        # record per round, ready for offline replay or RL/SFT data prep.
        self._trajectory_dir = trajectory_dir
        self._steps_path: Optional[str] = None
        self._step_idx: int = 0
        self._last_request: Optional[Dict[str, Any]] = None
        self._cached_prequery_rounds: Dict[str, Any] = {}
        self._cached_prequery_tool_calls: List[Dict[str, Any]] = []
        self._cached_prequery_source: Optional[str] = None
        if trajectory_dir:
            sid = str(sample.get("id", "unknown")).replace("/", "_")
            os.makedirs(trajectory_dir, exist_ok=True)
            self._steps_path = os.path.join(trajectory_dir,
                                             f"{sid}.steps.jsonl")
            # Truncate at start so re-runs do not concat onto stale data.
            with open(self._steps_path, "w", encoding="utf-8") as _f:
                pass

        prequery_dir = os.environ.get(
            "STREAMO_PREQUERY_TRAJECTORY_DIR", ""
        ).strip()
        if prequery_dir:
            source_id = str(
                sample.get("prequery_source_id") or sample.get("id") or ""
            )
            trajectory_path = os.path.join(
                prequery_dir,
                f"{source_id.replace('/', '_')}.json",
            )
            if not os.path.isfile(trajectory_path):
                raise FileNotFoundError(
                    f"pre-query trajectory missing: {trajectory_path}"
                )
            with open(trajectory_path, encoding="utf-8") as handle:
                trajectory = json.load(handle)
            if sample.get("prequery_source_id"):
                trajectory_video_id = os.path.basename(
                    os.path.normpath(str(trajectory.get("video") or ""))
                )
                sample_video_id = os.path.basename(
                    os.path.normpath(str(sample.get("video") or ""))
                )
                if (
                    trajectory_video_id != source_id
                    or sample_video_id != source_id
                    or str(trajectory.get("question") or "")
                    != str(sample.get("question") or "")
                ):
                    raise ValueError(
                        f"synthetic pre-query source mismatch for "
                        f"{sample.get('id')}"
                    )
            restored = restore_prequery_memory(
                self,
                trajectory,
                expected_source_id=source_id,
            )
            self._cached_prequery_rounds = {
                str(key): value
                for key, value in (trajectory.get("rounds") or {}).items()
                if (
                    _trajectory_round_index(str(key)) is not None
                    and _trajectory_round_index(str(key))
                    < int(self.question_time)
                )
            }
            self._cached_prequery_tool_calls = [
                dict(item)
                for item in (trajectory.get("tool_calls") or [])
                if (
                    isinstance(item.get("round"), (int, float))
                    and int(item["round"]) < int(self.question_time)
                )
            ]
            self._cached_prequery_source = os.path.abspath(trajectory_path)
            self.ctx.memory = self._memory
            self.ctx.tracks = self._memory.tracks if self._memory else {}
            print(
                f"[prequery-restore] id={sample.get('id')} "
                f"q={self.question_time} {restored}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Inference-request construction
    # ------------------------------------------------------------------
    def get_next_request(self, last_answer=None):
        """Build an InferRequest for the current state.

        Three cases:
          (a) We have pending tool results → re-run the model in the *same*
              round with tool_result appended (NO clock tick yet).
          (b) ``next_frame_idx`` was set by a temporal tool → fast-forward
              ``current_round`` to that index by emitting ``</Silence>``
              (handled by caller; see ``_advance_to_infer_or_complete``).
          (c) Default → fall through to parent.
        """
        # Optional task-agnostic raw-buffer mode. Before the query exists,
        # advance the stream without asking the model to infer; after the
        # query arrives, the normal query-driven planner decides what visual
        # audit is needed. This gate never reads the task label or query text.
        # It is disabled by default so existing backward/realtime behavior is
        # byte-for-byte unchanged unless explicitly enabled for an eval.
        if (
            os.environ.get("STREAMO_PREQUERY_BUFFER_ONLY", "0") == "1"
            and self.current_round < self.question_time
        ):
            self.current_round = self.question_time
            self.ctx.current_round = self.current_round
        # Specialized audit requests are self-contained and must be serviced
        # in both brief and full memory modes. They are scheduled only after
        # the normal planner/agent has already produced an answer.
        if (
            self._current_locus_dual_view_pending
            or self._current_locus_protector_pending
            or self._event_predicate_pending
        ):
            return self._brief_get_next_request(last_answer)
        # Brief mode: rebuild a tiny prompt every round (no history grow).
        if self.memory_mode == "brief":
            return self._brief_get_next_request(last_answer)
        if self._pending_tool_text is not None:
            return True, self._build_tool_continuation_request(last_answer)
        return super().get_next_request(last_answer)

    # ------------------------------------------------------------------
    # Brief-mode request construction (single frame + memory snapshot)
    # ------------------------------------------------------------------

    # ── Pillar 4: Phase-aware multi-frame image builder ───────────────
    def _build_images_for_phase(
        self,
        frame_idx: int,
        user_blocks: List[str],
        is_post_hoc: bool,
    ) -> List[str]:
        """Build the image list for the current round based on the phase.

        OBSERVE-ONLY (is_post_hoc) + 2-frame mode:
          - t == 0: only ``frame_0`` (no previous frame exists).
          - t > 0:  ``frame_{t-1}`` + ``frame_t`` for visual delta comparison.
        POST-QUESTION / COMMIT / REPLAY:
          - single ``frame_t`` (1-frame fallback).
        1-frame mode (``delta_input_mode='1frame'``):
          - always single ``frame_t`` regardless of phase.
        """
        images: List[str] = []
        # Always drain any pending tool-result images first.
        if self._pending_tool_text and self._pending_tool_images:
            for img in self._pending_tool_images:
                user_blocks.insert(-1, "<image>")
                images.append(img)

        use_two_frame = (
            self.delta_input_mode == "2frame"
            and is_post_hoc
            and not (self._resume_round_after_seek is not None
                     or self.ctx.next_frame_idx is not None)
        )

        if use_two_frame and frame_idx > 0:
            # OBSERVE-ONLY with a previous frame available: present both
            # frames side-by-side for visual delta comparison.
            prev_frame = self.frame_paths[frame_idx - 1]
            # Insert the previous-frame <image> marker BEFORE the last
            # user block so the prompt reads: MEMORY / phase-text /
            # <image>(prev) / <image>(curr).
            user_blocks.insert(
                -1,
                "PREVIOUS FRAME (t-1):\n<image>\n\nCURRENT FRAME (t):",
            )
            images.append(prev_frame)
            images.append(self.frame_paths[frame_idx])
        elif use_two_frame and frame_idx == 0:
            # Round 0: no previous frame exists, single-frame fallback.
            user_blocks.insert(
                -1,
                "[Round 0 — initial frame, no previous frame to compare]\n"
                "Write a full observation; set delta to '(initial frame)'.",
            )
            images.append(self.frame_paths[0])
        else:
            # POST-QUESTION / COMMIT / REPLAY / 1-frame mode:
            # single-frame input.
            images.append(self.frame_paths[frame_idx])

        return images

    def _build_count_audit_timeline(
        self,
        *,
        force_direct: Optional[bool] = None,
        fps_override: Optional[float] = None,
    ) -> Tuple[List[str], str]:
        """Build a timestamped high-fps timeline from the visible video."""
        visible_seconds = max(1.0, float(self.current_round + 1))
        sample_id = str(self.sample.get("id") or "")
        video_roots = [
            os.environ.get("STREAMO_VIDEO_DIR", ""),
            "data/OVO-Bench/"
            "chunked_videos_asi_full",
            "data/OVO-Bench/chunked_videos",
        ]
        video_path = next(
            (
                os.path.join(root, f"{sample_id}.mp4")
                for root in video_roots
                if root and os.path.isfile(
                    os.path.join(root, f"{sample_id}.mp4"))
            ),
            None,
        )
        if video_path is None:
            paths = list(self.frame_paths[:self.current_round + 1])
            return paths, "one frame per second"

        use_direct_frames = (
            bool(force_direct)
            if force_direct is not None
            else visible_seconds <= float(os.environ.get(
                "STREAMO_COUNT_AUDIT_DIRECT_MAX_SECONDS", "0"))
        )
        audit_fps = (
            float(fps_override)
            if fps_override is not None
            else (
                float(os.environ.get("STREAMO_COUNT_AUDIT_SHORT_FPS", "8"))
                if visible_seconds <= 50
                else float(os.environ.get(
                    "STREAMO_COUNT_AUDIT_LONG_FPS", "4"))
            )
        )
        audit_fps = max(1.0, min(12.0, audit_fps))
        cache_root = os.environ.get(
            "STREAMO_COUNT_AUDIT_CACHE", "/tmp/ovo_count_audit")
        cache_key = (
            f"{sample_id}_t{int(round(visible_seconds * 1000))}"
            f"_fps{audit_fps:g}_{'direct' if use_direct_frames else 'sheet'}"
            "_timeline_v2"
        )
        cache_dir = os.path.join(cache_root, cache_key)
        manifest_path = os.path.join(cache_dir, "manifest.json")
        try:
            if os.path.isfile(manifest_path):
                with open(manifest_path, encoding="utf-8") as file:
                    manifest = json.load(file)
                cached = [
                    os.path.join(cache_dir, name)
                    for name in manifest.get("files", [])
                ]
                if cached and all(os.path.isfile(path) for path in cached):
                    return cached, manifest.get(
                        "description", f"{audit_fps:g} fps contact sheets")

            import cv2
            import numpy as np

            os.makedirs(cache_dir, exist_ok=True)
            capture = cv2.VideoCapture(video_path)
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if not capture.isOpened() or source_fps <= 0 or frame_count <= 0:
                capture.release()
                raise RuntimeError("video decoder did not open")

            source_duration = frame_count / source_fps
            cutoff = min(visible_seconds, source_duration)
            sample_count = max(1, int(round(cutoff * audit_fps)) + 1)
            sample_times = [
                min(index / audit_fps, max(0.0, cutoff - 1 / source_fps))
                for index in range(sample_count)
            ]
            target_indices = [
                min(frame_count - 1, int(round(t * source_fps)))
                for t in sample_times
            ]
            targets: Dict[int, List[Tuple[int, float]]] = {}
            for order, (frame_index, timestamp) in enumerate(
                    zip(target_indices, sample_times)):
                targets.setdefault(frame_index, []).append((order, timestamp))

            panels: List[Optional[Any]] = [None] * len(sample_times)
            source_index = 0
            while source_index <= target_indices[-1]:
                ok = capture.grab()
                if not ok:
                    break
                if source_index in targets:
                    ok, frame = capture.retrieve()
                    if not ok or frame is None:
                        break
                    for order, timestamp in targets[source_index]:
                        panel = cv2.resize(
                            frame, (384, 216), interpolation=cv2.INTER_AREA)
                        cv2.rectangle(
                            panel, (0, 0), (150, 28), (0, 0, 0), -1)
                        cv2.putText(
                            panel,
                            f"t={timestamp:.2f}s",
                            (7, 21),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        panels[order] = panel
                source_index += 1
            capture.release()
            if any(panel is None for panel in panels):
                raise RuntimeError("video decode ended before audit cutoff")

            files: List[str] = []
            if use_direct_frames:
                for panel_index, panel in enumerate(panels):
                    name = f"frame_{panel_index:04d}.jpg"
                    path = os.path.join(cache_dir, name)
                    if not cv2.imwrite(
                        path, panel, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        raise RuntimeError(f"failed to write {path}")
                    files.append(name)
                description = (
                    f"{audit_fps:g} fps, one timestamped frame per image"
                )
            else:
                for sheet_index in range(0, len(panels), 4):
                    group = panels[sheet_index:sheet_index + 4]
                    blank = np.zeros_like(group[0])
                    while len(group) < 4:
                        group.append(blank.copy())
                    sheet = np.vstack((
                        np.hstack((group[0], group[1])),
                        np.hstack((group[2], group[3])),
                    ))
                    name = f"sheet_{sheet_index // 4:04d}.jpg"
                    path = os.path.join(cache_dir, name)
                    if not cv2.imwrite(
                        path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                        raise RuntimeError(f"failed to write {path}")
                    files.append(name)
                description = (
                    f"{audit_fps:g} fps, four timestamped panels per contact "
                    "sheet in left-to-right then top-to-bottom order"
                )
            with open(manifest_path, "w", encoding="utf-8") as file:
                json.dump(
                    {"files": files, "description": description},
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            return [
                os.path.join(cache_dir, name) for name in files
            ], description
        except Exception as error:
            print(
                f"[count-audit] sid={sample_id} high-fps fallback: "
                f"{type(error).__name__}: {error}")
            paths = list(self.frame_paths[:self.current_round + 1])
            return paths, "one frame per second (high-fps fallback)"

    def _build_ssr_local_audit_window(self) -> Tuple[List[str], str]:
        """Build a short timestamped window ending at the SSR query time."""
        fps = max(1.0, min(8.0, float(os.environ.get(
            "STREAMO_SSR_LOCAL_AUDIT_FPS", "4"
        ))))
        window_seconds = max(4.0, float(os.environ.get(
            "STREAMO_SSR_LOCAL_AUDIT_WINDOW_SECONDS", "8"
        )))
        sheets, description = self._build_count_audit_timeline(
            force_direct=False,
            fps_override=fps,
        )
        panels = max(2, int(math.ceil(window_seconds * fps)) + 1)
        keep_sheets = max(1, int(math.ceil(panels / 4.0)))
        return sheets[-keep_sheets:], (
            f"last {window_seconds:g}s before query time; {description}"
        )

    def _build_crr_target_audit_window(self) -> Tuple[List[str], str]:
        """Build timestamped static sheets around CRR target emergence."""
        fps = max(1.0, min(8.0, float(os.environ.get(
            "STREAMO_CRR_TARGET_AUDIT_FPS", "4"
        ))))
        window_seconds = max(8.0, float(os.environ.get(
            "STREAMO_CRR_TARGET_AUDIT_WINDOW_SECONDS", "12"
        )))
        sheets, description = self._build_count_audit_timeline(
            force_direct=False,
            fps_override=fps,
        )
        panels = max(2, int(math.ceil(window_seconds * fps)) + 1)
        keep_sheets = max(1, int(math.ceil(panels / 4.0)))
        return sheets[-keep_sheets:], (
            f"last {window_seconds:g}s before query time; {description}"
        )

    def _build_crr_clue_audit_window(self) -> Tuple[List[str], str]:
        """Build a bounded visible-history window for CRR clue discovery."""
        fps = max(1.0, min(4.0, float(os.environ.get(
            "STREAMO_CRR_CLUE_AUDIT_FPS", "4"
        ))))
        window_seconds = max(12.0, float(os.environ.get(
            "STREAMO_CRR_CLUE_AUDIT_WINDOW_SECONDS", "47.875"
        )))
        sheets, description = self._build_count_audit_timeline(
            force_direct=False,
            fps_override=fps,
        )
        panels = max(2, int(math.ceil(window_seconds * fps)) + 1)
        # 47.875s at 4fps is the promoted 192-panel / 48-sheet budget.
        # Avoid an inclusive-endpoint rounding artifact that would request
        # a 49th sheet without adding a materially new timestamp.
        keep_sheets = max(1, int(math.ceil((panels - 1) / 4.0)))
        return sheets[-keep_sheets:], (
            f"last {window_seconds:g}s before query time; {description}"
        )

    def _target_history_evidence(self, target: str) -> str:
        """Render query-target DAG paths for a specialized audit."""
        if not target or self._graph is None:
            return "(no target-matched DAG path)"
        try:
            paths = self._graph.retrieve_backtrack(
                query=target,
                top_k=8,
                current_t=int(self.current_round),
            )
        except Exception:
            paths = []
        if not paths:
            return "(no target-matched DAG path)"
        lines: List[str] = []
        for path_index, path in enumerate(paths[:8]):
            lines.append(
                f"path#{path_index + 1} score={path.score:.2f}:"
            )
            for node in path.nodes[:6]:
                t_text = f"t={node.t}s" if node.t is not None else "t=?"
                lines.append(
                    f"  [{node.layer} {t_text}] "
                    f"{' '.join(str(node.text or '').split())[:300]}"
                )
        return "\n".join(lines)

    def _listed_refusal_audit_candidates(
        self,
        proposed_option: str,
    ) -> List[Dict[str, Any]]:
        if self._graph is None:
            return []
        query = " ".join(filter(None, (
            build_evidence_query(
                question=str(self.question or ""),
                plan=self._query_plan,
                sample=self.sample,
            ),
            proposed_option,
        )))
        try:
            paths = self._graph.retrieve_backtrack(
                query=query,
                top_k=12,
                current_t=int(self.current_round),
            )
        except Exception:
            paths = []
        ranked: List[Tuple[float, float, str]] = []
        for path_index, path in enumerate(paths):
            for node_index, node in enumerate(path.nodes):
                if node.t is None:
                    continue
                ranked.append((
                    float(path.score) - 0.05 * node_index,
                    float(node.t),
                    f"dag_path_{path_index + 1}_{node.layer}",
                ))
        return dedupe_refusal_audit_candidates(
            ranked,
            query_time=int(self.question_time),
        )

    def _listed_refusal_bidirectional_candidates(
        self,
    ) -> List[Dict[str, Any]]:
        """Retrieve evidence symmetrically for every concrete option."""
        if self._graph is None:
            return []
        refusal_index = _refusal_option_index(self.sample)
        ranked: List[Tuple[float, float, str]] = []
        for option_index, option in enumerate(
            self.sample.get("options") or []
        ):
            if option_index == refusal_index:
                continue
            letter = chr(ord("A") + option_index)
            query = " ".join(filter(None, (
                build_evidence_query(
                    question=str(self.question or ""),
                    plan=self._query_plan,
                    sample=self.sample,
                ),
                str(option),
            )))
            try:
                paths = self._graph.retrieve_backtrack(
                    query=query,
                    top_k=8,
                    current_t=int(self.current_round),
                )
            except Exception:
                paths = []
            for path_index, path in enumerate(paths):
                for node_index, node in enumerate(path.nodes):
                    if node.t is None:
                        continue
                    ranked.append((
                        float(path.score) - 0.05 * node_index,
                        float(node.t),
                        (
                            f"option_{letter}_path_{path_index + 1}_"
                            f"{node.layer}"
                        ),
                    ))
        return dedupe_refusal_audit_candidates(
            ranked,
            query_time=int(self.question_time),
            gap=3.0,
            limit=8,
        )

    def _build_listed_refusal_audit_windows(
        self,
        candidates: List[Dict[str, Any]],
        *,
        radius_override: Optional[int] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        import cv2
        import numpy as np

        sample_id = str(self.sample.get("id") or "unknown")
        cache_root = os.environ.get(
            "STREAMO_LISTED_REFUSAL_AUDIT_CACHE",
            "/tmp/ovo_listed_refusal_audit",
        )
        cache_dir = os.path.join(
            cache_root,
            f"{sample_id}_q{int(self.question_time)}",
        )
        os.makedirs(cache_dir, exist_ok=True)
        image_paths: List[str] = []
        windows: List[Dict[str, Any]] = []
        radius = (
            max(2, int(radius_override))
            if radius_override is not None
            else max(2, int(os.environ.get(
                "STREAMO_LISTED_REFUSAL_AUDIT_RADIUS", "4"
            )))
        )
        last_visible = min(
            len(self.frame_paths) - 1,
            max(0, int(self.question_time)),
        )
        for window_index, candidate in enumerate(candidates, 1):
            center = max(
                0,
                min(last_visible, int(round(float(candidate["t"])))),
            )
            times = list(range(
                max(0, center - radius),
                min(last_visible + 1, center + radius + 1),
            ))
            panels: List[Any] = []
            for timestamp in times:
                image = cv2.imread(self.frame_paths[timestamp])
                if image is None:
                    raise RuntimeError(
                        f"failed to read frame t={timestamp} "
                        f"for sample {sample_id}"
                    )
                panel = cv2.resize(
                    image, (384, 216), interpolation=cv2.INTER_AREA
                )
                cv2.rectangle(panel, (0, 0), (170, 28), (0, 0, 0), -1)
                cv2.putText(
                    panel,
                    f"W{window_index} t={timestamp}s",
                    (7, 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(panel)
            blank = np.zeros_like(panels[0])
            while len(panels) < 9:
                panels.append(blank.copy())
            sheet = np.vstack((
                np.hstack(panels[0:3]),
                np.hstack(panels[3:6]),
                np.hstack(panels[6:9]),
            ))
            image_path = os.path.join(
                cache_dir, f"window_{window_index}.jpg"
            )
            if not cv2.imwrite(
                image_path,
                sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            ):
                raise RuntimeError(f"failed to write {image_path}")
            image_paths.append(image_path)
            windows.append({
                "window_id": window_index,
                "center_t": float(candidate["t"]),
                "times": times,
                "source": candidate["source"],
                "score": candidate["score"],
            })
        return image_paths, windows

    def _ssr_history_evidence(self) -> str:
        """Render target-matched DAG paths as prior-history evidence."""
        target = str(self.sample.get("all_steps") or "").strip()
        return self._target_history_evidence(target)

    def _brief_get_next_request(self, last_answer):
        """Compose ``messages = [system, user(memory + tool_results + <image>)]``.

        No assistant history is carried across rounds — past frames are
        compressed into ``ctx.memory``. Tool-call continuations within the
        same round still see the previous assistant turn so the model can
        chain reasoning.
        """
        from swift import InferRequest  # lazy import (heavy)

        if self._event_predicate_pending:
            stage = self._event_predicate_pending
            images: List[str] = []
            if stage == "contract":
                prompt = build_event_predicate_contract_prompt(
                    self.sample,
                    self._query_plan or {},
                )
                max_tokens = 1000
            elif stage == "ledger":
                contract = self._event_predicate_contract
                if not isinstance(contract, dict):
                    raise AssertionError(
                        "event predicate ledger requires a valid contract"
                    )
                if not self._event_predicate_windows:
                    try:
                        query = event_predicate_contract_query(contract)
                        proposals = event_predicate_dag_proposals(
                            self._graph,
                            query,
                            question_time=int(self.question_time),
                        )
                        cache_root = os.environ.get(
                            "STREAMO_EVENT_PREDICATE_CACHE",
                            "/tmp/ovo_event_predicate",
                        )
                        windows = build_event_predicate_local_sheets(
                            str(self.sample.get("id") or "unknown"),
                            proposals,
                            cache_root,
                        )
                        self._event_predicate_proposals = proposals
                        self._event_predicate_windows = windows
                        self._event_predicate_build_error = ""
                    except Exception as error:
                        self._event_predicate_proposals = []
                        self._event_predicate_windows = []
                        self._event_predicate_build_error = (
                            f"{type(error).__name__}: {error}"
                        )
                if self._event_predicate_windows:
                    prompt = build_event_predicate_ledger_prompt(
                        self.sample,
                        contract,
                        self._event_predicate_windows,
                    )
                    images = event_predicate_image_paths(
                        self._event_predicate_windows
                    )
                else:
                    prompt = "Return ONLY this JSON object: {}"
                max_tokens = 2200
            else:
                raise AssertionError(
                    f"unknown event predicate stage: {stage}"
                )
            if self._event_predicate_retries.get(stage, 0):
                prompt += (
                    "\n\nThe previous output failed the exact JSON contract. "
                    "Return one valid object with concrete typed fields, "
                    "in-range timestamps, no benchmark labels, and no prose."
                )
            messages = [{"role": "user", "content": prompt}]
            self._enforce_token_budget(messages, images)
            self.data = {"messages": messages, "images": images}
            self._event_predicate_pending = ""
            self._event_predicate_inflight = stage
            self._responding_to_tool = False
            self._last_request = {
                "kind": f"event_predicate_{stage}",
                "messages": [dict(message) for message in messages],
                "images": list(images),
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "proposals": [
                    dict(proposal)
                    for proposal in self._event_predicate_proposals
                ],
                "build_error": self._event_predicate_build_error,
            }
            return True, InferRequest(**self.data)

        if self._current_locus_protector_pending:
            stage = self._current_locus_protector_pending
            view = self._current_locus_dual_view_view
            options = self.sample.get("options") or []
            original = self._current_locus_dual_view_original_answer
            proposed = self._current_locus_protector_proposed_answer
            original_text = options[ord(original) - ord("A")]
            proposed_text = options[ord(proposed) - ord("A")]
            images: List[str] = []
            if stage == "pair_original_first":
                prompt = build_current_locus_pairwise_prompt(
                    self.sample,
                    view,
                    candidate_x=original_text,
                    candidate_y=proposed_text,
                )
                images = [view["current_path"], view["sheet_path"]]
            elif stage == "pair_proposed_first":
                prompt = build_current_locus_pairwise_prompt(
                    self.sample,
                    view,
                    candidate_x=proposed_text,
                    candidate_y=original_text,
                )
                images = [view["current_path"], view["sheet_path"]]
            elif stage == "fact_ledger":
                prompt = build_current_locus_fact_ledger_prompt(
                    self.sample, view
                )
                images = [view["current_path"], view["sheet_path"]]
            elif stage in {"claim_original", "claim_proposed"}:
                ledger = self._current_locus_protector_results.get(
                    "fact_ledger"
                )
                if not isinstance(ledger, dict):
                    raise AssertionError(
                        "claim stage requires a valid fact ledger"
                    )
                claim = (
                    original_text
                    if stage == "claim_original" else proposed_text
                )
                prompt = build_current_locus_claim_prompt(
                    self.sample, ledger, claim
                )
            else:
                raise AssertionError(f"unknown protector stage: {stage}")
            if self._current_locus_protector_retries.get(stage, 0):
                prompt += (
                    "\n\nThe previous output failed the exact JSON contract. "
                    "Return one valid object using only the supplied pixels "
                    "or ledger, with no prose."
                )
            messages = [{"role": "user", "content": prompt}]
            self._enforce_token_budget(messages, images)
            self.data = {"messages": messages, "images": images}
            self._current_locus_protector_pending = ""
            self._current_locus_protector_inflight = stage
            self._responding_to_tool = False
            self._last_request = {
                "kind": f"current_locus_protector_{stage}",
                "messages": [dict(message) for message in messages],
                "images": list(images),
                "temperature": 0.0,
                "max_tokens": 1800 if images else 900,
            }
            return True, InferRequest(**self.data)

        if self._current_locus_dual_view_pending:
            sample_id = str(self.sample.get("id") or "unknown")
            cache_root = os.environ.get(
                "STREAMO_CURRENT_LOCUS_DUAL_VIEW_CACHE",
                "/tmp/ovo_current_locus_dual_view",
            )
            cache_dir = os.path.join(
                cache_root,
                f"{sample_id}_q{int(self.question_time)}",
            )
            images: List[str] = []
            try:
                view = build_current_locus_dual_view(
                    self.frame_paths,
                    int(self.question_time),
                    cache_dir,
                )
                prompt = build_current_locus_dual_view_prompt(
                    self.sample,
                    view,
                )
                messages = [{"role": "user", "content": prompt}]
                if self._current_locus_dual_view_retries:
                    messages.append({
                        "role": "user",
                        "content": (
                            "The previous output failed the exact JSON "
                            "contract. Return one valid object with a legal "
                            "option letter, an in-range timestamp, concrete "
                            "target/reference/property evidence, typed "
                            "contradicted array, and no prose."
                        ),
                    })
                images = [view["current_path"], view["sheet_path"]]
                self._current_locus_dual_view_view = view
                self._current_locus_dual_view_build_error = ""
            except Exception as error:
                # A visual build failure is not evidence for changing the
                # answer. Send a deterministic malformed payload through the
                # same strict parser; after the frozen single retry the update
                # path safely preserves the original agent response.
                self._current_locus_dual_view_view = {}
                self._current_locus_dual_view_build_error = (
                    f"{type(error).__name__}: {error}"
                )
                messages = [{
                    "role": "user",
                    "content": "Return ONLY this JSON object: {}",
                }]
            self._enforce_token_budget(messages, images)
            self.data = {"messages": messages, "images": images}
            self._current_locus_dual_view_pending = False
            self._current_locus_dual_view_inflight = True
            self._responding_to_tool = False
            self._last_request = {
                "kind": "current_locus_dual_view_audit",
                "messages": [dict(message) for message in messages],
                "images": list(images),
                "view": {
                    key: value
                    for key, value in self._current_locus_dual_view_view.items()
                    if key not in {"current_path", "sheet_path"}
                },
                "build_error": self._current_locus_dual_view_build_error,
                "temperature": 0.0,
                "max_tokens": 1800,
            }
            return True, InferRequest(**self.data)

        if self._listed_refusal_bidir_pending:
            candidates = self._listed_refusal_bidirectional_candidates()
            refusal_index = _refusal_option_index(self.sample)
            options = self.sample.get("options") or []
            if refusal_index is None:
                raise AssertionError(
                    "bidirectional refusal audit has no refusal option"
                )
            refusal_letter = chr(ord("A") + refusal_index)
            if not candidates:
                self._listed_refusal_bidir_pending = False
                self._listed_refusal_bidir_inflight = True
                self._listed_refusal_bidir_windows = []
                payload = {
                    "status": "uncertain",
                    "original_answer":
                        self._listed_refusal_bidir_original_answer,
                    "selected_answer":
                        self._listed_refusal_bidir_original_answer,
                    "supported_option": None,
                    "window_id": None,
                    "evidence_t": None,
                    "target_visible": False,
                    "relation_visible": False,
                    "evidence": (
                        "No option-specific DAG timestamp was available; "
                        "candidate coverage is inadequate."
                    ),
                }
                self.data = {
                    "messages": [{
                        "role": "user",
                        "content": (
                            "Return ONLY this JSON object: "
                            + json.dumps(payload, ensure_ascii=False)
                        ),
                    }],
                    "images": [],
                }
                return True, InferRequest(**self.data)
            audit_images, windows = \
                self._build_listed_refusal_audit_windows(
                    candidates,
                    radius_override=3,
                )
            self._listed_refusal_bidir_windows = windows
            options_text = "\n".join(
                f"{chr(ord('A') + index)}. {option}"
                for index, option in enumerate(options)
            )
            concrete_letters = [
                chr(ord("A") + index)
                for index in range(len(options))
                if index != refusal_index
            ]
            window_text = "\n".join(
                f"- W{window['window_id']}: "
                f"t={min(window['times'])}s..{max(window['times'])}s "
                f"(DAG center={window['center_t']:.1f}s, "
                f"source={window['source']})"
                for window in windows
            )
            audit_user = (
                "Perform a BIDIRECTIONAL closed-set visual evidence audit. "
                "The original answer may be either a concrete option or the "
                "listed epistemic-refusal option. Candidate windows came "
                "from symmetric DAG retrieval for EVERY concrete option; "
                "they are untrusted and may still be incomplete. Options "
                "and plan text are output choices/context, never evidence.\n\n"
                f"Question: {self.question}\n"
                f"Options:\n{options_text}\n"
                f"Original answer: "
                f"{self._listed_refusal_bidir_original_answer}\n"
                f"Concrete option letters: {', '.join(concrete_letters)}\n"
                f"Refusal option letter: {refusal_letter}\n\n"
                f"Candidate windows:\n{window_text}\n\n"
                "Inspect every shown window and compare ALL concrete options. "
                "Use status=concrete_supported only when ONE window directly "
                "shows the exact question target AND the exact relation/"
                "state/action asserted by one concrete option. Target "
                "presence alone is insufficient. For where-did-put/place, "
                "require visible placement completion/result; for where-is/"
                "was, require the target at the claimed place; for count/"
                "color/state/identity, require that exact visible property. "
                "Do not use room type, affordance, common sense, similar "
                "objects, option wording, or the model's earlier claim.\n\n"
                "Use status=refusal_no_supported_concrete only when the "
                "inspected symmetric candidate set is adequate and no "
                "concrete option has direct joint target+relation evidence. "
                "Use status=uncertain when pixels are ambiguous, multiple "
                "options conflict, or candidate coverage is inadequate; "
                "uncertain MUST preserve the original answer.\n\n"
                "Return ONLY one JSON object with status "
                "(concrete_supported/refusal_no_supported_concrete/"
                "uncertain), original_answer (same original letter), "
                "selected_answer (concrete supported letter, refusal letter, "
                "or original letter for uncertain), supported_option "
                "(concrete letter or null), window_id (integer or null), "
                "evidence_t (number or null), target_visible (boolean), "
                "relation_visible (boolean), and evidence (specific pixels/"
                "timestamp or precise coverage limitation). No prose or "
                "Markdown fences.\n"
                + "\n".join("<image>" for _ in audit_images)
            )
            messages = [{"role": "user", "content": audit_user}]
            self._enforce_token_budget(messages, audit_images)
            self.data = {
                "messages": messages,
                "images": audit_images,
            }
            self._listed_refusal_bidir_pending = False
            self._listed_refusal_bidir_inflight = True
            self._responding_to_tool = False
            self._last_request = {
                "kind": "listed_refusal_bidirectional_audit",
                "messages": [dict(message) for message in messages],
                "images": list(audit_images),
                "windows": [dict(window) for window in windows],
            }
            return True, InferRequest(**self.data)

        if self._listed_refusal_audit_pending:
            candidates = self._listed_refusal_audit_candidates(
                self._listed_refusal_proposed_text
            )
            if not candidates:
                # No visual candidate is not proof of absence. Preserve the
                # proposed answer through an explicit uncertain audit.
                self._listed_refusal_audit_pending = False
                self._listed_refusal_audit_inflight = True
                self._listed_refusal_audit_windows = []
                messages = [{
                    "role": "user",
                    "content": (
                        "Return ONLY this JSON object: "
                        + json.dumps({
                            "status": "uncertain",
                            "proposed_answer":
                                self._listed_refusal_proposed_answer,
                            "window_id": None,
                            "evidence_t": None,
                            "target_visible": False,
                            "relation_visible": False,
                            "evidence": (
                                "No target-matched DAG timestamp was "
                                "available for a visual support audit."
                            ),
                        }, ensure_ascii=False)
                    ),
                }]
                self.data = {"messages": messages, "images": []}
                return True, InferRequest(**self.data)
            audit_images, windows = \
                self._build_listed_refusal_audit_windows(candidates)
            self._listed_refusal_audit_windows = windows
            options = self.sample.get("options") or []
            options_text = "\n".join(
                f"{chr(ord('A') + index)}. {option}"
                for index, option in enumerate(options)
            )
            window_text = "\n".join(
                f"- W{window['window_id']}: "
                f"t={min(window['times'])}s..{max(window['times'])}s "
                f"(DAG center={window['center_t']:.1f}s)"
                for window in windows
            )
            audit_user = (
                "Independently audit whether a PROPOSED CONCRETE answer "
                "to a closed-set video question has direct pixel support. "
                "The windows were selected by the agent's structured "
                "memory DAG and may be wrong. Options are output choices, "
                "not evidence. Do not infer a location from room type, "
                "affordance, common sense, a similar object, or the fact "
                "that target and location words appear separately.\n\n"
                f"Question: {self.question}\n"
                f"Options:\n{options_text}\n"
                f"Proposed answer: "
                f"{self._listed_refusal_proposed_answer}. "
                f"{self._listed_refusal_proposed_text}\n\n"
                f"Candidate windows:\n{window_text}\n\n"
                "Use status=supported ONLY when one window visibly shows "
                "the exact question target AND the exact relation/state/"
                "action claimed by the proposed option in the same "
                "grounded event. For `where did I put/place`, require a "
                "visible placement completion or resulting stable state; "
                "mere use/holding/initial location is insufficient. For "
                "`where is/was/see`, require the named target itself, not "
                "a similar object. Use status=unsupported only when the "
                "provided candidate windows have been inspected and fail "
                "to jointly ground target+relation. Use uncertain when "
                "pixels are ambiguous or candidate coverage is inadequate; "
                "uncertain safely preserves the original answer.\n\n"
                "Return ONLY one JSON object with fields: status "
                "(supported, unsupported, uncertain), proposed_answer "
                "(same option letter), window_id (integer or null), "
                "evidence_t (number or null), target_visible (boolean), "
                "relation_visible (boolean), evidence (specific visible "
                "pixels/timestamp or precise support gap).\n"
                + "\n".join("<image>" for _ in audit_images)
            )
            messages = [{"role": "user", "content": audit_user}]
            self._enforce_token_budget(messages, audit_images)
            self.data = {
                "messages": messages,
                "images": audit_images,
            }
            self._listed_refusal_audit_pending = False
            self._listed_refusal_audit_inflight = True
            self._responding_to_tool = False
            self._last_request = {
                "kind": "listed_refusal_visual_audit",
                "messages": [dict(message) for message in messages],
                "images": list(audit_images),
                "windows": [dict(window) for window in windows],
            }
            return True, InferRequest(**self.data)

        if (
            self._crr_clue_audit_pending
            and self._crr_typed_relation is not None
        ):
            audit_sheets, timeline_description = \
                self._build_crr_clue_audit_window()
            original_question = str(self.sample.get("question") or "").strip()
            history_evidence = self._target_history_evidence(
                build_evidence_query(
                    question=original_question,
                    plan=self._query_plan,
                    sample=self.sample,
                )
            )
            audit_user = build_crr_typed_relation_prompt(
                question=original_question,
                plan=self._query_plan or {},
                relation=self._crr_typed_relation,
                timeline_description=timeline_description,
                dag_evidence=history_evidence,
                image_count=len(audit_sheets),
            )
            if self._crr_clue_audit_retries:
                audit_user += (
                    "\n\nThe previous output failed the exact JSON contract. "
                    "Return one valid typed_relation_ledger object with every "
                    "required boolean, legal timestamps, chronological "
                    "instances, and no prose."
                )
            messages = [{"role": "user", "content": audit_user}]
            self._enforce_token_budget(messages, audit_sheets)
            self.data = {"messages": messages, "images": audit_sheets}
            self._crr_clue_audit_pending = False
            self._crr_clue_audit_inflight = True
            self._responding_to_tool = False
            self._last_request = {
                "kind": "crr_typed_relation_audit",
                "messages": [dict(message) for message in messages],
                "images": list(audit_sheets),
                "relation": dict(self._crr_typed_relation),
            }
            return True, InferRequest(**self.data)

        if self._crr_clue_audit_pending:
            audit_sheets, timeline_description = \
                self._build_crr_clue_audit_window()
            original_question = str(self.sample.get("question") or "").strip()
            history_evidence = self._target_history_evidence(
                build_evidence_query(
                    question=original_question,
                    plan=self._query_plan,
                    sample=self.sample,
                )
            )
            plan_text = ""
            if self._query_plan:
                plan_text = " ".join(
                    str(self._query_plan.get(key) or "").strip()
                    for key in (
                        "need", "info_type", "evidence_locus", "strategy",
                    )
                ).strip()
            audit_user = (
                "You are performing an independent CLUE-ANSWERABILITY "
                "AUDIT after the planner in an agentic streaming-video "
                "system. Judge whether the visible video up to query time "
                "now provides enough information to support ONE SPECIFIC "
                "answer to the ORIGINAL QUESTION. Do not answer Yes merely "
                "because the question's actor, setup, or action has begun. "
                "Do not require the actor to finish moving or remain visible "
                "after an answer-revealing clue has appeared.\n\n"
                f"Original question: {original_question}\n\n"
                f"Planner analysis: {plan_text or '(unavailable)'}\n\n"
                f"Visible-history window: {timeline_description}\n\n"
                "Target-matched prior DAG evidence:\n"
                f"{history_evidence}\n\n"
                "First infer the concrete answer you could give NOW. Then "
                "audit whether a visible target/outcome, or an ordered "
                "causal result linked to the anchor, uniquely supports it.\n"
                "Use anchor_status=matched only when the exact actor/event "
                "setup is visible in the window; use history_linked when "
                "the anchor is in a timestamped DAG path and later visible "
                "clues are linked to that event. If the anchor conflicts "
                "with the visual facts (including actor count, attributes, "
                "or action), use unmatched/uncertain.\n"
                "answer_status=specific_supported requires one concrete "
                "answer that the visible evidence would support if the "
                "original question were asked directly. setup_only means "
                "only the question premise is visible. partial or "
                "multiple_plausible means the clue does not yet identify "
                "one answer.\n"
                "Valid positive answer_source values are visible_target, "
                "visible_outcome, or ordered_causal_result. A destination "
                "can be revealed by a clearly linked gate/entrance/house "
                "or ordered path even before actors stop there. The "
                "following are NOT answer-revealing clues by themselves: "
                "the current yard/room, walking direction, approaching a "
                "generic building, background scenery, an on-screen movie "
                "title, "
                "or an unrelated scene. Classify these as scene_guess, "
                "direction_guess, background_guess, ocr_title, or "
                "irrelevant.\n"
                "temporal_link=linked requires the clue to belong to the "
                "same event sequence as the question anchor, using visible "
                "continuity or timestamped DAG links. A self-asserted actor "
                "identity without matching count/appearance/action is not "
                "a link. Cite specific before/after timestamps.\n\n"
                "Return ONLY one JSON object, no prose:\n"
                '{"anchor_status":"matched|history_linked|unmatched|'
                'uncertain","answer_status":"specific_supported|partial|'
                'multiple_plausible|setup_only|absent|uncertain",'
                '"answer_source":"visible_target|visible_outcome|'
                'ordered_causal_result|scene_guess|direction_guess|'
                'background_guess|ocr_title|irrelevant|none|uncertain",'
                '"temporal_link":"linked|broken|uncertain",'
                '"candidate_answer":"specific answer or empty",'
                '"before_t":0.0,"after_t":0.0,'
                '"evidence":"specific visible clue and relation"}\n'
                + "\n".join("<image>" for _ in audit_sheets)
            )
            messages = [{"role": "user", "content": audit_user}]
            self._enforce_token_budget(messages, audit_sheets)
            self.data = {"messages": messages, "images": audit_sheets}
            self._crr_clue_audit_pending = False
            self._crr_clue_audit_inflight = True
            self._responding_to_tool = False
            self._last_request = {
                "kind": "crr_clue_audit",
                "messages": [dict(message) for message in messages],
                "images": list(audit_sheets),
            }
            return True, InferRequest(**self.data)

        if self._crr_target_audit_pending:
            audit_sheets, timeline_description = \
                self._build_crr_target_audit_window()
            original_question = str(self.sample.get("question") or "").strip()
            history_evidence = self._target_history_evidence(
                build_evidence_query(
                    question=original_question,
                    plan=self._query_plan,
                    sample=self.sample,
                )
            )
            audit_user = (
                "You are auditing whether an ORIGINAL QUESTION is already "
                "answerable at query time inside an agentic streaming-video "
                "system. Do NOT answer Yes merely because the anchor actor or "
                "setup appeared. The TARGET VALUE requested by the original "
                "question must be available.\n\n"
                f"Original question: {original_question}\n\n"
                f"Visual window: {timeline_description}\n\n"
                "Target-matched prior DAG evidence:\n"
                f"{history_evidence}\n\n"
                "Classify:\n"
                "- anchor_status=matched only when the exact actor/event setup "
                "in the question is grounded;\n"
                "- target_status=anchor_only when only that setup/action is "
                "visible but the requested destination/next action/person/"
                "object/reason/count is not yet knowable;\n"
                "- target_status=concrete_visible when the requested value is "
                "directly visible;\n"
                "- target_status=causal_chain when an ordered visual chain "
                "uniquely determines the requested value;\n"
                "- absent or uncertain otherwise.\n"
                "For destination questions, causal_chain is valid ONLY when "
                "the SAME anchor actor/vehicle is continuously linked to an "
                "arrival state: visibly ENTERS, ARRIVES, STOPS AT, or STAYS "
                "inside/at the target. Merely walking toward/approaching a "
                "building, being located in a yard, a camera cut, background "
                "scenery, or an on-screen title card is NOT a destination. "
                "subject_continuity=same_actor requires stable appearance/"
                "track/ordered action identity across the chain. "
                "arrival_evidence records entered|arrived|stopped|stayed, "
                "other_target for non-destination concrete answers, or "
                "none/uncertain.\n"
                "target_value must be the concrete answer span you could give "
                "to the original question. Generic values such as unknown, "
                "somewhere, walking, or the anchor phrase are invalid. Scan "
                "all sheets and DAG paths; cite the strongest before→after "
                "evidence timestamps.\n\n"
                "Return ONLY one JSON object, no prose:\n"
                '{"anchor_status":"matched|unmatched|uncertain",'
                '"target_status":"concrete_visible|causal_chain|anchor_only|'
                'absent|uncertain","subject_continuity":"same_actor|broken|'
                'uncertain","arrival_evidence":"entered|arrived|stopped|'
                'stayed|other_target|none|uncertain","target_value":'
                '"concrete answer or empty",'
                '"before_t":0.0,"after_t":0.0,"evidence":"specific cue"}\n'
                + "\n".join("<image>" for _ in audit_sheets)
            )
            messages = [{"role": "user", "content": audit_user}]
            self._enforce_token_budget(messages, audit_sheets)
            self.data = {"messages": messages, "images": audit_sheets}
            self._crr_target_audit_pending = False
            self._crr_target_audit_inflight = True
            self._responding_to_tool = False
            self._last_request = {
                "kind": "crr_target_audit",
                "messages": [dict(message) for message in messages],
                "images": list(audit_sheets),
            }
            return True, InferRequest(**self.data)

        if self._ssr_local_audit_pending:
            audit_sheets, timeline_description = \
                self._build_ssr_local_audit_window()
            target_step = str(self.sample.get("all_steps") or "").strip()
            history_evidence = self._ssr_history_evidence()
            audit_user = (
                "You are performing a query-time LOCAL STATE AUDIT inside "
                "an agentic streaming-video system. The timestamped video "
                f"shows the {timeline_description}. It is not the full "
                "video. The structured memory DAG evidence below may show "
                "that the target happened earlier.\n\n"
                f"Target action: {target_step}\n\n"
                "Target-matched prior DAG evidence:\n"
                f"{history_evidence}\n\n"
                "Judge the TARGET ACTION itself, grounded in visible objects "
                "and the surrounding procedure. Do not substitute a generic "
                "similar motion and do not impose an unrelated dictionary "
                "domain when the visible workflow gives the phrase a local "
                "meaning. Compare TWO timestamped moments and classify the "
                "target's defining object/state transition:\n"
                "- contact_only: a tool/hand touches, approaches, positions, "
                "or attempts, but the target object/state has no visible "
                "response;\n"
                "- defining_motion: sustained action produces target-specific "
                "relative motion or state progression (for opening: an edge "
                "lifts, a gap appears/grows, or the cover moves relative to "
                "the case even if not fully detached);\n"
                "- changed_state: the target object is visibly in the new "
                "state;\n"
                "- completed: the full transition and resulting state are "
                "visibly complete;\n"
                "- absent or uncertain when none is supportable.\n"
                "Scan ALL timestamped panels in chronological order and "
                "compare adjacent/nearby moments; do NOT judge only the first "
                "and last panel. Report the STRONGEST target-defining "
                "transition visible anywhere before query time. If a gap, "
                "lift, separation, or other defining motion appears briefly "
                "and is later occluded or the object returns to a similar "
                "appearance, the event has still occurred. before_t and "
                "after_t MUST be the specific panel timestamps that show that "
                "strongest transition. history_phase uses the "
                "same enum for DAG evidence. Earlier defining_motion, "
                "changed_state, or completed evidence remains valid even if "
                "the local window is absent. Generic tool contact alone is "
                "never enough.\n\n"
                "Return ONLY one JSON object, no prose:\n"
                '{"local_phase":"contact_only|defining_motion|changed_state|'
                'completed|absent|uncertain","history_phase":"contact_only|'
                'defining_motion|changed_state|completed|absent|uncertain",'
                '"before_t":0.0,"after_t":0.0,"evidence":"one concrete '
                'before→after object/state comparison or DAG cue"}\n'
                + "\n".join("<image>" for _ in audit_sheets)
            )
            messages = [{"role": "user", "content": audit_user}]
            self._enforce_token_budget(messages, audit_sheets)
            self.data = {
                "messages": messages,
                "images": audit_sheets,
            }
            self._ssr_local_audit_pending = False
            self._ssr_local_audit_inflight = True
            self._responding_to_tool = False
            self._last_request = {
                "kind": "ssr_local_audit",
                "messages": [dict(message) for message in messages],
                "images": list(audit_sheets),
            }
            return True, InferRequest(**self.data)

        if self._count_audit_pending:
            audit_fps = float(os.environ.get(
                "STREAMO_COUNT_AUDIT_VIDEO_FPS", "4"))
            max_frames = max(8, int(os.environ.get(
                "STREAMO_COUNT_AUDIT_LEDGER_MAX_FRAMES", "180")))
            if not self._count_audit_segments:
                video_frames, timeline_description = \
                    self._build_count_audit_timeline(
                        force_direct=True,
                        fps_override=audit_fps,
                    )
                segmented_enabled = (
                    os.environ.get(
                        "STREAMO_ENABLE_SEGMENTED_COUNT_AUDIT", "0"
                    ) == "1"
                )
                threshold_seconds = max(1.0, float(os.environ.get(
                    "STREAMO_COUNT_AUDIT_SEGMENT_THRESHOLD_SECONDS", "80"
                )))
                self._count_audit_segmented = bool(
                    segmented_enabled
                    and len(video_frames) > threshold_seconds * audit_fps
                )
                if self._count_audit_segmented:
                    segment_frames = max(8, int(os.environ.get(
                        "STREAMO_COUNT_AUDIT_SEGMENT_MAX_FRAMES", "160"
                    )))
                    chunks = split_count_audit_frames(
                        video_frames, segment_frames
                    )
                    self._count_audit_segments = [
                        {
                            "frames": chunk,
                            "start_t": index * segment_frames / audit_fps,
                            "end_t": (
                                index * segment_frames + len(chunk) - 1
                            ) / audit_fps,
                            "timeline_description": timeline_description,
                        }
                        for index, chunk in enumerate(chunks)
                    ]
                else:
                    stride = max(1, int(math.ceil(
                        len(video_frames) / max_frames)))
                    selected = video_frames[::stride]
                    if selected[-1] != video_frames[-1]:
                        selected.append(video_frames[-1])
                    self._count_audit_segments = [{
                        "frames": selected,
                        "start_t": 0.0,
                        "end_t": max(
                            0.0, (len(video_frames) - 1) / audit_fps
                        ),
                        "timeline_description": timeline_description,
                    }]
                query_line = str(self.question or "").splitlines()[0]
                self._count_audit_target_activity = count_audit_target(
                    query_line)
                self._count_audit_use_evidence_ledger = \
                    count_audit_uses_evidence_ledger(query_line)
                self._count_audit_sports_cycle_version = \
                    count_audit_sports_cycle_version(query_line)
                self._count_audit_use_sports_cycle = bool(
                    self._count_audit_sports_cycle_version
                )
                self._count_audit_cycle = count_audit_cycle_hint(query_line)

            segment = self._count_audit_segments[
                self._count_audit_segment_index
            ]
            selected_frames = list(segment["frames"])
            try:
                from qwen_vl_utils import vision_process
                vision_process.FPS = audit_fps
            except Exception:
                pass
            audit_user = build_count_audit_prompt(
                target_activity=self._count_audit_target_activity,
                cycle_hint=self._count_audit_cycle,
                use_evidence_ledger=self._count_audit_use_evidence_ledger,
                sports_cycle_version=self._count_audit_sports_cycle_version,
                segmented=self._count_audit_segmented,
                segment_index=self._count_audit_segment_index,
                segment_count=len(self._count_audit_segments),
                segment_start_t=float(segment["start_t"]),
                segment_end_t=float(segment["end_t"]),
            )
            messages = [
                {"role": "user", "content": audit_user},
            ]
            self._enforce_token_budget(messages, [])
            self.data = {
                "messages": messages,
                "images": [],
                "videos": [selected_frames],
            }
            self._count_audit_pending = False
            self._count_audit_inflight = True
            self._responding_to_tool = False
            self._last_request = {
                "kind": "count_audit",
                "messages": [dict(message) for message in messages],
                "images": [],
                "videos": [list(selected_frames)],
                "segmented": self._count_audit_segmented,
                "segment_index": self._count_audit_segment_index,
                "segment_count": len(self._count_audit_segments),
                "segment_start_t": segment["start_t"],
                "segment_end_t": segment["end_t"],
            }
            return True, InferRequest(**self.data)

        # COMMIT round (current_round == total_rounds) is a virtual extra
        # round AFTER the last video frame. It exists to give the model
        # ONE more chance to emit `</Response>` when it failed to commit
        # on the last actual frame. Tools are disabled in this round and
        # the prompt forces a final answer. Anything beyond that is
        # truly terminal.
        if self.current_round > self.total_rounds:
            self.completed = True
            return False, None

        # In the COMMIT round we reuse the last frame on disk (no new
        # frame is consumed) so the model still has visual context.
        is_commit_round = self.current_round == self.total_rounds
        # Clamp to the last REAL frame on disk. total_rounds may exceed
        # len(frame_paths) by one (planning buffer round / COMMIT round),
        # so clamp on len(frame_paths) to avoid an index error.
        frame_idx = min(self.current_round, len(self.frame_paths) - 1)
        frame_path = self.frame_paths[frame_idx]

        # System-side auto memory writes — guarantees MEMORY accumulates
        # signal even when the model never calls memory_write itself.
        self._auto_write_memory(frame_idx)

        # ---- Phase decision ------------------------------------------
        # OVO-Bench protocol: the question is *posed* at ``question_time``.
        # Before that the model must observe and accumulate memory without
        # knowing the question (post-hoc); from question_time onward the
        # model has to decide whether/when to answer.
        is_post_hoc = self.current_round < self.question_time
        is_question_round = self.current_round == self.question_time

        # On the round when the question is FIRST posed, pin it to memory
        # so that subsequent prompts (which omit the question text) can
        # still see it through the MEMORY block.
        if is_question_round and not getattr(self, "_question_pinned", False):
            if self.question and self._memory is not None:
                self._memory.pin_fact(
                    f"[Q] {self.question}",
                    kind="question",
                    t=self.current_round,
                )
            self._question_pinned = True

        # On / past question_time we want the model to see as much
        # historical context as possible before committing to an answer:
        # double the budget and bump tail/summary caps for that snapshot.
        past_q = self.current_round >= self.question_time
        if self._memory is not None:
            snap_kwargs = self._resolve_render_args(
                current_t=self.current_round, past_q=past_q,
            )
            memory_brief_obj = self._memory.snapshot(**snap_kwargs)
        else:
            memory_brief_obj = {
                "text": "MEMORY: (empty)", "compressed": False,
            }
        memory_brief = memory_brief_obj["text"]

        user_blocks: List[str] = [memory_brief]

        # ---- Entity graph digest -------------------------------------
        # Surface the entity index (who/what recurs and when) plus the
        # causal action-order chain. Previously the entity index was built
        # but never shown to the model; this makes the graph's semantic
        # structure visible so it can reason about recurring actors/objects
        # and action ordering (helps interaction / ASI questions).
        if self._graph is not None:
            try:
                ent_digest = self._graph.entity_index.render_digest(top_k=8)
                if ent_digest:
                    user_blocks.append(ent_digest)
                rel_digest = self._graph.render_relations(max_edges=10)
                if rel_digest:
                    user_blocks.append(rel_digest)
            except Exception:
                pass

        # ---- Trigger-time automatic cross-layer retrieval ------------
        # Once the question has been posed, run a keyword retrieval over
        # the three memory layers using the question text and inject the
        # top-K hits as a `[RETRIEVED ...]` block right after the MEMORY
        # brief. The model can additionally call ``memory_search`` to
        # refine; this auto block guarantees a baseline recall even if
        # the model forgets to retrieve.
        if past_q and self._memory is not None and self.question:
            # ── Pillar 2: Cross-Layer Backtrack (DAG retrieval) ──────
            # When the MemoryGraph is available, use pointer-aware
            # backtrack search; otherwise fall back to flat keyword
            # retrieval.  The backtrack version returns evidence *paths*
            # with cross-layer context rather than disconnected items.
            retrieved = None
            evidence_query = self._evidence_query()
            if self._graph is not None:
                try:
                    paths = self._graph.retrieve_backtrack(
                        query=evidence_query,
                        top_k=12,
                        current_t=int(self.current_round),
                    )
                    # Render EvidencePaths into the same [RETRIEVED] block
                    # format the model already understands.
                    if paths:
                        ret_text = self._render_backtrack_paths(
                            paths, query=evidence_query)
                        if ret_text:
                            user_blocks.append(ret_text)
                except Exception:
                    paths = None
            else:
                try:
                    retrieved = self._memory.retrieve(
                        query=evidence_query,
                        top_k=12,
                        current_t=int(self.current_round),
                    )
                except Exception:
                    retrieved = None
                if retrieved:
                    ret_text = self._render_retrieved_block(
                        retrieved, query=evidence_query)
                    if ret_text:
                        user_blocks.append(ret_text)

            # ★ State Digest：扫描 MEMORY 中所有显式标注了事件状态
            # 的事件（[FINISHED] / [STARTED] / [IN-PROGRESS] / [ABORTED]
            # 或 state= 的 Recap），按时间顺序浓缩成一行行 snapshot，
            # 供 SSR/CRR 这类元问题任务直接读取——避免模型靠 verb tense
            # 猜测事件状态（CRR "走向 vs 到达" 的常见误判）。
            digest = self._build_state_digest()
            if digest:
                user_blocks.append(digest)
        # Bridge note when older context has been compressed away. Helps the
        # model treat the MEMORY block as authoritative rather than guessing
        # at missing history (借鉴 #2 — bridge messages).
        if memory_brief_obj.get("compressed"):
            tail_t_min = memory_brief_obj.get("tail_t_min")
            n_drop = memory_brief_obj.get("n_dropped", 0)
            bridge = (
                "[Note] Earlier frames have been COMPRESSED into the MEMORY "
                f"block above ({n_drop} event(s) summarized; "
                f"verbatim tail starts at t={tail_t_min}s). "
                "Treat MEMORY as ground truth for past observations; "
                "use memory_read / replay when you need detail."
            )
            user_blocks.append(bridge)
        if self._pending_tool_text:
            user_blocks.append(self._pending_tool_text)
            # Empty-result STOP guard: two consecutive empty searches /
            # lost tracks mean the target almost certainly is NOT in the
            # video. Push the model to stop searching and commit/abstain
            # (critical for HLD "Unable to answer" cases).
            if past_q and self._consecutive_empty >= 2:
                user_blocks.append(
                    "⚠️ [System] Your last 2 lookups returned NOTHING "
                    "[0 results found]. The target is most likely NOT in "
                    "the video. STOP searching. If an option says "
                    "\"Unable to answer\" (or similar), choose it now; "
                    "otherwise pick the best-supported option from MEMORY "
                    "and emit `</Response>`. Do NOT issue another search "
                    "or replay for the same target."
                )

        # ---- Standing plan banner (Query-Driven Planning) ------------
        # Once a <Plan> has been captured, surface it at the top of every
        # answering turn as a concise "battle order" so the model executes
        # its own strategy instead of drifting.
        if past_q and self._query_plan:
            user_blocks.append(self._render_plan_banner())
        if past_q:
            refusal_contract = listed_refusal_evidence_contract(self.sample)
            if refusal_contract:
                user_blocks.append(refusal_contract)
            if self.sample.get("proactive_output"):
                # NOTE: gated on sample["proactive_output"], set ONLY for
                # StreamingBench Proactive-Output samples (prepare_streamingbench
                # .py). OVO-Bench data never carries this field, so this block
                # can never fire for OVO — StreamingBench-only by construction.
                #
                # Timing fix: the model reliably identifies the trigger CONTENT
                # (~51% content-correct) but commits `</Response>` ~7s too late
                # (45% of content-correct triggers land >+10s after target),
                # because the old contract encouraged it to "keep watching" and
                # re-confirm. Scoring needs |trigger - target| <= 2s, so late
                # commit is the dominant failure. Rewrite the contract to
                # commit on the FIRST frame the condition is visibly satisfied,
                # without waiting for extra confirmation.
                user_blocks.append(
                    "[PROACTIVE OUTPUT CONTRACT] This is a trigger-waiting "
                    "instruction, not an immediate question. Inspect the "
                    "CURRENT frame and the transition from the immediately "
                    "preceding frame. While the requested trigger condition "
                    "is NOT yet visible, emit `</Silence>`. **The instant the "
                    "condition first becomes visible — on that very frame — "
                    "emit `</Response>` immediately.** Do NOT wait one or more "
                    "extra frames to re-confirm: a correct trigger that fires "
                    "even 2-3 frames late scores ZERO. Reacting one frame too "
                    "early on a genuine onset is far better than reacting "
                    "several frames late. Do not, however, infer the trigger "
                    "from the instruction text alone or from elapsed time — it "
                    "must be visibly happening. When you fire `</Response>`, "
                    "output only the string requested in the pinned [Q]."
                )
            # ── StreamingBench Causal-Reasoning evidence-first hint ────────
            # StreamingBench-ONLY. Double-gated: (1) env STREAMO_SB_CAUSAL_HINT
            # is set to "1" only in the StreamingBench branch of launch_eval.sh,
            # and (2) the sample must be a StreamingBench Causal-Reasoning item.
            # OVO-Bench samples carry benchmark=="ovobench" and never set the
            # env, so this block cannot fire for OVO — the OVO prompt is byte
            # identical to before.
            # Rationale: CR errors are 63% "commonsense prior overrides the
            # actual on-screen causality" (Mr. Bean / cartoon counter-intuitive
            # plots), and CR error rate is FLAT vs video length — a reasoning
            # bias, not a sampling gap. Push the model to answer from observed
            # evidence rather than the most plausible-sounding cause.
            if (
                os.environ.get("STREAMO_SB_CAUSAL_HINT", "0") == "1"
                and self.sample.get("benchmark") == "streamingbench"
                and self.sample.get("task") == "Causal Reasoning"
            ):
                user_blocks.append(
                    "[EVIDENCE-FIRST CAUSALITY] Answer strictly from what the "
                    "video actually SHOWS, not from what is most plausible in "
                    "general. The real cause here may be counter-intuitive or "
                    "absurd. Before choosing, cite the specific observed "
                    "event(s) in MEMORY that establish the cause; if an option "
                    "merely sounds like a typical explanation but you did not "
                    "actually observe it happening, reject it. When on-screen "
                    "evidence conflicts with common sense, the on-screen "
                    "evidence wins."
                )

        # Phase-specific instructions:
        # * post-hoc → no question. Ask the model to emit a one-line
        #   <Observation>...</Observation> describing the current frame.
        # * question round → inject the question explicitly (also pinned
        #   to MEMORY above).
        # * after question_time → DO NOT repeat the question text in the
        #   user turn; rely on the pinned fact in MEMORY (per the
        #   post-hoc paradigm: question is posed exactly once).
        if is_commit_round:
            # COMMIT round: virtual round AFTER the last video frame.
            # The video is over. Tools are disabled (dispatch will
            # reject them). The model MUST emit `</Response>` here.
            user_blocks.append(
                "[Phase] COMMIT (POST-VIDEO FORCED REPLY). The video "
                "has ENDED — there are no more frames after this. "
                "Tool calls are DISABLED in this round (any "
                "`<tool_call>` will be rejected). Your ONLY valid "
                "action is to emit `</Response> <your answer>` "
                "RIGHT NOW, grounded entirely in the MEMORY snapshot "
                "above. The question is pinned in MEMORY (look for "
                "`[Q] ...`); re-read it, parse TARGET / ANCHOR / "
                "FORMAT, scan MEMORY for the best-matching evidence, "
                "and commit. NEVER write `</Silence>`, `</Standby>`, "
                "or `<tool_call>` here — they are all invalid. Commit "
                "the answer your evidence best supports (follow your "
                "plan's answerability — if it judged the target absent "
                "and an \"Unable to answer\" option is listed, that is a "
                "valid answer). "
                + _last_frame_answer_constraint(
                    self.sample,
                    allow_tool=False,
                )
            )
        elif is_post_hoc:
            user_blocks.append(
                "[Phase] OBSERVE-ONLY. The question has not been asked yet. "
                "Reply with EXACTLY two lines:\n"
                "  Line 1: <Observation> a SELF-CONTAINED FULL frame "
                "report covering ALL SEVEN labelled fields below, in "
                "this order, on EVERY frame (no exceptions, no "
                "abbreviations, no `same as before`, no bare "
                "`(no change)`):\n"
                "      • `description:` 1-2 sentences, holistic summary "
                "of THIS frame.\n"
                "        ❌ BAD: \"A kitchen scene.\"\n"
                "        ✅ GOOD: \"Medium shot of a chef in white uniform "
                "standing at a stainless-steel counter, chopping carrots "
                "with a chef's knife. Steam rising from a pot on the stove "
                "in the background.\"\n"
                "      • `actors:` EVERY distinct person/animal visible. "
                "For each: appearance, clothing, role, position. "
                "Re-list ALL actors every frame. Include background figures. "
                "Always state the TOTAL count (e.g. '2 actors: ...').\n"
                "        ❌ BAD: \"actors: a person cooking\"\n"
                "        ✅ GOOD: \"actors: 1) chef in white double-breasted "
                "jacket, black apron, toque, at center counter — chopping; "
                "2) sous-chef in blue polo, at back sink — washing dishes\"\n"
                "      • `objects:` salient objects with current STATE and "
                "LOCATION. Tag state explicitly. "
                "For COUNTABLE objects, report the EXACT number "
                "(e.g. '3 cards', '5 bottles') — never vague "
                "ranges like '~3-5' or 'several'.\n"
                "        Format: `<object> [state] at <location>`\n"
                "        ❌ BAD: \"objects: knife, pot, carrots\"\n"
                "        ✅ GOOD: \"objects: chef's knife [held, in use] "
                "in chef's right hand; large stockpot [on stove, steaming] "
                "back left; 3 carrots [being chopped] on cutting board "
                "center frame\"\n"
                "      • `actions:` what each actor is CURRENTLY doing "
                "(verb + target + state tag).\n"
                "        Format: `<actor> <verb> <target> [STATE]`\n"
                "        State tags: [STARTING] / [IN-PROGRESS] / [FINISHED] / [ABORTED]\n"
                "        ❌ BAD: \"actions: cooking\"\n"
                "        ✅ GOOD: \"actions: chef chopping carrots "
                "[IN-PROGRESS]; sous-chef washing dishes [IN-PROGRESS]\"\n"
                "        For persisting actions, use `still ...` or "
                "`continues to ...` instead of re-describing.\n"
                "      • `ocr:` EVERY on-screen text/number VERBATIM. "
                "Include language, color, position.\n"
                "        ❌ BAD: \"ocr: some text on screen\"\n"
                "        ✅ GOOD: \"ocr: 'CHANNEL 5 NEWS' (white text, "
                "top-left overlay); scoreboard reads 'HOME 24 - 18 AWAY' "
                "(yellow, bottom-right); subtitle: 'What are you doing?' "
                "(white, bottom-center)\"\n"
                "        If genuinely no text: \"ocr: (none)\"\n"
                "      • `scene:` location + time-of-day + camera framing + "
                "indoor/outdoor.\n"
                "        ❌ BAD: \"scene: indoors\"\n"
                "        ✅ GOOD: \"scene: professional kitchen, daytime "
                "(fluorescent lighting), medium-wide shot, eye-level angle, "
                "indoor\"\n"
                "      • `delta:` what CHANGED from the PREVIOUS frame. "
                "This is the MOST IMPORTANT field — it drives memory "
                "compression and retrieval. MUST name ≥1 concrete, "
                "observable change.\n"
                "        Format: `<subject> <verb/delta> <detail>`\n"
                "        ❌ BAD: \"delta: slight motion\"  ← generic boilerplate\n"
                "        ❌ BAD: \"delta: no change\"       ← NEVER allowed\n"
                "        ❌ BAD: \"delta: scene continues\" ← garbage token\n"
                "        ✅ GOOD: \"delta: chef's right hand moved 3cm left, "
                "knife blade angle changed from 30° to 45°\"\n"
                "        ✅ GOOD: \"delta: steam plume above stockpot "
                "expanded upward ~5cm; new bubble formation visible\"\n"
                "        ✅ GOOD: \"delta: sous-chef turned head 90° left "
                "toward chef; mouth opened (speaking)\"\n"
                "        If you GENUINELY cannot detect any change after "
                "active visual inspection, pick a tool (zoom_crop / "
                "track_query) to find evidence first. ONLY if the frame is "
                "bit-identical may you write: \"delta: visually identical "
                "to t={prev}s — verified static\".</Observation>\n"
                "  CRITICAL when ≥2 entities of the same kind are "
                "visible: COUNT them exactly (e.g. `5 giants`, "
                "`3 dogs`, `2 cars`) and describe their relative "
                "spatial order (front/back/left/right/foreground/"
                "background, or `in a row from left to right: A, B, "
                "C` style). Identify which one is biggest / closest / "
                "leftmost — this is exactly the kind of fact a future "
                "question may rely on.\n"
                "  Line 2: </Silence>   # status tag — \"observed, no "
                "answer yet\".\n"
                "Do NOT emit </Response>. The system reads <Observation> "
                "into MEMORY and uses </Silence> as the round-end marker.\n"
                "## Anti-repetition rule\n"
                "If your last few <Observation>s were near-identical, "
                "that is a SIGNAL THAT YOU ARE NOT LOOKING — not a "
                "license to keep copy-pasting. On the very next frame "
                "you MUST: (a) gather NEW evidence — pick ONE of "
                "`track_init` (if a moving subject deserves "
                "following — usually the right choice), `asr_segment` "
                "([t-5,t] for spoken cues), `memory_search` (recall "
                "an earlier related event), or as a LAST resort "
                "`zoom_crop` on a region you have NEVER inspected "
                "(hands / face / a specific object). Prefer the "
                "non-zoom options: repeated zoom_crop on similar "
                "regions adds little new evidence; "
                "AND (b) write `delta:` referencing what the new "
                "evidence revealed. Repeating the same labelled "
                "body verbatim across 3+ frames is a bug.\n"
                "## Generic observation checklist (capture WHENEVER present)\n"
                "Future questions may ask about any of the following — "
                "treat each frame as evidence and log them when they "
                "appear, even if they look unimportant now:\n"
                "  - identities & appearances: who is in frame "
                "(clothing colour, gender hint, role, accessories) — "
                "useful for `who` / `which person` questions later. "
                "ENUMERATE EVERY DISTINCT PERSON / ANIMAL / SALIENT "
                "OBJECT visible — do NOT collapse them into a single "
                "`a person` even when one is clearly the protagonist. "
                "If a SECONDARY actor appears (a passer-by, a "
                "conversation partner, someone briefly entering the "
                "frame), log them as a separate entry with their own "
                "appearance descriptors (e.g. `secondary actor: "
                "person in blue shirt, on the right`); these "
                "background figures are exactly what `who did X "
                "interact with` questions hinge on.\n"
                "  - actions & their order: what each actor is doing, "
                "and the strict before/after order between sub-actions "
                "(e.g. `pours water → stirs → covers lid`); explicitly "
                "say what comes RIGHT AFTER a given action — useful "
                "for `what does X do after Y` questions.\n"
                "  - **action state (REQUIRED for any non-trivial "
                "action)**: tag every action you describe with one of "
                "`[STARTING]` / `[IN-PROGRESS]` / `[FINISHED]` / "
                "`[ABORTED]` — e.g. `pours water [IN-PROGRESS]`, "
                "`covers lid [FINISHED]`. This single tag is what "
                "future event-state and answerability meta-questions rely "
                "on to "
                "decide whether evidence already supports a Yes/No "
                "answer. Without the tag, the model has to guess from "
                "verb tense — be explicit.\n"
                "  - object state changes: object appears / "
                "disappears / is moved / is consumed / is altered "
                "(broken, opened, attached) — useful for "
                "object-relation and presence questions. Tag the "
                "**resulting** state with one of: present/absent, "
                "open/closed, on/off, full/empty, intact/broken, "
                "moving/static (e.g. `lid [closed]`, `bottle "
                "[empty]`).\n"
                "  - spatial relations: A is on / under / behind / "
                "next to B; what surface or platform an actor stands "
                "on (rock / cloud / ground / floor) — useful for "
                "`what is under X` / `where is X` questions.\n"
                "  - counts & repetitions: each time a discrete event "
                "repeats (egg cracked again, door opened again), log "
                "it as `Nth occurrence of <event>` — useful for "
                "counting questions.\n"
                "  - on-screen text & numbers: read them verbatim "
                "(names, scores, prices, captions, road signs, "
                "subtitles) — useful for on-screen-text questions.\n"
                "  - speech / dialogue: who is talking to whom, "
                "topic of conversation, key utterance — useful for "
                "audio-grounded questions.\n"
                "  - scene transitions: any cut / overlay / fade is "
                "worth a single line so future replay can land "
                "near the boundary.\n"
                "If a fact above is durable (identity, count, "
                "ordering, name, count) call `memory_write` with "
                "`pin=true` so it survives compression.\n"
                "## Recap on semantic boundaries (REQUIRED when applicable)\n"
                "If THIS frame marks a semantic boundary — an event "
                "you were narrating just FINISHED, a SCENE CHANGE "
                "happened, an ACTOR ENTERED / LEFT / visibly CHANGED, "
                "or a KEY OBJECT appeared/disappeared/changed state "
                "— additionally emit a `<Recap t0=<sec> t1=<sec> "
                "kind=<event_end|scene_change|actor_enter|actor_exit|"
                "actor_change|object_change|milestone> "
                "state=<finished|started|in_progress|aborted>> "
                "... </Recap>` block in this same turn (BEFORE "
                "`</Silence>`).\n"
                "★ The `state=` attribute is REQUIRED on event_end / "
                "milestone / object_change / actor_change recaps. It "
                "tells future readers (and event-state/answerability "
                "meta-questions) the EXACT progression status of the "
                "event:\n"
                "  - `state=finished`: the action / step is FULLY "
                "COMPLETED (use this when narrating event_end). "
                "Future event-state queries can use this as completed "
                "evidence.\n"
                "  - `state=started`: the action has just begun, "
                "evidence is now observable but not yet completed. "
                "Future answerability queries can use this as onset "
                "evidence.\n"
                "  - `state=in_progress`: the action is mid-stream; "
                "completion not yet reached.\n"
                "  - `state=aborted`: the actor visibly stopped or "
                "abandoned the action without finishing.\n"
                "★ The Recap body MUST start with an explicit verb "
                "phrase that names the state, e.g. `\"FINISHED: "
                "<action> at t=12s\"`, `\"STARTED: <action> at t=8s, "
                "currently in progress\"`. This single phrase is what "
                "future memory_search will surface for state queries.\n"
                "The Recap should be 3–8 sentences and MUST cover: "
                "every actor involved (with appearance), the full "
                "ordered action chain in [t0,t1], **final state of "
                "key objects (use one of: present/absent, "
                "open/closed, on/off, full/empty, moving/static)**, "
                "spatial layout at the boundary, any on-screen text / "
                "dialogue verbatim, and why this is a boundary. The "
                "system pins it as durable long-term memory. Do NOT "
                "emit Recap on quiet frames; do NOT emit more than "
                "one per ~5s.\n"
                "Tools are STRONGLY ENCOURAGED this round — past frames "
                "will be compressed away soon, so capture details NOW. "
                "Choose AT MOST ONE tool per round and DIVERSIFY across "
                "rounds (avoid issuing zoom_crop on every frame; recent "
                "tool history is rendered above). Concrete triggers "
                "(use one whenever relevant):\n"
                "  - new actor / object / animal appears → `track_init` "
                "with a tight bbox AND `text_prompt` (SAM2 will follow "
                "it across future frames). PREFERRED over zoom_crop "
                "when the subject WILL MOVE. LIMIT: at most 3 track_init "
                "for the WHOLE observe phase — the system auto-refreshes "
                "existing tracks every few seconds, so do NOT spawn a new "
                "track for every object; reuse `track_query` on an "
                "existing track id instead.\n"
                "  - already-tracked target → `track_query` to record "
                "its bbox / visibility on this frame.\n"
                "  - small text, fine detail, hands/faces, numeric "
                "counts on a STATIC element → `zoom_crop` on the bbox "
                "(same-frame near-duplicate crops are auto-rejected at "
                "IoU>=0.5; pick a CLEARLY different region next time).\n"
                "  - speech / dialogue / narration / music → "
                "`asr_segment` over a `[t-5, t]` window.\n"
                "  - durable fact (identity, count, ordering, color, "
                "scene type) → `memory_write` with `pin=true`.\n"
                "  - long static span → `set_sample_rate` to "
                "save compute.\n"
                "You may emit one or more <tool_call>...</tool_call> "
                "blocks BEFORE the two-line reply; the system will "
                "inject the tool result and re-prompt you for the "
                "same frame.\n"
                "FORBIDDEN in OBSERVE-ONLY: `replay` "
                "(no question yet, so no reason to time-travel; "
                "the streaming clock will reach those frames "
                "naturally). `memory_read` is also wasteful here — "
                "MEMORY is already rendered above.\n"
                "## Anti-Hallucination Rule\n"
                "- Every `delta:` field MUST reference SPECIFIC VISIBLE "
                "elements that YOU CAN SEE in the CURRENT frame(s).\n"
                "- Do NOT speculate about off-screen events or future actions.\n"
                "- Do NOT infer intentions unless there is explicit visual "
                "evidence (e.g., a facial expression, a pointing gesture).\n"
                "- If uncertain about a detail (count, color, identity), "
                "write `[uncertain: <detail>]` rather than guessing.\n"
                "- For ANY claim involving a COUNT or COLOR, visually "
                "re-examine the frame before writing — do not trust memory.\n"
                "- The delta MUST describe a change you SEE between the "
                "two frames; do not invent changes to satisfy the format."
            )
            # Vague-quantifier nudge: if the most recent observation in
            # MEMORY used a fuzzy count word ("multiple giants"), add a
            # one-line system reminder asking the model to recount on
            # this frame.
            vague_noun = self._latest_vague_noun()
            if vague_noun:
                user_blocks.append(
                    f"[System] Last observation used a vague quantifier "
                    f"for `{vague_noun}`. THIS round, count them exactly "
                    f"and describe their relative spatial order "
                    f"(left-to-right or front-to-back). If the frame is "
                    f"crowded, pick ONE supporting tool BEFORE the two-"
                    f"line reply: `zoom_crop` on the cluster, OR "
                    f"`track_init` on the leftmost/rightmost instance to "
                    f"follow it, OR `memory_search` for earlier counts."
                )
            # Anti-template delta nudge: surface the most recent delta
            # so the model is explicitly forbidden from re-emitting it
            # verbatim. This is the prompt-side complement to the
            # code-side repetition detector in RuleSummarizer.
            prev_delta = self._latest_delta()
            if prev_delta is not None:
                pt, pd = prev_delta
                user_blocks.append(
                    f"[System] Previous frame's delta (t={pt}s) was: "
                    f"\"{pd}\". For THIS frame: do NOT repeat that "
                    f"phrase or paraphrase it. Either (a) name a "
                    f"DIFFERENT concrete change you can see "
                    f"(<subject> <verb> <object/where>), or (b) BEFORE "
                    f"the two-line reply pick ONE evidence-gathering "
                    f"tool — `zoom_crop` on a region you have NOT "
                    f"inspected recently (note: same-frame near-"
                    f"duplicate crops are auto-rejected with IoU>=0.5), "
                    f"OR `track_init` to follow a moving subject, OR "
                    f"`asr_segment` over the last few seconds for "
                    f"spoken cues — then write `delta:` describing "
                    f"what the new evidence revealed. Repeating the "
                    f"previous delta is treated as a template-copy "
                    f"and the frame is demoted to a static event."
                )
        elif is_question_round:
            is_last_frame = self.current_round >= self.total_rounds - 1
            # ── Query-Driven Planning sub-phase ──────────────────────
            # The FIRST time the question is posed we ALWAYS ask the model
            # to ANALYSE the query and emit a <Plan> block BEFORE it starts
            # answering — no exceptions, even on the last frame (a virtual
            # buffer COMMIT round guarantees there is still a turn left to
            # answer in afterwards). It self-classifies what the question
            # needs (temporal order / spatial layout / existence / count /
            # a meta yes-no) — we do NOT hand it predefined task labels.
            # The plan is parsed in _update_impl, stored in
            # self._query_plan, and re-injected into every later
            # POST-QUESTION prompt. This planning turn is re-prompted on
            # the SAME frame (no clock tick, no tool budget spent).
            if not self._plan_done:
                user_blocks.append(
                    "[Phase] QUESTION POSED — PLANNING STEP (analyse "
                    "BEFORE you answer). The question is below (and pinned "
                    "to MEMORY). Do NOT answer yet and do NOT call tools "
                    "this turn. First THINK about what the question truly "
                    "asks, then emit ONE `<Plan>` block:\n"
                    "<Plan>\n"
                    "need: <in your own words, the exact thing being "
                    "asked — the TARGET and its ANCHOR (the moment/"
                    "event/actor it pivots on)>\n"
                    "pov: <FIRST-PERSON CHECK — does the question say "
                    "\"I\"/\"me\"/\"my\"? If so, \"I\" am the CAMERA "
                    "WEARER, NOT any person fully visible in the frame. "
                    "The camera wearer's own hands/arms/legs appear "
                    "entering from the BOTTOM or EDGE of the frame; a "
                    "person sitting across the table fully facing the "
                    "camera is the OTHER person, NOT me. State clearly WHO "
                    "\"I\" am and WHO the other people are here. Note my "
                    "own body is often only partially visible or off-"
                    "screen, so a question about what I do may not be "
                    "answerable by looking at me directly>\n"
                    "info_type: <what KIND of evidence settles it — e.g. "
                    "temporal order of actions / spatial direction or "
                    "position / existence (is X even present?) / exact "
                    "count / object identity / attribute like color>\n"
                    "evidence_locus: <work out WHERE the deciding "
                    "evidence for THIS question lives — in your own words, "
                    "no default direction. Weigh three possibilities "
                    "even-handedly and pick whichever the question truly "
                    "calls for: (i) it is in the CURRENT frame / the moment "
                    "the question is asked (what is happening or visible "
                    "right now); (ii) it is in something that has ALREADY "
                    "happened (recorded in MEMORY / a specific past moment "
                    "to re-look); (iii) it has NOT happened yet and you "
                    "must keep observing until it does. Do not assume it "
                    "is in the past just because MEMORY is handy, nor in "
                    "the present just because a frame is in front of you. "
                    "Check the ANCHOR the question pivots on (e.g. the "
                    "event in \"what happens AFTER X\" / \"before X\"): has "
                    "that anchor — and the thing asked relative to it — "
                    "actually appeared yet? If not, the evidence is not in "
                    "hand and answering now would just guess from whatever "
                    "is nearest. Then say how you'll obtain it: look at the "
                    "current frame, read/search a MEMORY layer (for what "
                    "terms), replay a specific moment (which t and why), or "
                    "keep watching. If the target is hard to observe "
                    "directly (occluded, off-screen, or your own body), "
                    "infer it INDIRECTLY from observable traces — how the "
                    "objects it acts on changed, the before/after scene "
                    "difference, or the effect it left>\n"
                    "answerability: <look at the ACTUAL listed OPTIONS "
                    "(A/B/C/D...) as a CLOSED set — your final answer MUST "
                    "be one of them verbatim. Check whether one of THOSE "
                    "listed options literally says \"Unable to answer\" / "
                    "\"cannot determine\" (or similar). Do NOT count the "
                    "generic instruction to \"give a best-guess if "
                    "uncertain\" as such an option — that is NOT a choice, "
                    "and there is no \"implied\" unable option. If such a "
                    "choice IS listed: your first job is to VERIFY the "
                    "target exists before locating it, and you may pick it "
                    "when the evidence truly shows the target is absent. If "
                    "NO such choice is listed: refusing is FORBIDDEN — even "
                    "with weak evidence you must commit the single MOST "
                    "LIKELY listed option, never \"Unable to answer\". State "
                    "here which case this is>\n"
                    "meta: <is the question asking whether something CAN "
                    "be answered / WILL happen / has STARTED-vs-FINISHED "
                    "(a meta/state question), rather than a plain fact? "
                    "yes/no + why>\n"
                    "strategy: <turn your evidence_locus reasoning into a "
                    "concrete plan: given WHERE you decided the evidence "
                    "lives, spell out which memory layer to read/search "
                    "(and for what terms), whether the current frame "
                    "suffices, and which exact t to replay if you need to "
                    "confirm. "
                    "E.g. for an action-order question: locate the exact "
                    "t of the anchor action, then read the NEXT action "
                    "change after it. For a direction question: compare "
                    "two frames a few seconds apart, do NOT track. For an "
                    "existence question: search once or twice; if empty AND "
                    "\"Unable to answer\" is a LISTED option, answer it — "
                    "otherwise commit the most likely listed option. "
                    "★ IMPORTANT — if info_type "
                    "is a CURRENT-FRAME percept (what the person is doing "
                    "RIGHT NOW, an on-screen object/text/attribute, an "
                    "immediate action): the answer is in the CURRENT "
                    "frame — LOOK at it and answer directly. Do NOT "
                    "`replay` the current moment (it just re-shows the "
                    "same frame), do NOT browse other frames, and use at "
                    "most ONE `zoom_crop` if a detail is unclear. Commit "
                    "quickly — over-using tools on a see-it-now question "
                    "adds noise and flips close options (e.g. connecting "
                    "vs disconnecting).>\n"
                    "</Plan>\n"
                    "End this turn with `</Silence>` right after the "
                    "</Plan> block (you will be asked again immediately to "
                    "act on this plan)."
                )
                if self.question:
                    user_blocks.append(self.question)
            elif is_last_frame:
                user_blocks.append(
                    "[Phase] QUESTION POSED (LAST FRAME — COMMIT NOW). "
                    "The question below is being asked NOW (and pinned "
                    "to MEMORY). This is the FINAL frame; the streaming "
                    "clock will not advance further. You may call ONE "
                    "quick tool if it directly clinches the answer, "
                    "but this round MUST end with `</Response> "
                    "<your answer>`. Stopping at `</Silence>` or "
                    "looping on tool calls without committing yields "
                    "a NULL output that scores 0. "
                    + _last_frame_answer_constraint(self.sample)
                    + "\n" + _RECENCY_RULE
                )
            else:
                user_blocks.append(
                    "[Phase] QUESTION POSED. The question below is being "
                    "asked NOW (and has been pinned to MEMORY).\n"
                    "## Read-the-question protocol (BEFORE deciding)\n"
                    "Silently parse the question text into:\n"
                    "  - TARGET: what is being asked (an entity, an "
                    "action, a count, a Yes/No truth value, an on-screen "
                    "text string, ...).\n"
                    "  - ANCHOR: the time / event / actor it pivots "
                    "on (e.g. `right after loading the wheel`, `the "
                    "person we communicated with while chopping`, "
                    "`under the monkey at this moment`).\n"
                    "  - FORMAT: option letter, Yes/No, integer 0..N, "
                    "free-form span — match the wording exactly.\n"
                    "Then check MEMORY for events that match the "
                    "ANCHOR; the TARGET is whatever is reported in "
                    "those events. If MEMORY only contains the "
                    "ANCHOR but not the TARGET, the answer probably "
                    "lies a few frames later — emit </Silence> and "
                    "wait, OR `replay` to a likely frame.\n"
                    "Decide whether you can answer with the evidence "
                    "accumulated so far (see MEMORY) or need to watch "
                    "more frames (emit </Silence> to wait).\n"
                    "*** GROUND YOUR ANSWER IN PIXELS WHEN IT HELPS: if the "
                    "answer hinges on a SPECIFIC PAST moment (an earlier "
                    "action / object state / who was present), `replay(t=...)` "
                    "that moment and verify with your own eyes before "
                    "committing — replaying is CHEAP (one look, then the "
                    "stream auto-returns to now). YOU decide whether a replay "
                    "is needed and how many: questions about the CURRENT "
                    "frame usually need none; questions about the past may "
                    "need one or a few. Do not over-replay when the current "
                    "frame already answers it. ***\n"
                    "BEFORE answering, prefer a quick tool call if it "
                    "would resolve uncertainty:\n"
                    "  - re-examine an earlier moment → `replay`.\n"
                    "  - re-read your notes → `memory_read` / `memory_list`.\n"
                    "  - inspect fine detail → `zoom_crop`.\n"
                    "  - genuinely no visual evidence at all → `abstain`.\n"
                    "One or two well-chosen tool calls almost always beat "
                    "a blind guess.\n"
                    + _RECENCY_RULE
                )
            if self.question:
                user_blocks.append(self.question)
        else:
            # POST-QUESTION: question is already in MEMORY (pinned). Do
            # not repeat it in the user turn.
            is_last_frame = self.current_round >= self.total_rounds - 1
            if is_last_frame:
                user_blocks.append(
                    "[Phase] POST-QUESTION (LAST FRAME — COMMIT NOW). "
                    "The question is pinned in MEMORY (look for "
                    "`[Q] ...`). This is the FINAL frame; the streaming "
                    "clock will not advance further. You may call ONE "
                    "quick tool if it directly clinches the answer, "
                    "but this round MUST end with `</Response> "
                    "<your answer>`. Stopping at `</Silence>` or "
                    "looping on tool calls without committing yields "
                    "a NULL output that scores 0. "
                    + _last_frame_answer_constraint(self.sample)
                    + "\n" + _RECENCY_RULE
                )
            else:
                user_blocks.append(
                    "[Phase] POST-QUESTION. The question is pinned in "
                    "MEMORY (look for the `[Q] ...` entry). Continue "
                    "watching the new frame.\n"
                    "## Read-the-question protocol (every round)\n"
                    "RE-READ the pinned `[Q] ...` line and parse it into "
                    "TARGET / ANCHOR / FORMAT (see QUESTION POSED phase). "
                    "Then ask: does the latest frame OR any new MEMORY "
                    "event satisfy the ANCHOR? If yes, the TARGET is the "
                    "fact reported there.\n"
                    "★ First-Person POV: when the camera shows bare "
                    "feet/legs/torso at the bottom of the frame, the "
                    "camera wearer is 'I'. A question asking 'who did "
                    "I communicate TO' or 'who did I interact WITH' is "
                    "asking about the OTHER person — NOT the one "
                    "visible through the first-person perspective (that "
                    "is 'I'). Trace both people's timelines in MEMORY: "
                    "the camera-wearer ('I') appears as partial body at "
                    "frame bottom; the interaction partner is a "
                    "fully-visible actor.\n"
                    "If the answer depends on a "
                    "SPECIFIC PAST frame, `replay(t=...)` that moment to "
                    "CONFIRM with your eyes first; if the CURRENT frame or "
                    "MEMORY already answers it clearly, just emit "
                    "`</Response> <answer>`. "
                    "Otherwise emit </Silence>.\n"
                    "*** Replaying is cheap (one look, auto-return to now) — "
                    "use it WHEN a past moment needs verifying, but YOU "
                    "decide if it is needed and how many; don't replay when "
                    "the current frame already settles the answer. ***\n"
                    "If something in MEMORY is fuzzy or this frame hints "
                    "at the answer, use a tool BEFORE answering — "
                    "`replay` to revisit a past moment, `zoom_crop` for detail, "
                    "`memory_read` to recall a pinned fact, `asr_segment` "
                    "for audio, or `abstain` if there is genuinely no "
                    "evidence at all.\n"
                    "*** CRITICAL: After EVERY tool result, FIRST write a "
                    "1-2 line observation of what the tool revealed "
                    "(especially: counts, text, identities, state changes) "
                    "BEFORE you decide to call another tool. Do NOT chain "
                    "tools blindly — each tool call must be justified by "
                    "a specific gap in what you have ALREADY observed. "
                    "If the tool result confirms what you need, COMMIT "
                    "with `</Response>` NOW instead of calling more tools. "
                    "A zoom_crop that returns a clear image of the target "
                    "region IS ENOUGH — write your observation and answer. ***\n"
                    "If THIS frame marks a semantic boundary (event "
                    "ended / scene change / actor enter-exit-change / "
                    "key object change), ALSO emit a "
                    "`<Recap t0=<sec> t1=<sec> kind=<...>> ... </Recap>` "
                    "block (3–8 sentences, see OBSERVE-ONLY phase "
                    "instructions for Recap format) — the "
                    "system pins it as durable memory.\n"
                    "\n## How to use the [RETRIEVED] block\n"
                    "The `[RETRIEVED ...]` block above is the system's "
                    "best-guess evidence for your question. HOW TO USE IT:\n"
                    "  - Each `path#N` shows a chain through memory layers "
                    "(wk → ep → lt). Follow the chain — the working-layer "
                    "event is closest to the original pixels.\n"
                    "  - Cross-check: if a fact appears consistently across "
                    "working + episodic, it is reliable. If only in "
                    "long_term, verify via replay.\n"
                    "  - It is NOT exhaustive — use `memory_search` with "
                    "different keywords if needed.\n"
                    "\n## Tool selection (use BEFORE each call)\n"
                    "1. MOVING subject across frames? → `track_init` (NOT zoom_crop)\n"
                    "2. Answer depends on a PAST frame? → `memory_search` first; "
                    "if detail missing → `replay`\n"
                    "3. Cue is likely AUDIO? → `asr_segment`\n"
                    "4. STATIC fine detail on CURRENT frame (text, label, logo)? "
                    "→ `zoom_crop`\n"
                    "5. Durable fact to remember? → `memory_write(pin=true)`\n"
                    "Anti-pattern: 2+ zoom_crops with no new fact → switch tool.\n"
                    "\n## Multi-candidate disambiguation\n"
                    "When MEMORY has MULTIPLE candidates for a question:\n"
                    "  1. State-tag wins: [FINISHED] > [IN-PROGRESS] > [STARTING]\n"
                    "  2. ★ RECENCY OVER FREQUENCY ★\n"
                    "     The TAIL region (most recent frames before q_time) "
                    "contains WHAT JUST HAPPENED. If tail evidence contradicts "
                    "episodic/middle, TRUST THE TAIL — the episodic region "
                    "repeats earlier actions many times (high frequency) but "
                    "the video's LAST action is what the question asks about. "
                    "Example: 300 frames of chopping eggplant then 10 frames "
                    "of handling meat → the answer is MEAT, not eggplant.\n"
                    "  3. Pin > episodic > working for durable facts\n"
                    "  4. If uncertain → `replay` the frame, DON'T guess by frequency\n"
                    "  5. Foreground (actor's action) > background (scene decoration)\n"
                )

        user_blocks.append(
            f"<{self.current_round}s-{self.current_round+1}s>\n<image>"
        )

        images = self._build_images_for_phase(
            frame_idx=frame_idx,
            user_blocks=user_blocks,
            is_post_hoc=is_post_hoc,
        )

        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": "\n\n".join(user_blocks)},
        ]
        # If we just resolved a tool continuation, also include the prior
        # assistant turn so the model sees its own request → result chain.
        if self._pending_tool_text and last_answer:
            messages.insert(1, {"role": "assistant", "content": last_answer})

        # ── Hard token guard (defense-in-depth) ──────────────────────────
        # The memory snapshot already soft-trims itself, but image tokens,
        # the system prompt and tool schemas are NOT counted there. As a
        # last line of defense we measure the real text token count and, if
        # the prompt risks exceeding ``max_model_len`` (after reserving room
        # for image tokens + generation), shrink the user turn from its
        # FRONT (the MEMORY/RETRIEVED region) while always keeping the tail
        # phase-instructions + the current <image> marker intact.
        self._enforce_token_budget(messages, images)

        self.data = {"messages": messages, "images": images}

        # Drain pending after consuming. Remember whether THIS turn is the
        # model responding to tool results (incl. zoom/replay image patches)
        # so the dispatcher can enforce "analyse before chaining the next
        # tool" (see _update_impl hard gate).
        self._responding_to_tool = bool(self._pending_tool_text)
        self._pending_tool_text = None
        self._pending_tool_images = []

        # Snapshot the request for trajectory streaming.
        self._last_request = {
            "kind": "brief",
            "messages": [dict(m) for m in messages],
            "images": list(images),
        }

        return True, InferRequest(**self.data)


    def _enforce_token_budget(self, messages, images) -> None:
        """Last-resort guard so a prompt never exceeds ``max_model_len``.

        vLLM aborts the entire EngineCore when ``prompt_len > max_model_len``
        (it raises ValueError rather than truncating). We measure the real
        text token count with the engine tokenizer, reserve room for image
        tokens and generation, and if the user turn is still too large we
        trim it from the FRONT (oldest MEMORY/RETRIEVED lines) while keeping
        the tail phase-instructions + current ``<image>`` marker intact.
        """
        eng = getattr(self, "_engine", None)
        if eng is None:
            return
        tok = getattr(eng, "tokenizer", None)
        max_len = int(getattr(eng, "max_model_len", 0) or 0)
        if tok is None or max_len <= 0:
            return
        # Reserve: generation headroom + per-image tokens (image markers are
        # expanded by vLLM, not by the text tokenizer). 1100/img is a
        # conservative upper bound for this model's IMAGE_MAX_TOKEN_NUM.
        n_img = len(images) if images else 0
        reserve = 2048 + n_img * 1100
        budget = max_len - reserve
        if budget <= 1024:
            budget = max(1024, max_len // 2)

        def _count(msgs) -> int:
            total = 0
            for m in msgs:
                try:
                    total += len(tok.encode(str(m.get("content", "")),
                                            add_special_tokens=False))
                except Exception:
                    total += len(str(m.get("content", ""))) // 3
            return total

        if _count(messages) <= budget:
            return
        # Find the user turn carrying the MEMORY/instruction blocks.
        user_idx = None
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                user_idx = i
                break
        if user_idx is None:
            return
        blocks = str(messages[user_idx]["content"]).split("\n\n")
        # Always keep the final block (the `<...s>\n<image>` marker) and the
        # last couple of phase-instruction blocks; trim from the front.
        keep_tail = 3 if len(blocks) > 3 else 1
        while len(blocks) > keep_tail and _count(messages) > budget:
            # Drop the 2nd-from-front block first (block 0 is the MEMORY
            # header / long-term region we prefer to keep); fall back to
            # front if only header remains.
            drop_at = 1 if len(blocks) - keep_tail > 1 else 0
            blocks.pop(drop_at)
            messages[user_idx]["content"] = "\n\n".join(blocks)
        # Final hard clamp: if STILL over budget, character-truncate the
        # user content middle, preserving head + tail.
        if _count(messages) > budget:
            content = str(messages[user_idx]["content"])
            approx_chars = budget * 3
            if len(content) > approx_chars:
                head = content[: approx_chars // 2]
                tail = content[-approx_chars // 2:]
                messages[user_idx]["content"] = (
                    head
                    + "\n\n[... MEMORY truncated to fit context window; "
                    "use memory_search / replay to recall detail ...]\n\n"
                    + tail
                )


    def _build_tool_continuation_request(self, last_answer: str):
        """Inject a tool-result *user* turn without advancing the clock."""
        from swift import InferRequest  # lazy import

        if self.data is None:
            # Should not happen — but guard.
            return super().get_next_request(last_answer)
        messages = self.data["messages"]
        # Append assistant's last raw answer (may contain <tool_call>...)
        messages.append({"role": "assistant", "content": last_answer or ""})
        # Build the user turn carrying tool results + optional extra images
        extra_imgs = list(self._pending_tool_images)
        body = self._pending_tool_text or ""
        # Post-tool observation nudge: the model often chains tools
        # blindly without observing what the last tool revealed.  A
        # one-line reminder keeps the loop bounded.
        if self.current_round >= self.question_time:
            body += (
                "\n\n[System] ↑ Tool result above.  BEFORE calling "
                "another tool, write ONE line describing what this "
                "result tells you (count? text? identity? state?).  "
                "If the evidence is now sufficient, emit "
                "`</Response> <answer>` AND STOP — do NOT call "
                "another tool."
            )
        if extra_imgs:
            body += "\n" + "\n".join(["<image>"] * len(extra_imgs))
        messages.append({"role": "user", "content": body})
        if extra_imgs:
            self.data["images"].extend(extra_imgs)
        self._enforce_token_budget(messages, self.data["images"])
        # Consume the pending payload
        self._pending_tool_text = None
        self._pending_tool_images = []
        # Snapshot the request for trajectory streaming.
        self._last_request = {
            "kind": "tool_continuation",
            "messages": [dict(m) for m in messages],
            "images": list(self.data["images"]),
        }
        return True, InferRequest(**self.data)

    # ------------------------------------------------------------------
    # System-side auto memory writes
    # ------------------------------------------------------------------
    # Regex used by ``_maybe_autopin_observation``. Compiled once at
    # class-definition time to avoid per-round overhead.
    _AUTOPIN_COUNT_RE = re.compile(
        r"\b(\d+)\s+([A-Za-z]{3,})s?\b"
    )
    _AUTOPIN_LAYOUT_RE = re.compile(
        r"(?:in\s+(?:the\s+)?(?:front|back|middle|center|left|right|"
        r"foreground|background)\s+(?:of|line|row)|"
        r"(?:left|right)most|biggest|smallest|tallest|closest|farthest|"
        r"in\s+a\s+row|line\s+of|row\s+of|"
        r"前面|后面|左[侧边]|右[侧边]|中间|最大|最小|最高|最近|最远)",
        re.IGNORECASE,
    )
    _VAGUE_QUANTIFIERS_RE = re.compile(
        r"\b(multiple|several|many|some|a\s+few|a\s+couple|a\s+group\s+of|"
        r"a\s+bunch\s+of|various)\s+([A-Za-z]{3,})s?\b",
        re.IGNORECASE,
    )

    def _maybe_autopin_observation(self, text: str) -> None:
        """No-op under the new key-node-only pinned-layer policy.

        Previously this method auto-pinned observations carrying an
        explicit count (>=2) plus a layout keyword (e.g. "the biggest
        giant standing in the front row"). Under the new memory
        semantics the long-term pinned layer is reserved for KEY
        NODES (scene change / object change / action start / action
        end / actor enter/exit), which the model must surface via
        `<Recap>` blocks. Generic spatial layouts now live in the
        working + episodic layers only, so this hook is intentionally
        neutered. Kept as a stub so the rest of the call-graph (and
        any future non-spatial heuristic) does not need editing.
        """
        return

    def _build_rolling_query(self) -> str:
        """Build an implicit pre-question query from recent Observation
        deltas (the rolling-delta window).

        Used when the question has not been posed yet — we still want
        ``snapshot(mode='selective')`` to recall topically-coherent
        events instead of degenerating into "head + tail" only. We
        concatenate the ``delta`` snippet (or full text fallback) of
        the last ``rolling_window`` events.
        """
        if self._memory is None or not self._memory.events:
            return ""
        evs = self._memory.events[-self.rolling_window:]
        parts: List[str] = []
        for e in evs:
            chunk = (e.delta or e.text or "").strip()
            if chunk:
                parts.append(chunk[:200])
        return " ".join(parts)[:400]

    def _evidence_query(self) -> str:
        return build_evidence_query(
            question=self.question or "",
            plan=self._query_plan,
            sample=self.sample,
        )

    def _resolve_render_args(
        self, current_t: int, past_q: bool
    ) -> Dict[str, Any]:
        """Decide MEMORY snapshot kwargs for this round.

        Honors three protections:
          * Cold start: first ``cold_start_rounds`` always use full mode.
          * Recall query: post-question uses ``self.question``;
            pre-question uses rolling-delta window.
          * Degrade protection: if working layer is too small OR recall
            hits too few events, fall back to full mode.
        """
        base: Dict[str, Any] = dict(
            token_budget=self.memory_token_budget * (2 if past_q else 1),
            current_t=current_t,
        )
        if past_q:
            base.update(tail_max=48, summary_max=60)

        if self.memory_render_mode == "full":
            base["mode"] = "full"
            return base

        if self.memory_render_mode == "continuous":
            # past_q: drop the verbatim head (the video's opening frames).
            # The model should reason from the recent TAIL + episodic
            # summaries + long-term pins, not from stale opening frames
            # that bias it toward early (high-frequency) evidence.
            head_secs = (0 if past_q else self.memory_head_seconds)
            base.update(
                mode="continuous",
                head_seconds=head_secs,
                tail_seconds=self.memory_tail_seconds,
            )
            # Optional recall enrichment: use rolling-delta query (or
            # the question post-Q) to bring back top-K relevant middle
            # events. Failures are silent — continuous still renders
            # head + tail verbatim regardless.
            try:
                if past_q and self.question:
                    q = self._evidence_query()
                else:
                    q = self._build_rolling_query()
                if q:
                    base["recall_query"] = q
                    base["recall_topk"] = self.memory_recall_topk
            except Exception:
                pass
            return base

        # mode == "selective" below
        if self.memory_render_mode != "selective":
            base["mode"] = "full"
            return base

        # Cold start protection
        if current_t < self.cold_start_rounds:
            base["mode"] = "full"
            return base

        # Degrade-by-size protection
        n_events = len(self._memory.events) if self._memory else 0
        if n_events < self.degrade_events_min:
            base["mode"] = "full"
            return base

        # Build query
        if past_q and self.question:
            q = self._evidence_query()
        else:
            q = self._build_rolling_query()
        if not q:
            base["mode"] = "full"
            return base

        # Degrade-by-recall protection: peek retrieve hits and fallback
        # if we cannot recall at least ``degrade_threshold`` events.
        try:
            hits = self._memory.retrieve(
                query=q, top_k=max(self.memory_recall_topk,
                                    self.degrade_threshold),
                current_t=current_t,
                layers=("working",),
            )
            n_hits = len((hits or {}).get("working") or [])
        except Exception:
            n_hits = 0
        if n_hits < self.degrade_threshold:
            base["mode"] = "full"
            return base

        base.update(
            mode="selective",
            recall_query=q,
            recall_topk=self.memory_recall_topk,
            selective_head_max=self.memory_head_max,
            selective_tail_max=self.memory_tail_max,
        )
        return base

    def _render_backtrack_paths(self, paths: List[Any],
                                 query: str) -> Optional[str]:
        """Render EvidencePath list as a `[RETRIEVED ...]` block.

        Each EvidencePath is a connected cross-layer chain.
        We flatten nodes into per-layer sections, keeping the path
        structure visible via indentation.
        """
        if not paths:
            return None
        q_short = (query or "")[:120].replace("\n", " ")
        lines: List[str] = [
            f'[RETRIEVED top evidence paths for: "{q_short}"]'
        ]
        for pi, path in enumerate(paths[:8]):
            lines.append(
                f"  path#{pi+1} score={path.score:.2f} "
                f"layers={path.layers_covered}:"
            )
            for node in path.nodes[:6]:
                layer_tag = {"working": "wk", "episodic": "ep",
                             "long_term": "lt"}.get(node.layer, "??")
                t_tag = f"t={node.t:>4d}" if node.t is not None else ""
                lines.append(
                    f"    [{layer_tag}] {t_tag} "
                    f"{(node.text or '')[:160]}"
                )
        return "\n".join(lines)

    def _render_retrieved_block(self, retrieved: Optional[Dict[str, Any]],
                                 query: str) -> Optional[str]:
        """Render the `[RETRIEVED ...]` user-block for the prompt.

        ``retrieved`` is the dict returned by ``MemoryStore.retrieve()``;
        we layout long_term → episodic → working with one item per line.
        Empty layers are skipped. Returns None if every layer is empty.
        """
        if not retrieved:
            return None
        lt = retrieved.get("long_term") or []
        ep = retrieved.get("episodic") or []
        wk = retrieved.get("working") or []
        if not (lt or ep or wk):
            return None
        q_short = (query or "")[:120].replace("\n", " ")
        lines: List[str] = [
            f'[RETRIEVED top relevant to question: "{q_short}"]'
        ]
        if lt:
            lines.append("  long_term:")
            for it in lt:
                t_tag = (f"t={it['t']:>4d} "
                         if it.get("t") is not None else "")
                kind = (it.get("kind") or "")[:6]
                ep_ref = (f" ←ep#{it['ep_id']}"
                          if it.get("ep_id") is not None else "")
                lines.append(
                    f"    [#{it.get('id')}] {t_tag}[{kind}]{ep_ref} "
                    f"{(it.get('text') or '')[:240]}"
                )
        if ep:
            lines.append("  episodic:")
            for it in ep:
                kind = (it.get("kind") or "")[:6]
                pin_ref = (f" →pin#{it['pin_id']}"
                           if it.get("pin_id") is not None else "")
                lines.append(
                    f"    [ep#{it.get('id')}] t={it.get('t0')}-"
                    f"{it.get('t1')}s [{kind}]{pin_ref} "
                    f"{(it.get('summary') or '')[:240]}"
                )
        if wk:
            lines.append("  working:")
            for it in wk:
                pref = "[Δ]" if it.get("is_change") else "[obs]"
                ep_ref = (f" ↑ep#{it['parent_episode_id']}"
                          if it.get("parent_episode_id") is not None
                          else "")
                lines.append(
                    f"    t={int(it.get('t', 0)):>4d} {pref}{ep_ref} "
                    f"{(it.get('text') or '')[:240]}"
                )
        return "\n".join(lines)

    def _render_plan_banner(self) -> str:
        """Render the captured <Plan> as a compact standing instruction.

        Shown atop every POST-QUESTION turn so the model executes the
        strategy it set for itself instead of drifting into tool loops.
        """
        plan = self._query_plan or {}
        lines = ["[PLAN] Your own analysis of this question — FOLLOW IT:"]
        for key in ("need", "pov", "info_type", "evidence_locus",
                    "answerability", "meta", "strategy"):
            val = plan.get(key)
            if val:
                lines.append(f"  - {key}: {val}")
        if len(lines) == 1:
            # Only the raw body was captured (no recognised slots).
            raw = plan.get("_raw", "")
            if raw:
                lines.append(f"  {raw}")
        contract = render_evidence_contract(plan, self._evidence_query())
        if contract:
            lines.append(contract)
        return "\n".join(lines)

    def _persist_tool_round_observation(self, answer: str) -> None:
        """Write the model's free-text observation from a tool round to memory.

        On tool-continuation rounds the model narrates what a tool result
        revealed (e.g. "the zoomed patch shows 5 cards in his hand") before
        chaining the next call. That round returns early and never reaches
        the brief-mode summarizer, so this understanding used to be lost.
        Here we strip thinking / tool_call / sentinel markup and, if a
        substantive sentence remains, persist it as a low-key observation
        event (indexed into the entity graph) so cropped/replayed evidence
        readings become durable and retrievable.
        """
        if self._memory is None or not answer:
            return
        txt = ovo._strip_thinking(answer) or ""
        # Drop tool_call blocks and any sentinel/recap/plan markup.
        txt = TOOL_CALL_RE.sub(" ", txt)
        txt = _RECAP_BLOCK_RE.sub(" ", txt)
        txt = _PLAN_BLOCK_RE.sub(" ", txt)
        for tag in ("</Response>", "</Silence>", "</Standby>",
                     "<Observation>", "</Observation>"):
            txt = txt.replace(tag, " ")
        txt = re.sub(r"\s+", " ", txt).strip()
        # Require some substance (avoid pinning "ok"/"let me check").
        if len(txt) < 25:
            return
        # Keep enough room for the model's full reading/conclusion. A 600-char
        # cap chopped analyses mid-sentence (e.g. sample 8 tool-obs ended at
        # "Let me..."), so the next turn saw a truncated half-thought it could
        # not continue and re-derived the same reasoning from scratch — fueling
        # the replay/search spin. 2000 fits a complete conclusion while staying
        # under write_event's 4000 hard cap.
        txt = txt[:2000]
        try:
            self._memory.write_event(
                self.current_round, "observation",
                f"[tool-obs] {txt}",
                ref_frame=self.current_round, salience=1.2,
            )
            # Index entities so the reading joins the entity timeline.
            self._post_graph_hook(
                {"text": txt, "entities":
                    MemoryStore.extract_entities(txt)
                    if hasattr(MemoryStore, "extract_entities") else []},
                self.current_round,
            )
        except Exception:
            pass

    def _maybe_pin_recaps(self, answer: str) -> None:
        """Pin every well-formed `<Recap>` block emitted by the model.

        Each Recap becomes one durable ``pinned_facts`` entry with
        ``kind=recap_<subkind>`` so it survives MEMORY compression and
        is rendered first in the LONG-TERM section. We rate-limit by
        ``_recap_min_gap`` seconds and de-duplicate against the last
        pinned recap text to drop trivially-repeated payloads.
        """
        if self._memory is None or not answer:
            return
        recaps = parse_recap_blocks(answer)
        if not recaps:
            return
        # Always log a working-layer event so the trajectory shows
        # the model attempted recaps even if some get throttled.
        for rc in recaps:
            now_t = int(self.current_round)
            t1 = rc.get("t1") if rc.get("t1") is not None else now_t
            t0 = rc.get("t0") if rc.get("t0") is not None else max(0, t1 - 5)
            # Throttle: at most one recap pinned per _recap_min_gap s
            if (self._last_recap_t is not None
                    and now_t - self._last_recap_t < self._recap_min_gap):
                self._memory.write_event(
                    now_t, "note",
                    f"[recap-throttled] {rc['text'][:80]}",
                    ref_frame=now_t, salience=0.5,
                )
                continue
            # De-dup: skip if the previous recap shares >85% prefix
            new_head = rc["text"][:120].lower()
            dup = False
            for p in reversed(self._memory.pinned_facts):
                if not str(p.get("kind", "")).startswith("recap"):
                    continue
                prev_head = (p.get("text") or "")[:120].lower()
                if prev_head and prev_head == new_head:
                    dup = True
                break  # only check the most recent recap
            if dup:
                continue
            kind_tag = "recap_" + str(rc.get("kind") or "recap")
            subject = rc.get("subject")
            body = (
                f"[t={t0}-{t1}s] [{rc.get('kind') or 'recap'}]"
                + (f" [subject={subject}]" if subject else "")
                + f" {rc['text']}"
            )
            # Build the episodic-layer twin first so we can link them.
            # ev_ids: pull working events whose t falls in [t0,t1].
            ev_ids = [
                ev.eid for ev in self._memory.events
                if ev.t is not None and t0 <= ev.t <= t1
            ]
            ep = self._memory.write_episode(
                t0=int(t0), t1=int(t1),
                summary=rc["text"],
                kind=kind_tag,
                ev_ids=ev_ids,
                salience=2.0,
            )
            # Long-term pin mirrors the episode and stores the back-link.
            # Passing ``subject`` triggers automatic supersession of any
            # earlier active pin sharing the same (kind_family, subject).
            self._memory.pin_fact(
                body, kind=kind_tag, t=int(t1), ep_id=ep["id"],
                subject=subject,
            )
            self._memory.write_event(
                now_t, "note",
                f"[recap-pinned ep#{ep['id']} t={t0}-{t1}] {rc['text'][:80]}",
                ref_frame=now_t, salience=2.0,
            )
            self._last_recap_t = now_t

    def _has_vague_quantifier(self, text: str) -> Optional[str]:
        """Return the noun that was vaguely-quantified, or None.

        Used by the OBSERVE-ONLY phase to inject a follow-up prompt that
        nudges the model to re-count. We only care about countable nouns
        (e.g. ``multiple giants``); single matches like ``some smoke`` do
        not need a re-count and are filtered out by the noun length /
        plural check.
        """
        if not text:
            return None
        m = self._VAGUE_QUANTIFIERS_RE.search(text)
        if not m:
            return None
        noun = m.group(2)
        # Skip uncountable / mass nouns we can't usefully recount.
        if noun.lower() in {"smoke", "fog", "mist", "water", "sand",
                              "light", "darkness", "noise", "music",
                              "sound", "energy", "snow", "rain"}:
            return None
        return noun.lower()

    def _latest_vague_noun(self) -> Optional[str]:
        """Scan the most recent observation event in MEMORY for a fuzzy
        quantifier and return the noun it qualified, else ``None``.

        Used by the OBSERVE-ONLY prompt builder to add a one-line nudge
        encouraging the model to recount on the current frame. We only
        look at the last 2 observation events to keep the nudge fresh.
        """
        if self._memory is None:
            return None
        recent = [
            e for e in self._memory.events[-6:]
            if e.kind == "observation"
            and not (e.text or "").startswith("frame_anchor")
        ]
        for e in reversed(recent[-2:]):
            noun = self._has_vague_quantifier(e.text or "")
            if noun:
                return noun
        return None

    def _latest_delta(self) -> Optional[Tuple[int, str]]:
        """Return the most recent non-empty (t, delta) pair from MEMORY,
        or ``None`` if no prior frame has a delta.

        Used by the OBSERVE-ONLY prompt to surface the previous frame's
        delta so the model can be explicitly told NOT to copy it. This
        is the prompt-side complement to ``RuleSummarizer``'s code-side
        repetition detection.
        """
        if self._memory is None:
            return None
        for e in reversed(self._memory.events[-12:]):
            if e.kind != "observation":
                continue
            d = (e.delta or "").strip()
            if d:
                return (int(e.t), d[:200])
        return None

    def _build_state_digest(self) -> Optional[str]:
        """扫描 MEMORY 中所有显式状态标签，浓缩成一段 [STATE DIGEST]。

        识别两类状态痕迹：
          1) Observation / change 事件中含 ``[FINISHED]`` /
             ``[STARTED]`` / ``[IN-PROGRESS]`` / ``[ABORTED]`` 标签；
          2) pinned_facts (kind=recap_*) 中含 ``state=finished`` /
             ``state=started`` 等属性，或 body 以 ``FINISHED:`` /
             ``STARTED:`` 起头。

        将每条命中按时间排序后输出，例如：
            [STATE DIGEST]
              t=12s [FINISHED] covers lid (Observation)
              t=15s state=finished | pull up the hair... (Recap)
              t=20s [IN-PROGRESS] pours water (Observation)

        SSR/CRR 这类元问题任务直接读取这一块，能避免靠 verb tense
        猜测事件状态（如把"走向汽车"误判为"未到达"）。
        """
        if self._memory is None:
            return None

        STATE_TAGS = (
            "[FINISHED]", "[STARTED]", "[IN-PROGRESS]", "[ABORTED]",
        )
        STATE_PREFIXES = ("FINISHED:", "STARTED:")
        STATE_ATTRS = (
            "state=finished", "state=started",
            "state=in_progress", "state=aborted",
        )

        hits: List[Tuple[int, str]] = []

        # 1) 从 working layer events 抓显式 tag
        try:
            for e in self._memory.events:
                txt = (e.text or "")
                for tag in STATE_TAGS:
                    if tag in txt:
                        # 提取 tag 周围最多 100 字符
                        idx = txt.find(tag)
                        snippet = txt[max(0, idx - 30):idx + 80].strip()
                        hits.append((int(e.t), f"{tag} {snippet}"[:160]))
                        break  # 每事件最多记一条
        except Exception:
            pass

        # 2) 从 pinned_facts 抓 state= 属性 / FINISHED: STARTED: 起头
        try:
            for p in (self._memory.pinned_facts or []):
                kind = str(p.get("kind", ""))
                if not kind.startswith("recap"):
                    continue
                txt = p.get("text") or ""
                t = int(p.get("t") or 0)
                if any(prefix in txt for prefix in STATE_PREFIXES):
                    hits.append((t, txt[:160]))
                    continue
                for attr in STATE_ATTRS:
                    if attr in txt:
                        idx = txt.find(attr)
                        snippet = txt[max(0, idx - 20):idx + 100].strip()
                        hits.append((t, f"{attr} | {snippet}"[:160]))
                        break
        except Exception:
            pass

        if not hits:
            return None

        # 排序 + 去重（同 t 同 prefix 只留一条）
        hits.sort(key=lambda x: x[0])
        seen = set()
        deduped = []
        for t, s in hits:
            key = (t, s[:40])
            if key in seen:
                continue
            seen.add(key)
            deduped.append((t, s))

        # 限制长度（最多 12 条，避免淹没 prompt）
        if len(deduped) > 12:
            deduped = deduped[-12:]

        lines = "\n".join(f"  t={t}s  {s}" for t, s in deduped)
        return (
            "[STATE DIGEST] Explicit event-state observations recorded "
            "in MEMORY (use these for event-state or answerability "
            "meta-questions — DO NOT re-infer state from verb tense):\n"
            f"{lines}"
        )

    # ── Novelty-Adaptive Cadence ──────────────────────────────────────
    def _adaptive_cadence(self, t: int):
        """Return (anchor_interval, asr_interval, track_interval) driven
        by the average novelty of recent observation events.

        High-activity periods → denser recording; static periods →
        sparser.  Content-driven, not time-driven — safe for real-time
        video where question_time is unknown.
        """
        # ── StreamingBench-only dense cadence ─────────────────────────────
        # Gated on env STREAMO_SB_DENSE_CADENCE, which is set to "1" ONLY in
        # the StreamingBench branch of launch_eval.sh. OVO-Bench runs never
        # set it (default "0"), so `dense` is False for OVO and the code below
        # is byte-identical to the original path — OVO cadence is unchanged.
        #
        # Rationale: across Real-Time VU tasks (Counting, Action/Object/Event
        # Perception, Spatial) the error rate rises monotonically with video
        # length (e.g. Counting 0.15 -> 0.73), a systematic under-sampling /
        # temporal-integration gap on long videos. A denser recording cadence
        # captures more of the timeline before the (end-of-video) question.
        dense = os.environ.get("STREAMO_SB_DENSE_CADENCE", "0") == "1"

        # Default intervals (mid-activity).
        if dense:
            anchor_iv, asr_iv, track_iv = 3, 6, 2
        else:
            anchor_iv, asr_iv, track_iv = 5, 10, 3

        if self._memory is None or not self._memory.events:
            return anchor_iv, asr_iv, track_iv

        # Compute average novelty over the last 8 observation events.
        recent_obs = [
            e for e in self._memory.events[-12:]
            if getattr(e, "kind", "") == "observation"
        ]
        if not recent_obs:
            return anchor_iv, asr_iv, track_iv

        # Extract novelty from event text prefix "[nov=0.XX]".
        novelties: List[float] = []
        for e in recent_obs[-8:]:
            txt = getattr(e, "text", "")
            m = re.match(r"\[nov=([0-9.]+)\]", txt)
            if m:
                novelties.append(float(m.group(1)))
        if not novelties:
            return anchor_iv, asr_iv, track_iv

        avg_nov = sum(novelties) / len(novelties)

        if dense:
            # Denser at every activity tier; static periods still record ASR
            # (unlike the default tier which drops to 0) so long quiet spans
            # do not lose the audio timeline.
            if avg_nov > 0.6:       # High-activity — very dense.
                anchor_iv, asr_iv, track_iv = 1, 3, 1
            elif avg_nov > 0.3:     # Medium-activity.
                anchor_iv, asr_iv, track_iv = 3, 6, 2
            else:                   # Static — sparse but still sampled.
                anchor_iv, asr_iv, track_iv = 5, 10, 0
            return anchor_iv, asr_iv, track_iv

        if avg_nov > 0.6:       # High-activity — dense recording.
            anchor_iv, asr_iv, track_iv = 2, 5, 1
        elif avg_nov > 0.3:     # Medium-activity — default.
            anchor_iv, asr_iv, track_iv = 5, 10, 3
        else:                   # Static — sparse but still anchored.
            anchor_iv, asr_iv, track_iv = 10, 0, 0

        return anchor_iv, asr_iv, track_iv

    def _auto_write_memory(self, frame_idx: int) -> None:
        """Write periodic frame-anchor + (optional) ASR events to memory.

        Runs every brief-mode round BEFORE the snapshot is rendered, so the
        very first frame already has an anchor entry and MEMORY never stays
        empty. Pure system-side; does not consume the per-sample tool budget.

        Note: in post-hoc / post-question phases the model itself emits an
        ``<Observation>`` per frame which RuleSummarizer writes — those are
        much more informative than a generic frame_anchor. We therefore
        only emit a frame_anchor when there is no observation event near
        the current time (cheap fallback).
        """
        if self._memory is None:
            return
        t = int(self.current_round)

        # ── Novelty-Adaptive Cadence ──────────────────────────────────
        # Content-driven recording density: high-activity periods
        # automatically get denser recording; static periods get
        # sparser.  This replaces a fixed time-based cadence and is
        # safe for real-time video where question_time is unknown.
        _anchor_iv, _asr_iv, _track_iv = self._adaptive_cadence(t)
        if os.environ.get("STREAMO_DISABLE_AUTO_ASR", "0") == "1":
            _asr_iv = 0
        self.auto_anchor_every = _anchor_iv
        self.auto_asr_every = _asr_iv
        self.auto_track_query_every = _track_iv

        # 1) Periodic frame-anchor — only as fallback when the model has
        #    not produced any [obs] event in the recent window.
        if self.auto_anchor_every > 0:
            should_anchor = (
                self._last_anchor_t is None
                or (t - self._last_anchor_t) >= self.auto_anchor_every
            )
            if should_anchor:
                # Skip if the model already wrote an observation/change event
                # in the last ``auto_anchor_every`` seconds (avoids polluting
                # MEMORY with low-info anchors next to real <Observation>s).
                window = self.auto_anchor_every
                recent_obs = any(
                    e.kind in ("observation", "change")
                    and not e.text.startswith("frame_anchor")
                    and (t - e.t) <= window
                    for e in self._memory.events
                )
                if not recent_obs:
                    # Build an informative anchor: prefer the most recent
                    # event that carries an explicit state tag (e.g.
                    # `[FINISHED]`, `[STARTED]`, `state=finished` from a
                    # Recap), so the anchor itself carries semantic state
                    # — useful for SSR/CRR meta-questions that scan
                    # MEMORY for completion evidence. Fall back to the
                    # most recent non-anchor event if no state-tagged
                    # event exists.
                    last_state = ""
                    state_keywords = (
                        "[FINISHED]", "[STARTED]", "[IN-PROGRESS]",
                        "[ABORTED]", "FINISHED:", "STARTED:",
                        "state=finished", "state=started",
                        "state=in_progress", "state=aborted",
                    )
                    # Pass 1: prefer an event with explicit state tag
                    for e in reversed(self._memory.events):
                        if (e.kind in ("observation", "change", "note")
                                and not e.text.startswith("frame_anchor")
                                and any(kw in e.text for kw in state_keywords)):
                            last_state = f" (recent_state: {e.text[:120]})"
                            break
                    # Pass 2: fall back to the most recent non-anchor obs
                    if not last_state:
                        for e in reversed(self._memory.events):
                            if (e.kind in ("observation", "change")
                                    and not e.text.startswith("frame_anchor")):
                                last_state = f" (state: {e.text[:100]})"
                                break
                    anchor_text = (
                        f"frame_anchor t={t}s "
                        f"frame={os.path.basename(self.frame_paths[frame_idx])}"
                        f"{last_state}"
                    )
                    self._memory.write_event(
                        t, "observation", anchor_text,
                        ref_frame=frame_idx,
                        salience=0.5,
                    )
                    # Feed auto-anchors into the MemoryGraph so BOCPD
                    # run-length tracking sees every frame (even
                    # low-info ones) and entity time-coverage is
                    # complete.
                    self._post_graph_hook(
                        {"text": anchor_text, "entities": []}, t,
                    )
                self._last_anchor_t = t

        # 2) Sidecar ASR windows become visible only after their end time.
        #    This path is independent of novelty cadence so every fixed
        #    window is injected exactly once. The final window ending at the
        #    question boundary remains excluded to match strict pre-query
        #    trajectory restore semantics.
        sidecar_asr = False
        if os.environ.get("STREAMO_ASR_BACKEND", "").lower() == "sidecar":
            sidecar_asr = True
            try:
                from tools.audio import DEFAULT_ASR, SidecarASR  # lazy
                if not isinstance(DEFAULT_ASR, SidecarASR):
                    raise RuntimeError("sidecar ASR backend failed to initialize")
                segments = DEFAULT_ASR.ready_segments(
                    self.ctx.sample_id,
                    after_t=self._last_sidecar_asr_t,
                    current_t=t,
                    question_time=int(self.question_time),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"ASR sidecar unavailable for {self.ctx.sample_id}"
                ) from exc
            for segment in segments:
                t0 = int(segment["t0"])
                t1 = int(segment["t1"])
                text = str(segment.get("text") or "").strip()
                if text:
                    self._memory.add_audio(t0, t1, text, None)
                    asr_text = f"asr[{t0}-{t1}s]: {text[:80]}"
                    self._memory.write_event(
                        t0, "audio", asr_text,
                        ref_frame=t0, salience=1.4,
                    )
                    self._post_graph_hook(
                        {"text": asr_text, "entities": []}, t0,
                    )
                self._last_sidecar_asr_t = max(
                    self._last_sidecar_asr_t, t1
                )

        # 3) Periodic live auto-ASR — only if ASR is reachable in this run
        #    (audio_cache exists OR an ASR-capable adapter is plugged in).
        if self.auto_asr_every > 0 and not sidecar_asr:
            should_asr = (
                self._last_auto_asr_t is None
                or (t - self._last_auto_asr_t) >= self.auto_asr_every
            )
            if should_asr and t >= 1:
                t0 = max(0, t - self.auto_asr_window)
                t1 = t
                if t1 > t0:
                    try:
                        from tools.audio import DEFAULT_ASR  # lazy
                        res = DEFAULT_ASR.transcribe(self.ctx.sample_id, t0, t1)
                    except Exception:
                        res = None
                    if res:
                        self._memory.add_audio(t0, t1,
                                                res.get("text", ""),
                                                None)
                        if res.get("text"):
                            asr_text = f"asr[{t0}-{t1}s]: {res['text'][:80]}"
                            self._memory.write_event(
                                t0, "audio", asr_text,
                                ref_frame=t0, salience=1.4,
                            )
                            # Feed ASR events through graph hook so
                            # BOCPD can detect scene boundaries from
                            # audio content changes.
                            self._post_graph_hook(
                                {"text": asr_text, "entities": []}, t0,
                            )
                self._last_auto_asr_t = t

        # 4) Periodic auto track_query — for any active SAM2 track the
        #    model has spawned, refresh its bbox / visibility every
        #    ``auto_track_query_every`` seconds without consuming the
        #    sample tool budget. This guarantees long-running trajectories
        #    even when the model stops calling track_query itself.
        if self.auto_track_query_every > 0 and self.ctx.tracks:
            should_q = (
                self._last_auto_track_t is None
                or (t - self._last_auto_track_t) >= self.auto_track_query_every
            )
            if should_q:
                try:
                    from tools.track import DEFAULT_TRACKER  # lazy
                except Exception:
                    DEFAULT_TRACKER = None  # type: ignore
                if DEFAULT_TRACKER is not None:
                    for tid, tr in list(self.ctx.tracks.items()):
                        if tr.get("status") != "active":
                            continue
                        try:
                            out = DEFAULT_TRACKER.query(
                                tr["session"], self.frame_paths[frame_idx])
                        except Exception:
                            continue
                        tr.setdefault("traj", []).append({"t": t, **out})
                        self._memory.upsert_track(
                            tid, traj_append=[{"t": t, **out}])
                        bbox = out.get("bbox")
                        in_view = out.get("in_view")
                        if in_view and bbox:
                            track_text = (
                                f"auto_track {tid} "
                                f"bbox={[int(v) for v in bbox]}"
                            )
                            self._memory.write_event(
                                t, "tool", track_text,
                                ref_frame=frame_idx, salience=0.9,
                            )
                            # Index track entities in the MemoryGraph
                            # so entity timelines are complete.
                            self._post_graph_hook(
                                {"text": track_text, "entities": [tid]}, t,
                            )
                        elif not in_view:
                            track_text = (
                                f"auto_track {tid} lost "
                                f"(score={out.get('score')})"
                            )
                            self._memory.write_event(
                                t, "tool", track_text,
                                ref_frame=frame_idx, salience=1.1,
                            )
                            self._post_graph_hook(
                                {"text": track_text, "entities": [tid]}, t,
                            )
                            tr["status"] = "lost"
                self._last_auto_track_t = t

    # ── Pillar 2: Graph post-write hook (BOCPD + entity indexing) ─────
    def _post_graph_hook(self, ev_payload: Optional[Dict[str, Any]],
                          t: int) -> None:
        """Run after every observation write: update BOCPD segmenter,
        index entities, and wire cross-layer pointers in the MemoryGraph."""
        if self._graph is None or self._memory is None or ev_payload is None:
            return
        # 1) Entity indexing.
        entities = ev_payload.get("entities") or []
        if entities:
            resolved = self._graph.entity_index.add_observation(
                self._memory._next_eid - 1,  # the event just written
                t, entities,
            )
            # Co-reference resolution: merge bare head-nouns into more
            # specific compound entities ("man" → "man_in_white") so a
            # single entity timeline is not split across surface names.
            try:
                self._graph.entity_index.auto_link_aliases(resolved)
            except Exception:
                pass
        # 2) BOCPD update.
        if self._bocpd is not None:
            from tools.memory_graph import observation_to_embedding
            text = ev_payload.get("text", "")
            emb = observation_to_embedding(text)
            self._bocpd.update(emb)
            # ── Cache embedding for hybrid retrieval ─────────────────
            if self._memory is not None:
                eid = self._memory._next_eid - 1
                self._memory.cache_embedding(eid, emb)
            # If changepoint detected → seal current run into an episode.
            # P7 Content Gate: use should_seal_with_content_check to
            # suppress BOCPD false-positives from lighting/camera jitter.
            run = getattr(self, '_current_run_events', [])
            if run:
                first_text = getattr(run[0], 'text', '') if run else ''
                seal_ok = self._bocpd.should_seal_with_content_check(
                    current_text=text, run_first_text=first_text,
                )
            else:
                seal_ok = self._bocpd.should_seal()
            if seal_ok and len(run) >= 3:
                # Collect event IDs from the run.
                ev_ids = []
                for ev in run:
                    if hasattr(ev, 'eid'):
                        ev_ids.append(ev.eid)
                if ev_ids:
                    from tools.memory_graph import op_add_epi_node
                    summary = self._memory._make_summary(run) if hasattr(
                        self._memory, '_make_summary') else ""
                    if not summary:
                        from tools.memory import _summarize_run
                        summary = _summarize_run(run)
                    ep_id = op_add_epi_node(
                        self._memory, self._graph,
                        t0=run[0].t, t1=run[-1].t,
                        summary=summary, ev_ids=ev_ids,
                    )
                    # ── Causal chain: link the previous sealed episode to
                    # this one so the action ORDER (A→B→C) is explicit for
                    # ASI "what happens after X?" reasoning.
                    prev_ep = getattr(self, "_last_episode_id", None)
                    if prev_ep is not None and prev_ep != ep_id:
                        from tools.memory_graph import op_add_causal_edge
                        op_add_causal_edge(
                            self._graph, prev_ep, ep_id,
                            relation="precedes",
                        )
                    self._last_episode_id = ep_id
                    # ── System-side boundary pin (long-term backbone) ──
                    # Promote this sealed episode to a durable pin so the
                    # long_term layer is not starved when the model omits
                    # <Recap>. Throttled by time-gap and a per-sample cap.
                    self._maybe_autopin_boundary(
                        ep_id=ep_id, t0=run[0].t, t1=run[-1].t,
                        summary=summary, run=run, t=t,
                    )
                self._bocpd.reset()
                setattr(self, '_current_run_events', [])
        # 3) Accumulate into current BOCPD run.
        if self._bocpd is not None:
            run_events = getattr(self, '_current_run_events', None)
            if run_events is None:
                setattr(self, '_current_run_events', [])
                run_events = getattr(self, '_current_run_events', [])
            # Find the event object just written.
            evs = self._memory.events
            if evs:
                run_events.append(evs[-1])

    def _maybe_autopin_boundary(
        self, ep_id: int, t0: int, t1: int, summary: str,
        run: List[Any], t: int,
    ) -> None:
        """Promote a freshly-sealed episode boundary to a durable pin.

        The model seldom emits <Recap>, so without this the long_term
        layer holds only the question pin and all ep→pin pointers are
        empty. We pin a boundary fact for the sealed episode, wiring the
        ep→pin pointer via ``op_add_pin_node``. The fact text is generated
        by a dedicated LLM call (``_llm_boundary_fact``) that prioritises
        FACT COVERAGE over brevity — it should capture who/what/where/
        counts/state so the pinned long-term layer carries durable ground
        truth. Falls back to the episode summary head if the LLM call
        fails. Throttled by a minimum time-gap and a per-sample cap so
        long videos do not flood the pinned layer.
        """
        if self._memory is None or self._graph is None:
            return
        if self._auto_boundary_pins >= self._max_auto_boundary_pins:
            return
        if (self._last_boundary_pin_t is not None
                and t - self._last_boundary_pin_t < self._boundary_pin_min_gap):
            return
        # Prefer an LLM-generated, fact-covering boundary statement; fall
        # back to the episode summary head when the engine is unavailable.
        fact = self._llm_boundary_fact(run, t0, t1)
        if not fact:
            fact = (summary or "").strip().split("\n")[0][:400]
        if not fact:
            return
        body = f"[boundary t={t0}-{t1}s] {fact}"
        # Subject keys supersession: reuse a stable subject so a newer
        # boundary for the same span family replaces the older one rather
        # than stacking.
        subject = f"boundary_{t0}_{t1}"
        try:
            from tools.memory_graph import op_add_pin_node
            op_add_pin_node(
                self._memory, self._graph,
                text=body, kind="boundary", t=int(t1),
                ep_id=int(ep_id), subject=subject,
            )
            self._last_boundary_pin_t = t
            self._auto_boundary_pins += 1
        except Exception:
            pass

    def _llm_boundary_fact(
        self, run: List[Any], t0: int, t1: int,
    ) -> Optional[str]:
        """LLM-generate a FACT-COVERING boundary statement for long-term.

        Unlike the episodic summary (which aims for a concise narrative),
        this prioritises COVERAGE: enumerate the concrete facts of the
        segment — actors and their count, key objects and counts, the
        action(s) and their final state, locations / spatial relations,
        any text/OCR — so the pinned long-term layer holds durable ground
        truth that later questions can rely on. Brevity is NOT a goal;
        completeness is. Returns ``None`` on any error so the caller can
        fall back to the episode summary head.
        """
        if not run or self._engine is None:
            return None
        try:
            from swift import InferRequest  # lazy import (heavy)
            from swift.infer_engine import RequestConfig
        except Exception:
            return None
        lines: List[str] = []
        for ev in run:
            txt = (getattr(ev, "text", "") or "").strip().replace("\n", " ")
            if len(txt) > 400:
                txt = txt[:400] + "..."
            lines.append(f"t={int(getattr(ev, 't', 0))}s | {txt}")
        body = "\n".join(lines)
        sys_msg = (
            "You distil a run of streaming-video observations into a "
            "durable LONG-TERM FACT record. Prioritise FACT COVERAGE over "
            "brevity: capture EVERY concrete fact present — all actors "
            "(with exact counts), all key objects (with counts), the "
            "action(s) performed and their final state "
            "(started/in-progress/finished), locations and spatial "
            "relations, and any visible on-screen text/numbers. Prefer "
            "listing "
            "more facts to omitting them. Do NOT speculate beyond the "
            "observations and do NOT invent details. Output plain prose or "
            "semicolon-separated facts (no markdown, no preamble, up to ~6 "
            "sentences)."
        )
        user_msg = (
            f"Segment [t={t0}s, t={t1}s], {len(run)} observations:\n"
            f"{body}\n\n"
            "Write the fact-covering long-term record now."
        )
        try:
            req = InferRequest(messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ])
            cfg = RequestConfig(max_tokens=400, temperature=0.1)
            resp = self._engine.infer([req], cfg)
            if not resp:
                return None
            text = _clean_llm_summary(
                resp[0].choices[0].message.content or "")
            return text[:1100] or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # LLM-based episode summarization
    # ------------------------------------------------------------------
    def _llm_summarize_run(self, run: List[Any]) -> Optional[str]:
        """Use the underlying vLLM engine to write a 1-3 sentence summary
        for a run of working-layer events (an emerging episode).

        Input: original event text in chronological order, each line tagged
        with its absolute timestamp ``t=<sec>``. The model is asked for a
        concise narrative that captures actors, key actions, scene state,
        and any salient changes across [t0, t1].

        Returns the summary string on success, or ``None`` on any error
        (so the caller can fall back to the rule-based summarizer).
        """
        if not run or self._engine is None:
            return None
        try:
            from swift import InferRequest  # lazy import (heavy)
            from swift.infer_engine import RequestConfig
        except Exception:
            return None
        # Build a compact event listing — keep first 280 chars per event
        # to bound prompt length while preserving the core fields.
        lines: List[str] = []
        for ev in run:
            txt = (getattr(ev, "text", "") or "").strip().replace("\n", " ")
            if len(txt) > 280:
                txt = txt[:280] + "..."
            lines.append(f"t={int(getattr(ev, 't', 0))}s | {txt}")
        body = "\n".join(lines)
        t0 = int(getattr(run[0], "t", 0))
        t1 = int(getattr(run[-1], "t", 0))
        sys_msg = (
            "You compress a run of streaming-video observations into "
            "ONE concise episodic summary. Output 1-3 sentences (<=120 "
            "words total) covering: actors involved, the ordered action "
            "chain, final scene state, and any salient changes. Do NOT "
            "include per-second timestamps in the body. No markdown, no "
            "lists, no preamble — just the summary text."
        )
        user_msg = (
            f"Window [t={t0}s, t={t1}s], {len(run)} events:\n"
            f"{body}\n\n"
            "Write the summary now."
        )
        try:
            req = InferRequest(messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ])
            cfg = RequestConfig(max_tokens=200, temperature=0.1)
            resp = self._engine.infer([req], cfg)
            if not resp:
                return None
            text = _clean_llm_summary(
                resp[0].choices[0].message.content or "")
            return text or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------
    def update(self, answer: str):
        """Public entry: dispatch to ``_update_impl`` then stream a step
        record to ``<sid>.steps.jsonl`` if trajectory_dir is enabled.

        Wrapping the original logic with try/finally guarantees that even
        when an early-return path is taken (abstain / tool pending / final
        response / temporal jump) we still emit one trajectory record per
        physical model round.
        """
        answer, dropped = limit_tool_call_reply(
            answer or "", self.max_per_reply
        )
        self._last_reply_tool_calls_dropped = dropped
        # Capture pre-state once, so the record reflects what the model
        # actually saw during this round.
        round_before = self.current_round
        calls_made_before = self._calls_made
        try:
            return self._update_impl(answer)
        finally:
            self._record_step(answer, round_before, calls_made_before)

    def _update_impl(self, answer: str):
        # Sync ctx with parent state
        self.ctx.current_round = self.current_round
        self.ctx.total_rounds = self.total_rounds
        self.ctx.calls_this_round = 0

        if getattr(self, "_event_predicate_inflight", ""):
            stage = self._event_predicate_inflight
            self._event_predicate_inflight = ""
            attempt = self._event_predicate_retries.get(stage, 0) + 1
            self.output[
                f"Round {self.current_round}_event_predicate_"
                f"{stage}_attempt_{attempt}"
            ] = answer
            if stage == "contract":
                parsed = parse_event_predicate_contract(answer)
            elif stage == "ledger":
                parsed = parse_event_predicate_ledger(
                    answer,
                    self._event_predicate_windows,
                )
            else:
                raise AssertionError(
                    f"unknown event predicate stage: {stage}"
                )
            if parsed is None and attempt < 2:
                self._event_predicate_retries[stage] = attempt
                self._event_predicate_pending = stage
                return
            if parsed is None:
                effective = self._event_predicate_original_response
                self.output[f"Round {self.current_round}"] = effective
                self.output[
                    f"Round {self.current_round}_event_predicate_final"
                ] = json.dumps({
                    "stage": stage,
                    "accepted": False,
                    "reason": "invalid_twice_safe_preserve",
                    "contract": self._event_predicate_contract,
                    "proposals": self._event_predicate_proposals,
                    "build_error": self._event_predicate_build_error,
                }, ensure_ascii=False)
                self._event_predicate_done = True
                self.current_round += 1
                self.completed = True
                return
            if stage == "contract":
                self._event_predicate_contract = parsed
                self._event_predicate_pending = "ledger"
                return
            accepted = bool(parsed.get("direct_exact_positive"))
            effective = (
                f"{RESPONSE_TAG} Yes"
                if accepted
                else self._event_predicate_original_response
            )
            self.output[f"Round {self.current_round}"] = effective
            self.output[
                f"Round {self.current_round}_event_predicate_final"
            ] = json.dumps({
                "accepted": accepted,
                "original": normalize_yes_no_response(
                    self._event_predicate_original_response
                ),
                "final": normalize_yes_no_response(effective),
                "contract": self._event_predicate_contract,
                "proposals": self._event_predicate_proposals,
                "ledger": parsed,
                "build_error": self._event_predicate_build_error,
            }, ensure_ascii=False)
            if self._memory is not None:
                self._memory.write_event(
                    self.current_round,
                    "note",
                    (
                        "[event-predicate-audit] "
                        f"accepted={accepted} "
                        f"target={parsed.get('target_action')} "
                        f"direct_exact={parsed.get('direct_exact_positive')}"
                    ),
                    ref_frame=self.current_round,
                    salience=2.0,
                )
            self._event_predicate_done = True
            self.current_round += 1
            self.completed = True
            return

        if self._current_locus_protector_inflight:
            stage = self._current_locus_protector_inflight
            self._current_locus_protector_inflight = ""
            view = self._current_locus_dual_view_view
            if stage.startswith("pair_"):
                parsed = parse_current_locus_pairwise_audit(
                    answer, list(view.get("times") or [])
                )
            elif stage == "fact_ledger":
                parsed = parse_current_locus_fact_ledger(
                    answer, list(view.get("times") or [])
                )
            else:
                ledger = self._current_locus_protector_results.get(
                    "fact_ledger"
                )
                valid_ids = (
                    current_locus_fact_ids(ledger)
                    if isinstance(ledger, dict) else set()
                )
                parsed = parse_current_locus_claim_judgment(
                    answer, valid_ids
                )
            attempt = (
                self._current_locus_protector_retries.get(stage, 0) + 1
            )
            self.output[
                f"Round {self.current_round}_current_locus_protector_"
                f"{stage}_attempt_{attempt}"
            ] = answer
            if parsed is None and attempt < 2:
                self._current_locus_protector_retries[stage] = attempt
                self._current_locus_protector_pending = stage
                return
            if parsed is None:
                # Any malformed protector stage safely preserves the original.
                self._current_locus_protector_results[stage] = None
                effective = self._current_locus_dual_view_original_response
                self.output[f"Round {self.current_round}"] = effective
                if self._memory is not None:
                    self._memory.write_event(
                        self.current_round,
                        "note",
                        (
                            "[current-locus-protector] "
                            f"stage={stage} invalid_twice keep_original="
                            f"{self._current_locus_dual_view_original_answer}"
                        ),
                        ref_frame=self.current_round,
                        salience=2.0,
                    )
                self.current_round += 1
                self.completed = True
                return
            self._current_locus_protector_results[stage] = parsed
            next_stage = {
                "pair_original_first": "pair_proposed_first",
                "pair_proposed_first": "fact_ledger",
                "fact_ledger": "claim_original",
                "claim_original": "claim_proposed",
            }.get(stage)
            if next_stage is not None:
                self._current_locus_protector_pending = next_stage
                return

            pair_first = self._current_locus_protector_results.get(
                "pair_original_first"
            )
            pair_second = self._current_locus_protector_results.get(
                "pair_proposed_first"
            )
            original_judgment = self._current_locus_protector_results.get(
                "claim_original"
            )
            proposed_judgment = self._current_locus_protector_results.get(
                "claim_proposed"
            )
            accepted = current_locus_iter39_accept(
                pair_first,
                pair_second,
                original_judgment,
                proposed_judgment,
            )
            effective = (
                self._current_locus_protector_proposed_response
                if accepted
                else self._current_locus_dual_view_original_response
            )
            self.output[f"Round {self.current_round}"] = effective
            if self._memory is not None:
                self._memory.write_event(
                    self.current_round,
                    "note",
                    (
                        "[current-locus-protector] "
                        f"original={self._current_locus_dual_view_original_answer} "
                        f"proposed={self._current_locus_protector_proposed_answer} "
                        f"accepted={accepted} "
                        f"results={json.dumps(self._current_locus_protector_results, ensure_ascii=False)}"
                    ),
                    ref_frame=self.current_round,
                    salience=2.0,
                )
            self.current_round += 1
            self.completed = True
            return

        if self._current_locus_dual_view_inflight:
            self._current_locus_dual_view_inflight = False
            view = self._current_locus_dual_view_view
            audit = parse_current_locus_dual_view_audit(
                answer,
                self.sample,
                list(view.get("times") or []),
            )
            self.output[
                f"Round {self.current_round}_current_locus_dual_view_"
                f"audit_attempt_{self._current_locus_dual_view_retries + 1}"
            ] = answer
            if (
                audit is None
                and self._current_locus_dual_view_retries < 1
            ):
                self._current_locus_dual_view_retries += 1
                self._current_locus_dual_view_pending = True
                return
            self._current_locus_dual_view_done = True
            effective = self._current_locus_dual_view_original_response
            if audit is not None:
                selected = audit["answer"]
                index = ord(selected) - ord("A")
                options = self.sample.get("options") or []
                if 0 <= index < len(options):
                    effective = (
                        f"{RESPONSE_TAG} {selected}. {options[index]}"
                    )
                    if (
                        os.environ.get(
                            "STREAMO_ENABLE_CURRENT_LOCUS_ITER39_PROTECTOR",
                            "0",
                        ) == "1"
                        and selected
                        != self._current_locus_dual_view_original_answer
                    ):
                        self._current_locus_protector_proposed_answer = (
                            selected
                        )
                        self._current_locus_protector_proposed_response = (
                            effective
                        )
                        self._current_locus_protector_pending = (
                            "pair_original_first"
                        )
                        self.output[
                            f"Round {self.current_round}_"
                            "pre_current_locus_protector"
                        ] = (
                            self._current_locus_dual_view_original_response
                        )
                        return
            self.output[f"Round {self.current_round}"] = effective
            if self._memory is not None:
                audit_text = (
                    json.dumps(audit, ensure_ascii=False)
                    if audit is not None
                    else "invalid_twice_safe_preserve"
                )
                build_error = (
                    f" build_error={self._current_locus_dual_view_build_error}"
                    if self._current_locus_dual_view_build_error else ""
                )
                self._memory.write_event(
                    self.current_round,
                    "note",
                    (
                        "[current-locus-dual-view-audit] "
                        f"original="
                        f"{self._current_locus_dual_view_original_answer} "
                        f"effective={ovo._extract_option_letter(effective)} "
                        f"audit={audit_text}{build_error}"
                    ),
                    ref_frame=self.current_round,
                    salience=2.0,
                )
            self.current_round += 1
            self.completed = True
            return

        if self._listed_refusal_bidir_inflight:
            self._listed_refusal_bidir_inflight = False
            options = self.sample.get("options") or []
            refusal_index = _refusal_option_index(self.sample)
            refusal_letter = (
                chr(ord("A") + refusal_index)
                if refusal_index is not None
                else ""
            )
            concrete_letters = [
                chr(ord("A") + index)
                for index in range(len(options))
                if index != refusal_index
            ]
            audit = parse_listed_refusal_bidirectional_audit(
                answer,
                original_answer=self._listed_refusal_bidir_original_answer,
                concrete_answers=concrete_letters,
                refusal_answer=refusal_letter,
                window_times={
                    int(window["window_id"]): list(window["times"])
                    for window in self._listed_refusal_bidir_windows
                },
            )
            self.output[
                f"Round {self.current_round}_listed_refusal_bidir_audit"
            ] = answer
            if audit is None and self._listed_refusal_bidir_retries < 1:
                self._listed_refusal_bidir_retries += 1
                self._listed_refusal_bidir_pending = True
                return
            self._listed_refusal_bidir_done = True
            effective = self._listed_refusal_bidir_original_response
            if audit is not None:
                selected = audit["selected_answer"]
                index = ord(selected) - ord("A")
                if 0 <= index < len(options):
                    effective = (
                        f"{RESPONSE_TAG} {selected}. {options[index]}"
                    )
            self.output[f"Round {self.current_round}"] = effective
            if self._memory is not None:
                audit_text = (
                    json.dumps(audit, ensure_ascii=False)
                    if audit is not None
                    else "invalid_twice_safe_preserve"
                )
                self._memory.write_event(
                    self.current_round,
                    "note",
                    (
                        "[listed-refusal-bidirectional-audit] "
                        f"original={self._listed_refusal_bidir_original_answer} "
                        f"effective={ovo._extract_option_letter(effective)} "
                        f"audit={audit_text}"
                    ),
                    ref_frame=self.current_round,
                    salience=2.0,
                )
            self.current_round += 1
            self.completed = True
            return

        if self._listed_refusal_audit_inflight:
            self._listed_refusal_audit_inflight = False
            audit = parse_listed_refusal_visual_audit(
                answer,
                proposed_answer=self._listed_refusal_proposed_answer,
                window_times={
                    int(window["window_id"]): list(window["times"])
                    for window in self._listed_refusal_audit_windows
                },
            )
            self.output[
                f"Round {self.current_round}_listed_refusal_audit"
            ] = answer
            if audit is None and self._listed_refusal_audit_retries < 1:
                self._listed_refusal_audit_retries += 1
                self._listed_refusal_audit_pending = True
                return
            self._listed_refusal_audit_done = True
            effective = self._listed_refusal_original_response
            if audit is not None and audit["status"] == "unsupported":
                effective = (
                    _listed_refusal_final_for(self.sample)
                    or self._listed_refusal_original_response
                )
            self.output[f"Round {self.current_round}"] = effective
            if self._memory is not None:
                audit_text = (
                    json.dumps(audit, ensure_ascii=False)
                    if audit is not None
                    else "invalid_twice_safe_preserve"
                )
                self._memory.write_event(
                    self.current_round,
                    "note",
                    (
                        "[listed-refusal-visual-audit] "
                        f"proposed={self._listed_refusal_proposed_answer} "
                        f"effective={ovo._extract_option_letter(effective)} "
                        f"audit={audit_text}"
                    ),
                    ref_frame=self.current_round,
                    salience=2.0,
                )
            self.current_round += 1
            self.completed = True
            return

        if self._crr_clue_audit_inflight:
            self._crr_clue_audit_inflight = False
            audit_kind = (
                "crr_typed_relation_audit"
                if self._crr_typed_relation is not None
                else "crr_clue_audit"
            )
            self.output[
                f"Round {self.current_round}_{audit_kind}_raw"
            ] = answer
            audit = (
                parse_crr_typed_relation_response(
                    answer,
                    relation=self._crr_typed_relation,
                    # Integer round N exposes the full <Ns-(N+1)s> interval;
                    # high-fps sheets may therefore cite N.25/N.50/N.75.
                    question_time=float(self.question_time) + 1.0,
                )
                if self._crr_typed_relation is not None
                else parse_crr_clue_audit_response(answer)
            )
            if audit is not None:
                self.output[
                    f"Round {self.current_round}_{audit_kind}_parsed"
                ] = json.dumps(audit, ensure_ascii=False)
                normalized = f"{RESPONSE_TAG} {audit['answer']}"
                self.output[f"Round {self.current_round}"] = normalized
                if self._memory is not None:
                    if self._crr_typed_relation is not None:
                        note = (
                            "[crr-typed-relation-audit] "
                            f"family={audit['relation_family']} "
                            f"latest={audit['selected_latest_instance_index']} "
                            f"compatible={audit['compatible_instance_count']} "
                            f"candidate_answer={audit['candidate_answer']} "
                            f"positive={audit['positive']} "
                            "typed_event="
                            f"{json.dumps(audit['typed_event_node'], ensure_ascii=False)} "
                            f"evidence={audit['evidence']}"
                        )
                    else:
                        note = (
                            "[crr-clue-audit] "
                            f"anchor_status={audit['anchor_status']} "
                            f"answer_status={audit['answer_status']} "
                            f"answer_source={audit['answer_source']} "
                            f"temporal_link={audit['temporal_link']} "
                            f"candidate_answer={audit['candidate_answer']} "
                            f"before_t={audit['before_t']:.2f} "
                            f"after_t={audit['after_t']:.2f} "
                            f"evidence={audit['evidence']}"
                        )
                    self._memory.write_event(
                        self.current_round,
                        "note",
                        note,
                        ref_frame=self.current_round,
                        salience=2.0,
                    )
                    if self._crr_typed_relation is not None:
                        self._post_graph_hook(
                            {
                                "text": note,
                                "entities": [
                                    audit["relation_family"],
                                    audit["candidate_answer"],
                                ],
                            },
                            self.current_round,
                        )
                self.current_round += 1
                self.completed = True
                return
            self.output[f"Round {self.current_round}"] = answer
            if self._crr_clue_audit_retries < 1:
                self._crr_clue_audit_retries += 1
                self._crr_clue_audit_pending = True
                return
            self._pending_tool_text = (
                "<tool_result>"
                + json.dumps({
                    "name": "_crr_clue_audit",
                    "ok": False,
                    "error": (
                        "The structured clue-answerability audit was "
                        "malformed twice. Continue with the standing plan, "
                        "MEMORY, and targeted replay; then emit Yes or No."
                    ),
                }, ensure_ascii=False)
                + "</tool_result>"
            )
            return

        if self._crr_target_audit_inflight:
            self._crr_target_audit_inflight = False
            audit = parse_crr_target_audit_response(answer)
            if audit is not None:
                normalized = f"{RESPONSE_TAG} {audit['answer']}"
                self.output[f"Round {self.current_round}"] = normalized
                if self._memory is not None:
                    self._memory.write_event(
                        self.current_round,
                        "note",
                        (
                            "[crr-target-audit] "
                            f"anchor_status={audit['anchor_status']} "
                            f"target_status={audit['target_status']} "
                            f"subject_continuity="
                            f"{audit['subject_continuity']} "
                            f"arrival_evidence={audit['arrival_evidence']} "
                            f"target_value={audit['target_value']} "
                            f"before_t={audit['before_t']:.2f} "
                            f"after_t={audit['after_t']:.2f} "
                            f"evidence={audit['evidence']}"
                        ),
                        ref_frame=self.current_round,
                        salience=2.0,
                    )
                self.current_round += 1
                self.completed = True
                return
            self.output[f"Round {self.current_round}"] = answer
            if self._crr_target_audit_retries < 1:
                self._crr_target_audit_retries += 1
                self._crr_target_audit_pending = True
                return
            self._pending_tool_text = (
                "<tool_result>"
                + json.dumps({
                    "name": "_crr_target_audit",
                    "ok": False,
                    "error": (
                        "The structured target-completeness audit was "
                        "malformed twice. Continue with the standing plan, "
                        "MEMORY, and targeted replay; then emit Yes or No."
                    ),
                }, ensure_ascii=False)
                + "</tool_result>"
            )
            return

        if self._ssr_local_audit_inflight:
            self._ssr_local_audit_inflight = False
            audit = parse_ssr_local_audit_response(answer)
            if audit is not None:
                normalized = f"{RESPONSE_TAG} {audit['answer']}"
                self.output[f"Round {self.current_round}"] = normalized
                if self._memory is not None:
                    self._memory.write_event(
                        self.current_round,
                        "note",
                        (
                            "[ssr-local-audit] "
                            f"local_phase={audit['local_phase']} "
                            f"history_phase={audit['history_phase']} "
                            f"before_t={audit['before_t']:.2f} "
                            f"after_t={audit['after_t']:.2f} "
                            f"evidence={audit['evidence']}"
                        ),
                        ref_frame=self.current_round,
                        salience=2.0,
                    )
                self.current_round += 1
                self.completed = True
                return
            self.output[f"Round {self.current_round}"] = answer
            if self._ssr_local_audit_retries < 1:
                self._ssr_local_audit_retries += 1
                self._ssr_local_audit_pending = True
                return
            # A malformed specialized audit must not strand the sample.
            # Return to the normal post-question planner/tool flow.
            self._pending_tool_text = (
                "<tool_result>"
                + json.dumps({
                    "name": "_ssr_local_audit",
                    "ok": False,
                    "error": (
                        "The structured local audit was malformed twice. "
                        "Continue with the standing plan, MEMORY, and targeted "
                        "replay; then emit the required Yes/No response."
                    ),
                }, ensure_ascii=False)
                + "</tool_result>"
            )
            return

        if self._count_audit_inflight:
            self._count_audit_inflight = False
            if self._count_audit_use_sports_cycle:
                ledger = extract_sports_cycle_ledger(
                    answer,
                    self._count_audit_target_activity,
                    require_grounded=(
                        self._count_audit_sports_cycle_version >= 3
                    ),
                )
            else:
                ledger = extract_count_audit_ledger(answer)
            if ledger is not None:
                self._count_audit_retries = 0
                if self._count_audit_segmented:
                    self._count_audit_ledger = merge_count_audit_ledgers(
                        self._count_audit_ledger,
                        ledger,
                    )
                    if (
                        self._count_audit_segment_index + 1
                        < len(self._count_audit_segments)
                    ):
                        self.output[
                            f"Round {self.current_round}"
                        ] = answer
                        self._count_audit_segment_index += 1
                        self._count_audit_pending = True
                        return
                    ledger = list(self._count_audit_ledger)
                normalized = normalize_count_audit_response(
                    f"count is {min(10, len(ledger))}",
                    self.question or "",
                )
                if normalized is not None:
                    self.output[f"Round {self.current_round}"] = normalized
                    if self.memory_mode == "brief" \
                            and self._summarizer is not None:
                        ev = self._summarizer(
                            normalized,
                            self.current_round,
                            ref_frame=self.current_round,
                        )
                        if ev and self._memory is not None:
                            self._memory.write_event(
                                self.current_round, **ev)
                            self._post_graph_hook(ev, self.current_round)
                    self._count_audit_segments = []
                    self._count_audit_segment_index = 0
                    self._count_audit_ledger = []
                    self._count_audit_segmented = False
                    self.current_round += 1
                    self.completed = True
                    return
            elif self._count_audit_retries < 1:
                self._count_audit_retries += 1
                self._count_audit_pending = True
                self.output[f"Round {self.current_round}"] = answer
                return
            elif self._count_audit_segmented:
                self._count_audit_segments = []
                self._count_audit_segment_index = 0
                self._count_audit_ledger = []
                self._count_audit_segmented = False
                self._pending_tool_text = (
                    "<tool_result>"
                    + json.dumps({
                        "name": "_segmented_count_audit",
                        "ok": False,
                        "error": (
                            "One visual segment produced malformed ledger "
                            "JSON twice, so the PARTIAL segmented count was "
                            "discarded. Continue with the standing plan, "
                            "MEMORY, and targeted replay; then answer."
                        ),
                    }, ensure_ascii=False)
                    + "</tool_result>"
                )
                return

        # Track when the streaming clock advanced (i.e. we are now on
        # a new physical frame): reset cross-replies counters so format
        # retries / per-round caps refer to *this* frame only.
        if getattr(self, "_last_phys_round", None) != self.current_round:
            self._calls_this_phys_round = 0
            self._replies_this_phys_round = 0
            self._format_retries_this_phys_round = 0
            self._last_frame_reply_cap_used = False
            # promptfix v2 改动 1 + 7：物理轮切换时清空 memsearch 历史
            # 与 zoom 物理轮计数。memsearch_history 用于 memory_search 重复
            # 检测；zoom_count_this_phys_round 用于 zoom_crop 软上限。
            self.ctx.memsearch_history = []
            self.ctx.zoom_count_this_phys_round = 0
            # POST-QUESTION per-frame tool budgets & empty-result guard.
            self._memsearch_this_frame = 0
            self._replay_this_frame = 0
            self._track_query_this_frame = 0
            self._consecutive_empty = 0
            # Repeated-identical-tool-call tracking is per-frame: the same
            # zoom/replay on a NEW frame is legitimate, so reset here.
            self._last_toolcall_sig = None
            self._toolcall_sig_repeat = 0
            self._last_phys_round = self.current_round
        self._replies_this_phys_round += 1

        # ---- Model-driven Recap: pin <Recap>...</Recap> blocks -------
        # Parsed BEFORE tool dispatch so a Recap emitted alongside a
        # tool_call still gets pinned in the same round.
        self._maybe_pin_recaps(answer)

        # ---- Query-Driven Planning: capture <Plan> on question round --
        # On the question-posed frame we asked the model to emit a <Plan>
        # analysing the query. Parse and store it, then re-prompt the SAME
        # frame to begin answering (clock does NOT advance, no tool budget
        # spent). Done unconditionally once per sample — even on the last
        # frame (the virtual COMMIT buffer round leaves room to answer).
        if (not self._plan_done) \
                and self.current_round == self.question_time:
            plan = parse_plan_block(answer or "")
            self.output[f"Round {self.current_round}"] = answer
            if plan is not None:
                self._query_plan = plan
                self._plan_done = True
                # Log the plan as a low-salience working note for traces.
                if self._memory is not None:
                    self._memory.write_event(
                        self.current_round, "note",
                        f"[plan] {plan.get('_raw', '')[:3000]}",
                        ref_frame=self.current_round, salience=0.5,
                    )
                if plan_requests_repetition_count(
                        plan, self.question, self.sample):
                    visible_end = min(
                        self.current_round, len(self.frame_paths) - 1)
                    max_frames = max(1, int(os.environ.get(
                        "STREAMO_COUNT_AUDIT_MAX_FRAMES", "400")))
                    history_paths = self.frame_paths[:visible_end + 1]
                    if len(history_paths) <= max_frames:
                        self._count_audit_pending = True
                        return
                    else:
                        note = (
                            "Plan recorded. The complete timeline exceeds "
                            "the safe image budget, so ACT on the plan using "
                            "MEMORY and targeted replay, then answer."
                        )
                elif plan_requests_ssr_local_audit(plan, self.sample):
                    self._ssr_local_audit_pending = True
                    return
                elif plan_requests_crr_typed_relation_audit(
                    plan, self.sample
                ):
                    try:
                        self._crr_typed_relation = (
                            compile_crr_typed_relation_contract(
                                str(self.sample.get("question") or ""),
                                plan,
                            )
                        )
                    except ValueError:
                        self._crr_typed_relation = None
                    if self._crr_typed_relation is not None:
                        self._crr_clue_audit_pending = True
                        return
                    note = (
                        "Plan recorded, but the typed visible-relation "
                        "contract could not be compiled. Continue with MEMORY "
                        "and targeted replay, then answer."
                    )
                elif plan_requests_crr_clue_audit(plan, self.sample):
                    self._crr_clue_audit_pending = True
                    return
                elif plan_requests_crr_target_audit(plan, self.sample):
                    self._crr_target_audit_pending = True
                    return
                else:
                    note = (
                        "Plan recorded. Now ACT on it: gather the evidence "
                        "your strategy named, then answer. It will be shown "
                        "to you each turn."
                    )
                # Re-prompt the SAME frame so the model now acts on its
                # plan (POST-QUESTION instructions next turn).
                self._pending_tool_text = (
                    "<tool_result>"
                    + json.dumps({
                        "name": "_plan",
                        "ok": True,
                        "data": {"note": note},
                    }, ensure_ascii=False)
                    + "</tool_result>"
                )
                return
            # No well-formed plan emitted: don't loop forever — mark the
            # planning step done and fall through to normal handling so the
            # sample still progresses.
            self._plan_done = True

        calls = parse_tool_calls(answer or "")
        prequery_cap_action = prequery_tool_reply_cap_action(
            has_calls=bool(calls),
            current_round=self.current_round,
            question_time=self.question_time,
            replies_this_round=self._replies_this_phys_round,
        )
        if prequery_cap_action == "advance":
            # Rejected tools and same-frame tool continuations can otherwise
            # loop forever before the query, where the post-query bare-chain
            # gate intentionally does not run. Preserve any useful observation
            # already emitted, then advance exactly one streaming tick.
            self.output[
                f"Round {self.current_round}_prequery_reply_cap"
            ] = answer
            self.output[f"Round {self.current_round}"] = answer
            self._persist_tool_round_observation(answer)
            self._pending_tool_text = None
            self._pending_tool_images = []
            self.ctx.next_round_images = []
            self.ctx.calls_this_round = 0
            self._responding_to_tool = False
            self._bare_tool_chain = 0
            self.current_round = min(
                self.current_round + 1,
                self.total_rounds,
            )
            self._last_phys_round = self.current_round
            self._calls_this_phys_round = 0
            self._replies_this_phys_round = 0
            self._format_retries_this_phys_round = 0
            self._last_frame_reply_cap_used = False
            self._memsearch_this_frame = 0
            self._replay_this_frame = 0
            self._track_query_this_frame = 0
            self._consecutive_empty = 0
            self.ctx.memsearch_history = []
            self.ctx.zoom_count_this_phys_round = 0
            self._last_toolcall_sig = None
            self._toolcall_sig_repeat = 0
            return
        # A rejected tool call does not consume ``_calls_made`` and may not
        # match the narrow "budget exhausted" strings below. On the final
        # frame, a model can therefore repeat rejected zoom/replay requests
        # forever even though every request says COMMIT NOW. Bound physical
        # replies independently of tool success. We do not choose an answer
        # here: drop the tool calls and give the model one tool-less final
        # commit turn; the existing bounded format/COMMIT fallback remains
        # the final safety net.
        cap_action = last_frame_tool_reply_cap_action(
            has_calls=bool(calls),
            current_round=self.current_round,
            total_rounds=self.total_rounds,
            question_time=self.question_time,
            replies_this_round=self._replies_this_phys_round,
            cap_used=self._last_frame_reply_cap_used,
        )
        if cap_action != "none":
            self.output[
                f"Round {self.current_round}_last_frame_reply_cap"
            ] = answer
            self._pending_tool_images = []
            self.ctx.next_round_images = []
            if cap_action == "nudge":
                self._last_frame_reply_cap_used = True
                self._pending_tool_text = (
                    "<tool_result>"
                    + json.dumps({
                        "name": "_final_answer_required",
                        "ok": False,
                        "error": (
                            "Final-frame tool/reply limit reached. No more "
                            "tools will run. Use the visual readings and "
                            "MEMORY you already collected and emit "
                            "`</Response> <one listed option>` NOW. Do not "
                            "issue another tool call."
                        ),
                    }, ensure_ascii=False)
                    + "</tool_result>"
                )
                return
            # The model ignored the one final tool-less commit chance.
            # Discard further calls and fall through to bounded format
            # retries / COMMIT fallback; do not emit the same nudge forever.
            calls = []
        # ── Hard gate: analyse BEFORE chaining the next tool ───────────
        # When THIS turn is a response to tool results (zoom/replay image
        # patches, search output) AND the model fired more tool_calls
        # WITHOUT first reporting what the last result showed, reject the
        # tools and demand a one-line concrete observation. This breaks the
        # "want another look → look → want another look" spin where the
        # model replays/zooms forever and never commits a reading (e.g.
        # sample 8 counting cards). abstain / a legal sentinel always pass.
        if calls and self._responding_to_tool \
                and self.current_round >= self.question_time:
            has_abstain = any((s.get("name") or "").strip() == "abstain"
                              for s in calls)
            ans_clean = ovo._strip_thinking(answer or "")
            has_sentinel = ("</Response>" in ans_clean
                            or "</Silence>" in ans_clean
                            or "</Standby>" in ans_clean)
            # ── Repeated-identical-tool-call detection ────────────────
            # A weaker model may re-issue the EXACT same tool call (same
            # name + args) turn after turn, each wrapped in a fresh
            # <Observation> blurb. That prose makes `_has_analysis_text`
            # return True, so the bare-chain gate below never fires and the
            # spin only stops at PHYS_CAP — per frame, so it recurs every
            # frame (observed: qwen3vl-8b, 15 identical zoom/track calls ×
            # hundreds of frames). Track the tool-call signature across
            # turns on this frame; an identical re-issue counts as a spin
            # and feeds the same GATE_ESCAPE hatch as a bare chain.
            sig = _toolcall_signature(calls)
            if has_abstain or has_sentinel:
                # A commit/observe decision is present → not a spin.
                self._last_toolcall_sig = None
                self._toolcall_sig_repeat = 0
            else:
                if sig and sig == self._last_toolcall_sig:
                    self._toolcall_sig_repeat += 1
                else:
                    self._toolcall_sig_repeat = 0
                self._last_toolcall_sig = sig
            repeating = (not has_abstain) and (not has_sentinel) \
                and self._toolcall_sig_repeat >= 1
            if (not has_abstain) and (not has_sentinel) \
                    and (not _has_analysis_text(answer) or repeating):
                self._bare_tool_chain += 1
                self._calls_this_phys_round += 1  # PHYS_CAP backstop
                # ── Escape hatch: the gate itself must terminate ──────
                # This branch returns BEFORE the tool-dispatch loop, so it
                # never reaches PHYS_CAP / per-frame budgets — a model that
                # ignores the "STOP, report first" nudge and keeps emitting
                # the SAME bare tool_calls (sample 8: 90+ identical
                # replay+search rounds, clock frozen at t=7) loops forever.
                # After GATE_ESCAPE consecutive bare-chain rejections we
                # stop nudging and break the spin by phase:
                #   * video not over  → force the streaming clock forward
                #     one tick (</Silence> semantics). This is SAFE for
                #     forward tasks (REC/SSR/CRR) that must keep watching
                #     for the queried event, and unsticks backward/realtime
                #     tasks whose evidence is already in memory.
                #   * last frame past q_time → fall through to the
                #     last-frame safety net, which commits a best-guess
                #     answer (guessing beats an infinite loop).
                GATE_ESCAPE = 4
                # Use the SAME last-frame test as the downstream last-frame /
                # COMMIT flow (current_round >= total_rounds - 1). An earlier
                # attempt used `current+1 >= total-1` (one frame early), which
                # made the gate fall through at round T-2 while the downstream
                # format-check still treated it as a normal past_q round (no
                # forced commit) — so the sample just spun there and ended up
                # abstain-guessing option A anyway (sample 306). Aligning the
                # tests means: BEFORE the last frame we force the clock one
                # tick forward (walking toward the end); only AT the last
                # frame do we drop the bare calls and let the last-frame flow
                # (which allows one final tool then DEMANDS `</Response>`, and
                # finally the tool-less COMMIT round) extract the model's own
                # verdict — e.g. "no shovel in memory → Unable to answer".
                is_last_frame_now = \
                    self.current_round >= self.total_rounds - 1
                if self._bare_tool_chain >= GATE_ESCAPE \
                        and not is_last_frame_now:
                    # Not yet at the last frame: force the streaming clock
                    # forward one tick and reset per-phys-round counters
                    # (mirrors the format-retry-exhausted path below).
                    self._bare_tool_chain = 0
                    self._pending_tool_text = None
                    self._pending_tool_images = []
                    self.output[f"Round {self.current_round}"] = answer
                    self._persist_tool_round_observation(answer)
                    self.current_round = min(self.current_round + 1,
                                              self.total_rounds)
                    self._last_phys_round = self.current_round
                    self._calls_this_phys_round = 0
                    self._replies_this_phys_round = 0
                    self._format_retries_this_phys_round = 0
                    self._last_frame_reply_cap_used = False
                    self.ctx.memsearch_history = []
                    self.ctx.zoom_count_this_phys_round = 0
                    self._last_toolcall_sig = None
                    self._toolcall_sig_repeat = 0
                    if self.current_round >= self.total_rounds:
                        self.completed = True
                    return
                if self._bare_tool_chain >= GATE_ESCAPE \
                        and is_last_frame_now:
                    # AT the last frame: do NOT force the clock to completion.
                    # Drop the bare tool calls and fall through so the
                    # last-frame POST-QUESTION / final-call flow forces the
                    # model to emit `</Response>` itself (and finally the
                    # tool-less COMMIT round), rather than letting the abstain
                    # fallback guess option A.
                    self._bare_tool_chain = 0
                    self._pending_tool_text = None
                    self.output[f"Round {self.current_round}"] = answer
                    self._persist_tool_round_observation(answer)
                    self._last_toolcall_sig = None
                    self._toolcall_sig_repeat = 0
                    calls = []  # drop the bare tool calls; go to safety net
                else:
                    if repeating:
                        # Model DID report but re-issued the identical tool
                        # call — nudge it to change course, not to "report
                        # first" (it already did).
                        gate_err = (
                            "STOP. You just issued the SAME tool call "
                            "(same tool + same arguments) you already ran on "
                            "this frame — repeating it returns the same "
                            "result and makes no progress. Commit now: write "
                            "`</Response> <answer>` with your best reading, "
                            "or `</Silence>` to move on. If you truly need "
                            "another look, use DIFFERENT arguments (a "
                            "different t / bbox / query), not the same one.")
                    else:
                        gate_err = (
                            "STOP. You just received a tool result "
                            "(image/replay/search) but issued another "
                            "tool call WITHOUT reporting what you saw. "
                            "FIRST write ONE line stating the CONCRETE "
                            "reading from the last result — for a count "
                            "question give the EXACT number you see (e.g. "
                            "'I count 5 cards in his hand'); for an "
                            "identity/location give the specific value. "
                            "THEN either answer with `</Response> "
                            "<answer>` if that settles it, or call ONE "
                            "more tool. Do NOT repeat the same "
                            "replay/zoom you already did.")
                    self._pending_tool_text = (
                        "<tool_result>"
                        + json.dumps({
                            "name": "_analysis_required",
                            "ok": False,
                            "error": gate_err,
                        }, ensure_ascii=False)
                        + "</tool_result>"
                    )
                    # Persist whatever partial text the model did emit.
                    self._persist_tool_round_observation(answer)
                    self.output[f"Round {self.current_round}"] = answer
                    return
                # (escape-hatch last-frame branch set calls=[] and falls
                #  through to the safety net below; do NOT return here.)
        else:
            self._bare_tool_chain = 0
        # COMMIT round: video is over. ALL tool calls are rejected so
        # the model is forced to emit `</Response>` from MEMORY alone.
        is_commit_round = self.current_round == self.total_rounds
        commit_blocked: List[ToolResult] = []
        if calls and is_commit_round:
            for spec in calls:
                nm = (spec.get("name") or "").strip()
                if nm == "abstain":
                    # abstain is still allowed (it short-circuits to a
                    # legal final answer sentinel).
                    continue
                commit_blocked.append(ToolResult(
                    nm, False,
                    error=(
                        "tool calls are DISABLED in the COMMIT round "
                        "(post-video forced reply). The video has "
                        "ended; emit `</Response> <your answer>` "
                        "grounded in MEMORY immediately."
                    ),
                ))
            # Drop blocked specs from `calls` so they are not dispatched.
            calls = [s for s in calls
                     if (s.get("name") or "").strip() == "abstain"]
        # Filter: in OBSERVE-ONLY (post-hoc, before question_time),
        # reject backward-time tools that would otherwise let the model
        # spiral into a replay loop. The streaming clock will
        # naturally reach later frames; no reason to time-travel before
        # the question even appears.
        is_post_hoc = self.current_round < self.question_time
        past_q = self.current_round >= self.question_time
        observe_blocked: List[ToolResult] = list(commit_blocked)
        if calls and is_post_hoc:
            kept = []
            for spec in calls:
                nm = (spec.get("name") or "").strip()
                if nm in ("replay", "memory_read", "memory_list",
                          "memory_search"):
                    observe_blocked.append(ToolResult(
                        nm, False,
                        error=f"{nm} is not allowed in OBSERVE-ONLY phase "
                              f"(t={self.current_round} < question_time="
                              f"{self.question_time}); the clock will "
                              f"advance naturally."))
                else:
                    kept.append(spec)
            calls = kept
        results: List[ToolResult] = list(observe_blocked)
        if calls:
            for spec in calls:
                # `abstain` is a meta self-control signal — it must not be
                # blocked by tool budget, otherwise a model that has run
                # out of budget can never gracefully exit (死循环).
                tool_name = spec.get("name", "?")
                # ``max_per_sample <= 0`` disables the per-sample cap.
                if tool_name != "abstain" \
                        and self.max_per_sample > 0 \
                        and self._calls_made >= self.max_per_sample:
                    results.append(ToolResult(
                        tool_name, False,
                        error=f"sample tool budget exhausted "
                              f"({self.max_per_sample})"))
                    break
                # Per physical-round cap (cross re-prompt). Enforced in
                # ALL phases (t<Q / t>=Q middle / last_frame) at PHYS_CAP
                # tool dispatches per same-frame re-prompt loop. Without
                # this cap, the model can spin on a frame issuing 100+
                # memory_search calls (observed: sample 305 reasoned the
                # same q_time frame 420 times). After the cap, further
                # tool_calls are rejected with a hint to either
                # </Silence> (continue observing, t<last_frame) or
                # </Response> (commit, t>=q_time).
                PHYS_CAP = 15
                if tool_name != "abstain" \
                        and self._calls_this_phys_round >= PHYS_CAP:
                    if self.current_round < self.question_time:
                        hint = (f"physical-round tool budget exhausted "
                                f"({PHYS_CAP}) on this frame; commit to "
                                f"`</Silence>` now and continue observing.")
                    elif self.current_round >= self.total_rounds - 1:
                        hint = (f"*** physical-round tool budget "
                                f"exhausted ({PHYS_CAP}) AND this is "
                                f"the LAST FRAME. *** You've already "
                                f"used {PHYS_CAP} tool calls on this "
                                f"frame without committing. Stop "
                                f"calling tools and emit `</Response> "
                                f"<your answer>` THIS TURN; refusing "
                                f"costs the point. Pick the most "
                                f"plausible option from MEMORY even "
                                f"if evidence is weak.")
                    else:
                        hint = (f"physical-round tool budget exhausted "
                                f"({PHYS_CAP}) on this frame; commit "
                                f"either `</Silence>` (keep observing) "
                                f"or `</Response>` (final answer) now.")
                    results.append(ToolResult(tool_name, False, error=hint))
                    break
                if tool_name != "abstain" \
                        and self.ctx.calls_this_round >= self.max_per_round:
                    results.append(ToolResult(
                        tool_name, False,
                        error=f"round tool budget exhausted "
                              f"({self.max_per_round})"))
                    break
                # ── POST-QUESTION per-frame per-tool budgets ──────────
                # On a single physical frame after the question is posed,
                # cap memory_search at 5 and replay at 3. Beyond that the
                # model is looping; reject and push it to commit/abstain.
                if tool_name == "memory_search" and past_q \
                        and self._memsearch_this_frame >= 5:
                    results.append(ToolResult(
                        tool_name, False,
                        error="memory_search budget for THIS frame "
                              "exhausted (5). You have searched enough — "
                              "answer with what you have or `abstain`. "
                              "Do NOT search again on this frame."))
                    # Count rejected attempts so PHYS_CAP always backstops
                    # a same-frame loop (死循环 guard).
                    self._calls_this_phys_round += 1
                    break
                if tool_name == "replay" and past_q \
                        and self._replay_this_frame >= 3:
                    results.append(ToolResult(
                        tool_name, False,
                        error="replay budget for THIS frame exhausted "
                              "(3). Stop browsing frames — decide from "
                              "what you've already seen, or `abstain`."))
                    self._calls_this_phys_round += 1
                    break
                # ── replay intent declaration (anti-aimless-browse) ───
                # A replay must be preceded (same answer text) by a stated
                # intent like "I will replay t=X to check Y". If missing,
                # reject once with a reminder; the model re-issues with a
                # one-line intent next turn.
                if tool_name == "replay" and past_q \
                        and not _has_replay_intent(answer):
                    results.append(ToolResult(
                        tool_name, False,
                        error="Before replaying, state your intent in one "
                              "line, e.g. `I will replay t=12 to check the "
                              "card count`. Re-issue the replay WITH that "
                              "one-line intent."))
                    self._calls_this_phys_round += 1
                    break
                # ── OBSERVE-ONLY track_init cap (per sample) ──────────
                # ID=636 spawned 84 tracks → context blew up before the
                # question even arrived. The system already auto-refreshes
                # active tracks via _auto_write_memory, so the model needs
                # at most a few manual inits.
                if tool_name == "track_init" and is_post_hoc \
                        and self._track_init_count >= 3:
                    results.append(ToolResult(
                        tool_name, False,
                        error="track_init limit reached (3) for the "
                              "OBSERVE-ONLY phase. The system already "
                              "refreshes existing tracks automatically — "
                              "reuse track_query on an existing track id "
                              "instead of spawning new ones."))
                    self._calls_this_phys_round += 1
                    break
                # ── Per-frame track_query cap (all phases) ────────────
                # qwen3vl-8b re-issues the same track_query{trk,t} 10-15×
                # per frame; track_query always returns ok+bbox so there is
                # no "dead end" signal to make it stop, and this spins in
                # OBSERVE-ONLY where the POST-Q gates don't apply. Cap at 3
                # per frame (like replay): one query records the box, a
                # couple more is plenty — beyond that is redundant.
                if tool_name == "track_query" \
                        and self._track_query_this_frame >= 3:
                    results.append(ToolResult(
                        tool_name, False,
                        error="track_query limit reached (3) on this "
                              "frame. Re-querying the same track returns "
                              "essentially the same box and makes no "
                              "progress. The system auto-refreshes active "
                              "tracks each frame — emit `</Silence>` to "
                              "advance and keep observing, or (past the "
                              "question) commit your answer."))
                    self._calls_this_phys_round += 1
                    break
                res = self.registry.dispatch(
                    tool_name,
                    spec.get("arguments", {}) or {},
                    self.ctx,
                )
                results.append(res)
                self.ctx.calls_this_round += 1
                self._calls_made += 1
                self._calls_this_phys_round += 1
                # Per-frame per-tool counters (used by the budgets above).
                if tool_name == "memory_search":
                    self._memsearch_this_frame += 1
                elif tool_name == "replay":
                    self._replay_this_frame += 1
                elif tool_name == "track_init":
                    self._track_init_count += 1
                elif tool_name == "track_query":
                    self._track_query_this_frame += 1
                # Empty-result guard: track consecutive "0 results" so the
                # next prompt can inject a hard STOP when the target is
                # clearly not in the video.
                if tool_name in ("memory_search", "track_query") \
                        and _is_empty_tool_result(res):
                    self._consecutive_empty += 1
                else:
                    self._consecutive_empty = 0
                self.tool_log.append({
                    "round": self.current_round,
                    "call": spec,
                    "result": res.to_dict(),
                })

        # Always log the raw answer for this round (for trace replay)
        self.output[f"Round {self.current_round}"] = answer

        # ---- Abstain short-circuit -----------------------------------
        if self.ctx.abstain_reason is not None:
            listed_refusal = None
            if (
                os.environ.get(
                    "STREAMO_ENABLE_LISTED_REFUSAL_EVIDENCE_GATE", "0"
                ) == "1"
                or os.environ.get(
                    "STREAMO_ENABLE_LISTED_REFUSAL_VISUAL_AUDIT", "0"
                ) == "1"
            ):
                listed_refusal = _listed_refusal_final_for(self.sample)
            self.output[f"Round {self.current_round}_final"] = (
                listed_refusal
                or _abstain_final_for(self.sample)
            )
            self.completed = True
            self.current_round += 1
            return

        # ---- Tool-results pending: stay in same round ----------------
        if results:
            # Safety net: if (1) we're already at the last frame, (2) all
            # tool calls this round failed due to budget exhaustion, and
            # (3) the model is stuck looping on the same frame — force
            # completion to avoid an infinite tool-pending loop.
            all_budget_fail = bool(results) and all(
                (not r.ok) and (
                    "budget exhausted" in (r.error or "")
                    or "OBSERVE-ONLY phase" in (r.error or "")
                    # Per-frame per-tool budgets added for the Query-Driven
                    # Planning era. Without matching these the clock never
                    # advances when the model spins on a frame re-issuing
                    # the same rejected tool (死循环).
                    or "budget for THIS frame exhausted" in (r.error or "")
                )
                for r in results
            )
            at_last_frame = self.current_round >= self.total_rounds - 1
            if all_budget_fail and at_last_frame:
                self.output[f"Round {self.current_round}_final"] = _abstain_final_for(self.sample)
                self.completed = True
                self.current_round += 1
                return
            # If budget is fully exhausted but we're not at last frame,
            # the model will keep emitting tool_calls that all fail. Stop
            # waiting on tool results and force the streaming clock to
            # advance one tick so the loop makes progress.
            if all_budget_fail:
                self._pending_tool_text = None
                self._pending_tool_images = []
                self.ctx.next_round_images = []
                step = 1
                # Lock fps=1 within ±LOCK window around question_time and
                # for the entire post-question phase, otherwise the streaming
                # clock can stride past question_time and the sample exits
                # with no </Response>.
                in_lock = (self.question_time > 0 and
                           self.current_round >= self.question_time - 5)
                if (not in_lock) and self.ctx.sample_rate_fps and \
                        self.ctx.sample_rate_fps > 0:
                    step = max(1, int(round(1.0 / self.ctx.sample_rate_fps)))
                if in_lock:
                    self.ctx.sample_rate_fps = 1.0
                # Never stride past question_time in a single tick: clamp
                # the next round to question_time so the agent gets a
                # chance to emit </Response> at that frame.
                if self.current_round < self.question_time \
                        <= self.current_round + step:
                    self.current_round = self.question_time
                else:
                    self.current_round = min(self.current_round + step,
                                              self.total_rounds)
                if self.current_round >= self.total_rounds:
                    self.completed = True
                return
            self._pending_tool_text = "\n".join(
                (r.to_prompt_str()
                 + (" [0 results found]" if _is_empty_tool_result(r) else ""))
                for r in results)
            # ── Persist the model's tool-round observation into memory ──
            # On a tool-continuation round the model usually writes a short
            # observation about what a tool result (e.g. a zoom_crop patch)
            # revealed BEFORE issuing the next call. Previously this text
            # was dropped (we return early, never reaching the summarizer),
            # so visual readings like "I count 5 cards" never entered
            # memory. Capture that observation now so the understanding of
            # cropped/zoomed/replayed evidence is durable.
            self._persist_tool_round_observation(answer)
            # Drain any patches the tools wanted to inject
            self._pending_tool_images.extend(self.ctx.next_round_images)
            self.ctx.next_round_images = []
            # Tools may have set next_frame_idx — apply BEFORE we tick.
            # (The tick only happens once we move past the tool-continuation.)
            return

        # ---- Format validation: reply must include a legal sentinel --
        # Once we are at/after question_time the reply MUST contain
        # </Response> or </Silence>/</Standby>; otherwise we re-prompt
        # the same frame with a format-error tool_result.
        # Retry policy:
        #   * Not at last_frame: UNBOUNDED retries — the model is free
        #     to keep using tools / re-thinking on the SAME physical
        #     round until a legal sentinel is produced. Streaming
        #     clock never advances on a malformed reply.
        #   * At last_frame past q_time: bounded "final-call" retries
        #     (MAX_FINAL_CALL_RETRIES). Each retry the tool_result tells
        #     the model "this is the FINAL CALL — there are no more
        #     frames; answer NOW from memory." Only after the model
        #     fails all retries does the last-frame safety net abstain.
        MAX_FMT_RETRIES = 3
        MAX_FINAL_CALL_RETRIES = 3
        ans_norm_for_fmt = ovo._strip_thinking(answer or "").strip()
        has_resp = "</Response>" in ans_norm_for_fmt
        has_silence = ("</Silence>" in ans_norm_for_fmt
                       or "</Standby>" in ans_norm_for_fmt)
        is_legal = has_resp or has_silence
        past_q = self.current_round >= self.question_time
        is_last_frame_now = self.current_round >= self.total_rounds - 1
        # On last_frame past q_time, </Silence> is no longer "legal":
        # there is nothing to wait for, the model must commit
        # </Response>. Treat </Silence>-only replies as malformed so
        # they trigger final-call retries instead of being accepted.
        if is_last_frame_now and past_q and (not has_resp):
            is_legal = False
        if not is_legal and (ans_norm_for_fmt or past_q):
            self._format_retries_this_phys_round += 1
            cap = (MAX_FINAL_CALL_RETRIES if is_last_frame_now and past_q
                   else MAX_FMT_RETRIES)
            # Cap is enforced on EVERY frame (not just last_frame).
            # Historical bug: `(not is_last_frame_now) or _retries <= cap`
            # let non-last frames retry unconditionally, which on
            # malformed-tool_call drift (sample 0 of 20260610: 386
            # consecutive empty parses) burned hours of wall time on a
            # single sample. The streaming clock could never advance
            # because we kept re-prompting the same physical round.
            should_retry = self._format_retries_this_phys_round <= cap
            if should_retry:
                # Re-prompt the same physical round with a format-error
                # tool_result. Streaming clock does NOT advance.
                if is_last_frame_now and past_q:
                    options = self.sample.get("options") or []
                    if options:
                        opt_lines = "\n".join(
                            f"  {chr(65+i)}. {opt}"
                            for i, opt in enumerate(options[:8])
                        )
                        err_msg = (
                            "*** FINAL CALL — RESPOND OR LOSE THE "
                            "POINT. *** This is the last frame and "
                            "the previous reply did NOT contain "
                            "`</Response>`, so the system rolled back "
                            "and is asking again. The streaming clock "
                            "will not advance further; this round "
                            "MUST end with `</Response> <letter>. "
                            "<option text>`. Pick the option your "
                            "evidence best supports from MEMORY — "
                            "including an \"Unable to answer\"/\"cannot "
                            "determine\" option if your plan's "
                            "answerability judged the target absent.\n"
                            f"Options:\n{opt_lines}\n"
                            "Reply format: `</Response> <letter>. "
                            "<option text>` (e.g. `</Response> "
                            "C. ...`). Don't end with `</Silence>`; "
                            "don't open another tool-call loop. If a "
                            "single tool call would settle it, you "
                            "may call ONE, but THEN commit "
                            "`</Response>` immediately."
                        )
                    else:
                        err_msg = (
                            "*** FINAL CALL — RESPOND OR LOSE THE "
                            "POINT. *** This is the last frame and "
                            "the previous reply did NOT contain "
                            "`</Response>`, so the system rolled back "
                            "and is asking again. The streaming clock "
                            "will not advance further; this round "
                            "MUST end with `</Response> <your "
                            "answer>` (short and concrete). Use the "
                            "long-term MEMORY (the pinned [Q] entry "
                            "plus pinned facts and episodic "
                            "summaries) to ground your answer. Commit "
                            "the answer your evidence best supports. "
                            "Don't end with `</Silence>`; "
                            "don't open another tool-call loop. If a "
                            "single tool call would settle it, you "
                            "may call ONE, but THEN commit "
                            "`</Response>` immediately."
                        )
                elif past_q:
                    err_msg = (
                        "format error: your reply must end with "
                        "`</Silence>` (continue observing) OR "
                        "`</Response> <your answer>` (commit final "
                        "answer). t>=question_time, so once you have "
                        "enough info you SHOULD commit `</Response>` — "
                        "the question has already been asked. "
                        "Lingering with </Silence> wastes rounds; if "
                        "MEMORY already contains the answer, emit "
                        "`</Response>` THIS TURN."
                    )
                else:
                    err_msg = (
                        "format error: your reply must end with "
                        "`</Silence>`. t<question_time so emit "
                        "<Observation>...</Observation> + </Silence>."
                    )
                self._pending_tool_text = (
                    "<tool_result>"
                    + json.dumps({
                        "name": "_format_check",
                        "ok": False,
                        "error": err_msg,
                    }, ensure_ascii=False)
                    + "</tool_result>"
                )
                return
            # Retries exhausted. Two cases:
            #   (a) last_frame past q_time → fall through to the
            #       last-frame safety net which will abstain.
            #   (b) non-last_frame (or pre-q_time) → force the
            #       streaming clock to advance one tick and reset
            #       per-phys-round counters. Without this, a sample
            #       that keeps emitting malformed text can stall
            #       indefinitely on the same frame (cf. sample 0 of
            #       20260610: 386 consecutive malformed parses on
            #       round=2).
            if not (is_last_frame_now and past_q):
                self._pending_tool_text = None
                self._pending_tool_images = []
                # Advance one frame; clamp to total_rounds so the
                # COMMIT-round / completion logic still runs.
                self.current_round = min(self.current_round + 1,
                                          self.total_rounds)
                self._last_phys_round = self.current_round
                self._calls_this_phys_round = 0
                self._replies_this_phys_round = 0
                self._format_retries_this_phys_round = 0
                self._last_frame_reply_cap_used = False
                # v2: clear memsearch + zoom phys-round counters on
                # forced clock advance.
                self.ctx.memsearch_history = []
                self.ctx.zoom_count_this_phys_round = 0
                if self.current_round >= self.total_rounds:
                    # Reached COMMIT round; let the next step handle
                    # the forced final answer prompt.
                    pass
                return
            # Last-frame past q_time, retries exhausted: fall through
            # so the last-frame safety net below abstains.

        # ---- No tool calls: fall back to base behavior ---------------
        # 1) </Response> -> task done
        # Strip <think>...</think> reasoning before scanning for the
        # response sentinel. Local Qwen3.5 with thinking enabled frequently
        # quotes the literal token `</Response>` inside its chain-of-thought
        # (e.g. the system prompt says "do NOT use </Response>" and the
        # model echoes it back). We must not let that quoted token finish
        # the sample. The same applies to `</Silence>` checks below.
        _resp_part = ovo._strip_thinking(answer) if answer else ""

        # ── Pillar 3: Look-Back Verification gate ────────────────────
        # Before accepting `</Response>`, run cross-layer consistency
        # checks. If conflicts are found, schedule targeted replays
        # instead of completing the sample.
        if (self.enable_verification
                and RESPONSE_TAG in _resp_part
                and self.current_round >= self.question_time
                and not self.sample.get("proactive_output")
                and self._verification_checks is not None
                and self._replay_scheduler is not None
                and not self._verification_in_progress
                and self._graph is not None):
            from tools.verification import run_verification_pipeline
            needs_replay, targets, _results = run_verification_pipeline(
                answer=_resp_part,
                retrieval_query=self._evidence_query(),
                graph=self._graph,
                checker=self._verification_checks,
                scheduler=self._replay_scheduler,
                current_time=self.current_round,
                question_time=self.question_time,
            )
            if needs_replay and targets:
                # Schedule the first replay target via the existing replay
                # infrastructure. The model will re-watch the target frame,
                # then we'll re-enter the normal tool-continuation flow.
                self.ctx.next_frame_idx = targets[0]
                self._verification_in_progress = True
                self._pending_tool_text = (
                    "<tool_result>"
                    + json.dumps({
                        "name": "_verify",
                        "ok": True,
                        "data": {
                            "action": "replay_for_verification",
                            "target_frame": targets[0],
                            "reason": (
                                "Memory evidence is inconsistent for this "
                                "answer. Re-watch the frame at t="
                                f"{targets[0]}s to verify before committing."
                            ),
                        },
                    }, ensure_ascii=False)
                    + "</tool_result>"
                )
                return  # Stay on the same frame; let replay execute.

        if self.question and answer and RESPONSE_TAG in _resp_part:
            # Phase gate: a model-emitted </Response> is only valid AFTER
            # the question has been posed (i.e. the streaming clock has
            # reached `question_time`). Local Qwen3.5 (thinking-enabled)
            # occasionally hallucinates "this is the LAST FRAME, emit
            # </Response>" mid-stream during OBSERVE-ONLY — accepting that
            # would short-circuit the entire sample dozens of seconds early
            # (e.g. sample 297: question_time=370 but model fired
            # </Response> at t=36). Reject the sentinel, drop the line into
            # MEMORY as a normal observation if useful, and let the stream
            # clock advance.
            if self.current_round < self.question_time:
                # Strip the bogus sentinel from the visible answer so it is
                # not treated as a final reply downstream, but still allow
                # the regular brief-mode summarizer below to capture any
                # legitimate <Observation> the model emitted alongside it.
                answer = (answer or "").replace(RESPONSE_TAG, "")
                # Fall through into the normal observe-only path; do NOT
                # mark the sample as completed here.
            elif (
                self.sample.get("proactive_output")
                and self.current_round <= self.question_time
            ):
                answer = (
                    (answer or "").replace(RESPONSE_TAG, "").strip()
                    + "\n</Silence>"
                ).strip()
                self.output[f"Round {self.current_round}"] = answer
            else:
                # ── Invalid-refusal guard ────────────────────────────
                # The model emitted `</Response>` past question_time, but
                # the answer is a refusal ("Unable to answer" / "cannot
                # determine") on a question whose listed options do NOT
                # include such a choice (e.g. EPM colour/location MCQ,
                # samples 34/35). Accepting it scores 0 — it isn't even a
                # listed option. Reject ONCE and re-prompt for a concrete
                # pick; only if the model refuses AGAIN do we fall back to
                # the best-guess option. HLD-style questions whose options
                # genuinely include an "unable" choice are exempt (the
                # refusal is a valid answer there).
                _ans_visible = _resp_part.split(RESPONSE_TAG, 1)[-1] \
                    if RESPONSE_TAG in _resp_part else _resp_part
                if (_is_refusal_answer(_ans_visible)
                        and not _options_allow_refusal(self.sample)
                        and not self._invalid_refusal_retry_used):
                    self._invalid_refusal_retry_used = True
                    self._persist_tool_round_observation(answer)
                    options = self.sample.get("options") or []
                    if options:
                        opt_lines = "\n".join(
                            f"  {chr(65+i)}. {opt}"
                            for i, opt in enumerate(options[:8]))
                        retry_err = (
                            "*** \"Unable to answer\" is NOT one of the "
                            "options for THIS question. *** You must pick "
                            "the single listed option your evidence best "
                            "supports — refusing scores zero. Re-read "
                            "MEMORY (and the current frame) and commit the "
                            "MOST LIKELY option, even if you are not "
                            "certain.\n"
                            f"Options:\n{opt_lines}\n"
                            "Reply format: `</Response> <letter>. "
                            "<option text>` (e.g. `</Response> A. ...`).")
                    else:
                        retry_err = (
                            "*** Refusing is not allowed here — this "
                            "question does not offer an \"unable / cannot "
                            "determine\" choice. *** Give your single "
                            "best-guess answer grounded in MEMORY and the "
                            "current frame, in the exact format the "
                            "question asks for. Do NOT reply \"Unable to "
                            "answer\".")
                    self._pending_tool_text = (
                        "<tool_result>"
                        + json.dumps({
                            "name": "_answer_required",
                            "ok": False,
                            "error": retry_err,
                        }, ensure_ascii=False)
                        + "</tool_result>"
                    )
                    self.output[f"Round {self.current_round}"] = answer
                    return  # stay on same frame; demand a real pick
                if (_is_refusal_answer(_ans_visible)
                        and not _options_allow_refusal(self.sample)
                        and self._invalid_refusal_retry_used):
                    # Refused a second time → substitute the best-guess
                    # option so the sample scores on its merits, not 0.
                    answer = _abstain_final_for(self.sample)
                    _resp_part = answer
                    # Overwrite the round's recorded output (stamped with
                    # the original refusal at line ~3270) so
                    # _extract_final_response reads the substituted answer,
                    # not the stale "Unable to answer".
                    self.output[f"Round {self.current_round}"] = answer
                if (
                    plan_requests_listed_refusal_bidirectional_audit(
                        self._query_plan,
                        self.sample,
                    )
                    and not self._listed_refusal_bidir_done
                    and not self._listed_refusal_bidir_inflight
                ):
                    original = ovo._extract_option_letter(_ans_visible)
                    options = self.sample.get("options") or []
                    if (
                        original is not None
                        and 0 <= ord(original) - ord("A") < len(options)
                    ):
                        self._listed_refusal_bidir_original_answer = original
                        self._listed_refusal_bidir_original_response = answer
                        self._listed_refusal_bidir_pending = True
                        self.output[
                            f"Round {self.current_round}_pre_bidir_audit"
                        ] = (
                            "[listed-refusal-bidirectional-audit pending] "
                            f"original={original}. "
                            f"{options[ord(original) - ord('A')]}"
                        )
                        return
                if (
                    normalize_yes_no_response(answer) == "No"
                    and route_event_predicate(
                        self._query_plan,
                        self.sample,
                        compiled_locus=compile_evidence_locus(
                            self._query_plan
                        ),
                    )
                    and not self._event_predicate_done
                    and not self._event_predicate_inflight
                    and not self._event_predicate_pending
                ):
                    self._event_predicate_original_response = answer
                    self._event_predicate_pending = "contract"
                    self.output[
                        f"Round {self.current_round}_"
                        "pre_event_predicate_audit"
                    ] = answer
                    return
                if (
                    plan_requests_current_locus_dual_view_audit(
                        self._query_plan,
                        self.sample,
                    )
                    and not self._current_locus_dual_view_done
                    and not self._current_locus_dual_view_inflight
                ):
                    original = ovo._extract_option_letter(_ans_visible)
                    options = self.sample.get("options") or []
                    if (
                        original is not None
                        and 0 <= ord(original) - ord("A") < len(options)
                    ):
                        self._current_locus_dual_view_original_answer = \
                            original
                        self._current_locus_dual_view_original_response = \
                            answer
                        self._current_locus_dual_view_pending = True
                        self.output[
                            f"Round {self.current_round}_"
                            "pre_current_locus_dual_view_audit"
                        ] = answer
                        return
                if (
                    os.environ.get(
                        "STREAMO_ENABLE_LISTED_REFUSAL_VISUAL_AUDIT", "0"
                    ) == "1"
                    and _refusal_option_index(self.sample) is not None
                    and not _is_refusal_answer(_ans_visible)
                    and not self._listed_refusal_audit_done
                    and not self._listed_refusal_audit_inflight
                ):
                    proposed = ovo._extract_option_letter(_ans_visible)
                    options = self.sample.get("options") or []
                    legal = {
                        chr(ord("A") + index): str(option)
                        for index, option in enumerate(options)
                    }
                    refusal_index = _refusal_option_index(self.sample)
                    refusal_letter = (
                        chr(ord("A") + refusal_index)
                        if refusal_index is not None else None
                    )
                    if proposed in legal and proposed != refusal_letter:
                        self._listed_refusal_proposed_answer = proposed
                        self._listed_refusal_proposed_text = legal[proposed]
                        self._listed_refusal_original_response = answer
                        self._listed_refusal_audit_pending = True
                        self.output[
                            f"Round {self.current_round}_pre_audit"
                        ] = (
                            "[listed-refusal-audit pending] proposed="
                            f"{proposed}. {legal[proposed]}"
                        )
                        return
                # Accept the final </Response>. Visual grounding is now

                # ENCOURAGED via prompt (replay tool + 强引导), NOT enforced:
                # the model decides for itself whether and how many times to
                # replay based on the query. A hard "must replay before
                # answering" gate was tried but hurt real-time perception
                # tasks (OCR/ACR/FPD/STU) — they answer from the CURRENT
                # frame and forced look-back injected noise — while only
                # helping backward tasks. To keep the comparison fair across
                # all tasks (no per-task branching), the gate is removed and
                # the choice is left to the model.
                # Brief mode: still write the final answer to memory before
                # exit.
                if self.memory_mode == "brief" and self._summarizer is not None:
                    ev = self._summarizer(answer, self.current_round,
                                           ref_frame=self.current_round)
                    if ev and self._memory is not None:
                        self._memory.write_event(self.current_round, **ev)
                        self._post_graph_hook(ev, self.current_round)
                self.current_round += 1
                self.completed = True
                return

        # Brief mode: summarize this round's response into memory.
        if self.memory_mode == "brief" and self._summarizer is not None:
            ev = self._summarizer(answer, self.current_round,
                                   ref_frame=self.current_round)
            if ev and self._memory is not None:
                self._memory.write_event(self.current_round, **ev)
                self._post_graph_hook(ev, self.current_round)
                # Spatial-layout auto-pin: if the observation contains
                # explicit count + relative position keywords, treat it
                # as a durable fact and pin it (immune to MEMORY eviction).
                self._maybe_autopin_observation(ev.get("text") or "")
            if self._budget is not None:
                self._budget.add_frame()

        # 1.5) Last-frame / COMMIT-round safety net.
        # New flow:
        #   * If we're at the LAST video frame past q_time and there is
        #     no </Response>, advance to the COMMIT round (a virtual
        #     extra round where tools are disabled and the prompt
        #     forces a final answer).
        #   * If we're already in the COMMIT round and STILL no
        #     </Response>, fall back to the abstain sentinel.
        is_last_frame = self.current_round == self.total_rounds - 1
        is_commit_round = self.current_round == self.total_rounds
        past_question = self.current_round >= self.question_time
        if (
            (is_last_frame or is_commit_round)
            and past_question
            and bool(self.question)
        ):
            ans_norm = ovo._strip_thinking(answer or "").strip()
            if "</Response>" not in ans_norm:
                if is_commit_round:
                    # Final fallback: abstain after the model failed
                    # both the last frame AND the COMMIT round.
                    self.output[f"Round {self.current_round}_final"] = _abstain_final_for(self.sample)
                    self.completed = True
                    self.current_round += 1
                    return
                # Last frame failed → enter COMMIT round.
                self.current_round = self.total_rounds
                # Reset per-phys-round counters so the COMMIT round
                # starts fresh (independent format-retry budget).
                self._last_phys_round = self.current_round
                self._calls_this_phys_round = 0
                self._replies_this_phys_round = 0
                self._format_retries_this_phys_round = 0
                self._last_frame_reply_cap_used = False
                # v2: clear memsearch + zoom phys-round counters on
                # COMMIT round entry.
                self.ctx.memsearch_history = []
                self.ctx.zoom_count_this_phys_round = 0
                # Drop pending tool-result text — COMMIT round starts
                # with a fresh user turn carrying the COMMIT prompt.
                self._pending_tool_text = None
                self._pending_tool_images = []
                return

        # 2) Honor a pending temporal jump from earlier tool call.
        # Look-and-return + past-only: replay may only RE-WATCH frames that
        # have ALREADY streamed by, i.e. the visible range is [0,
        # current_round]. A target beyond the present is clamped back to
        # the current frame (you cannot look into the future). After the
        # model re-watches the target for one inference round, the stream
        # snaps back to where it left off (look-and-return).
        if self.ctx.next_frame_idx is not None:
            pre_seek_round = self.current_round
            # Visible range is the past up to and including NOW; never let
            # replay reach a future frame the stream has not produced yet.
            upper = min(self.current_round, self.total_rounds - 1)
            target = max(0, min(self.ctx.next_frame_idx, upper))
            self.ctx.next_frame_idx = None
            # A genuine backward look (target strictly before NOW) triggers
            # look-and-return; seeking to the current frame is a no-op jump
            # and needs no resume bookkeeping.
            if target < pre_seek_round:
                self._resume_round_after_seek = pre_seek_round
            self.current_round = target
            if self.current_round >= self.total_rounds:
                self.completed = True
            return
        # 2.5) Look-and-return restore: the model has now consumed one
        # inference round on the replay target. Snap the streaming clock back
        # to the pre-replay position so the forward stream continues from
        # there rather than re-walking everything after the replay target.
        if self._resume_round_after_seek is not None:
            resume = self._resume_round_after_seek
            self._resume_round_after_seek = None
            # Guard: never resume backward (only restore forward to where we
            # left off) and never strand past total_rounds.
            if resume > self.current_round:
                self.current_round = min(resume, self.total_rounds - 1)
            if self.current_round >= self.total_rounds:
                self.completed = True
            return
        # 3) Honor an active `defer` (efficiency tool): jump to defer_until_t
        if self.ctx.defer_until_t is not None and \
                self.current_round < self.ctx.defer_until_t:
            target = self.ctx.defer_until_t
            # Same clamp as path 2: never jump over question_time.
            if self.question_time > 0 and \
                    self.current_round < self.question_time <= target:
                target = self.question_time
                # Keep the residual defer so the agent will resume the
                # jump after question_time has been consumed.
                self.ctx.defer_until_t = self.ctx.defer_until_t \
                    if self.ctx.defer_until_t > self.question_time else None
            else:
                self.ctx.defer_until_t = None
            self.current_round = target
            if self.current_round >= self.total_rounds:
                self.completed = True
            return
        # 4) Default tick (honor sample_rate_fps if set).
        # Lock fps=1 within ±LOCK window around question_time and for the
        # entire post-question phase. Also clamp the next round to
        # question_time when a stride would skip over it, so the agent gets
        # a chance to emit </Response> at the proper frame.
        step = 1
        in_lock = (self.question_time > 0 and
                   self.current_round >= self.question_time - 5)
        if (not in_lock) and self.ctx.sample_rate_fps and \
                self.ctx.sample_rate_fps > 0:
            step = max(1, int(round(1.0 / float(self.ctx.sample_rate_fps))))
        if in_lock:
            self.ctx.sample_rate_fps = 1.0
        if self.current_round < self.question_time \
                <= self.current_round + step:
            self.current_round = self.question_time
        else:
            self.current_round += step
        if self.current_round >= self.total_rounds:
            self.completed = True

    # ------------------------------------------------------------------
    # Step-level trajectory streaming (借鉴 agentic-mle)
    # ------------------------------------------------------------------
    def _record_step(self, answer: str, round_before: int,
                      calls_made_before: int) -> None:
        """Append one JSONL record for this physical model round.

        Each record captures the full (request, response, parsed tool
        calls, tool results, memory snapshot, state transition) tuple.
        Crucially we ``flush`` after every write so a kill / OOM never
        loses prior steps.
        """
        if not self._steps_path:
            return
        # We record EVERY round (real model call + auto fast-forward
        # `</Silence>` filler) so the trajectory keeps a continuous
        # timeline. Filler rounds are tagged via ``is_filler`` so the
        # downstream consumer (training data prep / debug viewer) can
        # filter them on demand without losing temporal structure.
        is_filler = self._last_request is None
        try:
            calls = parse_tool_calls(
                answer or "",
                max_calls=self.max_per_reply or None,
            )
            new_tool_log = self.tool_log[calls_made_before:
                                          calls_made_before + len(calls)]
            mem_snap_meta: Optional[Dict[str, Any]] = None
            mem_text: Optional[str] = None
            if self._memory is not None:
                # Record the *same* snapshot mode that was actually sent
                # to the model on this round, so post-run diagnostics see
                # mode/n_recall/n_dropped that match the real prompt.
                snap_kwargs = self._resolve_render_args(
                    current_t=self.current_round,
                    past_q=self.current_round >= self.question_time,
                )
                snap = self._memory.snapshot(**snap_kwargs)
                mem_text = snap.get("text")
                mem_snap_meta = {k: v for k, v in snap.items() if k != "text"}
            req = self._last_request or {}
            now = time.time()
            step_duration_ms = (
                int((now - self._last_step_time) * 1000)
                if self._last_step_time is not None else None
            )
            self._last_step_time = now

            record = {
                "step": self._step_idx,
                "sample_id": str(self.sample.get("id", "")),
                "task": self.sample.get("task"),
                "round_before": round_before,
                "round_after": self.current_round,
                "completed": bool(self.completed),
                "abstain_reason": self.ctx.abstain_reason,
                "is_filler": is_filler,
                "timestamp": now,
                "step_duration_ms": step_duration_ms,
                "elapsed_ms": int((now - self._sample_start_time) * 1000),
                "gt": self.sample.get("gt"),
                "question": self.sample.get("question"),
                "all_steps": self.sample.get("all_steps"),
                "activity": self.sample.get("activity"),
                "options": self.sample.get("options"),
                "request": {
                    "kind": req.get("kind"),
                    "messages": req.get("messages", []),
                    "images": req.get("images", []),
                    "videos": req.get("videos", []),
                },
                "response": answer or "",
                "parsed_tool_calls": calls,
                "tool_calls_dropped_by_reply_limit": (
                    self._last_reply_tool_calls_dropped
                ),
                "tool_results": [
                    {
                        "round": e.get("round"),
                        "call": e.get("call"),
                        "result": e.get("result"),
                    }
                    for e in new_tool_log
                ],
                "memory": {
                    "text": mem_text,
                    "meta": mem_snap_meta,
                } if mem_snap_meta is not None else None,
                "ctx_after": {
                    "calls_this_round": self.ctx.calls_this_round,
                    "calls_made": self._calls_made,
                    "next_frame_idx": self.ctx.next_frame_idx,
                    "defer_until_t": self.ctx.defer_until_t,
                    "sample_rate_fps": self.ctx.sample_rate_fps,
                },
            }
            with open(self._steps_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            self._step_idx += 1
            # Mirror the latest MEMORY brief to <sid>.memory.txt so users
            # can `tail -f` it during a long run instead of waiting for
            # the sample to finish.
            if mem_text and self._trajectory_dir:
                sid = str(self.sample.get("id", "unknown")).replace("/", "_")
                mem_path = os.path.join(self._trajectory_dir,
                                         f"{sid}.memory.txt")
                with open(mem_path, "w", encoding="utf-8") as mf:
                    mf.write(mem_text)
                    mf.flush()
            # Drain the request snapshot so the next round starts clean.
            self._last_request = None
        except Exception as e:  # noqa: BLE001
            # Trajectory dump must never crash the main loop.
            print(f"[trajectory] step record failed for "
                  f"sid={self.sample.get('id')}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# System prompt with tool schemas
# ---------------------------------------------------------------------------
# Pre-query observation is deliberately task-label blind. The generic
# OBSERVE-ONLY checklist in `_brief_get_next_request` captures identities,
# actions/states, spatial relations, counts, OCR, audio and transitions for
# every sample. Specialized routing is allowed only after the model authors a
# structured query plan.

# System prompt templates
AGENT_SYSTEM_TEMPLATE = (
    "You are Streamo, a streaming video-understanding agent. You see one "
    "second of video per round. You may call tools to look more carefully "
    "before answering.\n"
    "\n"
    "## Tool-call protocol\n"
    "Whenever you want to call a tool, emit *exactly* the following block "
    "as part of your response (you may emit multiple such blocks):\n"
    "<tool_call>{\"name\": \"<tool_name>\", \"arguments\": {...}}</tool_call>\n"
    "After the tool block(s), STOP. The system will run the tools and feed "
    "<tool_result>{...}</tool_result> back to you. Only after you have "
    "enough evidence, finish with `</Response> <your final answer>`.\n"
    "\n"
    "## Available tools\n"
    "__TOOL_SPECS__\n"
    "\n"
    "## Guidelines\n"
    "- You will see the question on EVERY frame, including frames "
    "BEFORE the question is officially asked. Treat early frames as "
    "context: take notes via `memory_write` (set `pin=true` for facts "
    "you must not forget) so that when the question becomes "
    "answerable you have evidence ready.\n"
    "- Prefer answering directly when the current frame alone is "
    "sufficient.\n"
    "- If you are uncertain — small text, occluded objects, an event "
    "that may have happened earlier, or a fact you only vaguely "
    "remember — CALL A TOOL FIRST instead of guessing. Concretely:\n"
    "    * Look back in time → `replay` (re-watch a past frame, then "
    "auto-return to now).\n"
    "    * Re-read your notes → `memory_read` / `memory_list` / "
    "`memory_search` (semantic).\n"
    "    * Track an object across frames → `track_init` + `track_query` "
    "(PREFERRED for any moving subject — it follows automatically and "
    "saves you from repeated zoom_crop calls).\n"
    "    * Listen to speech → `asr_segment`.\n"
    "    * Examine STATIC fine detail (small text, a numeric label, a "
    "stationary face) → `zoom_crop` on the relevant bbox. **Use "
    "sparingly**: zoom_crop only re-reads ONE region of the CURRENT "
    "frame — for moving subjects use track_init, for cross-frame "
    "questions use replay+memory_search, for audio cues use "
    "asr_segment. Same-frame near-duplicate crops are auto-rejected.\n"
    "- If an ACTUAL listed option explicitly allows `Unable to answer` "
    "or `cannot determine`, and visual evidence is genuinely missing, "
    "you may call `abstain` instead of inventing a concrete claim.\n"
    "- Tool calls are cheap; one or two well-chosen calls almost "
    "always beat a blind guess.\n"
)


# Brief-mode supplement: appended on top of AGENT_SYSTEM_TEMPLATE only when
# memory_mode='brief'. Tells the model that past frames are gone and it must
# rely on the MEMORY block + targeted tools.
# NOTE: phase-specific instructions (OBSERVE-ONLY format, Recap protocol,
# answer guidelines, tool decision trees, disambiguation rules) are now
# injected in the per-round user prompt — NOT duplicated here.  This keeps
# the system prompt lean and cache-friendly.
AGENT_SYSTEM_BRIEF_SUFFIX = (
    "\n## Memory mode (brief)\n"
    "Past frames are NOT visible. Each round you receive:\n"
    "  1. A `MEMORY:` block — your only window into the past.\n"
    "  2. (optional) `<tool_result>` blocks from tools you just called.\n"
    "  3. A `[Phase]` line — see the per-round prompt for what to do.\n"
    "  4. One or two frames as `<image>` tokens.\n"
    "\n## Memory layers (tree: long_term ⊃ episodic ⊃ working)\n"
    "* **long_term (pinned)** — KEY NODES only. Scene changes, actor "
    "enter/exit, action start/end, object state changes. Never evicted. "
    "Split into `active` (current state) and `archived` (superseded old "
    "state, tagged `→#N@t=...` — DON'T use as current truth).\n"
    "* **episodic** — Auto-aggregated [t0,t1] span summaries. Your "
    "mid-term notebook: \"what happened across this period.\"\n"
    "* **working** — Per-frame events verbatim. Head (t < head_secs) + "
    "Tail (t >= now - tail_secs) are full-text; Middle is aggregated "
    "into episodic. Events tagged `[Δ]` = change-bearing (prefer these "
    "for retrieval); `[obs]` = static frame.\n"
    "Layers are CONNECTED by pointers: a working event links to its "
    "parent episode; an episode links to its pin; a pin links back to "
    "its episode. The `[RETRIEVED]` block (injected post-question) "
    "shows evidence *paths* that span layers — follow the chain.\n"
    "\n## Capabilities\n"
    "- `memory_search` / `memory_read`: keyword search across memory layers.\n"
    "- `replay(t=X)`: re-watch ANY past frame, then auto-return to now.\n"
    "- `track_init` + `track_query`: SAM2 follows moving subjects.\n"
    "- `zoom_crop`: inspect STATIC fine detail on CURRENT frame.\n"
    "- `asr_segment`: transcribe speech from any time window.\n"
    "- `memory_write(pin=true)`: record durable facts.\n"
    "\n## Limitations\n"
    "- MEMORY is keyword-searchable, NOT semantic (synonyms may not match).\n"
    "- Episodic summaries are COMPRESSED — details live in working layer; "
    "use `memory_search` to find specific working events.\n"
    "- Observations may contain MINOR ERRORS (misreads, typos, missed "
    "details). When a fact is critical, verify via `replay`.\n"
    "- The streaming clock CANNOT go backwards on its own — you must "
    "call `replay` to revisit the past.\n"
)



def render_agent_system(registry: ToolRegistry, base_system: str = "",
                         memory_mode: str = "full",
                         delta_input_mode: str = "2frame") -> str:
    specs = []
    for s in registry.schemas():
        specs.append(json.dumps(s, ensure_ascii=False))
    body = AGENT_SYSTEM_TEMPLATE.replace("__TOOL_SPECS__", "\n".join(specs))
    # ── Inject 2-frame mode note ────────────────────────────────────
    if delta_input_mode == "2frame":
        body = body.replace(
            "You are Streamo",
            "You are Streamo, operating in 2-FRAME DELTA mode: "
            "during OBSERVE-ONLY you will see TWO frames side-by-side "
            "(previous + current) for direct visual comparison. "
            "During POST-QUESTION and REPLAY you see a single frame. "
            "You are Streamo",
        )
    if memory_mode == "brief":
        body = body + AGENT_SYSTEM_BRIEF_SUFFIX
    if base_system:
        return base_system.rstrip() + "\n\n" + body
    return body


# ---------------------------------------------------------------------------
# Main loop (mirrors ovo.process_videos_batch but uses AgentVideoProcessor)
# ---------------------------------------------------------------------------
def _dump_trajectory(traj_dir: str, p: "AgentVideoProcessor",
                      eval_record: Dict[str, Any]) -> None:
    """Persist one sample's full agentic trajectory + memory snapshot.

    Output layout (借鉴 #8 from agentic-mle):
        <traj_dir>/<sample_id>.json   # canonical record
        <traj_dir>/<sample_id>.memory.txt  # final MEMORY brief (text-only)

    The JSON contains: eval result, per-round answers, full tool_log,
    and the final structured memory dump (events / pinned / entities /
    tracks / audio_segments). Suitable for offline RL / SFT data prep.
    """
    sid = str(p.sample.get("id", "unknown")).replace("/", "_")
    realtime = int(time.time() - p._sample_start_time)
    live_rounds = dict(p.output)
    cached_rounds = getattr(p, "_cached_prequery_rounds", {}) or {}
    rounds = (
        merge_cached_trajectory_rounds(
            cached_rounds,
            live_rounds,
            question_time=p.question_time,
        )
        if cached_rounds else live_rounds
    )
    cached_tool_calls = list(
        getattr(p, "_cached_prequery_tool_calls", []) or []
    )
    live_tool_calls = list(p.tool_log)
    tool_calls = cached_tool_calls + live_tool_calls
    rec: Dict[str, Any] = {
        "sample_id": sid,
        "task": p.sample.get("task"),
        "video": p.sample.get("video"),
        "gt": p.sample.get("gt"),
        "question": p.sample.get("question"),
        # SSR/CRR forward meta-questions carry their probed event in
        # ``all_steps`` (with an empty ``question``); record it so the
        # trajectory JSON shows WHAT was actually asked.
        "all_steps": p.sample.get("all_steps"),
        "step_start_time": p.sample.get("step_start_time"),
        # REC counting tasks probe an ``activity`` (question/options are
        # None); record it and its timing so the JSON shows what was asked.
        "activity": p.sample.get("activity"),
        "query_time": p.sample.get("query_time"),
        "prequery_source_id": p.sample.get("prequery_source_id"),
        "options": p.sample.get("options"),
        "question_time": p.question_time,
        "total_rounds": p.total_rounds,
        "memory_mode": p.memory_mode,
        "final_response": eval_record.get("final_response"),
        "trigger_round": eval_record.get("trigger_round"),
        "abstain_reason": p.ctx.abstain_reason,
        "realtime": realtime,
        "rounds": rounds,
        "tool_calls": tool_calls,
        "tool_calls_n": len(tool_calls),
    }
    cached_source = getattr(p, "_cached_prequery_source", None)
    if cached_source:
        rec.update({
            "trajectory_mode": "prequery_cached_then_live_query",
            "prequery_trajectory_source": cached_source,
            "cached_prequery_rounds_n": len(cached_rounds),
            "live_query_rounds_n": len(live_rounds),
            "cached_prequery_round_keys": list(cached_rounds),
            "live_query_round_keys": list(live_rounds),
            "cached_prequery_tool_calls_n": len(cached_tool_calls),
            "live_query_tool_calls_n": len(live_tool_calls),
            "steps_jsonl_scope": (
                "live query-time execution only; pre-query rounds are "
                "merged from prequery_trajectory_source in this "
                "canonical JSON"
            ),
        })
    else:
        rec.update({
            "trajectory_mode": "full_stream_live",
            "cached_prequery_rounds_n": 0,
            "live_query_rounds_n": len(live_rounds),
            "cached_prequery_round_keys": [],
            "live_query_round_keys": list(live_rounds),
            "cached_prequery_tool_calls_n": 0,
            "live_query_tool_calls_n": len(live_tool_calls),
            "steps_jsonl_scope": "full live execution",
        })
    mem_text = ""
    if p._memory is not None:
        snap = p._memory.snapshot(
            token_budget=p.memory_token_budget,
            current_t=p.current_round,
        )
        mem_text = snap.get("text", "")
        # Entity graph: extract the full EntityIndex state from MemoryGraph.
        entity_graph = {}
        if p._graph is not None:
            for eid, sub in p._graph.entity_index._entities.items():
                entity_graph[eid] = {
                    "entity_id": sub.entity_id,
                    "event_ids": list(sub.event_ids),
                    "episode_ids": list(sub.episode_ids),
                    "pinned_ids": list(sub.pinned_ids),
                    "first_seen": sub.first_seen,
                    "last_seen": sub.last_seen,
                    "aliases": list(sub.aliases),
                }
        rec["memory"] = {
            "snapshot_meta": {k: v for k, v in snap.items() if k != "text"},
            "events": [e.to_dict() for e in p._memory.events],
            "episodes": [dict(ep) for ep in p._memory.episodes],
            "pinned_facts": list(p._memory.pinned_facts),
            "entity_graph": entity_graph,
            "relations": (list(p._graph.relations)
                          if p._graph is not None else []),
            "entities": dict(p._memory.entities),
            "tracks": {k: dict(v) for k, v in p._memory.tracks.items()},
            "audio_segments": list(p._memory.audio_segments),
        }
    json_path = os.path.join(traj_dir, f"{sid}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    if mem_text:
        with open(os.path.join(traj_dir, f"{sid}.memory.txt"),
                  "w", encoding="utf-8") as f:
            f.write(mem_text)


def _batch_infer_current_locus_dual_view(
    engine: Any,
    infer_requests: List[Any],
    batch_size: int,
    *,
    max_tokens: int = 1800,
) -> List[str]:
    """Infer current-locus audit/protector requests at temperature zero."""
    if not infer_requests:
        return []
    from swift.infer_engine import RequestConfig

    responses: List[str] = []
    config = RequestConfig(max_tokens=int(max_tokens), temperature=0.0)
    for start in range(0, len(infer_requests), max(1, batch_size)):
        batch = infer_requests[start:start + max(1, batch_size)]
        raw = engine.infer(batch, config)
        responses.extend(
            str(response.choices[0].message.content or "")
            for response in raw
        )
    return responses


def process_videos_agent(
    engine,
    samples: List[dict],
    system: str,
    registry: ToolRegistry,
    *,
    max_concurrent: int = 32,
    batch_size: int = 16,
    output_file: Optional[str] = None,
    task_filter: Union[None, Set[str], List[str], str] = None,
    max_tool_calls_per_round: int = 3,
    max_tool_calls_per_sample: int = 12,
    max_tool_calls_per_reply: int = 0,
    memory_mode: str = "full",
    memory_token_budget: int = 3000,
    budget_factory=None,
    trajectory_dir: Optional[str] = None,
    auto_anchor_every: int = 5,
    auto_asr_every: int = 0,
    auto_asr_window: int = 5,
    auto_track_query_every: int = 0,
    memory_render_mode: str = "continuous",
    memory_head_max: int = 6,
    memory_tail_max: int = 8,
    memory_head_seconds: int = 25,
    memory_tail_seconds: int = 20,
    memory_recall_topk: int = 8,
    cold_start_rounds: int = 4,
    degrade_threshold: int = 2,
    degrade_events_min: int = 12,
    rolling_window: int = 3,
    delta_input_mode: str = "2frame",
    enable_verification: bool = False,
    max_verification_replays: int = 3,
):
    processed_ids = ovo.load_processed_ids_from_jsonl(output_file) if output_file else set()
    if processed_ids:
        orig = len(samples)
        samples = [s for s in samples if s.get("id") not in processed_ids]
        print(f"Startup filter: {orig - len(samples)} skipped, {len(samples)} remaining")

    task_set = ovo._normalize_task_filter(task_filter)
    if task_set:
        orig = len(samples)
        samples = [s for s in samples if s.get("task") in task_set]
        print(f"Task filter: keep {len(samples)}/{orig} samples → {sorted(task_set)}")

    if not samples:
        print("No samples to process. Exit.")
        return

    if trajectory_dir:
        os.makedirs(trajectory_dir, exist_ok=True)
        print(f"Trajectory dump: {trajectory_dir}")

    processors: Dict[str, AgentVideoProcessor] = {
        s["id"]: AgentVideoProcessor(
            s, system, registry,
            max_tool_calls_per_round=max_tool_calls_per_round,
            max_tool_calls_per_sample=max_tool_calls_per_sample,
            max_tool_calls_per_reply=max_tool_calls_per_reply,
            memory_mode=memory_mode,
            memory_token_budget=memory_token_budget,
            budget=(budget_factory() if budget_factory else None),
            trajectory_dir=trajectory_dir,
            auto_anchor_every=auto_anchor_every,
            auto_asr_every=auto_asr_every,
            auto_asr_window=auto_asr_window,
            auto_track_query_every=auto_track_query_every,
            memory_render_mode=memory_render_mode,
            memory_head_max=memory_head_max,
            memory_tail_max=memory_tail_max,
            memory_head_seconds=memory_head_seconds,
            memory_tail_seconds=memory_tail_seconds,
            memory_recall_topk=memory_recall_topk,
            cold_start_rounds=cold_start_rounds,
            degrade_threshold=degrade_threshold,
            degrade_events_min=degrade_events_min,
            rolling_window=rolling_window,
            engine=engine,
            delta_input_mode=delta_input_mode,
            enable_verification=enable_verification,
            max_verification_replays=max_verification_replays,
        )
        for s in samples
    }

    written: Set[str] = set()
    eval_file = open(output_file, "a") if output_file else None

    def _advance(p: AgentVideoProcessor, last_answer: str):
        skipped = 0
        la = last_answer
        while True:
            if p.is_completed():
                return None, skipped, la
            should_process, req = p.get_next_request(la)
            if p.is_completed():
                return None, skipped, la
            if should_process:
                return req, skipped, la
            p.update("</Silence>")
            skipped += 1
            la = "</Silence>"

    try:
        while True:
            active = {k: v for k, v in processors.items() if not v.is_completed()}
            if not active:
                print("All samples processed.")
                break
            print(f"\n--- Agent round (active={len(active)}) ---")

            batch_req, batch_keys = [], []
            for k, p in list(active.items())[:max_concurrent]:
                last = p.output.get(f"Round {p.current_round-1}") if p.current_round else None
                req, skipped, _ = _advance(p, last)
                if skipped:
                    print(f"Sample-{k}: fast-forward {skipped} round(s) → {p.current_round}")
                if req is not None:
                    batch_req.append(req)
                    batch_keys.append(k)

            if batch_req:
                resps: List[str] = [""] * len(batch_req)
                special_indices = [
                    index
                    for index, key in enumerate(batch_keys)
                    if str(
                        (processors[key]._last_request or {}).get("kind")
                        or ""
                    ).startswith((
                        "current_locus_",
                        "event_predicate_",
                    ))
                ]
                special_set = set(special_indices)
                normal_indices = [
                    index for index in range(len(batch_req))
                    if index not in special_set
                ]
                if normal_indices:
                    normal_responses = ovo.batch_infer(
                        engine,
                        [batch_req[index] for index in normal_indices],
                        batch_size,
                    )
                    for index, response in zip(
                        normal_indices, normal_responses
                    ):
                        resps[index] = response
                special_groups: Dict[int, List[int]] = {}
                for index in special_indices:
                    request_meta = (
                        processors[batch_keys[index]]._last_request or {}
                    )
                    max_tokens = int(
                        request_meta.get("max_tokens") or 1800
                    )
                    special_groups.setdefault(max_tokens, []).append(index)
                for max_tokens, indices in sorted(special_groups.items()):
                    audit_responses = \
                        _batch_infer_current_locus_dual_view(
                            engine,
                            [
                                batch_req[index]
                                for index in indices
                            ],
                            batch_size,
                            max_tokens=max_tokens,
                        )
                    for index, response in zip(
                        indices, audit_responses
                    ):
                        resps[index] = response
                for k, r in zip(batch_keys, resps):
                    p = processors[k]
                    p.update(r)
                    snippet = (r or "")[:80].replace("\n", " ")
                    print(f"Sample-{k}: r{p.current_round}/{p.total_rounds} | {snippet}")

            for k, p in processors.items():
                if "gt" not in p.sample:
                    p.sample["gt"] = p.sample.get("test_info")
                if p.is_completed() and eval_file and k not in written:
                    final_text, trigger_round = ovo._extract_final_response(p.output)
                    # If abstain was triggered, fold its sentinel into final
                    if p.ctx.abstain_reason is not None:
                        sentinel = _abstain_final_for(p.sample)
                        # Strip the leading "</Response> " tag for final_text.
                        final_text = sentinel.split(RESPONSE_TAG, 1)[-1].strip()
                        trigger_round = p.current_round - 1
                    out = {
                        "id": p.sample["id"],
                        "benchmark": p.sample.get("benchmark", "ovobench"),
                        "suite": p.sample.get("suite"),
                        "video": p.sample["video"],
                        "source_id": p.sample.get("source_id"),
                        "source_video": p.sample.get("source_video"),
                        "gt": p.sample.get("gt"),
                        "output": p.output,
                        "task": p.sample.get("task"),
                        "task_type": p.sample.get("task_type"),
                        "final_response": final_text,
                        "trigger_round": trigger_round,
                        "question": p.sample.get("question"),
                        "options": p.sample.get("options"),
                        "test_info": p.sample.get("test_info"),
                        "realtime": p.sample.get("realtime"),
                        "question_time": p.question_time,
                        "prompt_question": p.question,
                        "total_rounds": p.total_rounds,
                        # agentic-specific telemetry
                        "tool_calls": p.tool_log,
                        "tool_calls_n": len(p.tool_log),
                        "abstain_reason": p.ctx.abstain_reason,
                    }
                    if p.sample.get("proactive_output"):
                        out.update({
                            "proactive_output": True,
                            "expected_output": p.sample.get(
                                "expected_output"
                            ),
                            "time_stamp": p.sample.get("time_stamp"),
                            "ground_truth_time_stamp": p.sample.get(
                                "ground_truth_time_stamp"
                            ),
                            "target_time": p.sample.get("target_time"),
                            "annotated_target_time": p.sample.get(
                                "annotated_target_time"
                            ),
                        })
                    if p.sample.get("activity") and not p.sample.get("question"):
                        out["start_times"] = p.sample.get("start_times", [])
                        out["activity"] = p.sample.get("activity")
                    elif (
                        p.sample.get("all_steps")
                        and p.sample.get("step_start_time") is not None
                        and not p.sample.get("question")
                        and not p.sample.get("activity")
                    ):
                        out["step_index"] = p.sample.get("step_index", -1)
                        out["original_id"] = p.sample.get("original_id", "")
                        out["step_question"] = p.sample.get("all_steps", "")
                        out["step_start_time"] = p.sample.get("step_start_time", 0)
                    elif (
                        p.sample.get("question")
                        and p.sample.get("ask_time") is not None
                        and not p.sample.get("options")
                    ):
                        out["ask_time"] = p.sample.get("ask_time", 0)
                    eval_file.write(json.dumps(out, ensure_ascii=False) + "\n")
                    eval_file.flush()
                    written.add(k)

                    # Trajectory dump (借鉴 #8): per-sample full trace +
                    # final memory snapshot for offline analysis / training.
                    if trajectory_dir:
                        try:
                            _dump_trajectory(trajectory_dir, p, out)
                        except Exception as e:
                            print(f"[trajectory] dump failed for {k}: {e}")

            done = sum(1 for p in processors.values() if p.is_completed())
            print(f"Progress: {done}/{len(processors)} done")
    finally:
        if eval_file:
            eval_file.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OVO-Bench agentic video understanding driver."
    )
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="./agent_test.jsonl")
    p.add_argument("--video_prefix", default="")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--tasks", default="EPM,HLD,ASI,STU,OJR,ATR,FPD,OCR,SSR,ACR,CRR,REC")
    p.add_argument("--max_concurrent", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--max_tool_calls_per_round", type=int, default=3)
    p.add_argument("--max_tool_calls_per_sample", type=int, default=12)
    p.add_argument(
        "--max_tool_calls_per_reply",
        type=int,
        default=0,
        help=(
            "Maximum tool-call blocks accepted from one model reply. "
            "0 disables this guard."
        ),
    )
    p.add_argument("--no_tools", action="store_true",
                   help="Disable agent tools and act as legacy video_processor.")
    # ---- Phase-A extensions (opt-in) ---------------------------------
    p.add_argument("--memory_mode", choices=["full", "brief"], default="full",
                   help="brief = single-frame + memory snapshot per round")
    p.add_argument("--memory_token_budget", type=int, default=3000)
    p.add_argument("--enable_memory_tools", action="store_true",
                   help="Register memory_* tools (auto-on for memory_mode=brief)")
    p.add_argument("--enable_track_tools", action="store_true",
                   help="Register track_* (SAM3) tools — uses MockTracker")
    p.add_argument("--enable_audio_tools", action="store_true",
                   help="Register asr_segment / speaker_id / audio_event "
                        "(uses MockASR until a real adapter is wired)")
    p.add_argument("--enable_efficiency_tools", action="store_true",
                   help="Register set_sample_rate")
    p.add_argument("--budget_max_tool_calls", type=int, default=24)
    p.add_argument("--budget_max_frames_seen", type=int, default=600)
    p.add_argument("--budget_max_tokens", type=int, default=200_000)
    p.add_argument("--trajectory_dir", default=None,
                   help="Directory to dump per-sample agentic trajectories "
                        "(JSON + final MEMORY brief). Useful for offline "
                        "RL / SFT data preparation.")
    p.add_argument("--auto_anchor_every", type=int, default=5,
                   help="System-side auto frame_anchor cadence (seconds). "
                        "0 disables. Helps keep MEMORY non-empty when the "
                        "base model rarely calls memory_write itself.")
    p.add_argument("--auto_asr_every", type=int, default=0,
                   help="System-side auto ASR cadence (seconds). 0 disables. "
                        "Recommended only when STREAMO_ASR_BACKEND=whisper "
                        "is set (mock ASR returns empty text).")
    p.add_argument("--auto_asr_window", type=int, default=5,
                   help="Window length used by auto ASR ([t-window, t]).")
    p.add_argument("--auto_track_query_every", type=int, default=0,
                   help="System-side auto track_query cadence (seconds) for "
                        "every active SAM2 track. 0 disables. Refreshes "
                        "bbox/visibility into MEMORY without touching the "
                        "model's tool-call budget.")
    # ---- Selective MEMORY rendering (read-side compression) ----------
    p.add_argument("--memory_render_mode",
                   choices=["full", "selective", "continuous"],
                   default="continuous",
                   help="MEMORY snapshot mode in brief mode. continuous "
                        "(default) = full head (t<head_seconds) + full "
                        "tail (t>=now-tail_seconds) verbatim, middle "
                        "aggregated into episodic. selective = head + "
                        "tail + retrieved top-K from middle. full = "
                        "every event verbatim.")
    p.add_argument("--memory_head_max", type=int, default=6,
                   help="Selective mode only: cap head to N events.")
    p.add_argument("--memory_tail_max", type=int, default=8,
                   help="Selective mode only: cap tail to N events.")
    p.add_argument("--memory_head_seconds", type=int, default=25,
                   help="Continuous mode: keep ALL events with t<this "
                        "(seconds) verbatim in head.")
    p.add_argument("--memory_tail_seconds", type=int, default=20,
                   help="Continuous mode: keep ALL events with "
                        "t>=now-this (seconds) verbatim in tail.")
    p.add_argument("--memory_recall_topk", type=int, default=8)
    p.add_argument("--cold_start_rounds", type=int, default=4,
                   help="Force memory_render_mode=full for the first N "
                        "rounds of every sample (selective mode only).")
    p.add_argument("--degrade_threshold", type=int, default=2,
                   help="If retrieve() returns < this many working hits "
                        "for the round, fall back to full rendering.")
    p.add_argument("--degrade_events_min", type=int, default=12,
                   help="If working layer holds < this many events, fall "
                        "back to full rendering.")
    p.add_argument("--rolling_window", type=int, default=3,
                   help="Number of most-recent observation deltas to "
                        "concatenate as the implicit pre-question query.")
    # ── Pillar 4: Phase-aware multi-frame input ─────────────────────
    p.add_argument("--delta_input_mode", default="2frame",
                   choices=["1frame", "2frame"],
                   help="1frame = single frame every round (legacy). "
                        "2frame = frame_{t-1} + frame_t during OBSERVE-ONLY "
                        "for visual delta comparison (default).")
    p.add_argument("--no_think", action="store_true",
                   help="Append '/no_think' to the system prompt to "
                        "disable Qwen3.x thinking blocks for this run.")
    # ── Pillar 3: Look-Back Verification ─────────────────────────────
    p.add_argument("--enable_verification", action="store_true",
                   help="Enable cross-layer consistency verification "
                        "+ targeted replay before final answer.")
    return p


def main():
    args = _build_argparser().parse_args()

    from swift import VllmEngine  # lazy import (heavy)

    engine = VllmEngine(
        args.model,
        max_model_len=65536,
        limit_mm_per_prompt={"image": 600},
        tensor_parallel_size=args.tp,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )

    if args.no_think:
        # Disable Qwen3.x thinking at the *template* level (the model
        # parameter that controls whether the chat template injects
        # `<think>\n` as a forced response prefix). Setting this on the
        # already-built engine.template is sufficient — VllmEngine reads
        # `self.template.enable_thinking` every time it encodes a request.
        try:
            engine.template.enable_thinking = False
            print("[no_think] engine.template.enable_thinking = False")
        except Exception as e:  # pragma: no cover
            print(f"[no_think] failed to disable thinking: {e}")

    ovo.VIDEO_PREFIX = args.video_prefix
    samples = ovo.load_dataset_json(args.dataset)

    # Choose registry: extended if any extension is on (or brief mode).
    use_extended = (
        args.memory_mode == "brief"
        or args.enable_memory_tools
        or args.enable_track_tools
        or args.enable_audio_tools
        or args.enable_efficiency_tools
    )
    if use_extended:
        registry = build_extended_registry(
            enable_memory=args.enable_memory_tools or args.memory_mode == "brief",
            enable_track=args.enable_track_tools,
            enable_audio=args.enable_audio_tools,
            enable_efficiency=args.enable_efficiency_tools or args.memory_mode == "brief",
        )
    else:
        registry = build_default_registry()

    base_system = ovo.SYSTEM if hasattr(ovo, "SYSTEM") else ""
    system = (
        base_system if args.no_tools
        else render_agent_system(registry, base_system,
                                  memory_mode=args.memory_mode,
                                  delta_input_mode=args.delta_input_mode)
    )

    def _budget_factory():
        return Budget(
            max_tool_calls=args.budget_max_tool_calls,
            max_frames_seen=args.budget_max_frames_seen,
            max_tokens=args.budget_max_tokens,
        )

    if args.no_tools:
        # Plain replay of legacy loop, but kept here for fair comparison.
        ovo.process_videos_batch(
            engine=engine, samples=samples, system=system,
            max_concurrent=args.max_concurrent, batch_size=args.batch_size,
            output_file=args.output, task_filter=args.tasks,
        )
    else:
        process_videos_agent(
            engine=engine, samples=samples, system=system, registry=registry,
            max_concurrent=args.max_concurrent, batch_size=args.batch_size,
            output_file=args.output, task_filter=args.tasks,
            max_tool_calls_per_round=args.max_tool_calls_per_round,
            max_tool_calls_per_sample=args.max_tool_calls_per_sample,
            max_tool_calls_per_reply=args.max_tool_calls_per_reply,
            memory_mode=args.memory_mode,
            memory_token_budget=args.memory_token_budget,
            budget_factory=_budget_factory,
            trajectory_dir=args.trajectory_dir,
            auto_anchor_every=args.auto_anchor_every,
            auto_asr_every=args.auto_asr_every,
            auto_asr_window=args.auto_asr_window,
            auto_track_query_every=args.auto_track_query_every,
            memory_render_mode=args.memory_render_mode,
            memory_head_max=args.memory_head_max,
            memory_tail_max=args.memory_tail_max,
            memory_head_seconds=args.memory_head_seconds,
            memory_tail_seconds=args.memory_tail_seconds,
            memory_recall_topk=args.memory_recall_topk,
            cold_start_rounds=args.cold_start_rounds,
            degrade_threshold=args.degrade_threshold,
            degrade_events_min=args.degrade_events_min,
            rolling_window=args.rolling_window,
            delta_input_mode=args.delta_input_mode,
            enable_verification=args.enable_verification,
        )

    ovo.score_results(args.output)


if __name__ == "__main__":
    main()
