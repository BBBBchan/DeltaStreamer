"""Typed relation audit for planner-routed open-question answerability."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


_THINK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>|<think\b[^>]*>.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_TASK_LABEL_RE = re.compile(
    r"\b(?:REC|SSR|CRR|HLD|EPM|ASI|OCR|ATR|STU|FPD|OJR|ACR)\b"
)
_INVALID_ANSWERS = {
    "",
    "unknown",
    "uncertain",
    "none",
    "none observed",
    "not shown",
    "somewhere",
    "not enough information",
    "cannot determine",
}
_RELATION_FAMILIES = {
    "target_relative_transition",
    "entity_encounter",
    "direct_reveal",
    "intent_followthrough",
    "destination_reveal",
    "count_reveal",
    "outcome_reveal",
}
_INSTANCE_FLAGS = (
    "hard_scene_cut_before_instance",
    "anchor_visible",
    "same_physical_anchor_track",
    "target_visible",
    "same_physical_target_track",
    "target_zone_reached_or_passed",
    "mere_approach_only",
    "distinct_target_from_anchor",
    "target_is_anchor_duplicate",
    "encounter_or_notice_boundary_visible",
    "direct_view_reveal_visible",
    "segment_viewpoint_aligned",
    "background_only",
    "own_vehicle_exterior_only",
    "goal_directed_followthrough_visible",
    "multi_step_followthrough_visible",
    "destination_boundary_visible",
    "count_state_visible",
    "relation_or_outcome_boundary_visible",
    "direct_answer_reveal_visible",
    "color_only_identity_conflict",
    "non_color_identity_contradiction",
    "direct_pixel_evidence",
    "explicit_contradiction",
)


def _clean(value: Any, limit: int = 1200) -> str:
    return " ".join(str(value or "").split())[:limit]


def _sanitize_task_labels(value: Any, limit: int = 1200) -> str:
    text = _clean(value, limit)
    replacements = (
        (r"\bSSR\s*/\s*CRR(?:-style)?\b", "event-state"),
        (r"\b(?:SSR|CRR)(?:-style)?\b", "answerability"),
        (r"\bREC(?:-style)?\b", "repetition-count"),
        (r"\bHLD(?:-style)?\b", "answerability"),
        (r"\bOCR(?:-style)?\b", "on-screen-text"),
        (r"\b(?:EPM|ASI|STU|OJR|ATR|FPD|ACR)(?:-style)?\b", "visual"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return _clean(text, limit)


def _json_object(response: str) -> Optional[Dict[str, Any]]:
    visible = _THINK_RE.sub("", response or "").strip()
    decoder = json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    index = 0
    while index < len(visible):
        start = visible.find("{", index)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(visible[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            found.append(value)
        index = start + max(1, consumed)
    return found[-1] if found else None


def _normalize_counterevidence(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return _clean(value, 1800) or None
    if isinstance(value, list) and len(value) <= 6:
        items = [_clean(item, 500) for item in value]
        if any(not item for item in items):
            return None
        return " ".join(items) if items else "none observed"
    return None


def compile_relation_contract(
    question: str,
    plan: Dict[str, Any],
) -> Dict[str, str]:
    """Compile relation semantics from query-visible text after planning."""
    question_text = _sanitize_task_labels(question, 1800)
    if not question_text:
        raise ValueError("visible question missing")
    plan_text = " ".join(
        _sanitize_task_labels(plan.get(key), 600)
        for key in ("need", "info_type", "evidence_locus", "strategy")
    )
    raw = f"{question_text} {plan_text}"
    lowered_question = question_text.lower()
    lowered = raw.lower()
    if re.search(
        r"\b(?:how many|number of (?:people|persons|objects|times))\b",
        lowered_question,
    ):
        family = "count_reveal"
        rule = (
            "Require a directly visible terminal count state after the named "
            "boundary; a partial or still-changing group is insufficient."
        )
    elif re.search(
        r"\b(?:want|intend|intention|purpose|why)\b",
        lowered_question,
    ):
        family = "intent_followthrough"
        rule = (
            "Require a visible goal-directed multi-step followthrough in the "
            "current anchor instance; arbitrary motion is insufficient."
        )
    elif re.search(
        r"\b(?:destination|where (?:do|does|did|are|is|they|he|she)|"
        r"next vehicle)\b",
        lowered_question,
    ):
        family = "destination_reveal"
        rule = (
            "Require a destination or next-vehicle boundary that uniquely "
            "reveals one concrete value, not direction or scenery alone."
        )
    elif re.search(
        r"\b(?:come across|first creature|first animal|who .*meet|"
        r"who .*find|what .*creature|encounter)\b",
        lowered_question,
    ):
        family = "entity_encounter"
        rule = (
            "Require the first distinct compatible target on the active path "
            "or a linked notice, proximity, stop, or pass boundary."
        )
    elif (
        re.search(
            r"\b(?:see|seen|find|looking at|what(?:'s| is) in)\b",
            lowered_question,
        )
        or (
            re.search(r"\b(?:window|outside|mirror|box|room)\b", lowered)
            and re.search(r"\b(?:what|who)\b", lowered_question)
        )
    ):
        family = "direct_reveal"
        rule = (
            "Require the requested target or value to be directly visible in "
            "the linked viewing episode and correct viewpoint."
        )
    elif re.search(
        r"\bwhat (?:action )?(?:does|do|did).*\b(?:to|with)\b",
        lowered_question,
    ):
        family = "target_relative_transition"
        rule = (
            "Require a completed target-relative boundary beyond approach; "
            "reach, pass, touch, enter, transfer, or social engagement may "
            "qualify without mandatory contact."
        )
    else:
        family = "outcome_reveal"
        rule = (
            "Require a concrete visible outcome boundary in the current "
            "anchor instance that uniquely supports one answer."
        )
    latest_instance_policy = (
        "Select the latest compatible anchor instance, not the strongest or "
        "earliest instance. A hard scene cut begins a new instance."
    )
    return {
        "relation_family": family,
        "subject_need": _sanitize_task_labels(plan.get("need"), 900),
        "completion_rule": rule,
        "latest_instance_policy": latest_instance_policy,
    }


def build_prompt(
    *,
    question: str,
    plan: Dict[str, Any],
    relation: Dict[str, str],
    timeline_description: str,
    dag_evidence: str,
    image_count: int,
) -> str:
    """Build a label-blind typed ledger prompt over pixels and DAG paths."""
    plan_visible = {
        key: _sanitize_task_labels(plan.get(key), 800)
        for key in (
            "need",
            "info_type",
            "evidence_locus",
            "answerability",
            "strategy",
        )
        if _sanitize_task_labels(plan.get(key), 800)
    }
    relation_visible = {
        str(key): _sanitize_task_labels(value, 1800)
        for key, value in relation.items()
    }
    timeline_visible = _sanitize_task_labels(timeline_description, 1000)
    dag_visible = _sanitize_task_labels(dag_evidence, 7000)
    return (
        "Build one TYPED RELATION LEDGER after a streaming-video planner has "
        "received an open original question. Use only the chronological "
        "timestamped ORIGINAL PIXELS supplied below and the timestamped "
        "bidirectional memory paths. Do not answer Yes or No. Do not use "
        "dataset IDs, hidden answers, or benchmark metadata.\n\n"
        f"Visible original question: {_sanitize_task_labels(question, 1800)}\n"
        f"Planner output: {json.dumps(plan_visible, ensure_ascii=False)}\n"
        f"Compiled visible relation: "
        f"{json.dumps(relation_visible, ensure_ascii=False)}\n"
        f"Pixel window: {timeline_visible}\n\n"
        "Target-matched memory paths (each path is a cross-layer linked "
        "proposal, not proof by itself):\n"
        f"{dag_visible}\n\n"
        "Enumerate every compatible anchor instance in chronological order, "
        "up to three. A hard scene cut begins a new instance. An instance "
        "must be listed when its anchor role is directly visible even if the "
        "requested target or outcome has not appeared. Track physical roles "
        "by body/object shape, position, motion, shot continuity, and scene "
        "geometry. Clothing or vehicle color differences alone cannot break "
        "an otherwise continuous physical track; non-color body, role, path, "
        "or geometry conflicts do break it.\n\n"
        "For a target-relative relation, mere approach is setup; reaching or "
        "passing the target zone, a transfer, touch, entry, or linked social "
        "engagement can complete the boundary. For a person encounter, the "
        "target must be physically distinct from the anchor and linked by "
        "same-shot active-path geometry or a shot/reverse-shot, gaze, or path "
        "continuity reveal; an anchor duplicate or unrelated bystander fails. "
        "For a direct reveal, require the requested value in the relevant "
        "viewpoint; generic background, an own-vehicle exterior, titles, or "
        "direction alone fail. For intent, require goal-directed multi-step "
        "followthrough. For destination, count, and general outcomes, require "
        "the corresponding concrete boundary and one specific candidate "
        "answer. Later disappearance does not erase a boundary inside the "
        "same instance, but a later compatible anchor instance supersedes an "
        "older one.\n\n"
        "DAG path IDs may establish temporal linkage but never replace direct "
        "pixel evidence. For each instance, dag_path_ids must be a JSON list "
        "of zero to eight path labels such as path#1. Use timestamps within "
        "the supplied pixel window. Use `none observed` for an absent "
        "candidate or counterevidence.\n\n"
        "Return ONLY one JSON object with verifier_type=`typed_relation_ledger`, "
        "relation_family exactly equal to the compiled value, instances "
        "(zero to three objects), evidence, and counterevidence. Every "
        "instance object must contain instance_index (zero-based chronological "
        "integer), before_t, after_t, candidate_answer, dag_path_ids, evidence, "
        "counterevidence, and all of these JSON booleans: "
        + ", ".join(_INSTANCE_FLAGS)
        + ". No final answer, prose, or Markdown.\n"
        + "\n".join("<image>" for _ in range(image_count))
    )


def _relation_match(family: str, flags: Dict[str, bool]) -> bool:
    if family == "target_relative_transition":
        return bool(
            flags["target_visible"]
            and flags["same_physical_target_track"]
            and flags["target_zone_reached_or_passed"]
            and flags["relation_or_outcome_boundary_visible"]
            and not flags["mere_approach_only"]
        )
    if family == "entity_encounter":
        return bool(
            flags["target_visible"]
            and flags["same_physical_target_track"]
            and flags["distinct_target_from_anchor"]
            and not flags["target_is_anchor_duplicate"]
            and flags["encounter_or_notice_boundary_visible"]
        )
    if family == "direct_reveal":
        return bool(
            flags["target_visible"]
            and flags["direct_view_reveal_visible"]
            and flags["segment_viewpoint_aligned"]
            and not flags["background_only"]
            and not flags["own_vehicle_exterior_only"]
        )
    if family == "intent_followthrough":
        return bool(
            flags["goal_directed_followthrough_visible"]
            and flags["multi_step_followthrough_visible"]
        )
    if family == "destination_reveal":
        return bool(flags["destination_boundary_visible"])
    if family == "count_reveal":
        return bool(flags["count_state_visible"])
    return bool(flags["relation_or_outcome_boundary_visible"])


def parse_response(
    response: str,
    *,
    relation: Dict[str, str],
    question_time: float,
    window_seconds: float = 47.875,
) -> Optional[Dict[str, Any]]:
    """Parse a typed ledger and compute answerability programmatically."""
    payload = _json_object(response)
    family = _clean(payload.get("relation_family"), 100).lower() \
        if payload else ""
    if (
        payload is None
        or payload.get("verifier_type") != "typed_relation_ledger"
        or family != relation.get("relation_family")
        or family not in _RELATION_FAMILIES
    ):
        return None
    raw_instances = payload.get("instances")
    if not isinstance(raw_instances, list) or len(raw_instances) > 3:
        return None
    top_evidence = _clean(payload.get("evidence"), 2400)
    top_counter = _normalize_counterevidence(payload.get("counterevidence"))
    if not top_evidence or top_counter is None:
        return None

    instances: List[Dict[str, Any]] = []
    window_start = max(0.0, float(question_time) - float(window_seconds))
    for expected_index, raw in enumerate(raw_instances):
        if (
            not isinstance(raw, dict)
            or raw.get("instance_index") != expected_index
        ):
            return None
        before = raw.get("before_t")
        after = raw.get("after_t")
        if (
            isinstance(before, bool)
            or isinstance(after, bool)
            or not isinstance(before, (int, float))
            or not isinstance(after, (int, float))
            or not (
                window_start
                <= float(before)
                <= float(after)
                <= float(question_time)
            )
        ):
            return None
        candidate = _clean(raw.get("candidate_answer"), 900)
        evidence = _clean(raw.get("evidence"), 1800)
        counter = _normalize_counterevidence(raw.get("counterevidence"))
        path_ids = raw.get("dag_path_ids")
        if (
            not evidence
            or counter is None
            or not isinstance(path_ids, list)
            or len(path_ids) > 8
            or any(
                not re.fullmatch(r"path#\d+", _clean(path_id, 40))
                for path_id in path_ids
            )
        ):
            return None
        flags: Dict[str, bool] = {}
        for key in _INSTANCE_FLAGS:
            if not isinstance(raw.get(key), bool):
                return None
            flags[key] = raw[key]
        anchor_compatible = bool(
            flags["anchor_visible"]
            and flags["same_physical_anchor_track"]
            and flags["direct_pixel_evidence"]
            and not flags["non_color_identity_contradiction"]
            and not flags["explicit_contradiction"]
        )
        relation_match = _relation_match(family, flags)
        specific_answer = bool(
            candidate.lower() not in _INVALID_ANSWERS
            and flags["direct_answer_reveal_visible"]
        )
        instances.append({
            "instance_index": expected_index,
            "before_t": float(before),
            "after_t": float(after),
            "candidate_answer": candidate,
            "dag_path_ids": [_clean(path_id, 40) for path_id in path_ids],
            **flags,
            "anchor_compatible": anchor_compatible,
            "relation_match": relation_match,
            "specific_answer": specific_answer,
            "evidence": evidence,
            "counterevidence": counter,
        })
    if any(
        later["before_t"] < earlier["before_t"]
        for earlier, later in zip(instances, instances[1:])
    ):
        return None

    compatible = [row for row in instances if row["anchor_compatible"]]
    latest = max(
        compatible,
        key=lambda row: (row["before_t"], row["after_t"]),
    ) if compatible else None
    positive = bool(
        latest
        and latest["relation_match"]
        and latest["specific_answer"]
    )
    all_text = " ".join([
        top_evidence,
        top_counter,
        *(
            value
            for row in instances
            for value in (
                row["candidate_answer"],
                row["evidence"],
                row["counterevidence"],
            )
        ),
    ])
    if _TASK_LABEL_RE.search(all_text):
        return None
    node = None
    if positive:
        parent_ids = latest["dag_path_ids"] or ["current_pixel_anchor"]
        node = {
            "node_type": family,
            "parent_anchor_ids": parent_ids,
            "reverse_anchor_ids": parent_ids,
            "before_t": latest["before_t"],
            "after_t": latest["after_t"],
            "direct_pixel_evidence": True,
            "reliability_verified": True,
        }
    return {
        "verifier_type": "typed_relation_ledger",
        "relation_family": family,
        "instances": instances,
        "compatible_instance_count": len(compatible),
        "selected_latest_instance_index": (
            latest["instance_index"] if latest else -1
        ),
        "candidate_answer": latest["candidate_answer"] if latest else "",
        "positive": positive,
        "answer": "Yes" if positive else "No",
        "evidence": top_evidence,
        "counterevidence": top_counter,
        "typed_event_node": node,
    }
