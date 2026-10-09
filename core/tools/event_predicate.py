"""Planner-after-query event predicate verification for production runs."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


_THINK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>|<think\b[^>]*>.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_PLACEHOLDER_RE = re.compile(
    r"(?:exact target|specific target|target predicate|target object|"
    r"temporal anchor|specific evidence|distinguishing cue)",
    re.IGNORECASE,
)
_TASK_LABEL_RE = re.compile(
    r"\b(?:REC|SSR|CRR|HLD|EPM|ASI|OCR|ATR|STU|FPD|OJR|ACR)\b"
)
_HEDGED_RE = re.compile(
    r"\b(?:consistent with|appears?|likely|suggests?|could be|might be|"
    r"possibly|typical)\b",
    re.IGNORECASE,
)
_EVENT_STATUSES = {
    "exact_action",
    "result_state_only",
    "related_or_setup",
    "contradicted",
    "uncertain",
}


def clean(value: Any, limit: int = 900) -> str:
    return " ".join(str(value or "").split())[:limit]


def is_event_state_schema(sample: Optional[Dict[str, Any]]) -> bool:
    sample = sample or {}
    return bool(
        sample.get("all_steps")
        and sample.get("step_start_time") is not None
        and not sample.get("question")
        and not sample.get("activity")
    )


def normalize_yes_no_response(response: str) -> Optional[str]:
    visible = _THINK_RE.sub("", response or "")
    if "</Response>" in visible:
        visible = visible.rsplit("</Response>", 1)[1]
    match = re.search(r"\b(yes|no)\b", visible, re.IGNORECASE)
    return match.group(1).capitalize() if match else None


def route_event_predicate(
    plan: Optional[Dict[str, str]],
    sample: Optional[Dict[str, Any]],
    *,
    compiled_locus: str,
) -> bool:
    if os.environ.get("STREAMO_ENABLE_EVENT_PREDICATE_AUDIT", "0") != "1":
        return False
    if not plan or not is_event_state_schema(sample):
        return False
    if compiled_locus not in {"past", "mixed"}:
        return False
    text = " ".join(
        str(plan.get(key) or "")
        for key in ("need", "info_type", "meta", "strategy")
    ).lower()
    return bool(re.search(
        r"\b(?:event|action|happen|occur|start|progress|yes/no|state)\b",
        text,
    ))


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


def build_contract_prompt(sample: Dict[str, Any], plan: Dict[str, str]) -> str:
    target = clean(sample.get("all_steps"), 1000)
    plan_text = json.dumps({
        key: plan.get(key)
        for key in (
            "need", "pov", "info_type", "evidence_locus",
            "answerability", "meta", "strategy",
        )
        if plan.get(key)
    }, ensure_ascii=False)
    return (
        "Compile an EVIDENCE CONTRACT after a streaming-video planner has "
        "received one visible event-state query. Do not answer Yes or No. "
        "No benchmark-only metadata, ID, or historical answer is supplied."
        "\n\n"
        f"Visible event query: {target}\n"
        f"Planner output: {plan_text}\n\n"
        "Separate the exact target, predicate/action, direct object or defining "
        "result, actor/reference, temporal anchor, and coordinate frame. List "
        "near-matches that must not count: setup, prerequisites, an already "
        "achieved result state, a related action, a different actor/object, or "
        "a repeated similar stage. Workflow order is not evidence.\n\n"
        "Return ONLY one JSON object with target, predicate, object_or_result, "
        "actor_or_reference, temporal_anchor, required_frame, "
        "direct_evidence_rule, forbidden_near_matches (1-6 strings), and "
        "retrieval_terms (2-12 concrete strings). No prose or Markdown."
    )


def parse_contract(response: str) -> Optional[Dict[str, Any]]:
    payload = _json_object(response)
    if payload is None:
        return None
    fields: Dict[str, str] = {}
    for key, limit in (
        ("target", 500),
        ("predicate", 500),
        ("object_or_result", 500),
        ("actor_or_reference", 500),
        ("temporal_anchor", 700),
        ("required_frame", 500),
        ("direct_evidence_rule", 1000),
    ):
        value = clean(payload.get(key), limit)
        if not value or _PLACEHOLDER_RE.fullmatch(value):
            return None
        fields[key] = value
    forbidden = payload.get("forbidden_near_matches")
    terms = payload.get("retrieval_terms")
    if (
        not isinstance(forbidden, list)
        or not 1 <= len(forbidden) <= 6
        or not isinstance(terms, list)
        or not 2 <= len(terms) <= 12
    ):
        return None
    forbidden_text = [clean(value, 500) for value in forbidden]
    term_text = [clean(value, 120) for value in terms]
    if any(not value for value in [*forbidden_text, *term_text]):
        return None
    if _TASK_LABEL_RE.search(" ".join([
        *fields.values(), *forbidden_text, *term_text,
    ])):
        return None
    return {
        **fields,
        "forbidden_near_matches": forbidden_text,
        "retrieval_terms": term_text,
    }


def contract_query(contract: Dict[str, Any]) -> str:
    return " ".join([
        contract["target"],
        contract["predicate"],
        contract["object_or_result"],
        contract["actor_or_reference"],
        contract["temporal_anchor"],
        *contract["retrieval_terms"],
    ])


def dag_proposals(
    graph: Any,
    query: str,
    *,
    question_time: int,
    maximum: int = 3,
    minimum_gap: float = 7.0,
) -> List[Dict[str, Any]]:
    try:
        paths = graph.retrieve_backtrack(
            query=query,
            top_k=12,
            current_t=int(question_time),
        ) if graph is not None else []
    except Exception:
        paths = []
    ranked: List[Dict[str, Any]] = []
    for path_index, path in enumerate(paths):
        timestamps = [
            float(node.t)
            for node in path.nodes
            if node.t is not None and 0 <= float(node.t) <= question_time
        ]
        if not timestamps:
            continue
        ranked.append({
            "t": timestamps[0],
            "score": float(path.score),
            "seed_layer": str(path.nodes[0].layer),
            "seed_id": int(path.nodes[0].node_id),
            "path_index": path_index,
            "path": [
                {
                    "layer": str(node.layer),
                    "id": int(node.node_id),
                    "t": node.t,
                    "kind": str(node.kind or ""),
                    "text": clean(node.text, 700),
                }
                for node in path.nodes[:6]
            ],
        })
    selected: List[Dict[str, Any]] = []
    for row in sorted(
        ranked,
        key=lambda item: (-item["score"], item["t"], item["path_index"]),
    ):
        if any(abs(row["t"] - old["t"]) < minimum_gap for old in selected):
            continue
        selected.append(row)
        if len(selected) >= maximum - 1:
            break
    if not any(
        abs(float(question_time) - row["t"]) < minimum_gap for row in selected
    ):
        selected.append({
            "t": float(question_time),
            "score": 0.0,
            "seed_layer": "query_time",
            "seed_id": None,
            "path_index": -1,
            "path": [],
        })
    return sorted(selected[:maximum], key=lambda item: item["t"])


def resolve_video(sample_id: str) -> Path:
    roots = (
        Path(os.environ.get("STREAMO_VIDEO_DIR", "data/OVO-Bench/chunked_videos_asi_full")),
        Path("data/OVO-Bench/chunked_videos"),
    )
    for root in roots:
        path = root / f"{sample_id}.mp4"
        if path.is_file():
            return path
    raise FileNotFoundError(f"video missing for {sample_id}")


def build_local_sheets(
    sample_id: str,
    proposals: Sequence[Dict[str, Any]],
    cache_root: str,
    *,
    fps: float = 4.0,
    radius: float = 4.0,
) -> List[Dict[str, Any]]:
    import cv2
    import numpy as np

    video_path = resolve_video(sample_id)
    capture = cv2.VideoCapture(str(video_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not capture.isOpened() or source_fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"video decoder did not open: {video_path}")
    duration = frame_count / source_fps
    output: List[Dict[str, Any]] = []
    for window_id, proposal in enumerate(proposals):
        center = min(
            max(0.0, float(proposal["t"])),
            max(0.0, duration - 1 / source_fps),
        )
        start = max(0.0, center - radius)
        end = min(duration - 1 / source_fps, center + radius)
        count = max(2, int(math.floor((end - start) * fps)) + 1)
        times = [min(end, start + index / fps) for index in range(count)]
        indices = [
            min(frame_count - 1, int(round(timestamp * source_fps)))
            for timestamp in times
        ]
        cache_dir = Path(cache_root) / sample_id / f"window_{window_id}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        panels = []
        for order, (timestamp, frame_index) in enumerate(zip(times, indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                raise RuntimeError(f"decode failed at t={timestamp:.2f}")
            panel = cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA)
            cv2.rectangle(panel, (0, 0), (125, 24), (0, 0, 0), -1)
            cv2.putText(
                panel, f"t={timestamp:.2f}s", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255),
                1, cv2.LINE_AA,
            )
            panels.append(panel)
        files: List[str] = []
        for start_index in range(0, len(panels), 16):
            group = list(panels[start_index:start_index + 16])
            blank = np.zeros_like(group[0])
            while len(group) < 16:
                group.append(blank.copy())
            sheet = np.vstack([
                np.hstack(group[row * 4:(row + 1) * 4])
                for row in range(4)
            ])
            path = cache_dir / f"sheet_{start_index // 16:02d}.jpg"
            if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                capture.release()
                raise RuntimeError(f"failed to write {path}")
            files.append(str(path))
        output.append({
            "window_id": window_id,
            "center_t": center,
            "times": times,
            "files": files,
            "proposal": dict(proposal),
        })
    capture.release()
    return output


def build_ledger_prompt(
    sample: Dict[str, Any],
    contract: Dict[str, Any],
    windows: Sequence[Dict[str, Any]],
) -> str:
    description = "\n".join(
        f"- W{window['window_id']}: center={window['center_t']:.2f}s, "
        f"visible={min(window['times']):.2f}s..{max(window['times']):.2f}s"
        for window in windows
    )
    image_count = sum(len(window["files"]) for window in windows)
    return (
        "Audit whether one EXACT EVENT is directly visible in any supplied "
        "chronological local window. Window times came from a memory DAG, but "
        "memory text and proposal timing are not evidence. Inspect timestamped "
        "pixels only. Do not answer from workflow order, setup, prerequisites, "
        "an already-achieved result state, a related action, or a repeated "
        "similar stage.\n\n"
        f"Visible event query: {clean(sample.get('all_steps'), 1000)}\n"
        f"Evidence contract: {json.dumps(contract, ensure_ascii=False)}\n"
        f"Windows:\n{description}\n\n"
        "For each window choose exact_action, result_state_only, "
        "related_or_setup, contradicted, or uncertain. exact_action requires "
        "exact_action_match, exact_object_result_match, and primary_event all "
        "true. It must show the target transition itself, not an already "
        "achieved state. Completion is not required when direct onset or "
        "in-progress pixels are visible. Finite windows cannot prove the event "
        "never happened.\n\n"
        "Return ONLY one JSON object with target_action, object_or_result, and "
        "windows (one entry per W in order). Each entry must contain window_id, "
        "status, exact_action_match, exact_object_result_match, primary_event "
        "(booleans), before_t, after_t, distinguishing_cue, and evidence. "
        "No benchmark labels, answer, prose, or Markdown.\n"
        + "\n".join("<image>" for _ in range(image_count))
    )


def normalize_window_id(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"\s*W(\d+)\s*", value, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def normalize_timestamp(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        match = re.fullmatch(
            r"\s*(-?\d+(?:\.\d+)?)\s*s?\s*", value, re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            return value if math.isfinite(value) else None
    return None


def parse_ledger(
    response: str,
    windows: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    payload = _json_object(response)
    if payload is None:
        return None
    target = clean(payload.get("target_action"), 500)
    object_result = clean(payload.get("object_or_result"), 500)
    entries = payload.get("windows")
    if (
        not target
        or not object_result
        or not isinstance(entries, list)
        or len(entries) != len(windows)
    ):
        return None
    parsed = []
    for expected, entry in zip(windows, entries):
        if not isinstance(entry, dict):
            return None
        window_id = normalize_window_id(entry.get("window_id"))
        if window_id != expected["window_id"]:
            return None
        status = clean(entry.get("status"), 80).lower()
        if status not in _EVENT_STATUSES:
            return None
        flags = {}
        for key in (
            "exact_action_match", "exact_object_result_match", "primary_event",
        ):
            if not isinstance(entry.get(key), bool):
                return None
            flags[key] = entry[key]
        if (status == "exact_action") != all(flags.values()):
            return None
        before_t = normalize_timestamp(entry.get("before_t"))
        after_t = normalize_timestamp(entry.get("after_t"))
        if (
            before_t is None
            or after_t is None
            or after_t < before_t
            or not min(expected["times"]) - 0.3
            <= before_t <= max(expected["times"]) + 0.3
            or not min(expected["times"]) - 0.3
            <= after_t <= max(expected["times"]) + 0.3
        ):
            return None
        cue = clean(entry.get("distinguishing_cue"), 900)
        evidence = clean(entry.get("evidence"), 1200)
        if not cue or not evidence or _PLACEHOLDER_RE.search(cue):
            return None
        parsed.append({
            "window_id": window_id,
            "status": status,
            **flags,
            "before_t": before_t,
            "after_t": after_t,
            "distinguishing_cue": cue,
            "evidence": evidence,
        })
    all_text = " ".join([
        target, object_result,
        *(value for row in parsed for value in (
            row["distinguishing_cue"], row["evidence"],
        )),
    ])
    if _TASK_LABEL_RE.search(all_text):
        return None
    direct = any(
        row["status"] == "exact_action"
        and not _HEDGED_RE.search(
            f"{row['distinguishing_cue']} {row['evidence']}"
        )
        for row in parsed
    )
    return {
        "target_action": target,
        "object_or_result": object_result,
        "windows": parsed,
        "exact_positive": any(
            row["status"] == "exact_action" for row in parsed
        ),
        "direct_exact_positive": direct,
    }


def image_paths(windows: Sequence[Dict[str, Any]]) -> List[str]:
    return [path for window in windows for path in window["files"]]
