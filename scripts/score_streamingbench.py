#!/usr/bin/env python3
"""Dependency-light coverage and metric audit for StreamingBench runs."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# 与官方 src/data/count.py 对齐：Proactive 主判定用 ±2s 时间窗 + 子串包含内容。
# 官方硬编码 ±2；这里保留 1/2/5 三档作报告（primary = 2）。
THRESHOLDS = (1, 2, 5)
PRIMARY_THRESHOLD = 2
_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE
)
_THINK_OPEN_TAIL_RE = re.compile(
    r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE
)


def strip_thinking(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    return _THINK_OPEN_TAIL_RE.sub("", cleaned)


def extract_final_response(output: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    final_text: Optional[str] = None
    trigger_round: Optional[int] = None
    for key, value in (output or {}).items():
        match = re.fullmatch(r"Round\s+(\d+)", str(key))
        if match is None or not value:
            continue
        visible = strip_thinking(str(value))
        if "</Response>" not in visible:
            continue
        trigger_round = int(match.group(1))
        final_text = visible.split("</Response>", 1)[1].strip()
    return final_text, trigger_round


def ensure_final_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    final, trigger = extract_final_response(record.get("output") or {})
    if final is not None:
        record["final_response"] = final
        record["trigger_round"] = trigger
    else:
        record.setdefault("final_response", None)
        record.setdefault("trigger_round", None)
    return record


def extract_option_letter(response: Optional[str]) -> Optional[str]:
    if response is None:
        return None
    text = str(response).strip()
    if len(text) == 1 and text.upper() in string.ascii_uppercase:
        return text.upper()
    match = re.match(r"\s*([A-Za-z])\s*[.)\s:：\-]", text)
    if match:
        return match.group(1).upper()
    match = re.search(
        r"(?:answer|option|choose|select|choice)\s+(?:is\s+)?"
        r"([A-Za-z])\b",
        text,
        re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


def normalize_output(text: Any) -> str:
    normalized = str(text or "").strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \t\r\n.,!?;:'\"`")


def result_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    workers = sorted(
        glob.glob(os.path.join(path, "streamingbench_*.worker*.jsonl"))
    )
    if workers:
        return workers
    return sorted(glob.glob(os.path.join(path, "streamingbench_*.jsonl")))


def load_records(path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    malformed = 0
    duplicates = 0
    files = result_files(path)
    for file_path in files:
        with open(file_path, encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                sample_id = record.get("id")
                if sample_id is None:
                    malformed += 1
                    continue
                key = str(sample_id)
                duplicates += int(key in records)
                records[key] = ensure_final_fields(record)
    return records, {
        "files": files,
        "malformed_lines": malformed,
        "duplicate_ids_last_wins": duplicates,
    }


def load_dataset(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    with open(path, encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise ValueError("dataset must be a JSON list")
    by_id = {str(sample["id"]): sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError(
            f"dataset has duplicate IDs: rows={len(samples)}, unique={len(by_id)}"
        )
    return samples, by_id


def mcq_correct(record: Dict[str, Any], sample: Dict[str, Any]) -> int:
    prediction = extract_option_letter(record.get("final_response"))
    return int(prediction is not None and prediction == str(sample["gt"]).upper())


def proactive_detail(
    record: Dict[str, Any],
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    response = record.get("final_response")
    expected = sample.get("expected_output", sample.get("gt"))
    norm_resp = normalize_output(response)
    norm_exp = normalize_output(expected)
    # 官方口径（count.py）：ground_truth_output in last_answer —— 子串包含。
    content_correct = bool(norm_exp) and norm_exp in norm_resp
    # 更严的参考口径：归一化后完全相等（保留供对比，不参与 primary）。
    content_correct_exact = norm_resp == norm_exp
    trigger = record.get("trigger_round")
    target = int(sample["target_time"])
    delay = int(trigger) - target if trigger is not None else None
    return {
        "content_correct": content_correct,
        "content_correct_exact": content_correct_exact,
        "trigger_round": trigger,
        "target_time": target,
        "delay_seconds": delay,
        "early": bool(delay is not None and delay < 0),
        "threshold_hits": {
            str(threshold): bool(
                content_correct
                and delay is not None
                and abs(delay) <= threshold
            )
            for threshold in THRESHOLDS
        },
    }


def accuracy_block(correct: int, total: int) -> Dict[str, Any]:
    return {
        "correct": correct,
        "wrong": total - correct,
        "total": total,
        "accuracy": 100.0 * correct / total if total else None,
    }


def summarize(
    records: Dict[str, Dict[str, Any]],
    samples: Iterable[Dict[str, Any]],
    source_meta: Dict[str, Any],
) -> Dict[str, Any]:
    sample_list = list(samples)
    expected = {str(sample["id"]) for sample in sample_list}
    completed = set(records) & expected
    per_suite_total: Counter[str] = Counter()
    per_suite_correct: Counter[str] = Counter()
    per_task_total: Counter[str] = Counter()
    per_task_correct: Counter[str] = Counter()
    proactive_total = 0
    proactive_content = 0
    proactive_content_exact = 0
    proactive_thresholds: Counter[int] = Counter()
    proactive_early = 0
    proactive_delays: List[int] = []
    details: List[Dict[str, Any]] = []

    for sample in sample_list:
        sample_id = str(sample["id"])
        if sample_id not in records:
            continue
        record = records[sample_id]
        suite = str(sample.get("suite") or "Unknown")
        task = str(sample.get("task") or "Unknown")
        detail: Dict[str, Any] = {
            "id": sample_id,
            "suite": suite,
            "task": task,
            "gt": sample.get("gt"),
            "final_response": record.get("final_response"),
            "trigger_round": record.get("trigger_round"),
        }
        per_suite_total[suite] += 1
        per_task_total[task] += 1
        if sample.get("proactive_output"):
            proactive_total += 1
            proactive = proactive_detail(record, sample)
            detail.update(proactive)
            proactive_content += int(proactive["content_correct"])
            proactive_content_exact += int(proactive["content_correct_exact"])
            proactive_early += int(proactive["early"])
            if proactive["delay_seconds"] is not None:
                proactive_delays.append(int(proactive["delay_seconds"]))
            for threshold in THRESHOLDS:
                proactive_thresholds[threshold] += int(
                    proactive["threshold_hits"][str(threshold)]
                )
            # 官方 primary = ±2s 时间窗 + 子串内容。
            correct = int(proactive["threshold_hits"][str(PRIMARY_THRESHOLD)])
        else:
            correct = mcq_correct(record, sample)
            detail["correct"] = correct
        per_suite_correct[suite] += correct
        per_task_correct[task] += correct
        details.append(detail)

    ordinary_total = sum(
        total
        for suite, total in per_suite_total.items()
        if suite != "Proactive Output"
    )
    ordinary_correct = sum(
        correct
        for suite, correct in per_suite_correct.items()
        if suite != "Proactive Output"
    )
    proactive_metrics = {
        "content": accuracy_block(proactive_content, proactive_total),
        "content_exact": accuracy_block(proactive_content_exact, proactive_total),
        "early_outputs": proactive_early,
        "delay_seconds": {
            "min": min(proactive_delays) if proactive_delays else None,
            "max": max(proactive_delays) if proactive_delays else None,
            "mean": (
                sum(proactive_delays) / len(proactive_delays)
                if proactive_delays else None
            ),
        },
        "within_seconds": {
            str(threshold): accuracy_block(
                proactive_thresholds[threshold],
                proactive_total,
            )
            for threshold in THRESHOLDS
        },
        "primary_threshold_seconds": PRIMARY_THRESHOLD,
        "primary_content_rule": "substring (official count.py: gt_output in answer)",
    }
    return {
        "source": source_meta,
        "coverage": {
            "expected": len(expected),
            "completed": len(completed),
            "missing": sorted(expected - set(records)),
            "extra": sorted(set(records) - expected),
        },
        "ordinary_mcq": accuracy_block(ordinary_correct, ordinary_total),
        "proactive_output": proactive_metrics,
        "primary_combined": accuracy_block(
            ordinary_correct + proactive_thresholds[PRIMARY_THRESHOLD],
            ordinary_total + proactive_total,
        ),
        "per_suite": {
            suite: accuracy_block(
                per_suite_correct[suite],
                per_suite_total[suite],
            )
            for suite in sorted(per_suite_total)
        },
        "per_task": {
            task: accuracy_block(
                per_task_correct[task],
                per_task_total[task],
            )
            for task in sorted(per_task_total)
        },
        "details": details,
    }


def print_summary(report: Dict[str, Any]) -> None:
    coverage = report["coverage"]
    ordinary = report["ordinary_mcq"]
    proactive = report["proactive_output"]
    combined = report["primary_combined"]
    print(
        f"coverage={coverage['completed']}/{coverage['expected']} "
        f"missing={len(coverage['missing'])} extra={len(coverage['extra'])}"
    )
    print(
        f"ordinary_mcq={ordinary['correct']}/{ordinary['total']}="
        f"{ordinary['accuracy'] or 0.0:.2f}%"
    )
    print(
        f"proactive_content(substr)="
        f"{proactive['content']['correct']}/{proactive['content']['total']}="
        f"{proactive['content']['accuracy'] or 0.0:.2f}% "
        f"(exact={proactive['content_exact']['correct']}/"
        f"{proactive['content_exact']['total']}) "
        f"early={proactive['early_outputs']}"
    )
    for threshold in THRESHOLDS:
        metric = proactive["within_seconds"][str(threshold)]
        print(
            f"proactive_within_{threshold}s="
            f"{metric['correct']}/{metric['total']}="
            f"{metric['accuracy'] or 0.0:.2f}%"
        )
    print(
        f"primary_combined_at_{PRIMARY_THRESHOLD}s="
        f"{combined['correct']}/{combined['total']}="
        f"{combined['accuracy'] or 0.0:.2f}%  "
        f"(official: MCQ exact-letter + proactive ±{PRIMARY_THRESHOLD}s/substr)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="Result directory or JSONL file")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--json-out")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    samples, _ = load_dataset(args.dataset)
    records, source_meta = load_records(args.run)
    report = summarize(records, samples, source_meta)
    print_summary(report)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"audit_json={output}")

    coverage = report["coverage"]
    if args.require_complete and coverage["missing"]:
        raise SystemExit(2)
    if args.strict and (
        source_meta["malformed_lines"]
        or source_meta["duplicate_ids_last_wins"]
        or coverage["extra"]
    ):
        print(
            "strict_failures="
            f"malformed={source_meta['malformed_lines']} "
            f"duplicates={source_meta['duplicate_ids_last_wins']} "
            f"unexpected={len(coverage['extra'])}"
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
