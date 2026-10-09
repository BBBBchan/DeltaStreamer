#!/usr/bin/env python3
"""Strict, dependency-light auditing for OVO-Bench agent runs.

Unlike the historical scorer, this module does not import ms-swift or vLLM.
It reports the full-dataset micro average used by the optimization target,
while retaining the historical category-macro score for continuity.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


BACKWARD_TASKS = ("EPM", "ASI", "HLD")
REALTIME_TASKS = ("OCR", "ACR", "ATR", "STU", "FPD", "OJR")
FORWARD_TASKS = ("REC", "SSR", "CRR")
TASKS = BACKWARD_TASKS + REALTIME_TASKS + FORWARD_TASKS
CATEGORIES = (
    ("Backward Tracing", BACKWARD_TASKS),
    ("Real-time Visual Perception", REALTIME_TASKS),
    ("Forward Active Responding", FORWARD_TASKS),
)

_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE
)
_THINK_OPEN_TAIL_RE = re.compile(
    r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE
)
_REFUSAL_RE = re.compile(
    r"\b(?:unable to answer|cannot determine|can't determine|"
    r"cannot answer|no answer|not enough information|i do not know|"
    r"i don't know)\b",
    re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    return _THINK_OPEN_TAIL_RE.sub("", cleaned)


def extract_final_response(output: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    def scan(canonical_only: bool) -> Tuple[Optional[str], Optional[int]]:
        final_text: Optional[str] = None
        trigger_round: Optional[int] = None
        for key, value in (output or {}).items():
            if not value:
                continue
            match = re.fullmatch(r"Round\s+(\d+)", str(key))
            if canonical_only and match is None:
                continue
            visible = strip_thinking(str(value))
            if "</Response>" not in visible:
                continue
            trigger_round = int(match.group(1)) if match is not None else None
            final_text = visible.split("</Response>", 1)[1].strip()
        return final_text, trigger_round

    canonical = scan(canonical_only=True)
    return canonical if canonical[0] is not None else scan(
        canonical_only=False
    )


def extract_option_letter(response: Optional[str]) -> Optional[str]:
    if response is None:
        return None
    text = response.strip()
    if not text:
        return None
    if len(text) == 1 and text.upper() in "ABCDEFGHIJK":
        return text.upper()
    match = re.match(r"\s*([A-Ka-k])\s*[.)\s:：\-]", text)
    if match:
        return match.group(1).upper()
    match = re.search(
        r"(?:answer|option|choose|select|choice)\s+(?:is\s+)?"
        r"([A-Ka-k])\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    letters = re.findall(r"\b([A-Ka-k])\b", text)
    return letters[-1].upper() if letters else None


def ensure_final_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    output = record.get("output") or {}
    final, trigger = extract_final_response(output)
    if final is not None:
        record["final_response"] = final
        record["trigger_round"] = trigger
    elif "final_response" not in record:
        record["final_response"] = None
        record["trigger_round"] = None
    return record


def score_record(record: Dict[str, Any]) -> int:
    ensure_final_fields(record)
    task = record.get("task")
    gt = record.get("gt")
    final = record.get("final_response")

    if task in BACKWARD_TASKS or task in REALTIME_TASKS:
        if final is None:
            return 0
        letter = extract_option_letter(str(final))
        if letter is None:
            return int(
                isinstance(gt, str) and gt.upper() in str(final).upper()
            )
        return int(isinstance(gt, str) and letter == gt.upper())

    if task == "REC":
        if final is None:
            return 0
        letter = extract_option_letter(str(final))
        if letter is not None:
            try:
                return int(ord(letter) - ord("A") == int(gt))
            except (TypeError, ValueError):
                return 0
        digits = "".join(re.findall(r"\d+", str(final)))
        try:
            return int(bool(digits) and int(digits) == int(gt))
        except (TypeError, ValueError):
            return 0

    if task in ("SSR", "CRR"):
        if final is None:
            prediction = "No"
        else:
            match = re.search(r"\b(yes|no)\b", str(final), re.IGNORECASE)
            prediction = match.group(1).capitalize() if match else "Yes"
        return int(isinstance(gt, str) and prediction == gt)

    return 0


def _candidate_jsonl_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    workers = sorted(glob.glob(os.path.join(path, "ovobench_*.worker*.jsonl")))
    if workers:
        return workers
    return sorted(glob.glob(os.path.join(path, "ovobench_*.jsonl")))


def load_records(path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    malformed = 0
    duplicates = 0
    files = _candidate_jsonl_files(path)
    for file_path in files:
        with open(file_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
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
                if key in records:
                    duplicates += 1
                records[key] = ensure_final_fields(record)
    return records, {
        "files": files,
        "malformed_lines": malformed,
        "duplicate_ids_last_wins": duplicates,
    }


def load_expected_ids(dataset_path: Optional[str]) -> Tuple[Set[str], Dict[str, str]]:
    if not dataset_path:
        return set(), {}
    with open(dataset_path, encoding="utf-8") as handle:
        samples = json.load(handle)
    return (
        {str(sample["id"]) for sample in samples},
        {str(sample["id"]): str(sample.get("task") or "UNK") for sample in samples},
    )


def load_filter_ids(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "ids" in data:
        data = data["ids"]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return {str(item["id"]) for item in data}
    return {str(item) for item in data}


def failure_signature(record: Dict[str, Any]) -> str:
    ensure_final_fields(record)
    final = record.get("final_response")
    task = str(record.get("task") or "")
    if final is None or not str(final).strip():
        return "empty"
    text = str(final).strip()
    if _REFUSAL_RE.search(text):
        options = " ".join(str(x) for x in (record.get("options") or []))
        return "legal_refusal" if _REFUSAL_RE.search(options) else "invalid_refusal"
    if task in BACKWARD_TASKS or task in REALTIME_TASKS:
        if extract_option_letter(text) is None:
            if re.fullmatch(r"\s*\d+\s*", text):
                return "numeric_non_rec"
            return "malformed_mcq"
    if task == "REC" and extract_option_letter(text) is None:
        if not re.search(r"\d+", text):
            return "malformed_rec"
    if task in ("SSR", "CRR") and not re.search(
        r"\b(?:yes|no)\b", text, re.IGNORECASE
    ):
        return "implicit_yes"
    return "valid"


def summarize(
    records: Dict[str, Dict[str, Any]],
    *,
    expected_ids: Optional[Set[str]] = None,
    id_filter: Optional[Set[str]] = None,
    source_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selected_ids = set(records)
    if id_filter is not None:
        selected_ids &= id_filter
    selected = [records[sample_id] for sample_id in sorted(selected_ids)]

    per_task_total: Counter[str] = Counter()
    per_task_correct: Counter[str] = Counter()
    signatures: Counter[str] = Counter()
    details: List[Dict[str, Any]] = []
    for record in selected:
        task = str(record.get("task") or "UNK")
        correct = score_record(record)
        signature = failure_signature(record)
        per_task_total[task] += 1
        per_task_correct[task] += correct
        signatures[signature] += 1
        details.append({
            "id": str(record.get("id")),
            "task": task,
            "gt": record.get("gt"),
            "final_response": record.get("final_response"),
            "correct": correct,
            "failure_signature": signature,
            "tool_calls_n": record.get(
                "tool_calls_n", len(record.get("tool_calls") or [])
            ),
        })

    total = len(selected)
    correct = sum(per_task_correct.values())
    per_task: Dict[str, Dict[str, Any]] = {}
    for task in TASKS:
        n_task = per_task_total[task]
        c_task = per_task_correct[task]
        per_task[task] = {
            "correct": c_task,
            "wrong": n_task - c_task,
            "total": n_task,
            "accuracy": 100.0 * c_task / n_task if n_task else None,
        }

    categories: Dict[str, Dict[str, Any]] = {}
    legacy_category_scores: List[float] = []
    for name, tasks in CATEGORIES:
        category_total = sum(per_task_total[task] for task in tasks)
        category_correct = sum(per_task_correct[task] for task in tasks)
        category_micro = (
            100.0 * category_correct / category_total if category_total else None
        )
        task_accuracies = [
            per_task[task]["accuracy"]
            for task in tasks
            if per_task[task]["accuracy"] is not None
        ]
        task_macro = (
            sum(task_accuracies) / len(task_accuracies)
            if task_accuracies else None
        )
        if task_macro is not None:
            legacy_category_scores.append(task_macro)
        categories[name] = {
            "correct": category_correct,
            "total": category_total,
            "micro_accuracy": category_micro,
            "task_macro_accuracy": task_macro,
        }

    expected = set(expected_ids or ())
    if id_filter is not None:
        expected = (expected & id_filter) if expected else set(id_filter)
    coverage = {
        "expected": len(expected) if expected else None,
        "completed": len(selected_ids & expected) if expected else total,
        "missing": sorted(expected - selected_ids) if expected else [],
        "extra": sorted(selected_ids - expected) if expected else [],
    }
    return {
        "source": source_meta or {},
        "coverage": coverage,
        "micro": {
            "correct": correct,
            "wrong": total - correct,
            "total": total,
            "accuracy": 100.0 * correct / total if total else 0.0,
        },
        "legacy_macro": {
            "accuracy": (
                sum(legacy_category_scores) / len(legacy_category_scores)
                if legacy_category_scores else 0.0
            ),
            "definition": (
                "mean of task accuracies within each category, then mean "
                "of available category scores"
            ),
        },
        "categories": categories,
        "per_task": per_task,
        "failure_signatures": dict(sorted(signatures.items())),
        "details": details,
    }


def compare(
    baseline: Dict[str, Dict[str, Any]],
    candidate: Dict[str, Dict[str, Any]],
    *,
    id_filter: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if id_filter is not None:
        baseline_ids &= id_filter
        candidate_ids &= id_filter
    common = baseline_ids & candidate_ids
    transitions: Dict[str, Counter[str]] = defaultdict(Counter)
    changed: List[Dict[str, Any]] = []
    for sample_id in sorted(common):
        old = score_record(baseline[sample_id])
        new = score_record(candidate[sample_id])
        task = str(candidate[sample_id].get("task") or "UNK")
        label = {
            (0, 0): "wrong_to_wrong",
            (0, 1): "wrong_to_correct",
            (1, 0): "correct_to_wrong",
            (1, 1): "correct_to_correct",
        }[(old, new)]
        transitions[task][label] += 1
        if old != new:
            changed.append({
                "id": sample_id,
                "task": task,
                "transition": label,
                "gt": candidate[sample_id].get("gt"),
                "baseline_response": baseline[sample_id].get("final_response"),
                "candidate_response": candidate[sample_id].get("final_response"),
                "baseline_tools": baseline[sample_id].get(
                    "tool_calls_n", len(baseline[sample_id].get("tool_calls") or [])
                ),
                "candidate_tools": candidate[sample_id].get(
                    "tool_calls_n", len(candidate[sample_id].get("tool_calls") or [])
                ),
            })

    per_task: Dict[str, Dict[str, Any]] = {}
    total_counter: Counter[str] = Counter()
    for task in TASKS:
        counter = transitions[task]
        total_counter.update(counter)
        per_task[task] = {
            **{key: counter[key] for key in (
                "wrong_to_wrong", "wrong_to_correct",
                "correct_to_wrong", "correct_to_correct",
            )},
            "net_correct": counter["wrong_to_correct"] - counter["correct_to_wrong"],
            "common": sum(counter.values()),
        }
    return {
        "common": len(common),
        "baseline_only": len(baseline_ids - candidate_ids),
        "candidate_only": len(candidate_ids - baseline_ids),
        "wrong_to_correct": total_counter["wrong_to_correct"],
        "correct_to_wrong": total_counter["correct_to_wrong"],
        "net_correct": (
            total_counter["wrong_to_correct"] - total_counter["correct_to_wrong"]
        ),
        "per_task": per_task,
        "changed": changed,
    }


def _print_summary(label: str, report: Dict[str, Any]) -> None:
    micro = report["micro"]
    coverage = report["coverage"]
    print(f"\n=== {label} ===")
    print(
        f"micro={micro['correct']}/{micro['total']}="
        f"{micro['accuracy']:.2f}%  legacy_macro="
        f"{report['legacy_macro']['accuracy']:.2f}%"
    )
    if coverage["expected"] is not None:
        print(
            f"coverage={coverage['completed']}/{coverage['expected']}  "
            f"missing={len(coverage['missing'])}  extra={len(coverage['extra'])}"
        )
    print("task       correct/total    accuracy")
    for task in TASKS:
        item = report["per_task"][task]
        acc = "-" if item["accuracy"] is None else f"{item['accuracy']:.2f}%"
        print(f"{task:<5} {item['correct']:>7}/{item['total']:<7} {acc:>10}")
    print(f"failure_signatures={report['failure_signatures']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="Result directory or JSONL file")
    parser.add_argument("--baseline", help="Optional baseline result directory/JSONL")
    parser.add_argument("--dataset", help="Expected full dataset JSON")
    parser.add_argument("--subset", help="Limit scoring/comparison to these IDs")
    parser.add_argument("--json-out", help="Write complete audit JSON")
    parser.add_argument(
        "--require-complete", action="store_true",
        help="Exit non-zero when expected IDs are missing",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Also reject malformed lines, duplicate IDs, and unexpected IDs",
    )
    args = parser.parse_args()

    records, meta = load_records(args.run)
    expected_ids, _ = load_expected_ids(args.dataset)
    explicit_filter_ids = load_filter_ids(args.subset)
    if expected_ids:
        filter_ids: Optional[Set[str]] = set(expected_ids)
        if args.subset:
            filter_ids &= explicit_filter_ids
    else:
        filter_ids = explicit_filter_ids if args.subset else None
    report = summarize(
        records,
        expected_ids=expected_ids,
        id_filter=filter_ids,
        source_meta=meta,
    )
    result: Dict[str, Any] = {"candidate": report}
    _print_summary("candidate", report)

    if args.baseline:
        baseline, baseline_meta = load_records(args.baseline)
        baseline_report = summarize(
            baseline,
            expected_ids=expected_ids,
            id_filter=filter_ids,
            source_meta=baseline_meta,
        )
        comparison = compare(
            baseline,
            records,
            id_filter=filter_ids,
        )
        result["baseline"] = baseline_report
        result["comparison"] = comparison
        _print_summary("baseline", baseline_report)
        print("\n=== comparison on common IDs ===")
        print(
            f"common={comparison['common']}  "
            f"wrong->correct={comparison['wrong_to_correct']}  "
            f"correct->wrong={comparison['correct_to_wrong']}  "
            f"net={comparison['net_correct']:+d}"
        )
        for task in TASKS:
            item = comparison["per_task"][task]
            if item["common"]:
                print(
                    f"{task:<5} common={item['common']:<4} "
                    f"gain={item['wrong_to_correct']:<3} "
                    f"loss={item['correct_to_wrong']:<3} "
                    f"net={item['net_correct']:+d}"
                )

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"audit_json={output_path}")

    if args.require_complete and report["coverage"]["missing"]:
        raise SystemExit(2)
    if args.strict:
        unexpected = (
            set(records) - expected_ids
            if expected_ids else set()
        )
        if (
            meta["malformed_lines"]
            or meta["duplicate_ids_last_wins"]
            or unexpected
        ):
            print(
                "strict_failures="
                f"malformed={meta['malformed_lines']} "
                f"duplicates={meta['duplicate_ids_last_wins']} "
                f"unexpected={len(unexpected)}"
            )
            raise SystemExit(2)


if __name__ == "__main__":
    main()
