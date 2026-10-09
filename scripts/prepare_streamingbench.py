#!/usr/bin/env python3
"""Prepare StreamingBench for the StreamDelta streaming inference runner."""

from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


EXPECTED_CSV_ROWS = {
    "Real_Time_Visual_Understanding.csv": 2500,
    "Sequential_Question_Answering.csv": 250,
    "Contextual_Understanding.csv": 500,
    "Omni_Source_Understanding.csv": 1000,
    "Proactive_Output.csv": 250,
}
EXPECTED_SUITE_COUNTS = {
    "Real-Time Visual Understanding": 2500,
    "Sequential Question Answering": 250,
    "Contextual Understanding": 500,
    "Omni-Source Understanding": 1000,
    "Proactive Output": 250,
}
EXPECTED_VIDEO_COUNTS = {
    "Real-Time Visual Understanding": 500,
    "Sequential Question Answering": 50,
    "Anomaly Context Understanding": 50,
    "Misleading Context Understanding": 50,
    "Emotion Recognition": 50,
    "Multimodal Alignment": 50,
    "Scene Understanding": 50,
    "Source Discrimination": 50,
    "Proactive Output": 50,
}
CSV_SPECS = (
    (
        "Real_Time_Visual_Understanding.csv",
        "Real-Time Visual Understanding",
    ),
    (
        "Sequential_Question_Answering.csv",
        "Sequential Question Answering",
    ),
    ("Contextual_Understanding.csv", "Contextual Understanding"),
    ("Omni_Source_Understanding.csv", "Omni-Source Understanding"),
    ("Proactive_Output.csv", "Proactive Output"),
)
TASK_TO_MEDIA_DIRECTORY = {
    "Misleading Context Recognition": "Misleading Context Understanding",
    "Anomaly Context Understanding": "Anomaly Context Understanding",
    "Emotion Recognition": "Emotion Recognition",
    "Multimodal Alignment": "Multimodal Alignment",
    "Scene Understanding": "Scene Understanding",
    "Source Discrimination": "Source Discrimination",
}
SUITE_TO_MEDIA_DIRECTORY = {
    "Real-Time Visual Understanding": "Real-Time Visual Understanding",
    "Sequential Question Answering": "Sequential Question Answering",
    "Proactive Output": "Proactive Output",
}
MEDIA_DIRECTORY_SLUGS = {
    name: re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    for name in EXPECTED_VIDEO_COUNTS
}
QUESTION_ID_RE = re.compile(r"_sample_(\d+)_([1-5])$")


def write_json_atomic(path: Path, value: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2 if pretty else None,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_timestamp(value: str) -> int:
    parts = str(value).strip().split(":")
    if len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds = float(parts[1])
    elif len(parts) == 3:
        hours, minutes = map(int, parts[:2])
        seconds = float(parts[2])
    else:
        raise ValueError(f"invalid StreamingBench timestamp: {value!r}")
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"invalid StreamingBench timestamp: {value!r}")
    return int(math.floor(hours * 3600 + minutes * 60 + seconds))


def normalize_options(raw: str) -> List[str]:
    options = ast.literal_eval(raw)
    if not isinstance(options, list) or not options:
        raise ValueError(f"invalid options list: {raw!r}")
    normalized: List[str] = []
    for index, value in enumerate(options):
        text = str(value).strip()
        expected = chr(ord("A") + index)
        text = re.sub(
            rf"^\s*{re.escape(expected)}\s*[.)]\s*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        normalized.append(text)
    return normalized


def parse_source_number(question_id: str) -> int:
    match = QUESTION_ID_RE.search(question_id)
    if not match:
        raise ValueError(f"cannot map question_id to source video: {question_id}")
    return int(match.group(1))


def media_directory_for(suite: str, task_type: str) -> str:
    if suite in SUITE_TO_MEDIA_DIRECTORY:
        return SUITE_TO_MEDIA_DIRECTORY[suite]
    try:
        return TASK_TO_MEDIA_DIRECTORY[task_type]
    except KeyError as exc:
        raise ValueError(
            f"no media mapping for suite={suite!r}, task_type={task_type!r}"
        ) from exc


def find_source_video(source_root: Path, directory: str, number: int) -> Path:
    sample_dir = source_root / directory / f"sample_{number}"
    canonical = sample_dir / "video.mp4"
    if canonical.is_file() and canonical.stat().st_size:
        return canonical
    candidates = sorted(
        path
        for path in sample_dir.glob("*.mp4")
        if path.is_file() and path.stat().st_size
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"expected one source video under {sample_dir}, found {candidates}"
    )


def source_key(directory: str, number: int) -> str:
    return f"{MEDIA_DIRECTORY_SLUGS[directory]}/sample_{number}"


def frame_marker(frame_dir: Path) -> Path:
    return (
        frame_dir.parents[1]
        / ".streamdelta_complete"
        / frame_dir.parent.name
        / f"{frame_dir.name}.json"
    )


def completed_frame_count(
    frame_dir: Path,
    source: Path,
    duration: int,
) -> int:
    marker = frame_marker(frame_dir)
    first = frame_dir / "000000.jpg"
    if not marker.is_file() or not first.is_file():
        return 0
    try:
        payload = read_json(marker)
        count = int(payload["frames"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0
    stat = source.stat()
    if (
        count < 1
        or not (frame_dir / f"{count - 1:06d}.jpg").is_file()
        or int(payload.get("duration", -1)) != duration
        or int(payload.get("source_size", -1)) != stat.st_size
        or int(payload.get("source_mtime_ns", -1)) != stat.st_mtime_ns
    ):
        return 0
    return count


def resolve_ffmpeg(explicit: str | None) -> str:
    if explicit:
        resolved = shutil.which(explicit) or explicit
        if Path(resolved).is_file():
            return resolved
        raise FileNotFoundError(f"ffmpeg not found: {explicit}")
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "ffmpeg is required; install ffmpeg or imageio-ffmpeg"
        ) from exc


def resolve_ffprobe(ffmpeg: str) -> str:
    system = shutil.which("ffprobe")
    if system:
        return system
    sibling = Path(ffmpeg).with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)
    raise FileNotFoundError("ffprobe is required to validate video durations")


def probe_duration(ffprobe: str, source: Path) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"ffprobe failed for {source}: {result.stderr[-2000:].strip()}"
        )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid video duration for {source}: {duration}")
    return duration


def extract_frames(
    *,
    ffmpeg: str,
    source: Path,
    frame_dir: Path,
    duration: int,
    force: bool,
) -> int:
    existing = completed_frame_count(frame_dir, source, duration)
    if existing and not force:
        return existing
    frame_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = frame_dir.with_name(f".{frame_dir.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        str(duration),
        "-vf",
        "fps=1",
        "-start_number",
        "0",
        "-q:v",
        "2",
        str(temporary / "%06d.jpg"),
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"frame extraction failed for {source}: "
                f"{result.stderr[-2000:].strip()}"
            )
        frames = sorted(temporary.glob("*.jpg"))
        if not frames or frames[0].name != "000000.jpg":
            raise RuntimeError(f"no 1 fps frames generated for {source}")
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        os.replace(temporary, frame_dir)
        stat = source.stat()
        write_json_atomic(
            frame_marker(frame_dir),
            {
                "frames": len(frames),
                "duration": duration,
                "source": str(source.resolve()),
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
            },
            pretty=True,
        )
        return len(frames)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(target.resolve(), link.parent.resolve())
    if link.is_symlink() and os.readlink(link) == relative and link.exists():
        return
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            raise IsADirectoryError(f"refusing to replace directory: {link}")
        link.unlink()
    temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    os.symlink(relative, temporary)
    os.replace(temporary, link)


def load_csv_rows(annotation_root: Path) -> List[Tuple[str, Dict[str, str]]]:
    output: List[Tuple[str, Dict[str, str]]] = []
    for filename, suite in CSV_SPECS:
        path = annotation_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing annotation CSV: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = EXPECTED_CSV_ROWS[filename]
        if len(rows) != expected:
            raise ValueError(
                f"{filename}: expected {expected} rows, got {len(rows)}"
            )
        output.extend((suite, row) for row in rows)
    return output


def build_sequential_context(
    grouped_rows: Dict[Tuple[str, int], List[Dict[str, str]]],
) -> Dict[str, List[Dict[str, str]]]:
    context: Dict[str, List[Dict[str, str]]] = {}
    for rows in grouped_rows.values():
        ordered = sorted(rows, key=lambda row: parse_timestamp(row["time_stamp"]))
        history: List[Dict[str, str]] = []
        for row in ordered:
            context[row["question_id"]] = [dict(item) for item in history]
            history.append({
                "question": row["question"].strip(),
                "time_stamp": row["time_stamp"].strip(),
            })
    return context


def prepare_samples(
    source_root: Path,
    output_root: Path,
    proactive_tail_seconds: int,
    *,
    ffprobe: str,
    probe_workers: int,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, int],
]:
    annotation_root = source_root / "StreamingBench"
    raw_rows = load_csv_rows(annotation_root)
    sequential_groups: Dict[Tuple[str, int], List[Dict[str, str]]] = (
        collections.defaultdict(list)
    )
    for suite, row in raw_rows:
        if suite == "Sequential Question Answering":
            sequential_groups[(suite, parse_source_number(row["question_id"]))].append(
                row
            )
    sequential_context = build_sequential_context(sequential_groups)

    pending: List[Tuple[str, Dict[str, str], str, int, Path, str, Path]] = []
    source_paths: Dict[str, Path] = {}
    for suite, raw in raw_rows:
        question_id = raw["question_id"].strip()
        task_type = raw["task_type"].strip()
        number = parse_source_number(question_id)
        media_directory = media_directory_for(suite, task_type)
        source = find_source_video(source_root, media_directory, number)
        key = source_key(media_directory, number)
        frame_dir = output_root / "frames" / key
        pending.append((
            suite,
            raw,
            task_type,
            number,
            source,
            key,
            frame_dir,
        ))
        source_paths[key] = source

    source_durations: Dict[str, float] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, probe_workers)
    ) as pool:
        futures = {
            pool.submit(probe_duration, ffprobe, path): key
            for key, path in source_paths.items()
        }
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            source_durations[key] = float(future.result())

    samples: List[Dict[str, Any]] = []
    sources: Dict[str, Dict[str, Any]] = {}
    required_seconds: Dict[str, int] = collections.defaultdict(int)

    for suite, raw, task_type, number, source, key, frame_dir in pending:
        question_id = raw["question_id"].strip()
        media_directory = media_directory_for(suite, task_type)
        source_duration = source_durations[key]
        available_last_second = max(0, int(math.ceil(source_duration)) - 1)
        annotated_question_time = parse_timestamp(raw["time_stamp"])
        question_time = min(annotated_question_time, available_last_second)
        sources[key] = {
            "source_id": key,
            "media_directory": media_directory,
            "sample_number": number,
            "video": str(source.resolve()),
            "frames": str(frame_dir.resolve()),
            "duration_seconds": source_duration,
            "available_last_second": available_last_second,
        }

        base: Dict[str, Any] = {
            "id": question_id,
            "benchmark": "streamingbench",
            "suite": suite,
            "task": task_type,
            "task_type": task_type,
            "question": raw["question"].strip(),
            "realtime": question_time,
            "annotated_realtime": annotated_question_time,
            "question_time_clamped": question_time != annotated_question_time,
            "time_stamp": raw["time_stamp"].strip(),
            "temporal_clue_type": raw["temporal_clue_type"].strip(),
            "frames_required": raw["frames_required"].strip(),
            "source_id": key,
            "source_video": str(source.resolve()),
            "video": str(frame_dir.resolve()),
        }
        if suite == "Proactive Output":
            annotated_target_time = parse_timestamp(
                raw["ground_truth_time_stamp"]
            )
            target_time = min(annotated_target_time, available_last_second)
            base.update({
                "proactive_output": True,
                "target_time": target_time,
                "annotated_target_time": annotated_target_time,
                "target_time_clamped": target_time != annotated_target_time,
                "ground_truth_time_stamp": raw[
                    "ground_truth_time_stamp"
                ].strip(),
                "expected_output": raw["ground_truth_output"].strip(),
                "gt": raw["ground_truth_output"].strip(),
            })
            base["stream_end_time"] = min(
                available_last_second + 1,
                target_time + proactive_tail_seconds + 1,
            )
            required_seconds[key] = max(
                required_seconds[key],
                int(base["stream_end_time"]),
            )
        else:
            options = normalize_options(raw["options"])
            answer = raw["answer"].strip().upper()
            if not re.fullmatch(r"[A-Z]", answer):
                raise ValueError(
                    f"{question_id}: invalid answer letter {answer!r}"
                )
            base.update({
                "options": options,
                "gt": answer,
                "stream_end_time": question_time + 1,
            })
            required_seconds[key] = max(
                required_seconds[key],
                question_time + 1,
            )
            if suite == "Sequential Question Answering":
                history = sequential_context[question_id]
                base["native_question"] = base["question"]
                base["sequential_context"] = history
                if history:
                    rendered = "\n".join(
                        (
                            f"- At {item['time_stamp']}, Q: "
                            f"{item['question']}"
                        )
                        for item in history
                    )
                    base["question"] = (
                        "Earlier questions from this same stream are listed "
                        "below only to resolve references such as 'the person "
                        "mentioned in the previous question'. Their answers "
                        "are not provided; infer the referents from the video "
                        "and MEMORY.\n"
                        f"{rendered}\n\n"
                        f"Current question: {base['native_question']}"
                    )
        samples.append(base)

    validate_samples(samples, sources)
    return samples, sources, dict(required_seconds)


def validate_samples(
    samples: Sequence[Dict[str, Any]],
    sources: Dict[str, Dict[str, Any]],
) -> None:
    ids = [str(sample["id"]) for sample in samples]
    if len(samples) != 4500 or len(set(ids)) != 4500:
        raise ValueError(
            f"expected 4500 unique QA rows, got rows={len(samples)}, "
            f"unique={len(set(ids))}"
        )
    suite_counts = collections.Counter(sample["suite"] for sample in samples)
    if dict(suite_counts) != EXPECTED_SUITE_COUNTS:
        raise ValueError(
            f"unexpected suite counts: {dict(suite_counts)}; "
            f"expected {EXPECTED_SUITE_COUNTS}"
        )
    source_counts = collections.Counter(
        source["media_directory"] for source in sources.values()
    )
    if dict(source_counts) != EXPECTED_VIDEO_COUNTS:
        raise ValueError(
            f"unexpected video counts: {dict(source_counts)}; "
            f"expected {EXPECTED_VIDEO_COUNTS}"
        )


def run_frame_jobs(
    jobs: Iterable[Tuple[str, Any]],
    workers: int,
) -> Dict[str, int]:
    job_list = list(jobs)
    results: Dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(function): name for name, function in job_list
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            results[name] = int(future.result())
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(job_list):
                print(
                    f"[prepare] frames completed {completed}/{len(job_list)}: "
                    f"{name}"
                )
    return results


def reconcile_sample_times(
    samples: Sequence[Dict[str, Any]],
    frame_counts: Dict[str, int],
    proactive_tail_seconds: int,
) -> None:
    for sample in samples:
        sample_id = str(sample["id"])
        source_id = str(sample["source_id"])
        frame_count = int(frame_counts[source_id])
        if frame_count < 1:
            raise ValueError(
                f"{sample_id}: source {source_id} produced no frames"
            )
        last_frame = frame_count - 1
        annotated_question_time = int(sample["annotated_realtime"])
        sample["realtime"] = min(annotated_question_time, last_frame)
        sample["question_time_clamped"] = (
            sample["realtime"] != annotated_question_time
        )
        if sample.get("proactive_output"):
            annotated_target_time = int(sample["annotated_target_time"])
            sample["target_time"] = min(annotated_target_time, last_frame)
            sample["target_time_clamped"] = (
                sample["target_time"] != annotated_target_time
            )
            sample["stream_end_time"] = min(
                frame_count,
                int(sample["target_time"]) + proactive_tail_seconds + 1,
            )
        else:
            sample["stream_end_time"] = int(sample["realtime"]) + 1


def audit_prepared(output_root: Path, dataset_path: Path) -> Dict[str, Any]:
    samples = read_json(dataset_path)
    sources_path = output_root / "sources.json"
    sources = read_json(sources_path)
    validate_samples(samples, sources)
    missing_frames: List[str] = []
    missing_audio: List[str] = []
    invalid_times: List[str] = []
    source_ids_with_frames: set[str] = set()
    frame_counts: Dict[str, int] = {}
    for sample in samples:
        frame_dir = Path(sample["video"])
        frame_key = str(frame_dir)
        if frame_key not in frame_counts:
            marker = frame_marker(frame_dir)
            try:
                frame_counts[frame_key] = int(read_json(marker)["frames"])
            except (
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                frame_counts[frame_key] = (
                    len(list(frame_dir.glob("*.jpg")))
                    if frame_dir.is_dir()
                    else 0
                )
        frame_count = frame_counts[frame_key]
        if not frame_count:
            missing_frames.append(str(sample["id"]))
        else:
            source_ids_with_frames.add(str(sample["source_id"]))
            stream_end_time = int(sample.get("stream_end_time") or 0)
            if stream_end_time < 1 or frame_count < stream_end_time:
                invalid_times.append(str(sample["id"]))
        audio = output_root / "audio_videos" / f"{sample['id']}.mp4"
        if not audio.is_file():
            missing_audio.append(str(sample["id"]))
    report = {
        "dataset": str(dataset_path.resolve()),
        "rows": len(samples),
        "unique_ids": len({str(sample["id"]) for sample in samples}),
        "source_videos": len(sources),
        "source_videos_with_frames": len(source_ids_with_frames),
        "suite_counts": dict(collections.Counter(
            sample["suite"] for sample in samples
        )),
        "task_counts": dict(collections.Counter(
            sample["task"] for sample in samples
        )),
        "clamped_question_times": [
            {
                "id": sample["id"],
                "annotated": sample["annotated_realtime"],
                "effective": sample["realtime"],
            }
            for sample in samples
            if sample.get("question_time_clamped")
        ],
        "clamped_proactive_target_times": [
            {
                "id": sample["id"],
                "annotated": sample["annotated_target_time"],
                "effective": sample["target_time"],
            }
            for sample in samples
            if sample.get("target_time_clamped")
        ],
        "missing_frame_samples": missing_frames,
        "missing_audio_videos": missing_audio,
        "insufficient_frame_samples": invalid_times,
        "passed": not missing_frames and not missing_audio and not invalid_times,
    }
    if not report["passed"]:
        raise RuntimeError(
            "StreamingBench media audit failed: "
            f"missing_frames={len(missing_frames)}, "
            f"missing_audio={len(missing_audio)}, "
            f"insufficient_frames={len(invalid_times)}"
        )
    return report


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/StreamingBench-source"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository / "data" / "StreamingBench",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--proactive-tail-seconds", type=int, default=10)
    parser.add_argument("--skip-frames", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    dataset_path = output_root / "streamingbench_streamdelta_eval.json"
    audit_path = output_root / "streamdelta_media_audit.json"
    if args.check_only:
        report = audit_prepared(output_root, dataset_path)
        write_json_atomic(audit_path, report, pretty=True)
        print(json.dumps(report, ensure_ascii=False))
        return
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.proactive_tail_seconds < 0:
        parser.error("--proactive-tail-seconds must be non-negative")
    if not source_root.is_dir():
        parser.error(f"source root not found: {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    samples, sources, required_seconds = prepare_samples(
        source_root,
        output_root,
        args.proactive_tail_seconds,
        ffprobe=resolve_ffprobe(ffmpeg),
        probe_workers=min(args.jobs, 32),
    )
    for key, source in sources.items():
        source["required_seconds"] = required_seconds[key]
    manifest = {
        "format": "streamdelta-streamingbench-v1",
        "status": "preparing",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "rows": len(samples),
        "source_videos": len(sources),
        "fps": 1,
        "proactive_tail_seconds": args.proactive_tail_seconds,
        "annotation_sha256": {
            filename: sha256_file(source_root / "StreamingBench" / filename)
            for filename in EXPECTED_CSV_ROWS
        },
        "suite_counts": dict(collections.Counter(
            sample["suite"] for sample in samples
        )),
        "task_counts": dict(collections.Counter(
            sample["task"] for sample in samples
        )),
    }
    write_json_atomic(output_root / "sources.json", sources, pretty=True)
    write_json_atomic(dataset_path, samples)
    write_json_atomic(
        dataset_path.with_suffix(".manifest.json"),
        manifest,
        pretty=True,
    )
    audio_root = output_root / "audio_videos"
    for sample in samples:
        replace_symlink(
            audio_root / f"{sample['id']}.mp4",
            Path(sample["source_video"]),
        )

    frame_counts: Dict[str, int] = {}
    if not args.skip_frames:
        jobs = []
        ordered_sources = sorted(
            sources.items(),
            key=lambda item: (
                int(item[1]["sample_number"]),
                str(item[1]["media_directory"]),
            ),
        )
        for key, source in ordered_sources:
            source_path = Path(source["video"])
            frame_dir = Path(source["frames"])
            duration = required_seconds[key]
            jobs.append((
                key,
                lambda source_path=source_path,
                frame_dir=frame_dir,
                duration=duration: extract_frames(
                    ffmpeg=ffmpeg,
                    source=source_path,
                    frame_dir=frame_dir,
                    duration=duration,
                    force=args.force,
                ),
            ))
        frame_counts = run_frame_jobs(jobs, args.jobs)
    else:
        for key, source in sources.items():
            frame_dir = Path(source["frames"])
            count = completed_frame_count(
                frame_dir,
                Path(source["video"]),
                required_seconds[key],
            )
            if not count:
                raise FileNotFoundError(
                    f"no completed frame extraction for {key}: {frame_dir}"
                )
            frame_counts[key] = count

    for key, source in sources.items():
        source["frame_count"] = frame_counts[key]
    reconcile_sample_times(
        samples,
        frame_counts,
        args.proactive_tail_seconds,
    )

    write_json_atomic(output_root / "sources.json", sources, pretty=True)
    write_json_atomic(dataset_path, samples)
    manifest["status"] = "ready"
    manifest["prepared_source_videos"] = len(frame_counts)
    write_json_atomic(
        dataset_path.with_suffix(".manifest.json"),
        manifest,
        pretty=True,
    )
    report = audit_prepared(output_root, dataset_path)
    write_json_atomic(audit_path, report, pretty=True)
    print(
        f"[prepare] complete: dataset={dataset_path}, rows={len(samples)}, "
        f"source_videos={len(sources)}, audit=passed"
    )


if __name__ == "__main__":
    main()
