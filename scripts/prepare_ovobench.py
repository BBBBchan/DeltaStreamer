#!/usr/bin/env python3
"""Build the 3035-sample StreamDelta OVO-Bench evaluation dataset.

The official annotation contains 1640 rows. REC, SSR, and CRR contain several
evaluation timestamps per row, so the production protocol expands them into
3035 samples. The script also extracts one frame per second and gives ASI
samples their full source video, matching the current production run.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


EXPECTED_COUNTS = {
    "EPM": 297,
    "HLD": 186,
    "ASI": 148,
    "STU": 178,
    "OJR": 184,
    "ATR": 116,
    "FPD": 101,
    "OCR": 149,
    "SSR": 629,
    "ACR": 109,
    "CRR": 240,
    "REC": 698,
}
FORWARD_TASKS = {"REC", "SSR", "CRR"}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def yes_no(value: Any) -> str:
    return "Yes" if int(value) else "No"


def expand_annotations(
    annotations: Sequence[Dict[str, Any]],
    regular_frames_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Expand the official 1640 rows into the production 3035-row protocol."""
    output: List[Dict[str, Any]] = []
    source_by_eval_id: Dict[str, Dict[str, Any]] = {}

    for source in annotations:
        task = str(source["task"])
        source_id = str(source["id"])
        rows: List[Dict[str, Any]] = []
        if task == "REC":
            for index, test in enumerate(source["test_info"]):
                query_time = int(test["realtime"])
                rows.append({
                    "id": f"{source_id}_{index}",
                    "task": task,
                    "activity": source["activity"],
                    "query_time": query_time,
                    # This field is consumed as a question-time fallback by the
                    # frozen runner. It intentionally is not the event ledger.
                    "start_times": [query_time],
                    "gt": str(test["count"]),
                })
        elif task == "SSR":
            for index, test in enumerate(source["test_info"]):
                rows.append({
                    "id": f"{source_id}_{index}",
                    "task": task,
                    "all_steps": test["step"],
                    "step_start_time": int(test["realtime"]),
                    "disable_ssr_expand": True,
                    "gt": yes_no(test["type"]),
                })
        elif task == "CRR":
            for index, test in enumerate(source["test_info"]):
                rows.append({
                    "id": f"{source_id}_{index}",
                    "task": task,
                    "question": source["question"],
                    "ask_time": int(test["realtime"]),
                    "gt": yes_no(test["type"]),
                })
        else:
            row = {
                "id": source_id,
                "task": task,
                "question": source["question"],
                "options": source["options"],
                "realtime": int(source["realtime"]),
                "gt": chr(ord("A") + int(source["gt"])),
            }
            if task == "ASI":
                row["ask_time"] = int(source["realtime"]) // 2
            rows.append(row)

        for row in rows:
            eval_id = str(row["id"])
            row["video"] = str((regular_frames_root / eval_id).resolve())
            source_by_eval_id[eval_id] = source
            output.append(row)

    validate_schema(output)
    return output, source_by_eval_id


def validate_schema(samples: Sequence[Dict[str, Any]]) -> None:
    ids = [str(sample["id"]) for sample in samples]
    counts = collections.Counter(str(sample["task"]) for sample in samples)
    if len(samples) != 3035 or len(set(ids)) != 3035:
        raise ValueError(
            f"expected 3035 unique samples, got rows={len(samples)}, "
            f"unique={len(set(ids))}"
        )
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(
            f"unexpected task counts: {dict(counts)}; "
            f"expected {EXPECTED_COUNTS}"
        )


def resolve_ffmpeg(explicit: str | None) -> str:
    if explicit:
        path = shutil.which(explicit) or explicit
        if Path(path).is_file():
            return path
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


def source_video(data_root: Path, source: Dict[str, Any]) -> Path:
    path = data_root / "src_videos" / str(source["video"])
    if not path.is_file():
        raise FileNotFoundError(f"missing source video: {path}")
    return path


def cutoff_seconds(sample: Dict[str, Any], source: Dict[str, Any]) -> int:
    task = str(sample["task"])
    if task == "REC":
        value = sample["query_time"]
    elif task == "SSR":
        value = sample["step_start_time"]
    elif task == "CRR":
        value = sample["ask_time"]
    else:
        value = source["realtime"]
    return max(1, int(math.ceil(float(value))))


def run_command(command: Sequence[str], label: str) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        tail = result.stderr[-2000:].strip()
        raise RuntimeError(f"{label} failed (rc={result.returncode}): {tail}")


def create_clip(
    *,
    ffmpeg: str,
    source: Path,
    output: Path,
    duration: int,
    force: bool,
) -> None:
    if output.is_file() and output.stat().st_size > 0 and not force:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp4")
    temporary.unlink(missing_ok=True)
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
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        run_command(command, f"clip {output.name}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced an empty clip: {temporary}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def frame_marker(directory: Path) -> Path:
    return (
        directory.parent
        / ".streamdelta_complete"
        / f"{directory.name}.json"
    )


def existing_frame_count(directory: Path) -> int:
    marker = frame_marker(directory)
    first = directory / "000000.jpg"
    if not marker.is_file() or not first.is_file():
        return 0
    try:
        count = int(read_json(marker)["frames"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0
    if count < 1 or not (directory / f"{count - 1:06d}.jpg").is_file():
        return 0
    return count


def extract_frames(
    *,
    ffmpeg: str,
    source: Path,
    output: Path,
    force: bool,
) -> int:
    count = existing_frame_count(output)
    if count and not force:
        return count

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
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
        "-vf",
        "fps=1",
        "-start_number",
        "0",
        "-q:v",
        "2",
        str(temporary / "%06d.jpg"),
    ]
    try:
        run_command(command, f"frames {output.name}")
        frames = sorted(temporary.glob("*.jpg"))
        if not frames or frames[0].name != "000000.jpg":
            raise RuntimeError(f"no valid 1 fps frames generated for {source}")
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
        write_json_atomic(
            frame_marker(output),
            {
                "frames": len(frames),
                "source": str(source.resolve()),
                "source_size": source.stat().st_size,
            },
            pretty=True,
        )
        return len(frames)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run_parallel(
    jobs: Iterable[Tuple[str, Callable[[], Any]]],
    workers: int,
) -> Dict[str, Any]:
    job_list = list(jobs)
    results: Dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(function): name for name, function in job_list
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            completed += 1
            if completed == 1 or completed % 50 == 0 or completed == len(job_list):
                print(f"[prepare] completed {completed}/{len(job_list)}: {name}")
    return results


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


def audit_prepared(data_root: Path, dataset_path: Path) -> Dict[str, Any]:
    samples = read_json(dataset_path)
    validate_schema(samples)
    missing_frames: List[str] = []
    missing_audio: List[str] = []
    for sample in samples:
        sample_id = str(sample["id"])
        frame_dir = Path(str(sample["video"]))
        if not frame_dir.is_dir() or not next(frame_dir.glob("*.jpg"), None):
            missing_frames.append(sample_id)
        audio = data_root / "chunked_videos_asi_full" / f"{sample_id}.mp4"
        if not audio.is_file():
            missing_audio.append(sample_id)
    report = {
        "dataset": str(dataset_path.resolve()),
        "rows": len(samples),
        "task_counts": dict(collections.Counter(
            str(sample["task"]) for sample in samples
        )),
        "missing_frame_dirs": missing_frames,
        "missing_audio_videos": missing_audio,
        "passed": not missing_frames and not missing_audio,
    }
    if not report["passed"]:
        raise RuntimeError(
            f"media audit failed: missing_frames={len(missing_frames)}, "
            f"missing_audio={len(missing_audio)}"
        )
    return report


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repository / "data" / "OVO-Bench",
    )
    parser.add_argument("--annotation", type=Path)
    parser.add_argument(
        "--video-mode",
        choices=("prechunked", "source"),
        default="prechunked",
        help=(
            "Use official pre-chunked videos, or rebuild all clips from "
            "src_videos"
        ),
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--skip-clips", action="store_true")
    parser.add_argument("--skip-frames", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    annotation = (
        args.annotation.resolve()
        if args.annotation
        else data_root / "ovo_bench_new.json"
    )
    final_json = data_root / "ovobench_streamo_eval_asi_full.json"
    audit_json = data_root / "streamdelta_media_audit.json"

    if args.check_only:
        report = audit_prepared(data_root, final_json)
        write_json_atomic(audit_json, report, pretty=True)
        print(json.dumps(report, ensure_ascii=False))
        return
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not annotation.is_file():
        parser.error(f"annotation not found: {annotation}")

    official = read_json(annotation)
    regular_frames = data_root / "ovobench_streamo_frames"
    asi_frames = data_root / "ovobench_streamo_frames_asi_full"
    regular_clips = data_root / "chunked_videos"
    audio_clips = data_root / "chunked_videos_asi_full"
    samples, source_by_eval_id = expand_annotations(official, regular_frames)
    ffmpeg = resolve_ffmpeg(args.ffmpeg)

    if not args.skip_clips and args.video_mode == "source":
        clip_jobs = []
        for sample in samples:
            eval_id = str(sample["id"])
            raw = source_by_eval_id[eval_id]
            src = source_video(data_root, raw)
            output = regular_clips / f"{eval_id}.mp4"
            duration = cutoff_seconds(sample, raw)
            clip_jobs.append((
                eval_id,
                lambda src=src, output=output, duration=duration: create_clip(
                    ffmpeg=ffmpeg,
                    source=src,
                    output=output,
                    duration=duration,
                    force=args.force,
                ),
            ))
        run_parallel(clip_jobs, args.jobs)

    missing_regular = [
        str(sample["id"])
        for sample in samples
        if not (regular_clips / f"{sample['id']}.mp4").is_file()
    ]
    if missing_regular:
        raise FileNotFoundError(
            f"missing {len(missing_regular)} regular clips under "
            f"{regular_clips}; first={missing_regular[:10]}. Extract the "
            "official pre-chunked archive or use --video-mode source."
        )

    if not args.skip_frames:
        frame_jobs = []
        for sample in samples:
            eval_id = str(sample["id"])
            clip = regular_clips / f"{eval_id}.mp4"
            output = regular_frames / eval_id
            frame_jobs.append((
                eval_id,
                lambda clip=clip, output=output: extract_frames(
                    ffmpeg=ffmpeg,
                    source=clip,
                    output=output,
                    force=args.force,
                ),
            ))
        run_parallel(frame_jobs, args.jobs)

    regular_json = data_root / "ovobench_streamo_eval.json"
    write_json_atomic(regular_json, samples)

    final_samples = [dict(sample) for sample in samples]
    asi_jobs = []
    for sample in final_samples:
        eval_id = str(sample["id"])
        raw = source_by_eval_id[eval_id]
        regular_clip = regular_clips / f"{eval_id}.mp4"
        if sample["task"] == "ASI":
            src = source_video(data_root, raw)
            output = asi_frames / eval_id
            if args.skip_frames:
                count = existing_frame_count(output)
                if not count:
                    raise FileNotFoundError(
                        f"missing completed ASI frame directory: {output}"
                    )
                sample["_full_frames"] = count
            else:
                asi_jobs.append((
                    eval_id,
                    src,
                    output,
                    sample,
                ))
            replace_symlink(audio_clips / f"{eval_id}.mp4", src)
        else:
            replace_symlink(
                audio_clips / f"{eval_id}.mp4",
                regular_clip,
            )

    if asi_jobs:
        extracted = run_parallel(
            [
                (
                    eval_id,
                    lambda src=src, output=output: extract_frames(
                        ffmpeg=ffmpeg,
                        source=src,
                        output=output,
                        force=args.force,
                    ),
                )
                for eval_id, src, output, _sample in asi_jobs
            ],
            args.jobs,
        )
        for eval_id, _src, _output, sample in asi_jobs:
            sample["_full_frames"] = int(extracted[eval_id])

    for sample in final_samples:
        if sample["task"] != "ASI":
            continue
        eval_id = str(sample["id"])
        count = int(sample["_full_frames"])
        sample["video"] = str((asi_frames / eval_id).resolve())
        sample["ask_time"] = max(0, count - 1)
        sample["_asi_full_rebuilt"] = True

    validate_schema(final_samples)
    write_json_atomic(final_json, final_samples)
    report = audit_prepared(data_root, final_json)
    report["annotation_sha256"] = hashlib.sha256(
        annotation.read_bytes()
    ).hexdigest()
    report["video_mode"] = args.video_mode
    write_json_atomic(audit_json, report, pretty=True)
    print(
        f"[prepare] complete: dataset={final_json}, rows={len(final_samples)}, "
        f"frames=3035/3035, audio=3035/3035"
    )


if __name__ == "__main__":
    main()
