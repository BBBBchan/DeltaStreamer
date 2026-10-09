"""Run one uploaded-video job through the repository's actual agent loop.

Launch with ``python -m demo.runner --job-dir PATH``. The web server owns job
creation and process cancellation; this worker owns media preparation and
model calls. All original core files remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any

from .engine import (
    Cancelled, EngineConfig, InferenceEngine, ModelCallLimit, RequestConfig,
    install_swift_protocol_bridge,
)

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION = re.compile(r"<Observation\b[^>]*>(.*?)</Observation>", re.I | re.S)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def diagnostic_error(exc: Exception) -> None:
    """Keep worker diagnostics useful without printing credentials or locals."""
    detail = str(exc)
    secret = os.environ.get("DEMO_API_KEY", "")
    if secret:
        detail = detail.replace(secret, "[redacted]")
    detail = re.sub(r"(?i)\bBearer\s+[^\s\"']+", "Bearer [redacted]", detail)
    print(f"[worker] {type(exc).__name__}: {detail}", file=sys.stderr, flush=True)
    # Source lines or frame locals can contain user input and API bodies.
    # File/function/line locations are enough to find an import/runtime error.
    for frame in traceback.extract_tb(exc.__traceback__):
        print(f"  {Path(frame.filename).name}:{frame.lineno} in {frame.name}", file=sys.stderr, flush=True)


def extract_frames(video: Path, output: Path, duration: float, cancel_check) -> list[Path]:
    """Keep a native-resolution image at t=0,1,... for core's seconds clock."""
    from .media import binary

    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for second in range(max(1, math.ceil(duration))):
        if cancel_check():
            raise Cancelled("Cancelled while preparing frames.")
        path = output / f"{second:06d}.jpg"
        try:
            subprocess.run([
                binary("ffmpeg"), "-nostdin", "-v", "error", "-y",
                "-threads", "2", "-protocol_whitelist", "file,pipe",
                "-ss", str(second), "-i", str(video), "-map", "0:v:0",
                "-frames:v", "1", "-threads", "2", "-q:v", "2", str(path),
            ], capture_output=True, timeout=60, check=True)
        except subprocess.SubprocessError:
            raise ValueError("The uploaded video could not be decoded into frames.") from None
        if not path.is_file():
            # Do not create a frozen duplicate frame at an undecodable tail.
            break
        frames.append(path)
    if not frames:
        raise ValueError("The video contains no decodable frames.")
    return frames


def load_core(job_dir: Path):
    """Import core with its original implementation and an HTTP data bridge."""
    # A web deployment must not accidentally inherit a benchmark run's cached
    # pre-query memory or task-specific audit switches.
    for key in list(os.environ):
        if key.startswith("STREAMO_"):
            os.environ.pop(key)
    os.environ["STREAMO_VIDEO_DIR"] = str(job_dir / "media")
    os.environ["STREAMO_MAX_TOOL_CALLS_PER_REPLY"] = "5"
    os.environ["STREAMO_ZOOM_DUP_IOU"] = "0.85"
    install_swift_protocol_bridge()
    sys.path.insert(0, str(ROOT / "core"))
    return importlib.import_module("agent")


def attach_replay_delivery(registry) -> None:
    """Deliver replay evidence in the immediate tool-result turn.

    The original brief-mode loop can answer before consuming next_frame_idx.
    Use its existing image injection path so a successful replay is an actual
    historical image input. Keep the present clock unchanged (look and return).
    This deployment adapter does not edit or monkey-patch the core source.
    """
    from tools.temporal import ReplayTool

    class DeliveredReplay(ReplayTool):
        def run(self, args, context):
            result = super().run(args, context)
            if result.ok:
                timestamp = int(result.data["t"])
                image_index = len(context.next_round_images)
                context.next_round_images.append(context.frame_paths[timestamp])
                context.next_frame_idx = None
                result.data["evidence"] = (
                    f"Tool-evidence image {image_index + 1} in the next request is the "
                    f"historical frame at {timestamp}s. The last image is the present frame."
                )
            return result

    registry.register(DeliveredReplay())


def public_plan(plan: dict | None) -> dict[str, str]:
    """Return the real plan; the frontend controls which fields are visible."""
    return {str(key): str(value) for key, value in (plan or {}).items()}


def public_text(text: Any) -> str:
    return str(text or "")


def memory_records(processor, split_fields) -> dict[str, list[dict[str, Any]]]:
    memory = processor._memory
    if memory is None:
        return {"working": [], "episodes": [], "pins": []}
    working = []
    for event in memory.events:
        text = public_text(event.text)
        fields = split_fields(text) if event.kind == "observation" else {}
        working.append({
            "id": f"working-{event.eid}", "timestamp": event.t,
            "text": text, "kind": event.kind, "fields": fields,
            "description": fields.get("description", ""),
            "delta": fields.get("delta", event.delta or ""),
        })
    episodes = [{
        "id": f"episode-{item['id']}", "timestamp": item.get("t0", 0),
        "endTime": item.get("t1", 0), "text": public_text(item.get("summary", "")),
        "kind": item.get("kind", "episode"),
    } for item in memory.episodes]
    pins = [{
        "id": f"pin-{item.get('id', index)}",
        "timestamp": item.get("t", item.get("t0", 0)),
        "endTime": item.get("t1", item.get("t", 0)),
        "text": public_text(item.get("text", item.get("summary", ""))),
        "kind": item.get("kind", "fact"),
    } for index, item in enumerate(memory.pinned_facts)]
    return {"working": working, "episodes": episodes, "pins": pins}


def run(job_dir: Path) -> int:
    job_dir = job_dir.resolve()
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    previous_status = {}
    if (job_dir / "status.json").is_file():
        previous_status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    status = {
        "id": job["id"], "videoId": job.get("videoId"), "status": "running", "phase": "preparing",
        "question": str(job["question"]),
        "createdAt": previous_status.get("createdAt", datetime.now(timezone.utc).isoformat()),
        "progress": {"current": 0, "total": 0, "message": "Preparing video frames"},
        "answer": "", "error": None, "observations": [],
        "memory": {"working": [], "episodes": [], "pins": []},
        "plan": {}, "events": [], "modelCalls": 0,
        "queryTime": None,
    }
    cancelled = lambda: (job_dir / "cancel.requested").exists()
    engine = None
    processor = None
    core = None
    trajectory_dir = job_dir / "trajectory"

    def persist():
        if engine is not None:
            status["modelCalls"] = engine.request_count
        atomic_json(job_dir / "status.json", status)

    persist()
    try:
        video = Path(job["video_path"]).resolve()
        if not video.is_file():
            raise ValueError("The uploaded video is no longer available.")
        question = str(job["question"]).strip()
        duration = float(job["duration"])
        if not question or not math.isfinite(duration) or duration <= 0:
            raise ValueError("A question and a positive video duration are required.")
        if job.get("enable_audio") or job.get("enable_tracking"):
            raise ValueError("This demo exposes visual tools; audio transcription and tracking are not enabled.")
        max_steps = int(job.get("max_steps", 600))
        max_model_calls = int(job.get("max_model_calls", 300))
        if max_steps < 1 or max_model_calls < 1:
            raise ValueError("Step and model-call limits must be positive.")
        frames = extract_frames(video, job_dir / "frames", duration, cancelled)
        query_time = min(max(0, math.floor(float(job["query_time"]))), len(frames) - 1)
        status["queryTime"] = query_time
        status["progress"]["total"] = len(frames)
        status["progress"]["message"] = "Loading the streaming agent"
        persist()

        media = job_dir / "media"
        media.mkdir(exist_ok=True)
        # Some optional audit helpers resolve <sample id>.mp4. A private link
        # provides that convention without copying or changing the upload.
        target = media / f"{job['id']}.mp4"
        if not target.exists():
            target.symlink_to(video)
        core = load_core(job_dir)
        from tools.memory import _split_fields

        engine = InferenceEngine(
            EngineConfig.from_env(), max_model_calls=max_model_calls,
            cancel_check=cancelled,
        )
        registry = core.build_extended_registry(
            enable_memory=True, enable_efficiency=True,
            enable_track=False, enable_audio=False,
        )
        attach_replay_delivery(registry)
        system = core.render_agent_system(registry, memory_mode="brief", delta_input_mode="2frame")
        system += (
            "\nDeployment tool availability: only tools in the Available tools schemas "
            "can be called. Audio transcription and object tracking are not enabled in "
            "this deployment. Use visual observations, memory, replay, and zoom instead."
        )
        sample = {
            "id": job["id"], "benchmark": "uploaded_video", "task": "demo",
            "video": str(job_dir / "frames"), "source_video": str(video),
            "question": question, "realtime": query_time,
        }
        # In core, ask_time without options means an answerability meta-query.
        # Open-ended user questions therefore use realtime and no ask_time.
        processor = core.AgentVideoProcessor(
            sample, system, registry, memory_mode="brief", memory_token_budget=1500,
            max_tool_calls_per_round=3, max_tool_calls_per_sample=0,
            max_tool_calls_per_reply=5,
            budget=core.Budget(max_tool_calls=24, max_frames_seen=0, max_tokens=200000),
            trajectory_dir=str(trajectory_dir), engine=engine,
            auto_anchor_every=5, auto_asr_every=0, auto_track_query_every=0,
            memory_render_mode="continuous", memory_head_seconds=0,
            memory_tail_seconds=20, memory_recall_topk=8,
            delta_input_mode="2frame", enable_verification=True,
        )
        status["queryTime"] = processor.question_time
        status["phase"] = "observing"
        seen_tools = 0
        seen_observation_events = set()
        steps = 0
        while not processor.is_completed():
            if cancelled():
                raise Cancelled("Cancelled by the user.")
            if steps >= max_steps:
                raise ModelCallLimit("The configured agent-step limit was reached before completion.")
            previous = processor.output.get(f"Round {processor.current_round - 1}") if processor.current_round else None
            should_process, request = processor.get_next_request(previous)
            if processor.is_completed():
                break
            round_before = processor.current_round
            request_metadata = dict(processor._last_request or {})
            if round_before < processor.question_time:
                status["phase"] = "observing"
                message = "Observing frames and updating visual memory"
            elif not processor._query_plan:
                status["phase"] = "planning"
                message = "Planning which evidence the question needs"
            else:
                status["phase"] = "reasoning"
                message = "Inspecting memory and visual evidence"
            status["progress"]["current"] = min(len(frames), max(status["progress"]["current"], round_before))
            status["progress"]["message"] = message
            persist()
            response = "</Silence>"
            if should_process:
                config = RequestConfig(max_tokens=int(request_metadata.get("max_tokens") or 4096), temperature=0.1)
                response = engine.infer([request], config)[0].choices[0].message.content
            if cancelled():
                raise Cancelled("Cancelled by the user.")
            processor.update(response)
            steps += 1
            matches = list(OBSERVATION.finditer(core.ovo._strip_thinking(response)))
            for match in matches:
                body = match.group(1).strip()
                fields = _split_fields(body)
                status["observations"].append({
                    "id": f"observation-{len(status['observations'])}",
                    "timestamp": round_before, "text": body, "fields": fields,
                    "description": fields.get("description", ""), "delta": fields.get("delta", ""),
                })
            # Core also accepts an observation when the model omits its XML
            # wrapper. Retain those genuine memory records in the Delta view.
            for event in processor._memory.events:
                if event.kind != "observation" or event.eid in seen_observation_events:
                    continue
                seen_observation_events.add(event.eid)
                if not matches:
                    fields = _split_fields(event.text)
                    status["observations"].append({
                        "id": f"observation-{len(status['observations'])}",
                        "timestamp": event.t, "text": event.text, "fields": fields,
                        "description": fields.get("description", ""),
                        "delta": fields.get("delta", event.delta or ""),
                    })
            for index, item in enumerate(processor.tool_log[seen_tools:], start=seen_tools):
                call = item.get("call") or {}
                result = item.get("result") or {}
                name = str(call.get("name", "tool"))
                # Only public tool summaries, never backend file paths.
                text = f"{name}: {'completed' if result.get('ok') else 'not completed'}"
                status["events"].append({
                    "id": f"tool-{index}", "timestamp": item.get("round", round_before),
                    "type": "tool", "name": name, "text": text,
                })
            seen_tools = len(processor.tool_log)
            status["memory"] = memory_records(processor, _split_fields)
            status["plan"] = public_plan(processor._query_plan)
            status["progress"]["current"] = min(len(frames), max(status["progress"]["current"], processor.current_round))
            persist()

        final, trigger_round = core.ovo._extract_final_response(processor.output)
        if not final:
            raise RuntimeError("The model finished without a final answer. Try a clearer question or a shorter video.")
        status.update(status="completed", phase="answered", answer=final)
        status["progress"]["message"] = "Answer complete"
        core._dump_trajectory(str(trajectory_dir), processor, {"final_response": final, "trigger_round": trigger_round})
    except Cancelled as exc:
        status.update(status="cancelled", phase="cancelled", error=str(exc))
        status["progress"]["message"] = "Cancelled"
    except ModelCallLimit as exc:
        status.update(status="failed", phase="error", error=str(exc))
        status["progress"]["message"] = "Run limit reached"
    except Exception as exc:
        diagnostic_error(exc)
        # Model output and local filesystem paths should not become API errors.
        safe = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else f"{type(exc).__name__}: the run could not complete."
        for private in (str(job_dir), str(job.get("video_path", "")), os.environ.get("DEMO_API_KEY", "")):
            if private:
                safe = safe.replace(private, "[private]")
        status.update(status="failed", phase="error", error=safe)
        status["progress"]["message"] = "Run failed"
    finally:
        if processor is not None and core is not None and status["status"] != "completed":
            try:
                core._dump_trajectory(str(trajectory_dir), processor, {"final_response": None, "trigger_round": None})
            except Exception:
                pass
        if trajectory_dir.exists():
            for suffix, name in (("json", "trajectory.json"), ("steps.jsonl", "steps.jsonl")):
                original = trajectory_dir / f"{job['id']}.{suffix}"
                if original.is_file():
                    shutil.copyfile(original, job_dir / name)
        if status["status"] in {"completed", "failed", "cancelled"}:
            status["finishedAt"] = datetime.now(timezone.utc).isoformat()
        persist()
    return 0 if status["status"] in {"completed", "cancelled"} else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.job_dir))


if __name__ == "__main__":
    main()
