"""Local upload-and-ask server. Run: python -m demo.server --port 7861."""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from PIL import Image
from pydantic import BaseModel, Field

from .config import REPO, ROOT, Settings
from .media import EXTENSIONS, prepare_video

TERMINAL = {"completed", "failed", "cancelled"}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def checked_id(value):
    if not re.fullmatch(r"[a-f0-9]{32}", value):
        raise HTTPException(404, "Item not found.")
    return value


class RunRequest(BaseModel):
    videoId: str
    question: str = Field(min_length=1, max_length=2000)
    queryTime: float | None = None


class RunManager:
    def __init__(self, settings, runner_command=None):
        self.settings = settings
        self.root = settings.data_dir / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processes = {}
        self.runner_command = runner_command or [sys.executable, "-m", "demo.runner"]

    def directory(self, run_id):
        directory = self.root / checked_id(run_id)
        if not (directory / "status.json").is_file():
            raise HTTPException(404, "Inference task not found.")
        return directory

    def _external_pid(self, directory):
        """Only recognize a surviving worker whose command names this exact job."""
        try:
            pid = int(json.loads((directory / "process.json").read_text())["pid"])
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
            if b"demo.runner" in command and str(directory).encode() in command:
                return pid
        except (OSError, ValueError, KeyError):
            pass
        return None

    def read(self, run_id):
        directory = self.directory(run_id)
        with self.lock:
            status = json.loads((directory / "status.json").read_text())
            process = self.processes.get(run_id)
            alive = process.poll() is None if process else bool(self._external_pid(directory))
            if status.get("status") not in TERMINAL and not alive:
                cancelled = (directory / "cancel.requested").exists()
                status.update(status="cancelled" if cancelled else "failed",
                              phase="cancelled" if cancelled else "error",
                              error=None if cancelled else "The inference worker stopped. See this task's worker.log on the server.",
                              finishedAt=now())
                atomic_json(directory / "status.json", status)
            return status

    def active_id(self):
        with self.lock:
            for path in sorted(self.root.glob("*/status.json")):
                if self.read(path.parent.name).get("status") not in TERMINAL:
                    return path.parent.name
        return None

    def start(self, video, question, query_time):
        with self.lock:
            active = self.active_id()
            if active:
                raise HTTPException(409, {"message": "Another inference task is active. View or cancel it before starting a new question.", "activeRunId": active})
            run_id = uuid.uuid4().hex
            directory = self.root / run_id
            directory.mkdir(mode=0o700)
            job = {"id": run_id, "videoId": video["id"], "video_path": video["original_path"],
                   "question": question, "query_time": query_time, "duration": video["duration"],
                   "max_steps": self.settings.max_model_calls * 2,
                   "max_model_calls": self.settings.max_model_calls,
                   "enable_audio": False, "enable_tracking": False}
            atomic_json(directory / "job.json", job)
            state = {"id": run_id, "videoId": video["id"], "status": "queued", "phase": "preparing",
                     "question": question, "queryTime": query_time, "createdAt": now(),
                     "progress": {"current": 0, "total": video["duration"], "message": "Preparing video frames."},
                     "answer": None, "error": None, "observations": [],
                     "memory": {"working": [], "episodes": [], "pins": []}, "plan": {}, "events": [], "modelCalls": 0}
            atomic_json(directory / "status.json", state)
            with (directory / "worker.log").open("wb") as log:
                try:
                    process = subprocess.Popen(self.runner_command + ["--job-dir", str(directory)],
                                               cwd=REPO, env=self.settings.worker_env(), stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as exc:
                    state.update(status="failed", phase="error", error="The inference worker could not start.")
                    atomic_json(directory / "status.json", state)
                    raise HTTPException(500, state["error"]) from exc
            self.processes[run_id] = process
            atomic_json(directory / "process.json", {"pid": process.pid})
            threading.Thread(target=self._watch, args=(run_id, process), daemon=True).start()
            return state

    def _watch(self, run_id, process):
        process.wait()
        self.read(run_id)

    def cancel(self, run_id):
        directory = self.directory(run_id)
        state = self.read(run_id)
        if state.get("status") in TERMINAL:
            return state
        (directory / "cancel.requested").touch()
        process = self.processes.get(run_id)
        # Allow a cooperative cancellation at the next model/tool boundary.
        if process:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=3)
        else:
            pid = self._external_pid(directory)
            if pid:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        with self.lock:
            state = json.loads((directory / "status.json").read_text())
            if state.get("status") not in TERMINAL:
                state.update(status="cancelled", phase="cancelled", error=None, finishedAt=now())
                atomic_json(directory / "status.json", state)
        return state

    def shutdown(self):
        for run_id, process in list(self.processes.items()):
            if process.poll() is None:
                self.cancel(run_id)


def create_app(settings=None, runner_command=None):
    settings = settings or Settings.from_env()
    videos = settings.data_dir / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    runs = RunManager(settings, runner_command)
    health_cache = {"time": 0, "value": None, "verified_until": 0}
    upload_lock = asyncio.Semaphore(1)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await asyncio.to_thread(runs.shutdown)

    app = FastAPI(title="DeltaStreamer Upload Demo", lifespan=lifespan)
    app.state.settings = settings
    app.state.runs = runs

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Use this application's own origin."}, status_code=403)
        return await call_next(request)

    def read_video(video_id, private=False):
        directory = videos / checked_id(video_id)
        path = directory / "metadata.json"
        if not path.is_file():
            raise HTTPException(404, "Uploaded video not found.")
        result = json.loads(path.read_text())
        return result if private else {k: v for k, v in result.items() if k != "original_path"}

    def headers():
        return {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}

    @app.get("/api/health")
    async def health():
        if (time.monotonic() > health_cache["verified_until"] and time.monotonic() - health_cache["time"] > 5) or health_cache["value"] is None:
            result = {"ready": False, "connectionStatus": "error"}
            try:
                async with httpx.AsyncClient(timeout=min(8, settings.api_timeout)) as client:
                    response = await client.get(settings.api_base_url + "/models", headers=headers())
                if response.status_code in {404, 405}:
                    result.update(ready=True, connectionStatus="unverified", message="This provider does not list models. Use Check connection to verify two-image input.")
                elif response.is_success:
                    models = [row.get("id") for row in response.json().get("data", [])]
                    found = settings.model in models
                    result.update(ready=found, connectionStatus="reachable" if found else "model_missing",
                                  message="Model endpoint is reachable." if found else "DEMO_MODEL was not found in the server's model list.")
                else:
                    result["message"] = f"Model endpoint returned HTTP {response.status_code}. Check the API URL and server-side key."
            except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                result["message"] = "Cannot reach the model endpoint. Check that the model server is running and DEMO_API_BASE_URL is correct."
            health_cache.update(time=time.monotonic(), value=result)
        return {"ok": True, "model": settings.model, "backend": settings.backend,
                "limits": {"maxUploadMB": settings.max_upload_mb, "maxDurationSeconds": settings.max_duration_seconds},
                "activeRunId": runs.active_id(), **health_cache["value"]}

    @app.post("/api/check")
    async def check_connection():
        content = [{"type": "text", "text": "Return exactly a JSON list with the dominant color of the first image, then the second image. Use English color names only."}]
        for color in ("red", "blue"):
            buffer = io.BytesIO()
            Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
            data = base64.b64encode(buffer.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}})
        body = {"model": settings.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": 128, "temperature": 0, "stream": False, **settings.extra_body}
        try:
            async with httpx.AsyncClient(timeout=settings.api_timeout) as client:
                response = await client.post(settings.api_base_url + "/chat/completions", headers=headers(), json=body)
            if not response.is_success:
                raise HTTPException(502, f"Vision check returned HTTP {response.status_code}. Check the model name, credentials, and provider options.")
            text = response.json()["choices"][0]["message"]["content"]
            match = re.search(r"\[[^\]]+\]", text or "")
            colors = json.loads(match.group()) if match else None
            correct = isinstance(colors, list) and [str(x).lower().strip() for x in colors] == ["red", "blue"]
        except httpx.HTTPError as exc:
            raise HTTPException(502, "The model endpoint could not complete the two-image check.") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise HTTPException(502, "The model returned no usable text for the two-image check. Check thinking-mode and output-token settings.") from exc
        if correct:
            health_cache.update(time=time.monotonic(), verified_until=time.monotonic() + 300,
                                value={"ready": True, "connectionStatus": "verified", "message": "Two-image input verified."})
        return {"ok": correct, "model": settings.model, "message": "Two-image input verified." if correct else "The endpoint responded, but the two-image color check did not pass. Verify vision support and provider options.", "response": settings.safe_error(text or "")}

    @app.post("/api/videos", status_code=201)
    async def upload_video(video: UploadFile = File(...)):
        name = Path((video.filename or "video.mp4").replace("\\", "/")).name[:240]
        extension = Path(name).suffix.lower()
        if extension not in EXTENSIONS:
            await video.close()
            raise HTTPException(415, "Use an MP4, MOV, MKV, WebM, AVI, or M4V video.")
        video_id = uuid.uuid4().hex
        directory = videos / video_id
        directory.mkdir(mode=0o700)
        original = directory / ("original" + extension)
        try:
            size = 0
            with original.open("wb") as target:
                while chunk := await video.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, f"Video exceeds the {settings.max_upload_mb} MB upload limit.")
                    target.write(chunk)
            if not size:
                raise HTTPException(400, "The uploaded file is empty.")
            async with upload_lock:
                details = await asyncio.to_thread(prepare_video, original, directory / "preview.mp4", settings.max_duration_seconds)
            metadata = {"id": video_id, "name": name, "url": f"/media/{video_id}/video.mp4",
                        **details, "size": size, "createdAt": now(), "original_path": str(original)}
            atomic_json(directory / "metadata.json", metadata)
            return read_video(video_id)
        except (HTTPException, ValueError) as exc:
            shutil.rmtree(directory, ignore_errors=True)
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(422, str(exc)) from exc
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            await video.close()

    @app.get("/api/videos/{video_id}")
    async def get_video(video_id: str):
        return read_video(video_id)

    @app.get("/media/{video_id}/video.mp4")
    async def video_media(video_id: str):
        read_video(video_id)
        return FileResponse(videos / video_id / "preview.mp4", media_type="video/mp4")

    @app.post("/api/runs", status_code=202)
    async def start_run(request: RunRequest):
        video = read_video(request.videoId, private=True)
        question = request.question.strip()
        if not question:
            raise HTTPException(422, "Enter a question.")
        query_time = video["duration"] if request.queryTime is None else request.queryTime
        if not math.isfinite(query_time) or not 0 <= query_time <= video["duration"]:
            raise HTTPException(422, "Query time must be within the uploaded video's duration.")
        return runs.start(video, question, query_time)

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        return runs.read(run_id)

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str):
        return await asyncio.to_thread(runs.cancel, run_id)

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7861, type=int)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
