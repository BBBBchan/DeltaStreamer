"""A real-core smoke test with a deterministic local HTTP endpoint.

No GPU/model is used. The fixture makes an explicit replay tool call; the
assertion checks that core actually sends the historical red frame afterward.
"""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from PIL import Image
import pytest

from demo.config import Settings
from demo.runner import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def make_video(path):
    from demo.media import binary
    frames = path.parent / "source-frames"
    frames.mkdir()
    for index, color in enumerate(["red", "green", "blue", "white", "black"]):
        Image.new("RGB", (64, 48), color).save(frames / f"{index:03d}.png")
    subprocess.run([
        binary("ffmpeg"), "-nostdin", "-v", "error", "-y", "-threads", "2",
        "-framerate", "1", "-i", str(frames / "%03d.png"),
        "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-r", "10", str(path),
    ], check=True, capture_output=True)


@pytest.mark.parametrize("query_time,actual_time,via_server", [(3.8, 3, False), (5.0, 4, False), (3.8, 3, True)])
def test_real_core_replay_sends_historical_image(tmp_path, query_time, actual_time, via_server):
    seen = []
    replay_requested = False
    replay_received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            nonlocal replay_requested
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            user_text = "\n".join(
                message["content"] if isinstance(message["content"], str)
                else "\n".join(p.get("text", "") for p in message["content"])
                for message in payload["messages"] if message["role"] == "user"
            )
            images = [part["image_url"]["url"] for message in payload["messages"]
                      if isinstance(message["content"], list) for part in message["content"]
                      if part.get("type") == "image_url"]
            if not images:
                answer = "A solid red frame was shown before the later colors."
            elif replay_requested:
                replay_received.extend(images)
                answer = "I rewatched the earlier frame and saw red. </Response> Red."
            elif "[Phase] OBSERVE-ONLY." in user_text:
                answer = ("<Observation>description: A colored background is visible; actors: none; "
                          "objects: background; actions: none; ocr: none; scene: solid background; "
                          f"delta: The visible color at this observation is recorded at call {len(seen)}."
                          "</Observation>\n</Silence>")
            elif "PLANNING STEP" in user_text:
                answer = ("<Plan>\nneed: Identify the earliest color\ninfo_type: color\n"
                          "evidence_locus: past frame at 0 seconds\nanswerability: need to replay\n"
                          "meta: ordinary question\nstrategy: replay 0 seconds then answer\n</Plan>\n</Silence>")
            else:
                replay_requested = True
                answer = ('The current background is white, but the question asks about the first frame. I will replay t=0 to check the initial color. '
                          '<tool_call>{"name":"replay","arguments":{"t":0}}</tool_call>')
            body = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    video = tmp_path / "test.mp4"
    make_video(video)
    job = tmp_path / "job"
    job.mkdir()
    atomic_json(job / "job.json", {
        "id": "fixture", "videoId": "fixture-video", "video_path": str(video),
        "question": "What color appeared at the beginning?", "query_time": query_time,
        "duration": 5, "max_steps": 20, "max_model_calls": 20,
        "enable_audio": False, "enable_tracking": False,
    })
    settings = Settings(
        data_dir=tmp_path / "server-data", api_base_url=f"http://127.0.0.1:{server.server_port}/v1",
        model="fixture", api_key="", max_model_calls=20,
    )
    env = settings.worker_env()
    try:
        if via_server:
            from fastapi.testclient import TestClient
            from demo.server import create_app

            with TestClient(create_app(settings)) as client:
                with video.open("rb") as handle:
                    uploaded = client.post("/api/videos", files={"video": ("test.mp4", handle, "video/mp4")})
                assert uploaded.status_code == 201, uploaded.text
                started = client.post("/api/runs", json={
                    "videoId": uploaded.json()["id"],
                    "question": "What color appeared at the beginning?", "queryTime": query_time,
                })
                assert started.status_code == 202, started.text
                run_id = started.json()["id"]
                job = settings.data_dir / "runs" / run_id
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    current = client.get(f"/api/runs/{run_id}")
                    assert current.status_code == 200
                    if current.json()["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(0.05)
                else:
                    pytest.fail("The real demo worker did not finish within 60 seconds.")
                result = SimpleNamespace(
                    returncode=0 if current.json()["status"] == "completed" else 1,
                    stdout=(job / "worker.log").read_text(), stderr="",
                )
        else:
            result = subprocess.run([sys.executable, "-m", "demo.runner", "--job-dir", str(job)],
                                    cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
    finally:
        server.shutdown()
        server.server_close()
    status = json.loads((job / "status.json").read_text())
    assert result.returncode == 0, (status, result.stdout[-1000:], result.stderr[-1000:])
    assert status["status"] == "completed"
    assert status["answer"] == "Red."
    assert status["queryTime"] == actual_time
    assert status["plan"]["strategy"] == "replay 0 seconds then answer"
    assert status["observations"] and status["memory"]["working"]
    assert any(event["name"] == "replay" for event in status["events"])
    assert replay_requested and replay_received
    historical = Image.open(io.BytesIO(base64.b64decode(replay_received[0].split(",", 1)[1])))
    red, green, blue = historical.getpixel((20, 20))
    assert red > 200 and green < 30 and blue < 30
    trajectory = json.loads((job / "trajectory.json").read_text())
    assert trajectory["question"] == "What color appeared at the beginning?"
    steps = [json.loads(line) for line in (job / "steps.jsonl").read_text().splitlines()]
    assert any(step["parsed_tool_calls"] for step in steps)
    assert not any("*** META-QUESTION" in json.dumps(item) for item in seen)
    assert status["modelCalls"] == len(seen)


def test_worker_diagnostics_redact_credentials(monkeypatch, capsys):
    from demo.runner import diagnostic_error

    monkeypatch.setenv("DEMO_API_KEY", "private-test-key")
    try:
        raise AttributeError("Provider failed for private-test-key and Bearer another-secret")
    except AttributeError as exc:
        diagnostic_error(exc)
    output = capsys.readouterr().err
    assert "AttributeError" in output and "test_runner.py" in output
    assert "private-test-key" not in output and "another-secret" not in output
