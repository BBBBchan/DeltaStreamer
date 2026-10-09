"""Upload and worker-lifecycle tests; no model or external API is called."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from demo.config import Settings
from demo.media import binary
from demo.server import create_app


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "video.mp4"
    subprocess.run([
        binary("ffmpeg"), "-v", "error", "-nostdin", "-y", "-f", "lavfi",
        "-i", "testsrc=size=160x90:rate=10:duration=2", "-c:v", "libx264",
        "-threads", "2", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return path.read_bytes()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", api_key="test-private-key", max_upload_mb=1)


def upload(client, video):
    response = client.post("/api/videos", files={"video": ("clip.mp4", video, "video/mp4")})
    assert response.status_code == 201, response.text
    return response.json()


def test_real_upload_playback_and_input_validation(settings, video):
    with TestClient(create_app(settings)) as client:
        item = upload(client, video)
        assert item["duration"] == 2
        assert (item["width"], item["height"]) == (160, 90)
        assert "original_path" not in item
        assert client.get("/api/videos/" + item["id"]).json() == item
        media = client.get(item["url"], headers={"Range": "bytes=0-99"})
        assert media.status_code == 206
        assert len(media.content) == 100
        assert media.headers["content-type"] == "video/mp4"
        assert client.get("/api/videos/not-a-valid-id").status_code == 404
        assert client.post("/api/runs", json={"videoId": item["id"], "question": "   "}).status_code == 422
        for stamp in (-1, 2.1):
            assert client.post("/api/runs", json={"videoId": item["id"], "question": "What changed?", "queryTime": stamp}).status_code == 422
        assert client.post("/api/videos", files={"video": ("script.html", b"<html>")}).status_code == 415
        assert client.post("/api/videos", files={"video": ("empty.mp4", b"")}).status_code == 400
        assert client.post("/api/videos", files={"video": ("fake.mp4", b"not video")}).status_code == 422
        assert client.post("/api/videos", files={"video": ("large.mp4", b"x" * (1024 * 1024 + 1))}).status_code == 413
        assert len(list((settings.data_dir / "videos").iterdir())) == 1


def test_duration_limit_cleans_rejected_upload(settings, video):
    with TestClient(create_app(replace(settings, max_duration_seconds=1))) as client:
        result = client.post("/api/videos", files={"video": ("long.mp4", video)})
        assert result.status_code == 422
        assert "limit" in result.json()["detail"]
        assert list((settings.data_dir / "videos").iterdir()) == []


def test_single_active_worker_and_cancellation(settings, video, tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("import time\ntime.sleep(30)\n")
    with TestClient(create_app(settings, [sys.executable, str(worker)])) as client:
        item = upload(client, video)
        job = {"videoId": item["id"], "question": "What changed?"}
        first = client.post("/api/runs", json=job)
        assert first.status_code == 202
        run_id = first.json()["id"]
        assert first.json()["queryTime"] == 2
        busy = client.post("/api/runs", json=job)
        assert busy.status_code == 409
        assert busy.json()["detail"]["activeRunId"] == run_id
        assert settings.api_key not in (settings.data_dir / "runs" / run_id / "job.json").read_text()
        state = client.post(f"/api/runs/{run_id}/cancel").json()
        assert state["status"] == "cancelled"
        assert client.get(f"/api/runs/{run_id}").json()["status"] == "cancelled"
        second = client.post("/api/runs", json=job)
        assert second.status_code == 202
        assert second.json()["id"] != run_id
        client.post(f"/api/runs/{second.json()['id']}/cancel")


def test_crashed_worker_is_reported(settings, video):
    with TestClient(create_app(settings, [sys.executable, "-c", "raise SystemExit(7)"])) as client:
        item = upload(client, video)
        run_id = client.post("/api/runs", json={"videoId": item["id"], "question": "What?"}).json()["id"]
        for _ in range(50):
            state = client.get(f"/api/runs/{run_id}").json()
            if state["status"] == "failed":
                break
            time.sleep(0.02)
        assert state["status"] == "failed"
        assert "worker stopped" in state["error"]


def test_connection_check_is_explicit_and_verified_result_persists(settings, monkeypatch):
    requests = []
    def handler(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(404, json={"error": "No model list"})
        body = json.loads(request.content)
        images = [x for x in body["messages"][0]["content"] if x["type"] == "image_url"]
        assert len(images) == 2
        assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
        return httpx.Response(200, json={"choices": [{"message": {"content": '["red", "blue"]'}}]})

    original = httpx.AsyncClient
    monkeypatch.setattr("demo.server.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    with TestClient(create_app(settings)) as client:
        first = client.get("/api/health").json()
        assert first["connectionStatus"] == "unverified"
        assert all(request.method == "GET" for request in requests)
        assert settings.api_key not in json.dumps(first)
        assert client.post("/api/check").json()["ok"] is True
        assert client.get("/api/health").json()["connectionStatus"] == "verified"
        assert client.post("/api/check", headers={"Origin": "https://unrelated.example"}).status_code == 403


def test_configuration_hides_key_and_rejects_protocol_overrides(monkeypatch):
    monkeypatch.setenv("DEMO_API_KEY", "test-private-key")
    monkeypatch.setenv("DEMO_EXTRA_BODY_JSON", '{"messages": []}')
    with pytest.raises(ValueError, match="cannot override"):
        Settings.from_env()
    monkeypatch.setenv("DEMO_EXTRA_BODY_JSON", "{}")
    settings = Settings.from_env()
    assert "test-private-key" not in repr(settings)
    assert settings.safe_error("failed: test-private-key") == "failed: [redacted]"
