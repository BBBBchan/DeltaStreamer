"""Transport checks; these use local synthetic images, never a model service."""
from pathlib import Path
import io
import json
import urllib.error

from PIL import Image
import pytest

from demo.engine import Cancelled, EngineConfig, InferRequest, InferenceEngine, ModelCallLimit, RequestConfig


def image(tmp_path: Path, name: str, color: str) -> Path:
    path = tmp_path / name
    Image.new("RGB", (24, 24), color).save(path)
    return path


def test_image_order_and_no_silent_drop(tmp_path):
    first, second = image(tmp_path, "a.png", "red"), image(tmp_path, "b.png", "blue")
    engine = InferenceEngine(EngineConfig("http://localhost:9999/v1", "test"))
    result = engine.messages(InferRequest(
        messages=[{"role": "user", "content": "previous <image> current <image>"}],
        images=[first, second],
    ))
    parts = result[0]["content"]
    assert parts[0]["text"] == "previous "
    assert parts[2]["text"] == " current "
    assert parts[1] == engine._image_part(first)
    assert parts[3] == engine._image_part(second)
    with pytest.raises(ValueError, match="no corresponding"):
        engine.messages(InferRequest(messages=[{"role": "user", "content": "text"}], images=[first]))
    with pytest.raises(ValueError, match="count do not match"):
        engine.messages(InferRequest(messages=[{"role": "user", "content": "<image>"}]))


def test_video_frame_list_and_invalid_remote_video(tmp_path):
    first = image(tmp_path, "frame.png", "red")
    engine = InferenceEngine(EngineConfig("http://localhost:9999/v1", "test"))
    message = engine.messages(InferRequest(messages=[{"role": "user", "content": "<video>"}], videos=[[first, first]]))
    assert len(message[0]["content"]) == 2
    with pytest.raises(ValueError, match="ordered list"):
        engine.messages(InferRequest(messages=[{"role": "user", "content": "<video>"}], videos=["https://example.com/a.mp4"]))


def test_limits_and_cancellation_prevent_http():
    request = InferRequest(messages=[{"role": "user", "content": "test"}])
    engine = InferenceEngine(EngineConfig("http://localhost:9999/v1", "test"), max_model_calls=0)
    with pytest.raises(ModelCallLimit):
        engine.infer([request], RequestConfig())
    engine = InferenceEngine(EngineConfig("http://localhost:9999/v1", "test"), cancel_check=lambda: True)
    with pytest.raises(Cancelled):
        engine.infer([request], RequestConfig())


def test_credentials_do_not_enter_repr_and_reserved_overrides_fail():
    config = EngineConfig("http://localhost:9999/v1", "test", "secret-value")
    assert "secret-value" not in repr(config)
    with pytest.raises(ValueError):
        EngineConfig("http://secret@localhost:9999/v1", "test").validate()
    with pytest.raises(ValueError):
        EngineConfig("http://localhost:9999/v1", "test", extra_body={"messages": []}).validate()


def test_summary_calls_count_toward_the_same_hard_limit(monkeypatch):
    sent = []

    def endpoint(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "A summary."}}]}).encode())

    monkeypatch.setattr("demo.engine.urllib.request.urlopen", endpoint)
    engine = InferenceEngine(EngineConfig("http://localhost:9999/v1", "test"), max_model_calls=2)
    request = InferRequest(messages=[{"role": "user", "content": "Summarize the observations."}])
    for _ in range(2):
        assert engine.infer([request], RequestConfig(max_tokens=200))[0].choices[0].message.content == "A summary."
    with pytest.raises(ModelCallLimit):
        engine.infer([request], RequestConfig(max_tokens=200))
    assert engine.request_count == len(sent) == 2


def test_api_error_body_and_cancellation_do_not_leak_credentials(monkeypatch):
    def failed(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "secret-value", {}, io.BytesIO(b"secret-value"))

    monkeypatch.setattr("demo.engine.urllib.request.urlopen", failed)
    engine = InferenceEngine(EngineConfig("http://localhost:9999/v1", "test", "secret-value"))
    request = InferRequest(messages=[{"role": "user", "content": "Test"}])
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        engine.infer([request], RequestConfig())
    assert "secret-value" not in str(error.value)
    assert engine.request_count == 1
    checks = iter([False, True])
    monkeypatch.setattr("demo.engine.urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(
        b'{"choices":[{"message":{"content":"Late answer"}}]}'
    ))
    engine = InferenceEngine(EngineConfig("http://localhost:9999/v1", "test"), cancel_check=lambda: next(checks))
    with pytest.raises(Cancelled):
        engine.infer([request], RequestConfig())
