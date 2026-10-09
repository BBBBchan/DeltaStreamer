"""OpenAI-compatible transport for the unchanged DeltaStreamer agent.

The small request classes are protocol containers, not model implementations.
Only the isolated demo worker installs them under ``swift``; model serving is
performed by the separately configured HTTP endpoint.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


@dataclass
class InferRequest:
    messages: list[dict[str, Any]]
    images: list[Any] = field(default_factory=list)
    videos: list[Any] = field(default_factory=list)
    audios: list[Any] = field(default_factory=list)


@dataclass
class RequestConfig:
    max_tokens: int = 4096
    temperature: float = 0.1


class InferStats:
    def __init__(self) -> None:
        self.requests = 0

    def compute(self) -> dict[str, int]:
        return {"http_requests": self.requests}


def install_swift_protocol_bridge() -> None:
    """Supply the three data types imported by core, in a worker process only.

    This does not import ms-swift, transformers, vLLM, or a model. Refuse to
    replace an already imported inference framework in a shared process.
    """
    existing = sys.modules.get("swift")
    if existing is not None:
        if getattr(existing, "_deltastreamer_http_bridge", False):
            return
        raise RuntimeError("Run the demo worker in a fresh process; swift is already imported.")
    swift = ModuleType("swift")
    swift.__path__ = []
    swift._deltastreamer_http_bridge = True
    swift.InferRequest = InferRequest
    swift.InferStats = InferStats
    infer_engine = ModuleType("swift.infer_engine")
    infer_engine.RequestConfig = RequestConfig
    swift.infer_engine = infer_engine
    sys.modules["swift"] = swift
    sys.modules["swift.infer_engine"] = infer_engine


@dataclass
class EngineConfig:
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout: float = 180.0
    max_tokens: int = 4096
    max_pixels: int = 100352
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "EngineConfig":
        raw = os.environ.get("DEMO_EXTRA_BODY_JSON", "{}")
        try:
            extra = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("DEMO_EXTRA_BODY_JSON must be a JSON object.") from exc
        if not isinstance(extra, dict):
            raise ValueError("DEMO_EXTRA_BODY_JSON must be a JSON object.")
        return cls(
            base_url=os.environ.get("DEMO_API_BASE_URL", "http://127.0.0.1:8000/v1"),
            model=os.environ.get("DEMO_MODEL", "DeltaStreamer-VLM"),
            api_key=os.environ.get("DEMO_API_KEY", ""),
            timeout=float(os.environ.get("DEMO_API_TIMEOUT", "180")),
            max_tokens=int(os.environ.get("DEMO_MAX_TOKENS", "4096")),
            max_pixels=int(os.environ.get("DEMO_MAX_PIXELS", "100352")),
            extra_body=extra,
        )

    def validate(self) -> None:
        url = urllib.parse.urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("The API base URL must use http:// or https://.")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Keep API credentials in DEMO_API_KEY, not the URL.")
        if not self.model.strip():
            raise ValueError("DEMO_MODEL is required.")
        if self.timeout <= 0 or self.max_tokens <= 0 or self.max_pixels < 3136:
            raise ValueError("Timeout, token limit, and image pixel limit must be positive.")
        if set(self.extra_body) & {"messages", "model", "stream", "tools", "tool_choice"}:
            raise ValueError("API extra parameters cannot override messages/model or the agent tool protocol.")


class Cancelled(BaseException):
    """Escape core's optional-summary error fallback after cancellation."""


class ModelCallLimit(BaseException):
    """Escape core's optional-summary error fallback when the budget ends."""


class InferenceEngine:
    """Preserve agent messages and visual evidence while calling a vision API."""

    # Core can optionally use a local tokenizer to enforce context limits.
    # HTTP model tokenization belongs to the configured model server.
    tokenizer = None
    max_model_len = 0

    def __init__(self, config: EngineConfig, *, max_model_calls: int = 300, cancel_check=None):
        config.validate()
        self.config = config
        self.max_model_calls = max_model_calls
        self.cancel_check = cancel_check or (lambda: False)
        self.request_count = 0
        self.image_count = 0
        self.elapsed_seconds = 0.0

    def _image_part(self, path: Any) -> dict[str, Any]:
        from PIL import Image

        if not isinstance(path, (str, Path)) or not Path(path).is_file():
            raise ValueError("An agent image is missing or is not a local file.")
        with Image.open(path) as source:
            picture = source.convert("RGB")
            pixels = picture.width * picture.height
            if pixels > self.config.max_pixels:
                ratio = math.sqrt(self.config.max_pixels / pixels)
                picture = picture.resize(
                    (max(1, int(picture.width * ratio)), max(1, int(picture.height * ratio))),
                    Image.Resampling.LANCZOS,
                )
            stream = io.BytesIO()
            picture.save(stream, format="JPEG", quality=90)
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}

    def messages(self, request: InferRequest) -> list[dict[str, Any]]:
        """Replace image markers in place, retaining previous/current order.

        The optional core counting audit supplies a video as an ordered image
        list. These images are expanded in place; actual video URLs and audio
        are rejected instead of silently being omitted by a text-only API.
        """
        if request.audios:
            raise ValueError("This HTTP transport does not accept audio inputs.")
        images, videos = iter(request.images or []), iter(request.videos or [])
        image_used, video_used = 0, 0
        messages: list[dict[str, Any]] = []
        for original in request.messages:
            message = dict(original)
            content = message.get("content", "")
            if not isinstance(content, str):
                raise ValueError("Expected the core agent's textual messages with image markers.")
            if message.get("role") != "user":
                messages.append(message)
                continue
            pieces = re.split(r"(<image>|<video>)", content)
            if len(pieces) == 1:
                messages.append(message)
                continue
            parts = []
            for piece in pieces:
                if piece == "<image>":
                    try:
                        picture = next(images)
                    except StopIteration as exc:
                        raise ValueError("Image marker and image count do not match.") from exc
                    parts.append(self._image_part(picture))
                    image_used += 1
                elif piece == "<video>":
                    try:
                        frames = next(videos)
                    except StopIteration as exc:
                        raise ValueError("Video marker and video count do not match.") from exc
                    if not isinstance(frames, (list, tuple)) or not frames:
                        raise ValueError("Video evidence must be a nonempty ordered list of local frame files.")
                    for frame in frames:
                        parts.append(self._image_part(frame))
                        image_used += 1
                    video_used += 1
                elif piece:
                    parts.append({"type": "text", "text": piece})
            message["content"] = parts
            messages.append(message)
        if next(images, None) is not None or next(videos, None) is not None:
            raise ValueError("Visual evidence has no corresponding image/video marker.")
        return messages

    def infer(self, requests: list[InferRequest], config: RequestConfig, metrics=None):
        responses = []
        for request in requests:
            if self.cancel_check():
                raise Cancelled("Cancelled by the user.")
            if self.request_count >= self.max_model_calls:
                raise ModelCallLimit("The configured model-call limit was reached before completion.")
            messages = self.messages(request)
            payload = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": min(config.max_tokens, self.config.max_tokens),
                "temperature": config.temperature,
                "stream": False,
                **self.config.extra_body,
            }
            url = self.config.base_url.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            http_request = urllib.request.Request(
                url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers, method="POST",
            )
            started = time.monotonic()
            self.request_count += 1
            try:
                with urllib.request.urlopen(http_request, timeout=self.config.timeout) as response:
                    raw = response.read(32 * 1024 * 1024 + 1)
                if len(raw) > 32 * 1024 * 1024:
                    raise RuntimeError("The model API response exceeded 32 MiB.")
                result = json.loads(raw)
            except urllib.error.HTTPError as exc:
                # Server error bodies can include requests or credentials.
                # Keep UI errors actionable without echoing those bodies.
                raise RuntimeError(
                    f"Model API returned HTTP {exc.code}. Check the server model name, "
                    "vision support, credentials, and context limit."
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(
                    "The model API could not be reached or timed out. Check the configured endpoint and server."
                ) from None
            except json.JSONDecodeError:
                raise RuntimeError("The model API returned invalid JSON.") from None
            try:
                message = result["choices"][0]["message"]
                content = message["content"]
                if isinstance(content, list):
                    content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError
            except (KeyError, IndexError, TypeError, ValueError):
                raise RuntimeError("The model API returned no text answer in choices[0].message.content.") from None
            self.elapsed_seconds += time.monotonic() - started
            if self.cancel_check():
                raise Cancelled("Cancelled by the user.")
            self.image_count += sum(
                part.get("type") == "image_url"
                for message in messages if isinstance(message.get("content"), list)
                for part in message["content"]
            )
            for metric in metrics or []:
                metric.requests += 1
            responses.append(SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))]))
        return responses
