"""Environment-only configuration; API credentials never enter browser state."""
from dataclasses import dataclass, field
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def positive(name, default, cast=int):
    value = cast(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / ".data"
    backend: str = "local"
    api_base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = field(default="", repr=False)
    model: str = "DeltaStreamer-VLM"
    api_timeout: float = 180
    max_tokens: int = 4096
    max_pixels: int = 100352
    extra_body: dict = field(default_factory=dict)
    max_upload_mb: int = 512
    max_duration_seconds: float = 300
    max_model_calls: int = 600

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        backend = os.environ.get("DEMO_BACKEND", "local").strip().lower()
        if backend not in {"local", "remote"}:
            raise ValueError("DEMO_BACKEND must be local or remote")
        base = os.environ.get("DEMO_API_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("DEMO_API_BASE_URL must be an HTTP(S) base URL without credentials or query parameters")
        model = os.environ.get("DEMO_MODEL", "DeltaStreamer-VLM").strip()
        if not model:
            raise ValueError("DEMO_MODEL is required")
        extra = json.loads(os.environ.get("DEMO_EXTRA_BODY_JSON", "{}"))
        if not isinstance(extra, dict):
            raise ValueError("DEMO_EXTRA_BODY_JSON must be a JSON object")
        # These fields belong to the adapter, not provider-specific options.
        reserved = {"model", "messages", "stream", "tools", "tool_choice"}
        if reserved.intersection(extra):
            raise ValueError("DEMO_EXTRA_BODY_JSON cannot override model/messages/stream/tools")
        data_dir = Path(os.environ.get("DEMO_DATA_DIR", str(ROOT / ".data"))).expanduser().resolve()
        return cls(data_dir=data_dir, backend=backend, api_base_url=base,
                   api_key=os.environ.get("DEMO_API_KEY", ""), model=model,
                   api_timeout=positive("DEMO_API_TIMEOUT", 180, float),
                   max_tokens=positive("DEMO_MAX_TOKENS", 4096),
                   max_pixels=positive("DEMO_MAX_PIXELS", 100352), extra_body=extra,
                   max_upload_mb=positive("DEMO_MAX_UPLOAD_MB", 512),
                   max_duration_seconds=positive("DEMO_MAX_DURATION_SECONDS", 300, float),
                   max_model_calls=positive("DEMO_MAX_MODEL_CALLS", 600))

    def worker_env(self):
        env = dict(os.environ)
        env.update(DEMO_API_BASE_URL=self.api_base_url, DEMO_API_KEY=self.api_key,
                   DEMO_MODEL=self.model, DEMO_API_TIMEOUT=str(self.api_timeout),
                   DEMO_MAX_TOKENS=str(self.max_tokens), DEMO_MAX_PIXELS=str(self.max_pixels),
                   DEMO_EXTRA_BODY_JSON=json.dumps(self.extra_body), PYTHONUNBUFFERED="1")
        # GPU base images may inject an older system torch/lib into the
        # loader path. The worker must load libraries matching this venv.
        spec = importlib.util.find_spec("torch")
        if spec and spec.submodule_search_locations:
            library = Path(next(iter(spec.submodule_search_locations))) / "lib"
            if library.is_dir():
                env["LD_LIBRARY_PATH"] = str(library) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        return env

    def safe_error(self, error):
        value = str(error)
        if self.api_key:
            value = value.replace(self.api_key, "[redacted]")
        return value[:500]
