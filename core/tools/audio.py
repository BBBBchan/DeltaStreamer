"""Audio / ASR tools (Phase-A: Mock implementation, opt-in real Whisper).

Provides a thin :class:`ASRAdapter` interface and a deterministic
:class:`MockASR` that returns an empty transcript with neutral metadata
so the agent main loop can be developed without a real STT service.

When the env var ``STREAMO_ASR_BACKEND=whisper`` is set, ``DEFAULT_ASR``
is swapped for :class:`WhisperASR`, which:

* Locates the source mp4 via ``$STREAMO_VIDEO_DIR/<sample_id>.mp4``
  (default ``data/OVO-Bench/chunked_videos_asi_full``).
* Lazily extracts the audio track once per ``sample_id`` to a temp wav.
* Lazily loads ``whisper.load_model(STREAMO_WHISPER_MODEL or 'base')``.
* For ``transcribe``: runs whisper on the [t_start, t_end] sub-clip.
* ``speaker`` / ``event`` are not natively supported by Whisper and
  return neutral fallbacks (the model ignores empty results).

If anything fails (no audio track, ffmpeg missing, ckpt missing, ...)
each call returns the same shape as :class:`MockASR` so the agent loop
stays robust.

``STREAMO_ASR_BACKEND=sidecar`` selects :class:`SidecarASR`, a read-only
adapter for precomputed fixed-window Whisper transcripts. It validates the
sample ID, question time, and source-media fingerprint before exposing only
windows whose end time is already visible to the streaming agent.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import Tool, ToolContext, ToolResult


_LEGACY_PATH_PREFIXES = ()


def _path_prefix_rewrites() -> List[tuple[str, str]]:
    """Return configured and built-in mount-prefix relocations."""
    rewrites: List[tuple[str, str]] = []
    raw = os.environ.get("STREAMO_PATH_PREFIX_REWRITES", "")
    for item in raw.split(","):
        if "=" not in item:
            continue
        old, new = (part.strip().rstrip("/") for part in item.split("=", 1))
        if old and new:
            rewrites.append((old, new))
    rewrites.extend(_LEGACY_PATH_PREFIXES)
    return rewrites


def resolve_relocated_media_path(path: str) -> Optional[str]:
    """Resolve a media path even when a symlink retains an old mount root."""
    candidates = [os.path.abspath(path)]
    if os.path.islink(path):
        target = os.readlink(path)
        if not os.path.isabs(target):
            target = os.path.join(os.path.dirname(path), target)
        candidates.append(os.path.abspath(target))
    for candidate in list(candidates):
        for old, new in _path_prefix_rewrites():
            if candidate == old or candidate.startswith(old + os.sep):
                candidates.append(new + candidate[len(old):])
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    return None


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------
class ASRAdapter:
    def transcribe(self, sample_id: str, t_start: int, t_end: int
                   ) -> Dict[str, Any]:
        raise NotImplementedError

    def speaker(self, sample_id: str, t: int) -> Dict[str, Any]:
        raise NotImplementedError

    def event(self, sample_id: str, t: int) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class MockASR(ASRAdapter):
    def transcribe(self, sample_id, t_start, t_end):
        return {"text": "", "lang": "und", "confidence": 0.0,
                "t0": t_start, "t1": t_end}

    def speaker(self, sample_id, t):
        return {"speaker": None, "confidence": 0.0, "t": t}

    def event(self, sample_id, t):
        return {"label": None, "confidence": 0.0, "t": t}


# ---------------------------------------------------------------------------
# Precomputed fixed-window adapter (opt-in)
# ---------------------------------------------------------------------------
class SidecarASR(ASRAdapter):
    """Read precomputed ASR windows without loading Whisper at runtime."""

    def __init__(
        self,
        sidecar_dir: Optional[str] = None,
        video_dir: Optional[str] = None,
    ) -> None:
        self.sidecar_dir = sidecar_dir or os.environ.get(
            "STREAMO_ASR_SIDECAR_DIR", ""
        )
        if not self.sidecar_dir:
            raise ValueError(
                "STREAMO_ASR_SIDECAR_DIR is required for sidecar ASR"
            )
        self.video_dir = video_dir or os.environ.get(
            "STREAMO_VIDEO_DIR",
            "data/"
            "OVO-Bench/chunked_videos_asi_full",
        )
        self._payload_cache: Dict[str, Dict[str, Any]] = {}

    def _load(self, sample_id: str) -> Dict[str, Any]:
        sid = str(sample_id)
        if sid in self._payload_cache:
            return self._payload_cache[sid]
        path = os.path.join(self.sidecar_dir, f"{sid}.json")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            not isinstance(payload, dict)
            or str(payload.get("sample_id")) != sid
            or not isinstance(payload.get("question_time"), int)
            or not isinstance(payload.get("segments"), list)
        ):
            raise ValueError(f"invalid ASR sidecar: {path}")

        previous_end = 0
        for segment in payload["segments"]:
            if not isinstance(segment, dict):
                raise ValueError(f"invalid ASR sidecar window: {path}")
            t0 = segment.get("t0")
            t1 = segment.get("t1")
            if (
                not isinstance(t0, int)
                or not isinstance(t1, int)
                or t0 < 0
                or t1 <= t0
                or t0 < previous_end
                or t1 > payload["question_time"]
            ):
                raise ValueError(f"invalid ASR sidecar window: {path}")
            previous_end = t1

        media = payload.get("media") or {}
        source = resolve_relocated_media_path(
            os.path.join(self.video_dir, f"{sid}.mp4")
        )
        if source is None:
            raise FileNotFoundError(f"ASR sidecar media missing: {sid}")
        stat = os.stat(source)
        if (
            os.path.realpath(source) != str(media.get("path") or "")
            or stat.st_size != media.get("size")
            or stat.st_mtime_ns != media.get("mtime_ns")
        ):
            raise ValueError(f"stale ASR sidecar media fingerprint: {sid}")

        self._payload_cache[sid] = payload
        return payload

    def ready_segments(
        self,
        sample_id: str,
        *,
        after_t: int,
        current_t: int,
        question_time: int,
    ) -> List[Dict[str, Any]]:
        """Return newly visible windows, never the question-time boundary."""
        payload = self._load(sample_id)
        if int(payload["question_time"]) != int(question_time):
            raise ValueError(
                f"ASR sidecar question time mismatch for {sample_id}: "
                f"{payload['question_time']} != {question_time}"
            )
        return [
            dict(segment)
            for segment in payload["segments"]
            if (
                int(segment["t1"]) > int(after_t)
                and int(segment["t1"]) <= int(current_t)
                and int(segment["t1"]) < int(question_time)
            )
        ]

    def transcribe(self, sample_id, t_start, t_end):
        """Approximate an arbitrary past-only request from complete windows."""
        payload = self._load(str(sample_id))
        visible_end = min(int(t_end), int(payload["question_time"]) - 1)
        segments = [
            segment
            for segment in payload["segments"]
            if (
                int(segment["t1"]) <= visible_end
                and int(segment["t1"]) > int(t_start)
                and str(segment.get("text") or "").strip()
            )
        ]
        text = " ".join(
            str(segment.get("text") or "").strip()
            for segment in segments
        )
        return {
            "text": text,
            "lang": (
                str(segments[-1].get("lang") or "und")
                if segments else "und"
            ),
            "confidence": 1.0 if text else 0.0,
            "t0": int(t_start),
            "t1": int(t_end),
            "sidecar_windows": len(segments),
        }

    def speaker(self, sample_id, t):
        return {"speaker": None, "confidence": 0.0, "t": t}

    def event(self, sample_id, t):
        return {"label": None, "confidence": 0.0, "t": t}


# ---------------------------------------------------------------------------
# Whisper-backed adapter (opt-in)
# ---------------------------------------------------------------------------
class WhisperASR(ASRAdapter):
    """Real ASR using openai-whisper. Lazy model + per-sample audio cache."""

    def __init__(self, model_name: str = "base",
                 video_dir: Optional[str] = None,
                 ffmpeg_bin: Optional[str] = None,
                 download_root: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.video_dir = video_dir or os.environ.get(
            "STREAMO_VIDEO_DIR",
            "data/OVO-Bench/chunked_videos_asi_full",
        )
        self.download_root = download_root or os.environ.get(
            "STREAMO_WHISPER_DOWNLOAD_ROOT",
            "checkpoints/whisper",
        )
        # torch_npu's CUDA compatibility layer can make Whisper select an NPU
        # implicitly. Keep ASR on CPU unless a validated device is requested.
        self.device = device or os.environ.get(
            "STREAMO_WHISPER_DEVICE", "cpu"
        )
        self.ffmpeg_bin = ffmpeg_bin or self._resolve_ffmpeg()
        # whisper.load_audio() invokes a bare `ffmpeg` subprocess and
        # ignores any explicit binary path, so we must make the resolved
        # binary discoverable on PATH.
        self._inject_ffmpeg_on_path()
        self._model = None
        self._lock = threading.Lock()
        # sample_id -> path to extracted 16k mono wav (or "" if no audio)
        self._wav_cache: Dict[str, str] = {}
        self._transcript_cache: Dict[tuple[str, int, int], Dict[str, Any]] = {}
        self.cache_dir = os.environ.get(
            "STREAMO_ASR_CACHE_DIR",
            os.path.join(
                os.path.dirname(os.path.abspath(self.video_dir)),
                ".cache",
                "whisper_windows",
            ),
        )
        # tmpdir for wav extracts (lazy)
        self._tmpdir: Optional[str] = None

    def _inject_ffmpeg_on_path(self) -> None:
        if not self.ffmpeg_bin or not os.path.exists(self.ffmpeg_bin):
            return
        # If the resolved binary is named exactly `ffmpeg` already, just
        # prepend its directory to PATH. Otherwise, drop a symlink named
        # `ffmpeg` into a private dir and prepend that dir.
        bin_dir = os.path.dirname(self.ffmpeg_bin)
        bin_name = os.path.basename(self.ffmpeg_bin)
        if bin_name == "ffmpeg":
            shim_dir = bin_dir
        else:
            shim_dir = tempfile.mkdtemp(prefix="streamo_ffmpeg_shim_")
            shim_path = os.path.join(shim_dir, "ffmpeg")
            try:
                if not os.path.exists(shim_path):
                    os.symlink(self.ffmpeg_bin, shim_path)
            except OSError:
                # symlink may not be permitted; copy instead
                try:
                    shutil.copy2(self.ffmpeg_bin, shim_path)
                    os.chmod(shim_path, 0o755)
                except Exception:
                    return
        cur = os.environ.get("PATH", "")
        if shim_dir not in cur.split(os.pathsep):
            os.environ["PATH"] = shim_dir + os.pathsep + cur

    # -- helpers ------------------------------------------------------
    @staticmethod
    def _resolve_ffmpeg() -> str:
        env = os.environ.get("STREAMO_FFMPEG")
        if env and os.path.exists(env):
            return env
        which = shutil.which("ffmpeg")
        if which:
            return which
        try:
            import imageio_ffmpeg  # type: ignore
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"  # last-resort, will fail loudly

    def _ensure_tmpdir(self) -> str:
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="streamo_asr_")
        return self._tmpdir

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import whisper  # lazy import (heavy)
                    if self.download_root:
                        os.makedirs(self.download_root, exist_ok=True)
                        self._model = whisper.load_model(
                            self.model_name,
                            download_root=self.download_root,
                            device=self.device,
                        )
                    else:
                        self._model = whisper.load_model(
                            self.model_name,
                            device=self.device,
                        )
        return self._model

    def _extract_audio(self, sample_id: str) -> str:
        if sample_id in self._wav_cache:
            return self._wav_cache[sample_id]
        # Try a few common id->mp4 mappings (OVO-Bench: chunked_videos/<id>.mp4)
        candidates = [os.path.join(self.video_dir, f"{sample_id}.mp4")]
        # strip trailing "_step_*" suffixes if present
        if "_step_" in sample_id:
            base = sample_id.split("_step_")[0]
            candidates.append(os.path.join(self.video_dir, f"{base}.mp4"))
        mp4 = next(
            (
                resolved
                for candidate in candidates
                if (resolved := resolve_relocated_media_path(candidate))
            ),
            None,
        )
        if not mp4:
            self._wav_cache[sample_id] = ""
            return ""
        wav = os.path.join(self._ensure_tmpdir(), f"{sample_id}.wav")
        if os.path.exists(wav):
            self._wav_cache[sample_id] = wav
            return wav
        # ffmpeg: -vn drop video, 16k mono pcm wav
        cmd = [self.ffmpeg_bin, "-y", "-i", mp4, "-vn",
               "-ac", "1", "-ar", "16000", "-f", "wav", wav]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=180)
            if r.returncode != 0 or not os.path.exists(wav):
                self._wav_cache[sample_id] = ""
                return ""
        except Exception:
            self._wav_cache[sample_id] = ""
            return ""
        self._wav_cache[sample_id] = wav
        return wav

    def _cache_path(
        self,
        sample_id: str,
        t_start: int,
        t_end: int,
    ) -> str:
        source = resolve_relocated_media_path(
            os.path.join(self.video_dir, f"{sample_id}.mp4")
        )
        if source is None:
            fingerprint = "missing"
        else:
            stat = os.stat(source)
            fingerprint = hashlib.sha256(
                (
                    f"{os.path.realpath(source)}\0{stat.st_size}\0"
                    f"{stat.st_mtime_ns}"
                ).encode()
            ).hexdigest()[:20]
        safe_id = hashlib.sha256(str(sample_id).encode()).hexdigest()[:16]
        filename = (
            f"{safe_id}_{fingerprint}_{self.model_name}_"
            f"{int(t_start)}_{int(t_end)}.json"
        )
        return os.path.join(self.cache_dir, filename)

    def _load_cached_transcript(
        self,
        sample_id: str,
        t_start: int,
        t_end: int,
    ) -> Optional[Dict[str, Any]]:
        key = (str(sample_id), int(t_start), int(t_end))
        if key in self._transcript_cache:
            return dict(self._transcript_cache[key])
        path = self._cache_path(sample_id, t_start, t_end)
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("sample_id") != str(sample_id)
            or payload.get("model") != self.model_name
            or payload.get("t0") != int(t_start)
            or payload.get("t1") != int(t_end)
            or not isinstance(payload.get("result"), dict)
        ):
            return None
        result = dict(payload["result"])
        result["cache_hit"] = True
        self._transcript_cache[key] = result
        return dict(result)

    def _store_cached_transcript(
        self,
        sample_id: str,
        t_start: int,
        t_end: int,
        result: Dict[str, Any],
    ) -> None:
        cache_key = (str(sample_id), int(t_start), int(t_end))
        stored = {
            field: value
            for field, value in result.items()
            if field != "cache_hit"
        }
        self._transcript_cache[cache_key] = dict(stored)
        path = self._cache_path(sample_id, t_start, t_end)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                prefix=os.path.basename(path) + ".",
                suffix=".tmp",
                dir=os.path.dirname(path),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "sample_id": str(sample_id),
                            "model": self.model_name,
                            "t0": int(t_start),
                            "t1": int(t_end),
                            "result": stored,
                        },
                        handle,
                        ensure_ascii=False,
                    )
                    handle.write("\n")
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except OSError:
            # Cache failure must not make ASR unavailable.
            return

    # -- adapter methods ---------------------------------------------
    def transcribe(self, sample_id, t_start, t_end):
        cached = self._load_cached_transcript(sample_id, t_start, t_end)
        if cached is not None:
            return cached
        wav = self._extract_audio(sample_id)
        if not wav:
            return {"text": "", "lang": "und", "confidence": 0.0,
                    "t0": t_start, "t1": t_end, "no_audio": True}
        try:
            import numpy as np  # noqa: F401
            import whisper
            audio = whisper.load_audio(wav)
            sr = 16000
            i0 = max(0, int(t_start) * sr)
            i1 = min(len(audio), int(t_end) * sr)
            if i1 <= i0:
                return {"text": "", "lang": "und", "confidence": 0.0,
                        "t0": t_start, "t1": t_end}
            clip = audio[i0:i1]
            model = self._ensure_model()
            # Whisper only supports fp16 on CUDA; CPU must stay fp32.
            use_fp16 = str(self.device).startswith("cuda")
            res = model.transcribe(clip, fp16=use_fp16)
            text = (res.get("text") or "").strip()
            lang = res.get("language") or "und"
            result = {
                "text": text,
                "lang": lang,
                "confidence": 1.0 if text else 0.0,
                "t0": t_start,
                "t1": t_end,
            }
            self._store_cached_transcript(
                sample_id, t_start, t_end, result
            )
            return result
        except Exception as e:  # noqa: BLE001
            return {"text": "", "lang": "und", "confidence": 0.0,
                    "t0": t_start, "t1": t_end, "error": str(e)[:160]}

    def speaker(self, sample_id, t):
        # whisper does not provide speaker diarization; return neutral.
        return {"speaker": None, "confidence": 0.0, "t": t,
                "note": "whisper has no speaker-id; returning neutral"}

    def event(self, sample_id, t):
        return {"label": None, "confidence": 0.0, "t": t,
                "note": "whisper has no audio-event detection"}


def _build_default_asr() -> ASRAdapter:
    backend = os.environ.get("STREAMO_ASR_BACKEND", "mock").lower()
    if backend == "sidecar":
        try:
            return SidecarASR()
        except Exception:
            return MockASR()
    if backend == "whisper":
        try:
            return WhisperASR(
                model_name=os.environ.get("STREAMO_WHISPER_MODEL", "base"),
            )
        except Exception:
            return MockASR()
    return MockASR()


DEFAULT_ASR: ASRAdapter = _build_default_asr()


def _ensure_audio_cache(ctx: ToolContext) -> Dict[str, Any]:
    if ctx.audio_cache is None:
        ctx.audio_cache = {"asr": [], "speakers": [], "events": []}
    return ctx.audio_cache


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
class ASRSegmentTool(Tool):
    name = "asr_segment"
    description = (
        "Run automatic speech recognition on the audio span "
        "[t_start, t_end]. Result text is cached in memory.audio_segments "
        "for later retrieval; useful whenever the visible query depends "
        "on speech, identity, event details, or repetition evidence."
    )
    parameters = {
        "type": "object",
        "properties": {
            "t_start": {"type": "integer"},
            "t_end":   {"type": "integer"},
        },
        "required": ["t_start", "t_end"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            t0 = int(args["t_start"])
            t1 = int(args["t_end"])
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(self.name, False, error=f"bad args: {e}")
        if t1 <= t0:
            return ToolResult(self.name, False, error="t_end must be > t_start")
        cache = _ensure_audio_cache(ctx)
        try:
            res = DEFAULT_ASR.transcribe(ctx.sample_id, t0, t1)
        except Exception as e:  # noqa: BLE001
            return ToolResult(self.name, False, error=f"asr: {e}")
        cache["asr"].append(res)
        if ctx.memory is not None:
            ctx.memory.add_audio(t0, t1, res.get("text", ""), None)
            if res.get("text"):
                ctx.memory.write_event(
                    t0, "audio",
                    f"asr[{t0}-{t1}s]: {res['text'][:80]}",
                    ref_frame=t0, salience=1.5,
                )
        return ToolResult(self.name, True, data=res)


class SpeakerIDTool(Tool):
    name = "speaker_id"
    description = (
        "Identify which speaker is talking at time `t` (returns a stable "
        "label such as `S0`, `S1`, ...). Cached per sample."
    )
    parameters = {
        "type": "object",
        "properties": {"t": {"type": "integer"}},
        "required": ["t"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            t = int(args["t"])
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(self.name, False, error=f"bad t: {e}")
        cache = _ensure_audio_cache(ctx)
        try:
            res = DEFAULT_ASR.speaker(ctx.sample_id, t)
        except Exception as e:  # noqa: BLE001
            return ToolResult(self.name, False, error=f"asr: {e}")
        cache["speakers"].append(res)
        return ToolResult(self.name, True, data=res)


class AudioEventTool(Tool):
    name = "audio_event"
    description = (
        "Detect non-speech audio event at time `t` (laughter, vehicle, "
        "music, ...). Returns label + confidence."
    )
    parameters = {
        "type": "object",
        "properties": {"t": {"type": "integer"}},
        "required": ["t"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            t = int(args["t"])
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(self.name, False, error=f"bad t: {e}")
        cache = _ensure_audio_cache(ctx)
        try:
            res = DEFAULT_ASR.event(ctx.sample_id, t)
        except Exception as e:  # noqa: BLE001
            return ToolResult(self.name, False, error=f"asr: {e}")
        cache["events"].append(res)
        return ToolResult(self.name, True, data=res)
