"""Validate uploads and create browser-playable previews with FFmpeg."""
import json
import math
from pathlib import Path
import shutil
import subprocess

EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def binary(name):
    path = shutil.which(name)
    if not path:
        raise ValueError(f"{name} is required. Install FFmpeg and add it to PATH.")
    return path


def prepare_video(original: Path, preview: Path, max_duration: float):
    try:
        result = subprocess.run([
            binary("ffprobe"), "-v", "error", "-protocol_whitelist", "file,pipe",
            "-show_entries", "format=duration:stream=codec_type,width,height,duration",
            "-of", "json", str(original)
        ], capture_output=True, text=True, timeout=30, check=True)
        data = json.loads(result.stdout)
        stream = next(s for s in data.get("streams", []) if s.get("codec_type") == "video")
        duration = float(data.get("format", {}).get("duration") or stream.get("duration") or 0)
        width, height = int(stream["width"]), int(stream["height"])
    except (subprocess.SubprocessError, ValueError, KeyError, StopIteration) as exc:
        if isinstance(exc, ValueError) and "required" in str(exc):
            raise
        raise ValueError("The file could not be read as a video.") from exc
    if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise ValueError("The video must have a valid duration and image size.")
    if duration > max_duration:
        raise ValueError(f"Video exceeds the {max_duration:g}-second limit.")
    # Keep the original for actual inference. Only the browser preview is resized.
    temporary = preview.with_name("preview.partial.mp4")
    try:
        subprocess.run([
            binary("ffmpeg"), "-nostdin", "-v", "error", "-y", "-threads", "2",
            "-protocol_whitelist", "file,pipe", "-i", str(original), "-map", "0:v:0",
            "-map", "0:a:0?", "-vf", "scale=w='min(1280,iw)':h='min(1280,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(temporary)
        ], capture_output=True, text=True, timeout=max(120, duration * 4), check=True)
    except subprocess.SubprocessError as exc:
        temporary.unlink(missing_ok=True)
        raise ValueError("FFmpeg could not create a browser-playable video preview.") from exc
    temporary.replace(preview)
    return {"duration": duration, "width": width, "height": height}
