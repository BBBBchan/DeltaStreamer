"""Spatial tools: zoom_crop.

Currently the only spatial tool that requires no external model — pure PIL
crop + resize. The cropped patch is written to a temp dir and its path is
appended to ``ctx.next_round_images`` so that the agent main loop can inject
it as an extra ``<image>`` token in the next user turn.

Throttling (added 2026-06):
    The 27B agent tends to over-call zoom_crop (77-89% of all tool dispatches
    in 240-sample evals), starving track_* / asr_segment / memory_search.
    To mitigate this without changing the main loop, we now track per-frame
    zoom history on ``ctx.zoom_history`` and reject crops whose bbox is
    near-identical (IoU>=DUP_IOU) to a previous crop on the SAME frame.
    Optional per-frame / per-sample call caps are also available but
    DISABLED by default (set to 0). Tunables (env vars):
      * ``STREAMO_MAX_ZOOM_PER_FRAME``  (default 0 = unlimited)
      * ``STREAMO_MAX_ZOOM_PER_SAMPLE`` (default 0 = unlimited)
      * ``STREAMO_ZOOM_DUP_IOU``        (default 0.5)
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List

from .base import Tool, ToolContext, ToolResult

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is a hard dep elsewhere
    Image = None  # type: ignore


_DEFAULT_CACHE = os.path.join(tempfile.gettempdir(), "agent_zoom")
os.makedirs(_DEFAULT_CACHE, exist_ok=True)


# Throttling caps (env-overridable). 0 / negative disables that dimension.
def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, default))
        return v
    except (TypeError, ValueError):
        return default


_MAX_PER_FRAME = _env_int("STREAMO_MAX_ZOOM_PER_FRAME", 0)
_MAX_PER_SAMPLE = _env_int("STREAMO_MAX_ZOOM_PER_SAMPLE", 0)
# IoU threshold above which two crops are considered duplicates. Relaxed
# to 0.85 (was 0.5): near-duplicates are now RE-SERVED rather than
# rejected, so only almost-identical boxes are flagged as duplicates.
_DUP_IOU = float(os.environ.get("STREAMO_ZOOM_DUP_IOU", "0.85"))


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _normalize_bbox(bbox, w: int, h: int):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [x1,y1,x2,y2]")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    # Allow normalized coords in [0,1]
    if max(x1, y1, x2, y2) <= 1.0:
        x1 *= w; x2 *= w; y1 *= h; y2 *= h
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate bbox after clamp: {[x1, y1, x2, y2]}")
    return x1, y1, x2, y2


class ZoomCropTool(Tool):
    name = "zoom_crop"
    description = (
        "Crop a sub-region of the CURRENT frame at time `t` and upscale "
        "it to `target_size` square pixels. The cropped patch will be "
        "appended as an additional <image> in the very next round so "
        "you can read STATIC fine details (small printed text, a "
        "numeric label, a logo, a stationary face).\n"
        "**When NOT to use zoom_crop**:\n"
        "  - Subject is MOVING or you want to follow it across frames "
        "→ use `track_init` (zoom_crop's bbox is stale by next frame).\n"
        "  - Question depends on an EARLIER frame → use `replay` "
        "+ `memory_search` (zoom_crop only re-reads the "
        "CURRENT frame).\n"
        "  - Cue is AUDIO (speech / narration) → use `asr_segment`.\n"
        "  - You already inspected this region (high IoU with a prior "
        "crop on the same frame) → request will be auto-rejected; "
        "switch to a different tool family.\n"
        "**Best practice**: ONE well-chosen crop per question is "
        "usually enough. If the first crop didn't reveal the answer, "
        "the answer is probably NOT in the pixels — switch tool, do "
        "NOT chain more zooms."
    )
    parameters = {
        "type": "object",
        "properties": {
            "t": {"type": "number", "description": "Frame time in seconds."},
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "[x1,y1,x2,y2] in absolute pixels OR normalized [0,1]."
                ),
            },
            "target_size": {
                "type": "integer",
                "description": "Output side length, default 448.",
            },
        },
        "required": ["t", "bbox"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        # Hard kill-switch (ablation): if STREAMO_DISABLE_ZOOM_CROP is set,
        # reject every call so the model is forced to rely on track_init /
        # asr_segment / memory_search instead.
        if os.environ.get(
            "STREAMO_DISABLE_ZOOM_CROP", ""
        ).strip().lower() in ("1", "true", "yes"):
            return ToolResult(
                self.name, False,
                error=(
                    "zoom_crop is DISABLED in this run. Use "
                    "`track_init` to follow a salient subject across "
                    "frames, `asr_segment` for spoken cues, or "
                    "`memory_search` to recall earlier observations."
                ),
            )
        if Image is None:
            return ToolResult(self.name, False, error="PIL not available")
        try:
            t = int(round(float(args["t"])))
            bbox = args["bbox"]
            target = int(args.get("target_size", 448))
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(self.name, False, error=f"bad args: {e}")
        if not (0 <= t < len(ctx.frame_paths)):
            return ToolResult(self.name, False, error=f"t out of range: {t}")
        src = ctx.frame_paths[t]
        try:
            img = Image.open(src).convert("RGB")
            w, h = img.size
            x1, y1, x2, y2 = _normalize_bbox(bbox, w, h)
        except Exception as e:  # noqa: BLE001
            return ToolResult(self.name, False, error=f"image fail: {e}")

        # ---- Throttling: per-sample / per-frame / dedupe -------------
        if ctx.zoom_history is None:
            ctx.zoom_history = {}
        # Helper: append a "MUST commit" tail when the model is at the
        # last frame past question_time so the model can't keep retrying.
        is_last_frame = (
            ctx.total_rounds > 0
            and ctx.current_round >= ctx.total_rounds - 1
        )
        past_q = (
            int(getattr(ctx, "question_time", 0) or 0) > 0
            and ctx.current_round >= int(ctx.question_time)
        )
        commit_tail = (
            " *** This is the LAST FRAME — after at most ONE more "
            "supporting tool call, you MUST emit `</Response> "
            "<your answer>`. Do not stop at `</Silence>`."
            if is_last_frame and past_q else ""
        )
        # 1) per-sample cap
        total = sum(len(v) for v in ctx.zoom_history.values())
        if _MAX_PER_SAMPLE > 0 and total >= _MAX_PER_SAMPLE:
            return ToolResult(
                self.name, False,
                error=(
                    f"zoom_crop sample cap reached "
                    f"({_MAX_PER_SAMPLE}); switch to memory_search / "
                    f"track_init / asr_segment to gather new evidence."
                    f"{commit_tail}"
                ),
            )
        frame_hist = ctx.zoom_history.get(t, [])
        # 2) per-frame cap
        if _MAX_PER_FRAME > 0 and len(frame_hist) >= _MAX_PER_FRAME:
            return ToolResult(
                self.name, False,
                error=(
                    f"zoom_crop per-frame cap reached "
                    f"({_MAX_PER_FRAME} on t={t}s); inspect a DIFFERENT "
                    f"frame, or call track_init/memory_search/"
                    f"asr_segment instead.{commit_tail}"
                ),
            )
        # 2.5) phys-round soft cap (改动 7 promptfix v2): 同一物理轮内
        # zoom_crop 调用次数 >= 3 时拒收。错误信息引导切换工具家族——
        # 不会消耗 PHYS_CAP/per_round 配额（被 throttle 在工具内层）。
        # 防止 STU id=641 (87 次)、HLD id=321 (28 次)、FPD id=1135
        # (31 次) 这种 zoom 滥用；上限降到 3 同时压低单轮注入图峰值，
        # 避免超长视频 (如 Sample-545, 309 帧) 把 prompt token 撑爆
        # max_model_len 导致 max_tokens 变负、vLLM 崩溃整个 worker。
        _PHYS_ROUND_ZOOM_CAP = 3
        if ctx.zoom_count_this_phys_round >= _PHYS_ROUND_ZOOM_CAP:
            return ToolResult(
                self.name, False,
                error=(
                    f"zoom_crop phys-round cap reached "
                    f"({_PHYS_ROUND_ZOOM_CAP} on this physical round); "
                    f"further visual inspection is unlikely to help. "
                    f"SWITCH: (a) `replay(t=<frame>)` to revisit a "
                    f"different moment; (b) `memory_search` with "
                    f"question keywords; (c) commit `</Response>` "
                    f"with current evidence.{commit_tail}"
                ),
            )
        # 3) near-duplicate detection (IoU>=DUP_IOU on same frame).
        # Instead of REJECTING a near-duplicate (which left the model
        # blind and spinning when it genuinely needed another look at the
        # same region, e.g. counting overlapping cards in a hand), we now
        # RE-SERVE the crop: the pixels are re-cropped and returned with a
        # nudge to look carefully THIS time. The IoU threshold is also
        # relaxed (default 0.85) so only almost-identical boxes count.
        cur_box = [float(x1), float(y1), float(x2), float(y2)]
        is_dup = False
        dup_iou = 0.0
        for prev in frame_hist:
            iou = _bbox_iou(prev, cur_box)
            if iou >= _DUP_IOU:
                is_dup = True
                dup_iou = iou
                break

        # All checks passed — perform the actual crop.
        try:
            patch = img.crop((x1, y1, x2, y2)).resize(
                (target, target), Image.BICUBIC)
        except Exception as e:  # noqa: BLE001
            return ToolResult(self.name, False, error=f"image fail: {e}")

        out_name = (
            f"{ctx.sample_id.replace('/', '_')}_t{t}_"
            f"{x1}-{y1}-{x2}-{y2}_{int(time.time()*1000)}.jpg"
        )
        out_path = os.path.join(_DEFAULT_CACHE, out_name)
        patch.save(out_path, quality=92)
        ctx.next_round_images.append(out_path)
        # Record the bbox so future calls can be deduped.
        ctx.zoom_history.setdefault(t, []).append(cur_box)
        # v2 改动 7: 累计本物理轮成功 zoom 次数（用于 5 次软上限）
        ctx.zoom_count_this_phys_round += 1
        if ctx.memory is not None:
            ctx.memory.write_event(
                ctx.current_round, "tool",
                f"zoom_crop t={t}s bbox=[{x1},{y1},{x2},{y2}] ({target}px)"
                + (" [re-served duplicate]" if is_dup else ""),
                ref_frame=t, salience=1.3,
            )
        note = "patch will appear as an extra <image> in the next round"
        if is_dup:
            note = (
                f"You already cropped this region (IoU={dup_iou:.2f}); "
                "the SAME zoomed patch is shown again. Look CAREFULLY and "
                "state a concrete reading (e.g. an exact count) THIS time, "
                "then act on it — do not request the identical crop again."
            )
        return ToolResult(
            self.name,
            True,
            data={
                "t": t,
                "bbox": [x1, y1, x2, y2],
                "saved": out_path,
                "shape": [target, target, 3],
                "duplicate": is_dup,
                "note": note,
            },
        )
