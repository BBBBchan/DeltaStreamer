"""Tracking tools (Phase-A: Mock implementation, opt-in real SAM2.1).

A real :class:`SAM2Tracker` is provided that uses
``sam2.SAM2ImagePredictor`` with the SAM2.1-Hiera-Large checkpoint to
predict object masks frame by frame. Tracking is a simple bbox-prompt
chain (re-prompt each frame with the previous frame's predicted bbox);
sufficient for "where did this object go" queries on streaming tasks.

Activation: set ``STREAMO_TRACKER_BACKEND=sam2`` (else falls back to
:class:`MockTracker`). Optional env vars:

* ``STREAMO_SAM2_CKPT`` (default
  ``checkpoints/sam2.1_hiera_large.pt``)
* ``STREAMO_SAM2_CONFIG`` (default ``configs/sam2.1/sam2.1_hiera_l.yaml``)
* ``STREAMO_SAM2_DEVICE`` (default ``cuda`` if available else ``cpu``)
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import Tool, ToolContext, ToolResult

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


# ---------------------------------------------------------------------------
# Tracker adapter interface
# ---------------------------------------------------------------------------
class TrackerAdapter:
    """Minimal interface every tracker (mock or SAM2) must satisfy."""

    def init(self, frame_path: str, bbox: List[float],
             text_prompt: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def query(self, session: Dict[str, Any],
              frame_path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def end(self, session: Dict[str, Any]) -> Dict[str, Any]:
        return {"finalized": True}


@dataclass
class MockTracker(TrackerAdapter):
    decay_per_step: float = 0.04
    cutoff: float = 0.2

    def init(self, frame_path, bbox, text_prompt=None):
        if Image is None or not os.path.exists(frame_path):
            w = h = 0
        else:
            try:
                w, h = Image.open(frame_path).size
            except Exception:
                w = h = 0
        return {
            "init_frame": frame_path,
            "init_bbox": list(bbox),
            "prompt": text_prompt,
            "frame_size": [w, h],
            "step": 0,
            "score": 1.0,
        }

    def query(self, session, frame_path):
        session["step"] = int(session.get("step", 0)) + 1
        score = max(0.0, 1.0 - self.decay_per_step * session["step"])
        session["score"] = score
        in_view = score >= self.cutoff
        return {
            "bbox": session["init_bbox"] if in_view else None,
            "score": round(score, 3),
            "in_view": in_view,
        }


# ---------------------------------------------------------------------------
# SAM2-backed adapter (opt-in)
# ---------------------------------------------------------------------------
class SAM2Tracker(TrackerAdapter):
    """SAM2.1 image-predictor based tracker.

    Strategy (deliberately simple):
      * On ``init``: build SAM2 predictor lazily; run on the init frame
        with the user-provided bbox prompt; remember the resulting mask
        bbox as ``last_bbox``.
      * On each ``query(t)``: re-set image to frame_t, prompt with
        ``last_bbox`` (treated as a box prompt → SAM2 returns a refined
        mask); update ``last_bbox`` for the next call.
      * Confidence is taken from SAM2's IoU prediction.
      * If the predicted mask is empty / very small, mark as out of view.

    Note: this is **not** the full video propagation predictor (which
    requires all frames preloaded). For our agent's "look forward 2-3
    seconds, did the object leave the frame" use case it's accurate
    enough and runs on a single frame at a time.
    """

    def __init__(self, ckpt_path: Optional[str] = None,
                 config_file: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        self.ckpt_path = ckpt_path or os.environ.get(
            "STREAMO_SAM2_CKPT",
            "checkpoints/sam2.1_hiera_large.pt",
        )
        self.config_file = config_file or os.environ.get(
            "STREAMO_SAM2_CONFIG",
            "configs/sam2.1/sam2.1_hiera_l.yaml",
        )
        self.device = device or os.environ.get("STREAMO_SAM2_DEVICE")
        self._predictor = None
        self._lock = threading.Lock()

    # -- helpers ------------------------------------------------------
    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_predictor(self):
        if self._predictor is None:
            with self._lock:
                if self._predictor is None:
                    from sam2.build_sam import build_sam2
                    from sam2.sam2_image_predictor import SAM2ImagePredictor
                    device = self._resolve_device()
                    model = build_sam2(
                        config_file=self.config_file,
                        ckpt_path=self.ckpt_path,
                        device=device,
                    )
                    self._predictor = SAM2ImagePredictor(model)
                    self._device = device
        return self._predictor

    @staticmethod
    def _mask_to_bbox(mask) -> Optional[List[float]]:
        import numpy as np
        if mask is None:
            return None
        m = mask.astype(bool) if hasattr(mask, "astype") else mask
        ys, xs = np.where(m)
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return [float(x0), float(y0), float(x1), float(y1)]

    def _predict(self, frame_path: str, box: List[float]
                 ) -> Tuple[Optional[List[float]], float, Optional[List[int]]]:
        import numpy as np
        from PIL import Image as _PIL
        predictor = self._ensure_predictor()
        img = np.array(_PIL.open(frame_path).convert("RGB"))
        h, w = img.shape[:2]
        # Clamp / sanitize box
        x0, y0, x1, y1 = box
        x0 = max(0, min(int(x0), w - 1))
        x1 = max(0, min(int(x1), w))
        y0 = max(0, min(int(y0), h - 1))
        y1 = max(0, min(int(y1), h))
        if x1 <= x0 or y1 <= y0:
            return None, 0.0, [w, h]
        try:
            import torch
            ctx_mgr = (torch.inference_mode() if torch is not None
                       else _NullCtx())
        except Exception:
            ctx_mgr = _NullCtx()
        with ctx_mgr:
            predictor.set_image(img)
            masks, ious, _ = predictor.predict(
                box=np.array([x0, y0, x1, y1]),
                multimask_output=False,
            )
        if masks is None or len(masks) == 0:
            return None, 0.0, [w, h]
        mask = masks[0]
        iou = float(ious[0]) if ious is not None and len(ious) else 0.0
        bbox = self._mask_to_bbox(mask)
        return bbox, iou, [w, h]

    # -- adapter methods ---------------------------------------------
    def init(self, frame_path, bbox, text_prompt=None):
        bbox = [float(v) for v in (bbox or [0, 0, 0, 0])]
        try:
            new_bbox, score, size = self._predict(frame_path, bbox)
            sess = {
                "init_frame": frame_path,
                "init_bbox": list(bbox),
                "last_bbox": new_bbox or list(bbox),
                "prompt": text_prompt,
                "frame_size": size or [0, 0],
                "step": 0,
                "score": score,
                "lost": new_bbox is None,
            }
        except Exception as e:  # noqa: BLE001
            # Fall back to mock-style session so caller stays alive.
            sess = {
                "init_frame": frame_path,
                "init_bbox": list(bbox),
                "last_bbox": list(bbox),
                "prompt": text_prompt,
                "frame_size": [0, 0],
                "step": 0,
                "score": 1.0,
                "lost": False,
                "error": f"sam2 init: {type(e).__name__}: {e}",
            }
        return sess

    def query(self, session, frame_path):
        session["step"] = int(session.get("step", 0)) + 1
        prev = session.get("last_bbox") or session.get("init_bbox")
        if not prev or session.get("lost"):
            return {"bbox": None, "score": 0.0, "in_view": False}
        try:
            new_bbox, score, _ = self._predict(frame_path, prev)
        except Exception as e:  # noqa: BLE001
            return {"bbox": None, "score": 0.0, "in_view": False,
                    "error": f"sam2 query: {type(e).__name__}: {e}"}
        if new_bbox is None or score < 0.3:
            session["lost"] = True
            return {"bbox": None, "score": round(score, 3), "in_view": False}
        session["last_bbox"] = new_bbox
        session["score"] = score
        return {"bbox": new_bbox, "score": round(float(score), 3),
                "in_view": True}


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _build_default_tracker() -> TrackerAdapter:
    backend = os.environ.get("STREAMO_TRACKER_BACKEND", "mock").lower()
    if backend == "sam2":
        try:
            return SAM2Tracker()
        except Exception:
            return MockTracker()
    return MockTracker()


DEFAULT_TRACKER: TrackerAdapter = _build_default_tracker()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def _ensure_tracks(ctx: ToolContext) -> Dict[str, Any]:
    if ctx.tracks is None:
        ctx.tracks = {}
    return ctx.tracks


def _new_track_id(ctx: ToolContext) -> str:
    n = len(_ensure_tracks(ctx)) + 1
    return f"trk{n:02d}"


class TrackInitTool(Tool):
    name = "track_init"
    description = (
        "Start a SAM3 tracking session on the frame at time `t`. Provide "
        "either an explicit `bbox` (pixel or normalized) or a "
        "`text_prompt` describing the target. Returns a `track_id` you "
        "must pass to subsequent `track_query` / `track_end` calls."
    )
    parameters = {
        "type": "object",
        "properties": {
            "t": {"type": "integer"},
            "bbox": {"type": "array",
                      "items": {"type": "number"},
                      "minItems": 4, "maxItems": 4},
            "text_prompt": {"type": "string"},
        },
        "required": ["t"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            t = int(args["t"])
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(self.name, False, error=f"bad t: {e}")
        if not (0 <= t < len(ctx.frame_paths)):
            return ToolResult(self.name, False,
                              error=f"t out of range: {t}")
        bbox = args.get("bbox")
        prompt = args.get("text_prompt")
        if bbox is None and not prompt:
            return ToolResult(self.name, False,
                              error="bbox or text_prompt required")
        try:
            sess = DEFAULT_TRACKER.init(
                ctx.frame_paths[t],
                list(bbox) if bbox is not None else [0, 0, 0, 0],
                prompt,
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(self.name, False, error=f"tracker init: {e}")
        tid = _new_track_id(ctx)
        ctx.tracks[tid] = {
            "session": sess, "start_t": t, "prompt": prompt,
            "init_bbox": list(bbox) if bbox is not None else None,
            "traj": [], "status": "active",
        }
        if ctx.memory is not None:
            ctx.memory.upsert_track(tid, start_t=t, prompt=prompt,
                                     init_bbox=ctx.tracks[tid]["init_bbox"])
            ctx.memory.write_event(
                t, "tool",
                f"track_init {tid} prompt={prompt or 'bbox'}",
                ref_frame=t, salience=1.2,
            )
        return ToolResult(self.name, True,
                          data={"track_id": tid, "start_t": t})


class TrackQueryTool(Tool):
    name = "track_query"
    description = (
        "Query an active SAM3 track at time `t`. Returns the predicted "
        "bbox, confidence, and whether the target is still in view."
    )
    parameters = {
        "type": "object",
        "properties": {
            "track_id": {"type": "string"},
            "t": {"type": "integer"},
        },
        "required": ["track_id", "t"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tid = args.get("track_id")
        if not tid or ctx.tracks is None or tid not in ctx.tracks:
            return ToolResult(self.name, False,
                              error=f"unknown track_id: {tid}")
        try:
            t = int(args["t"])
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(self.name, False, error=f"bad t: {e}")
        if not (0 <= t < len(ctx.frame_paths)):
            return ToolResult(self.name, False,
                              error=f"t out of range: {t}")
        tr = ctx.tracks[tid]
        if tr.get("status") != "active":
            return ToolResult(self.name, False,
                              error=f"track {tid} not active")
        out = DEFAULT_TRACKER.query(tr["session"], ctx.frame_paths[t])
        tr["traj"].append({"t": t, **out})
        if ctx.memory is not None:
            ctx.memory.upsert_track(tid, traj_append=[{"t": t, **out}])
            # Also write a retrievable working-layer event so the bbox /
            # visibility survives past the current physical round and can be
            # surfaced by memory_search / snapshot. Without this, the per-
            # frame track_query result only lives in the tracks side-index
            # (which retrieve()/search() never scan) and evaporates once the
            # streaming clock ticks — forcing the model to re-query the same
            # target repeatedly. Mirror the auto_track event format.
            bbox = out.get("bbox")
            in_view = out.get("in_view")
            if in_view and bbox:
                ctx.memory.write_event(
                    t, "tool",
                    f"track_query {tid} bbox={[int(v) for v in bbox]} "
                    f"score={out.get('score')}",
                    ref_frame=t, salience=1.0,
                )
            elif not in_view:
                ctx.memory.write_event(
                    t, "tool",
                    f"track_query {tid} lost (score={out.get('score')})",
                    ref_frame=t, salience=1.1,
                )
        return ToolResult(self.name, True, data={"track_id": tid, **out})


class TrackEndTool(Tool):
    name = "track_end"
    description = (
        "End a SAM3 tracking session and obtain its summary "
        "(duration, sample count, max confidence drop)."
    )
    parameters = {
        "type": "object",
        "properties": {"track_id": {"type": "string"}},
        "required": ["track_id"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tid = args.get("track_id")
        if not tid or ctx.tracks is None or tid not in ctx.tracks:
            return ToolResult(self.name, False,
                              error=f"unknown track_id: {tid}")
        tr = ctx.tracks[tid]
        if tr.get("status") != "active":
            return ToolResult(self.name, False,
                              error=f"track {tid} already ended")
        DEFAULT_TRACKER.end(tr["session"])
        traj = tr["traj"]
        duration = (traj[-1]["t"] - tr["start_t"]) if traj else 0
        scores = [s.get("score", 0.0) for s in traj]
        summary = {
            "duration_s": duration,
            "samples": len(traj),
            "score_min": round(min(scores), 3) if scores else None,
            "score_max": round(max(scores), 3) if scores else None,
            "lost_at": next((s["t"] for s in traj if not s.get("in_view")), None),
        }
        tr["status"] = "ended"
        tr["summary"] = summary
        if ctx.memory is not None:
            ctx.memory.upsert_track(tid, status="ended", summary=summary)
            ctx.memory.write_event(
                tr["start_t"] + duration, "tool",
                f"track_end {tid} dur={duration}s lost@{summary['lost_at']}",
                salience=1.2,
            )
        return ToolResult(self.name, True,
                          data={"track_id": tid, "summary": summary})
