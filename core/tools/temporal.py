"""Temporal-control tools: replay / wait / skip.

These tools mutate :attr:`ToolContext.next_frame_idx` (or trigger a
fast-forward of ``current_round``) in the agent main loop.

They are *zero-dependency*: pure clock manipulation, no external models.
"""
from __future__ import annotations

from typing import Any, Dict

from .base import Tool, ToolContext, ToolResult


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class ReplayTool(Tool):
    name = "replay"
    description = (
        "RE-WATCH a past moment: replay the frame at absolute time `t` "
        "(seconds), then automatically return to the present. This is your "
        "PRIMARY way to ground an answer in what you actually SAW — "
        "prefer it over guessing from MEMORY text. "
        "Whenever a question hinges on a specific past moment (an action, "
        "an object's state, a color, a count, who was present), `replay` "
        "that moment and verify with your own eyes BEFORE answering. "
        "You can ONLY replay the PAST — `t` must lie within "
        "[0, current time]; frames after NOW have not streamed in yet and "
        "a future `t` is clamped back to the current frame. "
        "The very next inference round is conditioned on the frame at `t` "
        "so you can RE-WATCH that exact moment; after that single round "
        "the streaming clock snaps BACK to where you left off and the "
        "stream continues forward (look-and-return — it does NOT restart "
        "the video from `t`, so replaying is CHEAP: one look, then back). "
        "When in doubt, replay first; an answer backed by a re-watched "
        "frame beats an answer guessed from memory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "t": {
                "type": "number",
                "description": "Absolute time in seconds (>=0, <= current time).",
            }
        },
        "required": ["t"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            t = int(round(float(args["t"])))
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(self.name, False, error=f"invalid t: {e}")
        if ctx.total_rounds <= 0:
            return ToolResult(self.name, False, error="empty video")
        # Past-only: the visible range is [0, current_round]. Frames after
        # NOW have not streamed in yet, so a future target is clamped back
        # to the current frame and the model is told it cannot look ahead.
        upper = max(0, min(ctx.current_round, ctx.total_rounds - 1))
        requested = t
        t = _clamp(t, 0, upper)
        clamped_future = requested > upper
        ctx.next_frame_idx = t
        if ctx.memory is not None:
            ctx.memory.write_event(
                ctx.current_round, "tool",
                f"replay t={t}s (from r{ctx.current_round})",
                ref_frame=t, salience=1.1,
            )
        data = {"t": t, "from_round": ctx.current_round}
        if clamped_future:
            data["note"] = (
                f"requested t={requested}s is in the FUTURE (you are at "
                f"t={ctx.current_round}s); you can only re-watch frames "
                f"that already streamed by. Clamped to t={t}s."
            )
        return ToolResult(self.name, True, data=data)


class WaitTool(Tool):
    name = "wait"
    description = (
        "Skip ahead by `seconds` without producing intermediate output, "
        "i.e. fast-forward the streaming clock. Useful for long videos where "
        "the next informative segment is known to be later."
    )
    parameters = {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "How many seconds to skip ahead (>=1).",
            }
        },
        "required": ["seconds"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            sec = int(round(float(args["seconds"])))
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(self.name, False, error=f"invalid seconds: {e}")
        if sec <= 0:
            return ToolResult(self.name, False, error="seconds must be > 0")
        target = _clamp(ctx.current_round + sec, 0, max(0, ctx.total_rounds - 1))
        ctx.next_frame_idx = target
        if ctx.memory is not None:
            ctx.memory.write_event(
                ctx.current_round, "tool",
                f"wait {sec}s -> t={target}s",
                ref_frame=target, salience=0.8,
            )
        return ToolResult(self.name, True, data={"t": target, "skipped": sec})


# Skip is an alias of Wait, kept for prompt fluency
class SkipTool(WaitTool):
    name = "skip"
    description = (
        "Alias of `wait`: skip ahead by `seconds`. Provided for prompt fluency."
    )
