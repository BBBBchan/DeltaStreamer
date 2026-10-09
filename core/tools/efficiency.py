"""Efficiency-control tools: set_sample_rate / sleep / defer / resume.

Plus a :class:`Budget` dataclass that the agent main loop checks each
round to decide whether to short-circuit a sample.

These are pure clock / counter manipulations and require no external
services. They mutate :class:`ToolContext`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import Tool, ToolContext, ToolResult


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
@dataclass
class Budget:
    max_tool_calls: int = 24
    max_frames_seen: int = 600
    max_tokens: int = 200_000

    tool_calls: int = 0
    frames_seen: int = 0
    tokens_used: int = 0

    def add_call(self) -> None:
        self.tool_calls += 1

    def add_frame(self) -> None:
        self.frames_seen += 1

    def add_tokens(self, n: int) -> None:
        self.tokens_used += int(n)

    def exhausted(self) -> bool:
        # Any limit set to <=0 means "unlimited" for that dimension.
        if self.max_tool_calls > 0 and self.tool_calls >= self.max_tool_calls:
            return True
        if self.max_frames_seen > 0 and self.frames_seen >= self.max_frames_seen:
            return True
        if self.max_tokens > 0 and self.tokens_used >= self.max_tokens:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        def _fmt(used: int, cap: int) -> str:
            return f"{used}/{cap if cap > 0 else 'inf'}"
        return {
            "tool_calls": _fmt(self.tool_calls, self.max_tool_calls),
            "frames_seen": _fmt(self.frames_seen, self.max_frames_seen),
            "tokens_used": _fmt(self.tokens_used, self.max_tokens),
        }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
_VALID_FPS = (0.25, 0.5, 1.0, 2.0)
# Lock window around question_time: within ``LOCK_BEFORE_Q`` rounds before
# question_time, and any time at/after question_time, ``set_sample_rate``
# is rejected and fps is forced back to 1.0. This prevents the agent from
# striding past question_time with a slow sample rate, which would skip
# the only frame at which </Response> is allowed and produce a null result.
LOCK_BEFORE_Q = 5


class SetSampleRateTool(Tool):
    name = "set_sample_rate"
    description = (
        "Change how often the streaming clock advances per actual second. "
        "Use 0.25 fps for static / repetitive scenes to save budget; use "
        "2.0 fps only when fast motion is critical. Default is 1.0."
    )
    parameters = {
        "type": "object",
        "properties": {
            "fps": {"type": "number",
                     "description": "One of 0.25, 0.5, 1.0, 2.0."},
        },
        "required": ["fps"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            fps = float(args["fps"])
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(self.name, False, error=f"bad fps: {e}")
        # Reject any rate change within the lock window around question_time:
        # the streaming clock must hit question_time at fps=1.0 so the model
        # gets a chance to emit </Response>. Also reset fps back to 1.0.
        q = int(getattr(ctx, "question_time", 0) or 0)
        if q > 0 and ctx.current_round >= q - LOCK_BEFORE_Q:
            ctx.sample_rate_fps = 1.0
            return ToolResult(
                self.name, False,
                error=(
                    f"set_sample_rate is locked from t={q - LOCK_BEFORE_Q} "
                    f"(question_time={q}) onward; fps forced back to 1.0."
                ),
            )
        # Clamp to the allowed set
        fps = min(_VALID_FPS, key=lambda x: abs(x - fps))
        ctx.sample_rate_fps = fps
        return ToolResult(self.name, True, data={"fps": fps})


class SleepTool(Tool):
    name = "sleep"
    description = (
        "Skip ahead by `seconds` (alias of `wait`, but also marks the span "
        "as 'low-info' so the summarizer will not pollute memory)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "seconds": {"type": "number"},
        },
        "required": ["seconds"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        # Disabled: sleep can stride past question_time and cause null
        # answers. Use set_sample_rate (locked near question_time) instead.
        return ToolResult(
            self.name, False,
            error="sleep is disabled in OVO-Bench evaluation (it can "
                  "skip past question_time). Use set_sample_rate.",
        )


class DeferTool(Tool):
    name = "defer"
    description = (
        "Postpone answering until streaming reaches time `t` (seconds). "
        "Until then the agent will not invoke the model on intermediate "
        "frames (they are silently fast-forwarded)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "t": {"type": "integer"},
        },
        "required": ["t"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        # Disabled: defer can stride past question_time and cause null
        # answers. Use set_sample_rate (locked near question_time) instead.
        return ToolResult(
            self.name, False,
            error="defer is disabled in OVO-Bench evaluation (it can "
                  "skip past question_time). Use set_sample_rate.",
        )


class ResumeTool(Tool):
    name = "resume"
    description = "Cancel an active `defer` and resume normal streaming."
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        # Disabled: defer is disabled, so resume has nothing to cancel.
        return ToolResult(
            self.name, False,
            error="resume is disabled (defer is disabled too).",
        )
