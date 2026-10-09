"""Tool base classes & registry for agentic video understanding.

Design notes
------------
* Tools are pure-Python objects with a JSON-Schema-flavored ``parameters`` dict
  so we can render them into the system prompt as an OpenAI-style tool spec.
* ``run(args, ctx)`` returns a :class:`ToolResult` with ``ok`` flag and a
  JSON-serializable ``data`` payload. Some tools (e.g. ``zoom_crop``) may
  additionally append items to ``ctx.next_round_images`` to inject extra
  visual tokens into the next round of the streaming dialog.
* ``ToolContext`` is a thin handle the *agent main loop* (agent.py)
  passes in. Tools must NOT import video_processor.py to avoid coupling.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


log = logging.getLogger("agent.tools")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    name: str
    ok: bool
    data: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "ok": self.ok}
        if self.data is not None:
            d["data"] = self.data
        if self.error:
            d["error"] = self.error
        return d

    def to_prompt_str(self) -> str:
        return f"<tool_result>{json.dumps(self.to_dict(), ensure_ascii=False)}</tool_result>"


# ---------------------------------------------------------------------------
# Per-sample context handed to tools
# ---------------------------------------------------------------------------
@dataclass
class ToolContext:
    """Mutable per-sample state shared between agent loop and tool impls."""

    sample_id: str
    task: str
    # Streaming clock state
    current_round: int = 0
    total_rounds: int = 0
    question_time: int = 0
    # Frames available on disk (1-second granularity)
    frame_paths: List[str] = field(default_factory=list)
    # Next round of inference may pull frame_idx from here (replay/skip)
    next_frame_idx: Optional[int] = None
    # Images tools want to inject into the *next* round's prompt as
    # additional <image> tokens (e.g. zoom_crop output). Cleared after consume.
    next_round_images: List[str] = field(default_factory=list)
    # Free-form extra text to append to next user turn
    next_round_extra_text: List[str] = field(default_factory=list)
    # Set by abstain() to short-circuit the loop with a fixed answer
    abstain_reason: Optional[str] = None
    # Counters / log
    tool_call_log: List[Dict[str, Any]] = field(default_factory=list)
    # Per-round budget (reset by agent loop each round)
    calls_this_round: int = 0

    # ------------------------------------------------------------------
    # Optional extensions (Phase-A scaffolding). All default to None so
    # legacy tools / call sites that don't know about them keep working.
    # See report_doc/Agentic流式视频理解_Memory与工具体系设计.md for design.
    # ------------------------------------------------------------------
    memory: Optional[Any] = None         # tools.memory.MemoryStore
    tracks: Optional[Dict[str, Any]] = None    # active SAM3 sessions
    audio_cache: Optional[Dict[str, Any]] = None
    budget: Optional[Any] = None         # tools.efficiency.Budget
    # Sampling rate hint (frames-per-second) honored by efficiency tool
    sample_rate_fps: Optional[float] = None
    # Pending defer target (seconds); agent loop fast-forwards until reached
    defer_until_t: Optional[int] = None
    # Per-sample zoom_crop history: { t_frame: [ [x1,y1,x2,y2], ... ] }.
    # Used by ZoomCropTool to dedupe near-identical crops (IoU >= 0.5)
    # and enforce per-frame / per-sample call caps so the model cannot
    # spam zoom_crop and starve other tools.
    zoom_history: Optional[Dict[int, List[List[float]]]] = None

    # Per-physical-round tool counters used by promptfix v2 anti-loop
    # heuristics. The agent main loop is responsible for resetting these
    # whenever a new physical round starts (cf. _last_phys_round handling
    # in agent.py). Tools read & write them to:
    #   * memsearch_history: detect repeated memory_search results and
    #     redirect the model to replay (改动 1).
    #   * zoom_count_this_phys_round: enforce a soft 5-call cap on
    #     zoom_crop per physical round (改动 7).
    memsearch_history: List[Dict[str, Any]] = field(default_factory=list)
    zoom_count_this_phys_round: int = 0


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------
class Tool:
    name: str = ""
    description: str = ""
    # JSON-Schema-like dict; only top-level "type": "object" with "properties"
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:  # noqa: D401
        raise NotImplementedError

    # -- prompt rendering -------------------------------------------------
    def to_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError(f"Tool {tool!r} has empty name")
        if tool.name in self._tools:
            log.warning("Overriding existing tool %s", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def dispatch(self, name: str, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(name=name, ok=False, error=f"unknown tool: {name}")
        try:
            res = tool.run(args or {}, ctx)
            if not isinstance(res, ToolResult):
                res = ToolResult(name=name, ok=True, data=res)
            return res
        except Exception as e:  # noqa: BLE001
            log.exception("Tool %s failed", name)
            return ToolResult(name=name, ok=False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------
def build_default_registry() -> ToolRegistry:
    """Build the MVP toolset (no external models required).

    Honors the env var ``STREAMO_DISABLE_ZOOM_CROP`` (any truthy value:
    "1", "true", "yes") to drop ``ZoomCropTool`` from the registry. This
    is used for ablations that probe whether zoom_crop is genuinely
    necessary or whether the 27B agent over-relies on it.
    """
    import os
    from .temporal import ReplayTool
    from .meta import AbstainTool
    from .spatial import ZoomCropTool

    disable_zoom = os.environ.get(
        "STREAMO_DISABLE_ZOOM_CROP", ""
    ).strip().lower() in ("1", "true", "yes")

    reg = ToolRegistry()
    tools = [ReplayTool(), AbstainTool()]
    if not disable_zoom:
        tools.append(ZoomCropTool())
    for t in tools:
        reg.register(t)
    return reg
