"""Meta / self-control tools: abstain, confidence_estimate."""
from __future__ import annotations

from typing import Any, Dict

from .base import Tool, ToolContext, ToolResult


class AbstainTool(Tool):
    name = "abstain"
    description = (
        "Declare that the visual evidence is insufficient to answer the "
        "question. The agent loop will short-circuit and emit a fallback "
        "answer chosen by the calling driver (typically '</Response> "
        "Unable to answer', or a best-guess option when an explicit option "
        "list was given). Use this when the question explicitly allows "
        "'no answer' (e.g. hallucination-detection style prompts) and "
        "MEMORY truly has no relevant evidence; on standard questions "
        "with a fixed answer space, prefer answering with </Response> "
        "directly so the model commits to a real choice."
    )
    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Short rationale for abstaining (logged, not shown to user).",
            }
        },
        "required": [],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.abstain_reason = str(args.get("reason", ""))[:256]
        return ToolResult(self.name, True, data={"reason": ctx.abstain_reason})
