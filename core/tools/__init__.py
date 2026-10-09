"""Agentic tool protocol for Streamo OVO-Bench.

This subpackage is *additive*: it does NOT modify video_processor.py.
The agent main loop lives in core/agent.py.
"""
from .base import Tool, ToolRegistry, ToolContext, ToolResult, build_default_registry
from .memory import (
    MemoryStore,
    Event,
    RuleSummarizer,
    MemoryWriteTool,
    MemoryReadTool,
    MemoryListTool,
    MemoryRetrieveTool,
)
from .memory_graph import (
    EntityIndex,
    EntitySubgraph,
    MemoryGraph,
    EvidencePath,
    BOCPDSegmenter,
    observation_to_embedding,
    op_add_obs_node,
    op_add_epi_node,
    op_add_pin_node,
    op_correct_node,
    op_add_entity_edge,
    op_add_causal_edge,
)
from .verification import (
    Claim,
    ClaimGroup,
    VerificationResult,
    ConsistencyChecker,
    ReplayScheduler,
    run_verification_pipeline,
    apply_corrections,
)
from .efficiency import (
    Budget,
    SetSampleRateTool,
)
from .track import (
    TrackerAdapter,
    MockTracker,
    DEFAULT_TRACKER,
    TrackInitTool,
    TrackQueryTool,
    TrackEndTool,
)
from .audio import (
    ASRAdapter,
    MockASR,
    DEFAULT_ASR,
    ASRSegmentTool,
    SpeakerIDTool,
    AudioEventTool,
)


def build_extended_registry(
    enable_memory: bool = True,
    enable_track: bool = True,
    enable_audio: bool = True,
    enable_efficiency: bool = True,
) -> ToolRegistry:
    """Default registry + Phase-A extensions (memory / track / audio /
    efficiency). Legacy callers using :func:`build_default_registry` are
    unaffected.
    """
    reg = build_default_registry()
    if enable_memory:
        for t in (MemoryWriteTool(), MemoryReadTool(), MemoryListTool(),
                  MemoryRetrieveTool()):
            reg.register(t)
    if enable_efficiency:
        for t in (SetSampleRateTool(),):
            reg.register(t)
    if enable_track:
        for t in (TrackInitTool(), TrackQueryTool(), TrackEndTool()):
            reg.register(t)
    if enable_audio:
        for t in (ASRSegmentTool(), SpeakerIDTool(), AudioEventTool()):
            reg.register(t)
    return reg
