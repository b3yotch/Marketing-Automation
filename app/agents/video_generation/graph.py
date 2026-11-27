"""
Graph shape - identical to Agent 4's (see Image_generation.md / graph.py):

    start --> generate --> validate --+--> [more themes?] --+--> advance_theme --> generate (next theme)
                 ^                    |                      |
                 |                    v                      +--> finalize --> END (all themes done)
                 +----- bump_retry <--+ (invalid, retries remain)
                                      |
                                      +--> advance_theme (retries exhausted - skip this theme, same choice Agent 4 made)

Same two-loop shape (retry loop + theme loop) for the same reason: this
agent also iterates a variable number of themes. Reuses Agent 4's exact
_after_validate / _after_advance routing logic, just re-pointed at this
agent's state field names (retries/error -> same names, current_theme_index
-> driven by prompts.prompt_sets length instead of prompts stored under a
different key).

build_graph(checkpointer=...) - added for the same reason as Agent 4's:
this is the single most expensive stage in the pipeline (up to ~4500s per
attempt), so a crash after video 1 of 2 succeeded should never mean redoing
video 1. The module-level video_generation_graph below stays uncheckpointed
- test_full_pipeline_live.py builds its own checkpointed instance
separately, since the checkpointer's connection lifecycle belongs to
whoever's running the pipeline, not to this module.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.core.config import settings
from app.agents.video_generation.state import VideoGenerationState
from app.agents.video_generation.nodes import (
    start_node,
    generate_node,
    validate_node,
    bump_retry_node,
    advance_theme_node,
    finalize_node,
)


def _after_validate(state: VideoGenerationState) -> str:
    has_error = bool(state.get("error"))

    if has_error:
        if state.get("retries", 0) < settings.max_video_gen_retries:
            return "bump_retry"
        # Retries exhausted for this theme - skip it (see module docstring).
        return "advance_theme"

    return "advance_theme"


def _after_advance(state: VideoGenerationState) -> str:
    prompt_sets = state["prompts"].prompt_sets
    if state.get("current_theme_index", 0) < len(prompt_sets):
        return "generate"
    return "finalize"


def _after_start(state: VideoGenerationState) -> str:
    # A reopened thread can arrive with every theme already done (review kept
    # all existing videos). start_node then parks the index at the end, so
    # going straight to generate would index past prompt_sets.
    prompt_sets = state["prompts"].prompt_sets
    if state.get("current_theme_index", 0) >= len(prompt_sets):
        return "finalize"
    return "generate"


def build_graph(checkpointer=None):
    graph = StateGraph(VideoGenerationState)

    graph.add_node("start", start_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("bump_retry", bump_retry_node)
    graph.add_node("advance_theme", advance_theme_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("start")
    graph.add_conditional_edges(
        "start",
        _after_start,
        {"generate": "generate", "finalize": "finalize"},
    )
    graph.add_edge("generate", "validate")

    graph.add_conditional_edges(
        "validate",
        _after_validate,
        {"bump_retry": "bump_retry", "advance_theme": "advance_theme"},
    )
    graph.add_edge("bump_retry", "generate")

    graph.add_conditional_edges(
        "advance_theme",
        _after_advance,
        {"generate": "generate", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


# Compiled once at import time, matching the pattern the other agents'
# graphs already use. Uncheckpointed - see module docstring.
video_generation_graph = build_graph()