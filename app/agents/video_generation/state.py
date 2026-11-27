"""
State for the Video Generation subgraph (Agent 6).

Mirrors ImageGenerationState's shape and naming closely. Three things worth
calling out:

1. This agent takes TWO upstream inputs, not one - `prompts` (Agent 3's
   PromptGenerationOutput, for video_prompt/motion_description per theme)
   AND `selection` (Agent 5's ImageSelectionOutput, for the already-chosen
   source frame per theme). This is NOT Agent 4's raw ImageGenerationOutput
   - the whole point of Agent 5 existing is that this agent shouldn't have
   to pick a candidate itself anymore. Agents 1-4 each only ever consumed
   the single immediately-prior agent's output; this is the first agent in
   the chain that needs two, and its second input comes from Agent 5, not
   Agent 4 directly, now that Agent 5 sits between them.
2. `_pending_result` is declared here on purpose - see Image_generation.md
   Challenge 7. That bug (a returned-but-undeclared state key silently
   dropped by LangGraph, making validation always fail despite generation
   succeeding every time) is exactly the kind of mistake worth guarding
   against explicitly when porting a pattern to a new agent, not just
   trusting muscle memory to remember it this time.
3. The upstream-images field was renamed from `images: ImageGenerationOutput`
   to `selection: ImageSelectionOutput` - not just a rename, the type
   changed too. Left as `images` of the old type, a stale read like
   `theme_result.images[0]` would still type-check against the wrong
   schema and fail silently at runtime instead of being caught up front.
   Renaming forces every read site in nodes.py to be touched deliberately.
"""

from __future__ import annotations

from typing import TypedDict

from app.agents.prompt_gen_agent.schema import PromptGenerationOutput  # Agent 3's output
from app.agents.image_selection_agent.schema import ImageSelectionOutput  # Agent 5's output
from app.agents.video_generation.schema import ThemeVideoResult, VideoGenerationOutput


class VideoGenerationState(TypedDict, total=False):
    # Inputs
    prompts: PromptGenerationOutput
    selection: ImageSelectionOutput

    # NOTE: human review decisions are NOT state fields. They ride in the
    # graph config (`configurable.theme_allowlist` / `theme_overrides`)
    # because a resumed thread invokes with ainvoke(None) and would ignore
    # any fresh state input. See generate_node.

    # Progress through prompt_sets (driven by prompts.prompt_sets, same as
    # Agent 4 drives off its own prompt_sets length)
    current_theme_index: int
    completed_theme_results: list[ThemeVideoResult]

    # Set once in start_node
    run_id: str
    video_workflow_template: dict  # loaded once, deep-copied per theme in generate_node

    # Per-attempt scratch state - field names match Agent 4's exactly
    # (retries/error/current_seed) for the same reason Agent 4 matched
    # Agents 1-3's names: existing pipeline-runner code that reads
    # result.get("retries", 0) etc. keeps working unmodified.
    retries: int
    error: str | None
    current_seed: int | None
    _pending_result: ThemeVideoResult | None

    # Final output
    videos: VideoGenerationOutput | None