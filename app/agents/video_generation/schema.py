"""
Output contract for the Video Generation Agent (Agent 6).

Mirrors Agent 4's schema.py shape deliberately (GeneratedVideo /
ThemeVideoResult / VideoGenerationOutput <-> GeneratedImage /
ThemeGenerationResult / ImageGenerationOutput) rather than inventing a
different structure - same reasoning either way: one wrapper per theme,
the wrapper carries provenance + timing + retry count, the leaf object
carries the actual generated-file details.

Two real differences from Agent 4's shape, not oversights:

1. No list[GeneratedVideo] per theme - Agent 4 keeps every candidate in a
   batch because Agent 5 (Image Selection) picks a winner from that batch
   before this agent ever runs; by the time Agent 6 sees a theme, there is
   already exactly one chosen source frame. Video generation itself also
   deliberately produces exactly one video per theme (see
   Video_generation.md Challenge 2 discussion - video's cost profile
   doesn't reward batching the way cheap image batches did), so `video` is
   a single optional field, not a list.
2. `status` distinguishes the two non-generation outcomes from "success" -
   Agent 4 has no equivalent because it always has *something* to attempt
   (a prompt). This agent can hit a theme with no valid source frame to
   animate (that theme was itself skipped upstream - either Agent 4
   exhausted its own retries, or Agent 5 had nothing valid left to select
   from after its own deterministic filter - see Image_generation.md
   Challenge 3 and Image_selection.md) - that's not a generation failure
   worth retrying, so it's recorded explicitly rather than silently
   absent. `skipped_not_approved` is the same shape but for the HITL gate:
   a human rejected this theme's image in observability/review.py while
   approving others for the same product, so no video is generated for it.
   A genuine retries-exhausted generation failure, by contrast, is NOT
   recorded here at all - same as Agent 4, the theme is simply absent from
   theme_results.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GeneratedVideo(BaseModel):
    """One rendered video, for one theme."""

    filename: str = Field(description="Filename as reported by ComfyUI's save node - kept for traceability.")
    subfolder: str = Field(default="", description="Subfolder under ComfyUI's own output/ dir, if any.")
    local_path: str = Field(
        description="Path under this project's own output directory - the path to actually read from, same convention as GeneratedImage.local_path."
    )
    seed: int = Field(description="Seed actually used, for reproducibility/debugging.")
    source_image_local_path: str = Field(
        description=(
            "Which image was used as the source frame - echoes Agent 5's "
            "ThemeSelectionResult.selected_local_path verbatim for this "
            "theme. No longer a placeholder: Agent 5 (Image Selection) "
            "judges Agent 4's candidates and picks this before Agent 6 "
            "ever runs, replacing the earlier theme_result.images[0] "
            "stand-in that was used before Agent 5 existed."
        )
    )


class ThemeVideoResult(BaseModel):
    """Everything produced (or explicitly not produced) for one theme."""

    source_setting: str = Field(
        description="Echoes ThemeSelectionResult.source_setting verbatim, so this traces back to its origin theme."
    )
    video: GeneratedVideo | None = Field(
        default=None,
        description=(
            "None when status is 'skipped_no_source_image' or "
            "'skipped_not_approved'."
        ),
    )
    generation_time_seconds: float = 0.0
    retries_used: int = 0
    status: Literal[
        "success", "skipped_no_source_image", "skipped_not_approved"
    ] = "success"


class VideoGenerationOutput(BaseModel):
    """Final output for one product."""

    source_url: str
    theme_results: list[ThemeVideoResult] = Field(default_factory=list)