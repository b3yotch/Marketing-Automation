from __future__ import annotations

from typing import TypedDict

from app.agents.creative_strategy_agent.schema import CreativeDirection
from app.agents.prompt_gen_agent.schema import PromptGenerationOutput, ThemePromptSet


class PromptGenState(TypedDict, total=False):
    """
    State threaded through the prompt generation subgraph. `total=False`
    because most fields only exist after their producing node has run.

    No `scraped`-equivalent field here (same reasoning as
    CreativeStrategyState) - this agent's input is already a validated
    CreativeDirection object from Agent 2, not something that needs fetching.

    Themes are generated one call each, keyed by their index into
    creative.visual_themes - so a retry only re-calls the themes that
    actually failed, not the ones that already came back valid.
    """

    creative: CreativeDirection

    theme_sets: dict[int, ThemePromptSet]  # valid, validated results so far
    theme_models: dict[int, str]  # which model (primary/fallback) produced each - for observability
    theme_errors: dict[int, str]  # PROVIDER failures from the last generate pass (both models failed)
    theme_feedback: dict[int, str]  # VALIDATION failures from the last validate pass - the only
    # thing fed back into the next prompt, since a provider failure has nothing useful to feed back

    prompts: PromptGenerationOutput | None  # assembled only once every theme is valid
    model_used: str | None  # unique models used across themes, comma-joined

    retries: int
    error: str | None