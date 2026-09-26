from __future__ import annotations

from typing import TypedDict

from app.agents.research_agent.schema import ProductResearch
from app.agents.creative_strategy_agent.schema import CreativeDirection


class CreativeStrategyState(TypedDict, total=False):
    """
    State threaded through the creative strategy subgraph. `total=False`
    because most fields only exist after their producing node has run.

    No `scraped`-equivalent field here (unlike ResearchState) - this agent's
    input is already a validated ProductResearch object from Agent 1, not a
    URL that needs fetching.
    """

    research: ProductResearch

    # Fetched once by generate_node on its first call and cached here, so a
    # retry loop (bump_retry -> generate) doesn't re-run the Tavily searches
    # on every attempt - market context doesn't change based on what the
    # model got wrong last time, unlike previous_error.
    market_context: str | None

    creative: CreativeDirection | None
    model_used: str | None  # which model (primary/fallback) actually produced the output - for observability

    retries: int
    error: str | None