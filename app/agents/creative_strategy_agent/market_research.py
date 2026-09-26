"""
Market context enrichment for the Creative Strategy Agent.

The underlying LLM has a training cutoff - it has no way to know what ad
creative is actually working right now, or what competitors are currently
saying. This module fills that gap with two targeted Tavily searches (trend
context, competitor context) run concurrently, and folds the results into
the generation prompt as grounding.

Best-effort by design: a failed, slow, or unconfigured Tavily call should
never block creative generation. If a search errors out, times out, or no
API key is set, the agent proceeds without market context rather than
failing the whole pipeline over an enrichment step that was never a hard
requirement.
"""

from __future__ import annotations

import asyncio

from tavily import AsyncTavilyClient

from app.core.config import settings
from app.agents.research_agent.schema import ProductResearch


async def _safe_search(client: AsyncTavilyClient, *, query: str, **kwargs) -> str:
    """Run one Tavily search, returning its synthesized answer or '' on any failure."""
    try:
        response = await client.search(query=query, include_answer=True, **kwargs)
        return response.get("answer") or ""
    except Exception:  # noqa: BLE001 - enrichment is best-effort, never break the pipeline over this
        return ""


async def fetch_market_context(research: ProductResearch) -> str:
    """
    Run a trend search and a competitor search in parallel and return a
    combined block of context to fold into the creative-generation prompt.

    - Trend search uses topic="news" to bias toward recent/dated content -
      evergreen "how to market X" pages aren't the point, current activity is.
    - Competitor search uses search_depth="advanced" - "who's out there and
      how are they positioned" is a harder synthesis question than a trend
      headline, worth the extra credit cost here specifically.
    - Run concurrently (not sequentially) specifically to keep the added
      latency close to the cost of the SLOWER of the two calls, not their sum.

    Returns "" if no Tavily key is configured, or if both searches fail -
    callers should treat this as optional grounding, never a required input.
    """
    if not settings.tavily_api_key:
        return ""

    client = AsyncTavilyClient(settings.tavily_api_key)

    trend_query = f"{research.title} marketing trends 2026"
    competitor_query = (
        f"{research.brand} competitors" if research.brand else f"{research.title} alternative brands"
    )

    trend_answer, competitor_answer = await asyncio.gather(
        _safe_search(client, query=trend_query, topic="news", max_results=5),
        _safe_search(client, query=competitor_query, search_depth="advanced", max_results=5),
    )

    parts: list[str] = []
    if trend_answer:
        parts.append(f"Current market/trend context: {trend_answer}")
    if competitor_answer:
        parts.append(f"Competitor context: {competitor_answer}")

    return "\n\n".join(parts)