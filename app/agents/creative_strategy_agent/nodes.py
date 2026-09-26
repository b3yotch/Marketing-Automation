from __future__ import annotations

from app.core.config import settings
from app.core.llm import LLMExtractionError, structured_chat_groq
from app.agents.creative_strategy_agent.market_research import fetch_market_context
from app.agents.creative_strategy_agent.prompts import SYSTEM_PROMPT, build_user_prompt
from app.agents.creative_strategy_agent.schema import CreativeDirection
from app.agents.creative_strategy_agent.state import CreativeStrategyState


async def generate_node(state: CreativeStrategyState) -> CreativeStrategyState:
    """
    Generate creative direction from the product research (plus optional
    market context).

    Market context is fetched at most once per pipeline run: if
    state["market_context"] is already set (from this node's own earlier
    call, on a retry), it's reused as-is rather than re-querying Tavily -
    trend/competitor grounding doesn't change based on what the model got
    wrong last attempt, so re-fetching on every retry would just burn extra
    API credits for no benefit. `None` (key absent) means "not fetched yet";
    `""` (empty string) means "fetched, but nothing came back or no key is
    configured" - both are distinguished from a real non-empty result.

    Tries the primary model first (gpt-oss-120b - strongest copy/structure in
    testing) and falls back to the secondary model (llama-3.3-70b-versatile)
    on ANY failure from the primary - schema validation error, API error,
    rate limit. This model-fallback chain is deliberately separate from the
    retry loop below: a transient provider failure isn't a signal that the
    *approach* needs correcting, so it doesn't consume a retry or get a
    previous_error message. The retry loop is reserved for "a model produced
    output, but it didn't pass validation."
    """
    research = state["research"]

    market_context = state.get("market_context")
    if market_context is None:
        market_context = await fetch_market_context(research)

    previous_error = state.get("error")
    user_prompt = build_user_prompt(research, market_context=market_context, previous_error=previous_error)

    last_error = "No model was attempted"
    for model in (settings.groq_primary_model, settings.groq_fallback_model):
        try:
            creative = await structured_chat_groq(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=CreativeDirection,
                temperature=settings.groq_temperature,
            )
            creative.source_url = research.source_url
            return {
                **state,
                "creative": creative,
                "error": None,
                "model_used": model,
                "market_context": market_context,
            }
        except LLMExtractionError as exc:
            last_error = str(exc)
            continue

    return {
        **state,
        "creative": None,
        "error": last_error,
        "model_used": None,
        "market_context": market_context,
    }


def _missing_fields(creative: CreativeDirection | None) -> list[str]:
    if creative is None:
        return ["hooks", "captions"]
    missing = []
    if not creative.hooks:
        missing.append("hooks")
    if not creative.captions:
        missing.append("captions")
    return missing


def _is_valid(state: CreativeStrategyState) -> bool:
    return not _missing_fields(state.get("creative"))


def validate_node(state: CreativeStrategyState) -> CreativeStrategyState:
    """
    Sanity-check the generation. Low bar, same philosophy as Agent 1: catch
    outright failures (no hooks, no captions at all), not grade creative
    quality.

    Writes a specific failure reason the moment validation fails - retries
    remaining or not - so generate_node's previous_error mechanism actually
    has something useful to feed back into the next attempt's prompt.
    """
    if _is_valid(state):
        return {**state, "error": None}

    missing = _missing_fields(state.get("creative"))
    if state.get("creative") is None:
        reason = state.get("error") or "Model call failed on both primary and fallback"
    else:
        reason = f"Output was missing required field(s): {', '.join(missing)}"

    retries = state.get("retries", 0)
    if retries >= settings.max_creative_retries:
        return {**state, "error": f"{reason} (gave up after {retries} retries)"}

    return {**state, "error": reason}


def bump_retry_node(state: CreativeStrategyState) -> CreativeStrategyState:
    """Increments the retry counter, then loops back to generate_node."""
    return {**state, "retries": state.get("retries", 0) + 1}


def route_after_validation(state: CreativeStrategyState) -> str:
    """Conditional edge: retry generation, or stop."""
    if _is_valid(state):
        return "done"
    if state.get("retries", 0) >= settings.max_creative_retries:
        return "done"
    return "retry"