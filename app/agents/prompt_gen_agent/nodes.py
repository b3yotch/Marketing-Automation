from __future__ import annotations

import asyncio

from app.core.config import settings
from app.core.llm import LLMExtractionError, structured_chat_groq
from app.agents.creative_strategy_agent.schema import CreativeDirection
from app.agents.prompt_gen_agent.prompts import SYSTEM_PROMPT, build_user_prompt
from app.agents.prompt_gen_agent.schema import PromptGenerationOutput, ThemePromptSet
from app.agents.prompt_gen_agent.state import PromptGenState

# ~200 words - rough proxy for the video text encoder's ~226-token ceiling,
# catching an obviously oversized prompt before it silently truncates.
MAX_VIDEO_BASE_PROMPT_CHARS = 1200


async def _generate_theme(
    creative: CreativeDirection, theme_index: int, previous_error: str | None
) -> tuple[ThemePromptSet | None, str | None, str | None]:
    """
    One theme, with the primary/fallback model chain. Returns
    (theme_set, model_used, provider_error).

    Same chain semantics as Creative Strategy's generate_node: a provider
    failure (API error, rate limit, schema validation error) falls through
    to the fallback model immediately and does NOT consume a retry.
    """
    user_prompt = build_user_prompt(creative, theme_index, previous_error=previous_error)

    last_error = "No model was attempted"
    for model in (settings.prompt_gen_primary_model, settings.prompt_gen_fallback_model):
        try:
            theme_set = await structured_chat_groq(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=ThemePromptSet,
                temperature=settings.prompt_gen_temperature,
            )
        except LLMExtractionError as exc:
            last_error = str(exc)
            continue

        # Don't trust the model to echo this - it's a traceability key, and
        # we know the right value.
        theme_set.source_setting = creative.visual_themes[theme_index].setting
        return theme_set, model, None

    return None, None, last_error


async def generate_node(state: PromptGenState) -> PromptGenState:
    """
    Generate prompts for every theme that doesn't have a valid result yet,
    one call per theme, concurrently.

    Why one call per theme instead of one call returning all themes: asked
    for "exactly N entries" in a single response, the model repeatedly
    returned fewer (a theme merged or dropped) - three attempts in a row,
    so a systematic tendency rather than a fluke. With per-theme calls the
    theme count is guaranteed by this loop, each response is a third the
    size, and a retry only re-calls the themes that actually failed. Costs
    N requests per product instead of 1, which is fine at this project's
    scale (see config.py's note on Groq's request ceiling).
    """
    creative = state["creative"]
    theme_sets = dict(state.get("theme_sets", {}))
    theme_models = dict(state.get("theme_models", {}))
    feedback = state.get("theme_feedback", {})

    pending = [i for i in range(len(creative.visual_themes)) if i not in theme_sets]
    results = await asyncio.gather(
        *(_generate_theme(creative, i, feedback.get(i)) for i in pending)
    )

    provider_errors: dict[int, str] = {}
    for i, (theme_set, model, err) in zip(pending, results):
        if theme_set is None:
            provider_errors[i] = err or "unknown provider failure"
        else:
            theme_sets[i] = theme_set
            theme_models[i] = model

    return {
        **state,
        "theme_sets": theme_sets,
        "theme_models": theme_models,
        "theme_errors": provider_errors,
        "model_used": ", ".join(sorted(set(theme_models.values()))) or None,
    }


def _theme_problems(theme_set: ThemePromptSet) -> list[str]:
    """Every reason one theme's output isn't good enough, not just the first."""
    problems: list[str] = []
    if not theme_set.image_prompt.positive_prompt.strip():
        problems.append("image_prompt.positive_prompt is empty")
    if not theme_set.video_prompt.base_prompt.strip():
        problems.append("video_prompt.base_prompt is empty")
    if not theme_set.video_prompt.motion_description.strip():
        problems.append("video_prompt.motion_description is empty")
    if len(theme_set.video_prompt.base_prompt) > MAX_VIDEO_BASE_PROMPT_CHARS:
        problems.append(
            f"video_prompt.base_prompt is too long "
            f"({len(theme_set.video_prompt.base_prompt)} chars, likely to truncate)"
        )
    return problems


def validate_node(state: PromptGenState) -> PromptGenState:
    """
    Sanity-check each theme's output. Same low-bar philosophy as Agents 1
    and 2 - catches outright failures (missing theme, empty prompt fields),
    not prompt quality; that judgment belongs to the Review/Critic agent
    once it can look at what generation actually produces.

    A theme that fails validation is removed from theme_sets and its
    specific reason goes into theme_feedback the moment it fails (not just
    once retries are exhausted), so the next generate pass re-calls only
    that theme with something useful to act on. Provider failures
    (theme_errors) are reported in the error message but NOT fed back.
    """
    creative = state["creative"]
    total = len(creative.visual_themes)
    retries = state.get("retries", 0)

    if total == 0:
        return {**state, "prompts": None, "error": "CreativeDirection has no visual_themes to translate"}

    theme_sets = dict(state.get("theme_sets", {}))
    provider_errors = state.get("theme_errors", {})

    feedback: dict[int, str] = {}
    for i in list(theme_sets):
        problems = _theme_problems(theme_sets[i])
        if problems:
            feedback[i] = "; ".join(problems)
            del theme_sets[i]

    missing = [i for i in range(total) if i not in theme_sets]
    if not missing:
        ordered = [theme_sets[i] for i in range(total)]
        output = PromptGenerationOutput(
            source_url=creative.source_url,
            prompt_sets=ordered,
            extraction_confidence=min(ts.extraction_confidence for ts in ordered),
        )
        return {
            **state,
            "theme_sets": theme_sets,
            "theme_errors": {},
            "theme_feedback": {},
            "prompts": output,
            "error": None,
        }

    reason = "; ".join(
        f"theme {i + 1}: {feedback.get(i) or provider_errors.get(i) or 'no output'}"
        for i in missing
    )
    if retries >= settings.max_prompt_gen_retries:
        reason += f" (gave up after {retries} retries)"

    return {
        **state,
        "theme_sets": theme_sets,
        "theme_feedback": feedback,
        "prompts": None,
        "error": reason,
    }


def bump_retry_node(state: PromptGenState) -> PromptGenState:
    """Increments the retry counter, then loops back to generate_node."""
    return {**state, "retries": state.get("retries", 0) + 1}


def route_after_validation(state: PromptGenState) -> str:
    """Conditional edge: retry generation, or stop. validate_node always
    runs first and sets `error` iff any theme is still missing."""
    if not state.get("error"):
        return "done"
    if state.get("retries", 0) >= settings.max_prompt_gen_retries:
        return "done"  # give up, error is already set on state by validate_node
    return "retry"