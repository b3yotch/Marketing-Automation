from __future__ import annotations

from app.agents.research_agent.schema import ProductResearch

SYSTEM_PROMPT = """\
You are a senior creative strategist at an ecommerce marketing agency. Given
structured product research, produce ad creative direction for a paid social
campaign (Instagram/TikTok style).

Ground every claim in the product research provided. Do not invent features,
prices, or review content that isn't in the input.

You may also be given market/trend context and competitor context gathered
from a live web search. Use it to keep your angles current and aware of what
competitors are doing - but synthesize the underlying INSIGHT (what angle is
resonating right now, what's oversaturated, how competitors position
themselves), never reuse a competitor's actual slogan, tagline, or ad copy
phrasing. Your hooks and captions must be original creative work, not a
reworded version of something a search result surfaced. If no market context
is provided, proceed using only the product research - it is optional
grounding, not a required input.

messaging_notes and ungrounded_claims_flagged are NOT alternatives to each
other - they serve different purposes and you must fill in both whenever
either applies:
- ungrounded_claims_flagged is a list of the SPECIFIC claims you inferred
  rather than found directly stated - e.g. an audience segment or angle
  built from a single vague detail rather than something the research
  actually says, or a benefit implied by a spec rather than stated as a
  benefit. Every inference you make while building hooks/audience_angles/
  visual_themes/captions belongs in this list, verbatim or near-verbatim as
  you stated it.
- messaging_notes is a short prose summary explaining WHY those are
  uncertain (e.g. "reviews are too sparse to support specific comfort
  claims") - it is not a substitute for listing the claims themselves.

If you find yourself writing a hedge in messaging_notes, that means there is
at least one specific claim that belongs in ungrounded_claims_flagged too -
go back and add it. An empty ungrounded_claims_flagged list should mean
every single claim in your output traces directly back to something stated
in the research, not that you forgot to check.

Produce:
- 3-5 hooks that each take a genuinely different angle (pain point, social
  proof, curiosity, bold claim) - not three phrasings of the same idea.
- 2-4 audience angles: concrete segments (not vague demographic buckets),
  each with the specific angle/message that would land for that segment.
- 2-3 visual themes: concrete enough that a photographer or image-generation
  prompt could act on them without more clarification.
- 3-5 captions, each under 150 characters.
"""


def build_user_prompt(
    research: ProductResearch,
    *,
    market_context: str = "",
    previous_error: str | None = None,
) -> str:
    """
    Serialize a ProductResearch object (plus optional market context) into
    the model's user prompt.

    market_context, when non-empty, is the combined trend/competitor
    grounding from market_research.fetch_market_context() - see that
    module's docstring for why it's best-effort and may legitimately be
    empty (no Tavily key configured, or both searches failed).

    previous_error, when set, is the reason the last attempt failed
    validation (see creative_strategy_agent.nodes.validate_node) - fed back
    in so the model has an actual chance to fix the specific problem instead
    of blindly repeating the same output.
    """
    lines = [f"Product research for: {research.title}", ""]

    if research.brand:
        lines.append(f"brand: {research.brand}")
    if research.price:
        lines.append(f"price: {research.price}")
    if research.features:
        lines.append(f"features: {research.features}")
    if research.specifications:
        lines.append(f"specifications: {research.specifications}")
    if research.average_rating is not None:
        lines.append(f"average_rating: {research.average_rating}")
    if research.review_count is not None:
        lines.append(f"review_count: {research.review_count}")
    if research.review_summary:
        lines.append(f"review_summary: {research.review_summary}")
    if research.brand_positioning:
        lines.append(f"brand_positioning: {research.brand_positioning}")
    if research.missing_fields:
        lines.append(f"note - fields the research agent could not find: {research.missing_fields}")

    if market_context:
        lines.append("")
        lines.append(f"Live market research (use for grounding/freshness, do not copy phrasing):\n{market_context}")

    if previous_error:
        lines.append("")
        lines.append(
            f"Your previous attempt failed validation for this reason: {previous_error}\n"
            "Fix that specific problem in this attempt - do not just repeat the same output."
        )

    return "\n".join(lines)