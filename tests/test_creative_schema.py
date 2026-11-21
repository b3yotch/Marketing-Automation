"""Regression tests for the CreativeDirection contract.

The failure this covers: a real gpt-oss-120b call returned valid JSON that
ended after `visual_themes` - no `captions`, `messaging_notes` or
`ungrounded_claims_flagged`. Because those fields had defaults, Pydantic
accepted the object, `validate_node` rejected the empty captions, and the
node's fallback model was never tried (nothing raised LLMExtractionError).
`hooks` and `captions` are now required + min_length=1, so that same output
becomes a validation error and the primary -> fallback chain runs.

Trace: e9d1328904e14226e350c37bb71dd74e (run 2026-11-21T13:06Z).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.creative_strategy_agent import nodes
from app.agents.creative_strategy_agent.nodes import generate_node, validate_node
from app.agents.creative_strategy_agent.schema import CreativeDirection
from app.agents.research_agent.schema import ProductResearch
from app.core.config import settings
from app.core.llm import LLMExtractionError

# The shape observed in the failing run: valid JSON, properly closed, ending
# after visual_themes. Punctuation normalised to ASCII; field set and values
# are as returned by the model.
OBSERVED_OUTPUT = {
    "source_url": "https://example.com/product/black-turtle-neck-sweater",
    "extraction_confidence": 0.94,
    "hooks": [
        "Freezing mornings? Slip into the high-collar snug fit that locks in heat without bulk.",
        "4.86* rating from real shoppers - see why this black turtle-neck is the go-to winter layer.",
        "Ever wonder how a single sweater can be both a sleek solo piece and a perfect layering hero?",
        "Bold claim: this black turtle-neck is the only sweater you'll need all season.",
        "From office to after-hours: drape it over your shoulders for instant street-style swagger.",
    ],
    "audience_angles": [
        {
            "segment": "Urban commuting professionals",
            "angle": "Stay sharp on the train with a sleek, high-collar sweater that resists wind and looks polished.",
            "rationale": "The high-collar, snug fit and versatile style match the brand's positioning of comfortable, modern winter wear for everyday consumers.",
        },
        {
            "segment": "College students on a budget",
            "angle": "Get runway-ready winter style for under 1800 - versatile enough for class, coffee dates, and campus parties.",
            "rationale": "Price point (1799) and the sweater's ability to be worn alone or layered align with student needs for affordable, stylish pieces.",
        },
        {
            "segment": "Remote-working parents",
            "angle": "Comfort that lasts through Zoom calls and bedtime stories - snug fit, easy care, and timeless black.",
            "rationale": "Regular fit, easy care instructions, and comfort-focused features support a home-office lifestyle.",
        },
    ],
    "visual_themes": [
        {
            "setting": "Evening city street, soft streetlights, model standing at a crosswalk wearing the sweater, high collar visible, black sweater against muted gray buildings.",
            "mood": "confident, urban chic",
            "color_palette": "deep charcoal, cool blue accents, amber streetlight glow",
            "focal_elements": ["high-collar sweater", "city backdrop", "streetlight", "briefcase"],
        },
        {
            "setting": "Cozy home office nook, natural light from window, sweater draped over a wooden chair, coffee mug on side table, model typing on laptop.",
            "mood": "relaxed, productive",
            "color_palette": "warm neutrals, soft whites, black sweater as focal",
            "focal_elements": ["sweater on chair", "laptop", "coffee mug", "soft throw"],
        },
    ],
}

COMPLETE_OUTPUT = {
    **OBSERVED_OUTPUT,
    "captions": [
        "The high-collar knit that locks in heat without the bulk.",
        "One sweater, three looks: office, weekend, after-hours.",
        "4.86 stars from shoppers who actually wear it.",
    ],
    "messaging_notes": "Review count is low; comfort and warmth claims are inferred from the knit composition.",
    "ungrounded_claims_flagged": ["resists wind"],
}


def _error_types(payload: dict) -> list[tuple[tuple, str]]:
    with pytest.raises(ValidationError) as exc:
        CreativeDirection.model_validate(payload)
    return [(e["loc"], e["type"]) for e in exc.value.errors()]


def test_observed_output_is_rejected_for_missing_captions():
    """The exact real-world failure: no captions key at all."""
    assert (("captions",), "missing") in _error_types(OBSERVED_OUTPUT)


def test_empty_captions_are_rejected():
    assert (("captions",), "too_short") in _error_types({**OBSERVED_OUTPUT, "captions": []})


def test_empty_hooks_are_rejected():
    assert (("hooks",), "too_short") in _error_types({**COMPLETE_OUTPUT, "hooks": []})


def test_complete_output_is_accepted():
    creative = CreativeDirection.model_validate(COMPLETE_OUTPUT)
    assert creative.captions and creative.hooks


def test_validate_node_still_reports_empty_captions():
    """Defence in depth: an object that bypasses validation (e.g. an old
    checkpoint loaded with model_construct) is still caught by validate_node."""
    creative = CreativeDirection.model_construct(
        source_url="https://shop.example/x", hooks=["a"], captions=[]
    )
    out = validate_node({"creative": creative, "retries": 0})
    assert "captions" in out["error"]


async def test_missing_captions_falls_through_to_fallback_model(monkeypatch):
    """Before the schema change this payload validated, so model_used stayed the
    primary model and the product failed at validate. Now it raises, and
    generate_node tries the fallback model in the same attempt."""
    calls: list[str] = []

    async def fake_structured_chat_groq(*, model, system_prompt, user_prompt, schema, temperature):
        calls.append(model)
        payload = COMPLETE_OUTPUT if model == settings.groq_fallback_model else OBSERVED_OUTPUT
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise LLMExtractionError(f"Model output failed schema validation: {exc}") from exc

    monkeypatch.setattr(nodes, "structured_chat_groq", fake_structured_chat_groq)

    research = ProductResearch(
        source_url="https://shop.example/black-turtle-neck-sweater",
        title="Black Turtle-Neck Sweater",
    )
    result = await generate_node({"research": research, "market_context": "", "retries": 0})

    assert calls == [settings.groq_primary_model, settings.groq_fallback_model]
    assert result["error"] is None
    assert result["model_used"] == settings.groq_fallback_model
    assert result["creative"] is not None
    assert result["creative"].captions == COMPLETE_OUTPUT["captions"]
    # source_url from the model is a placeholder and is overwritten by the node.
    assert result["creative"].source_url == research.source_url
