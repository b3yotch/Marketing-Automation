"""
Output contract for the Creative Strategy Agent.

Field shapes here aren't guessed upfront - they're informed by the 3-way
model comparison run before implementation (Scout / Llama 3.3 70B /
gpt-oss-120b against the same sample ProductResearch inputs). All three
models, unprompted, naturally separated audience angles into
segment/angle/rationale and visual themes into setting/mood/palette - so
those became structured sub-models rather than free text, the same reasoning
Agent 1 used for its explicit ProductResearch fields over a free-text blob.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AudienceAngle(BaseModel):
    """One audience segment and the specific angle tailored to it."""

    segment: str = Field(
        description="A concrete audience segment (e.g. 'remote-working parents', 'digital nomads') - not a vague demographic bucket like 'young professionals'."
    )
    angle: str = Field(description="The specific message/hook angle that lands for this segment.")
    rationale: str = Field(
        default="",
        description="Why this angle resonates with this segment, grounded in the product research (features, reviews, or positioning already stated).",
    )


class VisualTheme(BaseModel):
    """One concrete visual direction for imagery/video generation downstream."""

    setting: str = Field(description="Concrete setting/location for the shot - specific enough to act on without more clarification.")
    mood: str = Field(description="Overall mood/tone of the visual (e.g. 'adventurous, pristine', 'relaxed, everyday elegance').")
    color_palette: str = Field(description="Color palette direction (e.g. 'muted earth tones, forest green, slate gray').")
    focal_elements: list[str] = Field(
        default_factory=list,
        description="Specific objects/elements that should be in frame - what a photographer or image-generation prompt needs to include.",
    )


class CreativeDirection(BaseModel):
    """Final, validated creative strategy output for one product."""

    source_url: str

    # Required (no default) on purpose. With a default, the field is optional in
    # the JSON Schema handed to Groq, and a real gpt-oss-120b run returned a
    # complete, valid object that simply ended after visual_themes - hooks and
    # captions omitted - because nothing forced it to emit them. Pydantic then
    # filled in [], validate_node rejected it, and the fallback model was never
    # tried (no exception was raised). Required + min_length=1 makes a missing or
    # empty value a ValidationError, which structured_chat_groq wraps as
    # LLMExtractionError and generate_node handles by trying the fallback model.
    hooks: list[str] = Field(
        ...,
        min_length=1,
        description="3-5 distinct scroll-stopping opening hooks for video/image ads. Each must take a genuinely different angle (pain point, social proof, curiosity, bold claim) - not restatements of the same idea.",
    )
    audience_angles: list[AudienceAngle] = Field(
        default_factory=list,
        description="2-4 distinct audience segments this product could be marketed to, each with its own tailored angle.",
    )
    visual_themes: list[VisualTheme] = Field(
        default_factory=list,
        description="2-3 concrete visual directions for imagery/video generation downstream.",
    )
    captions: list[str] = Field(
        ...,
        min_length=1,
        description="3-5 short-form captions ready to pair with a post, each under 150 characters.",
    )
    messaging_notes: str = Field(
        default="",
        description="Caveats flagging any claims the product research doesn't support strongly enough to state outright (e.g. no reviews to cite, positioning is inferred not stated).",
    )

    extraction_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Self-reported confidence 0-1 that this creative direction is grounded in the research provided, not invented.",
    )
    ungrounded_claims_flagged: list[str] = Field(
        default_factory=list,
        description="Any specific claims in hooks/captions that are inferred rather than directly supported by the research - so downstream/human review knows what to double-check before publishing.",
    )

    @field_validator("captions", mode="after")
    @classmethod
    def _trim_long_captions(cls, value: list[str]) -> list[str]:
        """
        Real model output occasionally runs a caption a few characters past
        the soft 150-char guidance (see the gpt-oss-120b test output, which
        added emoji/hashtags that pushed a couple over). Trim rather than
        reject - a slightly-too-long caption is still usable creative
        material for a human to shorten; failing validation over it would
        waste a retry on otherwise-good output. Same "normalize in code, not
        an ever-stricter prompt" philosophy as Agent 1's specifications
        coercion validator.
        """
        trimmed: list[str] = []
        for caption in value:
            if len(caption) > 150:
                trimmed.append(caption[:147].rstrip() + "...")
            else:
                trimmed.append(caption)
        return trimmed