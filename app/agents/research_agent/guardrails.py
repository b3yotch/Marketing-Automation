"""
Pipeline guardrails: one function, called between Research and Creative
Strategy.

    result = await check_guardrails(research)
    if not result.passed:
        ...skip/flag this product, record result.reason in the manifest

Two checks, cheapest first:
  1. Page validity: a plain heuristic on the ProductResearch fields. No model
     call. A page with no usable title/features isn't a real product listing.
  2. Appropriateness: Llama Guard 3 via Ollama, only run if the page was valid
     (no point spending a model call on something that isn't a product).

Why Llama Guard has its own client code instead of core/llm.py's
structured_chat(): it's fine-tuned to a fixed contract. It replies with
"safe" or "unsafe" on the first line, then comma-separated hazard codes
(S1-S13) on a second line if unsafe. It can't be steered with a JSON-schema
prompt, so we parse its native format.

Known limitation: Llama Guard was trained on chat safety, not ecommerce
catalogs. It covers its hazard taxonomy (weapons, hate, sexual content,
etc.) but has no special awareness of ecommerce-specific problems such as
counterfeit goods. Treat it as a useful first-pass filter, not complete
moderation.
"""

from __future__ import annotations

import ollama
from pydantic import BaseModel

from app.agents.research_agent.schema import ProductResearch
from app.core.config import settings

# Meta's MLCommons hazard taxonomy that Llama Guard 3 is trained against.
LLAMA_GUARD_CATEGORIES: dict[str, str] = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
}


class LlamaGuardError(Exception):
    """Llama Guard's response couldn't be parsed into a verdict."""


class GuardrailResult(BaseModel):
    source_url: str
    passed: bool
    # One human-readable line for the batch manifest. None when passed.
    reason: str | None = None
    # Which check stopped the product: "invalid_page", "unsafe", "guard_error",
    # or None when passed.
    blocked_by: str | None = None
    violated_categories: list[str] = []
    raw_verdict: str | None = None


def parse_verdict(raw: str) -> tuple[bool, list[str]]:
    """Parse Llama Guard's output: "safe", or "unsafe\\nS9,S2"."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        raise LlamaGuardError(f"Empty response from Llama Guard: {raw!r}")

    verdict = lines[0].lower()
    if verdict == "safe":
        return True, []
    if verdict == "unsafe":
        categories = [c.strip().upper() for c in lines[1].split(",") if c.strip()] if len(lines) > 1 else []
        return False, categories

    raise LlamaGuardError(f"Unrecognized Llama Guard verdict: {lines[0]!r} (full response: {raw!r})")


def check_validity(research: ProductResearch) -> str | None:
    """Return a reason string if the page isn't a usable product listing, else None."""
    if not research.title or not research.title.strip():
        return "No product title was extracted"
    if research.extraction_confidence < settings.min_valid_extraction_confidence:
        return f"Extraction confidence too low ({research.extraction_confidence:.2f})"
    if not research.features and not research.specifications:
        return "No features or specifications found - likely not a real product page"
    return None


async def classify_content(text: str) -> tuple[bool, list[str], str]:
    """Send text to Llama Guard. Returns (is_safe, category_codes, raw_response)."""
    client = ollama.AsyncClient(host=settings.ollama_host)
    response = await client.chat(
        model=settings.llama_guard_model,
        messages=[{"role": "user", "content": text}],
    )
    raw = response["message"]["content"] or ""
    is_safe, categories = parse_verdict(raw)
    return is_safe, categories, raw


async def check_guardrails(research: ProductResearch) -> GuardrailResult:
    url = research.source_url

    invalid_reason = check_validity(research)
    if invalid_reason:
        return GuardrailResult(source_url=url, passed=False, blocked_by="invalid_page", reason=invalid_reason)

    description = (
        "Product listing to evaluate for ad creative generation:\n"
        f"Title: {research.title}\n"
        f"Brand: {research.brand or 'unknown'}\n"
        f"Features: {'; '.join(research.features) or 'none listed'}\n"
        f"Positioning: {research.brand_positioning or 'none stated'}"
    )

    try:
        is_safe, categories, raw = await classify_content(description)
    except LlamaGuardError as exc:
        # Fail closed: an unparseable verdict blocks the product rather than
        # silently waving it through.
        return GuardrailResult(
            source_url=url,
            passed=False,
            blocked_by="guard_error",
            reason=f"Llama Guard response could not be parsed, failing closed: {exc}",
        )

    if is_safe:
        return GuardrailResult(source_url=url, passed=True, raw_verdict=raw)

    names = ", ".join(f"{c} ({LLAMA_GUARD_CATEGORIES.get(c, 'unknown')})" for c in categories) or "unspecified category"
    return GuardrailResult(
        source_url=url,
        passed=False,
        blocked_by="unsafe",
        reason=f"Flagged by Llama Guard: {names}",
        violated_categories=categories,
        raw_verdict=raw,
    )