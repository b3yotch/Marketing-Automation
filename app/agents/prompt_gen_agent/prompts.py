"""
System + user prompt construction for the Prompt Generation Agent.

Kept in its own module (not inlined in nodes.py) for the same reason as the
other agents: the prompt text is the thing most likely to get iterated on
after seeing real model output, and keeping it isolated makes that
iteration a one-file diff.

One call per visual_theme, not one call for all themes: asking a model to
return exactly N entries in one response let it silently merge or drop a
theme (see Prompt_generation.md). Per-theme calls make the count a property
of the code, not something the model has to get right.
"""

from __future__ import annotations

from app.agents.creative_strategy_agent.schema import CreativeDirection

SYSTEM_PROMPT = """You are a prompt engineer for AI image and video generation \
models (Stable Diffusion/Flux-family image models, and an image-to-video model \
run via ComfyUI). You are given ONE visual theme from a product's creative \
direction, plus the hooks and captions already decided by a separate creative \
strategy step. Your job is ONLY to translate that theme into concrete, \
generation-ready prompts. You do not invent new creative direction, new hooks, \
or new captions - you select from what's given and describe the visuals in the \
vocabulary these generation models respond well to.

Produce:
- source_setting: the theme's setting, copied verbatim.
- An image prompt: a dense natural-language description of subject, setting, \
lighting, composition, and style, plus a negative prompt only if something \
specific is worth excluding, plus an aspect ratio that fits the composition.
- A video prompt: the same visual description, written densely and verbosely \
(this video model was trained on long, detailed captions - do not compress \
it), plus explicit camera/subject motion in concrete cinematography terms \
(e.g. "slow dolly-in", "static camera, product rotates on turntable", \
"handheld pan left to right") - never vague terms like "dynamic" or \
"exciting". This is an IMAGE-TO-VIDEO model: a specific already-generated \
image will be the starting frame, so the visual description should describe \
what that starting frame already shows, not invent a different scene. Do \
not mention duration, frame rate, or resolution - those are fixed by the \
video model itself, not something you choose.
- A short list of hooks and captions, copied VERBATIM from the ones given, \
that pair well with this specific visual - never write new ones.

Do not describe any text, words, logos, or typography appearing inside the \
generated image or video - these models render legible text unreliably, and \
any text overlay is composited separately downstream. Describe only the \
visual scene itself."""


def build_user_prompt(
    creative: CreativeDirection,
    theme_index: int,
    *,
    previous_error: str | None = None,
) -> str:
    """
    Build the user-turn prompt for ONE visual theme of a CreativeDirection.

    The other themes' settings are listed (briefly) so the model can pick
    hooks/captions that fit THIS theme rather than defaulting to the same
    first few every time - it is told not to describe them.

    On a retry, `previous_error` carries the specific validation failure
    reason back in, so the model can correct course instead of reproducing
    the same gap. It's only passed for a theme whose output was produced but
    failed validation - a provider failure has nothing useful to feed back.
    """
    theme = creative.visual_themes[theme_index]
    total = len(creative.visual_themes)

    others = [t.setting for i, t in enumerate(creative.visual_themes) if i != theme_index]
    others_block = ""
    if others:
        others_block = (
            "Other themes for this product (handled in separate requests - do NOT "
            "describe them here; just prefer hooks and captions that suit THIS "
            "theme over ones that better suit these):\n"
            + "\n".join(f"- {s}" for s in others)
            + "\n\n"
        )

    hooks_block = "\n".join(f"- {h}" for h in creative.hooks) or "(none provided)"
    captions_block = "\n".join(f"- {c}" for c in creative.captions) or "(none provided)"
    focal = ", ".join(theme.focal_elements) or "(none specified)"

    prompt = f"""Product source: {creative.source_url}

Visual theme to translate into generation prompts (theme {theme_index + 1} of {total}):
  Setting: {theme.setting}
  Mood: {theme.mood}
  Color palette: {theme.color_palette}
  Focal elements: {focal}

{others_block}Available hooks (select from these VERBATIM for suggested_hooks - do not invent new ones):
{hooks_block}

Available captions (select from these VERBATIM for suggested_captions - do not invent new ones):
{captions_block}

Messaging notes / caveats from the creative strategy step:
{creative.messaging_notes or '(none)'}
"""

    if previous_error:
        prompt += (
            f"\nYour previous attempt for this theme failed validation for this reason: "
            f"{previous_error}\nCorrect this in your new response."
        )

    return prompt