"""Regression tests for fitting CogVideoX prompts into its 226-token cap.

Token counts measured with ComfyUI's own T5 tokenizer on the real outputs:

    jeans theme 1   911 chars -> 223 tokens   succeeded
    jeans theme 0   996 chars -> 237 tokens   failed
    sweater theme 1 1076 chars -> 273 tokens  failed
    sweater theme 0 1204 chars -> 286 tokens  failed

ComfyUI raised "Prompt is too long, max tokens supported is 226 or less"
rather than truncating, so every retry burned ~3 minutes of GPU time. The old
prompt-generation validator measured only base_prompt (and allowed up to 1200
chars), so motion_description pushed the combined prompt over the limit. Agent
3 now targets an 800-char combined budget (~4.1 chars/token) and the video node
clips as a last resort for already-cached prompts.
"""

from __future__ import annotations

from app.agents.creative_strategy_agent.schema import CreativeDirection, VisualTheme
from app.agents.prompt_gen_agent.nodes import validate_node
from app.agents.prompt_gen_agent.schema import (
    ImageGenerationPrompt,
    ThemePromptSet,
    VideoGenerationPrompt,
)
from app.agents.video_generation.nodes import _fit_video_prompt
from app.core.config import settings


def test_short_prompt_is_unchanged():
    prompt, clipped = _fit_video_prompt("a scene", "slow pan")
    assert prompt == "a scene slow pan"
    assert clipped is False


def test_long_prompt_is_clipped_to_budget_and_keeps_motion():
    base = " ".join(["scene"] * 200)  # 1199 chars
    motion = "slow dolly-in toward the product, then a gentle pan"
    prompt, clipped = _fit_video_prompt(base, motion)
    assert clipped is True
    assert len(prompt) <= settings.comfyui_video_max_prompt_chars
    assert motion in prompt
    assert not prompt.endswith((" ", ",", ";", ":", "-"))


def test_observed_sweater_prompt_fits():
    # The exact lengths that failed in the real run: 883 + 320 = 1204 chars,
    # which measured 286 T5 tokens against a 226-token cap.
    base = ("b " * 441).strip()  # 883 chars
    motion = ("m " * 160).strip()  # 319 chars
    prompt, clipped = _fit_video_prompt(base, motion)
    assert clipped is True
    assert len(prompt) <= settings.comfyui_video_max_prompt_chars
    assert prompt.startswith("b b")


def test_observed_successful_prompt_is_left_alone_at_its_length():
    # 911 chars was the one prompt that fit (223 tokens), but it has almost no
    # margin, so the 800-char budget deliberately trims it too. A prompt under
    # the budget must pass through untouched.
    base = "s " * 380  # 760 chars
    prompt, clipped = _fit_video_prompt(base, "slow pan")
    assert clipped is False
    assert prompt == f"{base}slow pan"


def test_empty_motion_still_fits():
    prompt, clipped = _fit_video_prompt("scene " * 300, "")
    assert clipped is True
    assert 0 < len(prompt) <= settings.comfyui_video_max_prompt_chars


def test_tiny_budget_never_overflows():
    # Guards the clamp: motion used to get a hard 100-char slice, which
    # overflowed any budget below that.
    for budget in (1, 10, 50, 99, 100):
        prompt, clipped = _fit_video_prompt(
            "base " * 100, "motion " * 50, max_chars=budget
        )
        assert clipped is True
        assert len(prompt) <= budget


def _state(base: str, motion: str) -> dict:
    creative = CreativeDirection(
        source_url="https://shop.example/x",
        hooks=["hook"],
        captions=["caption"],
        visual_themes=[VisualTheme(setting="s", mood="m", color_palette="p")],
    )
    theme_set = ThemePromptSet(
        source_setting="s",
        image_prompt=ImageGenerationPrompt(positive_prompt="p", aspect_ratio="1:1"),
        video_prompt=VideoGenerationPrompt(
            base_prompt=base, motion_description=motion
        ),
    )
    return {
        "creative": creative,
        "theme_sets": {0: theme_set},
        "theme_errors": {},
        "retries": 0,
    }


def test_validation_rejects_combined_prompt_over_budget():
    out = validate_node(_state("b" * 700, "m" * 200))
    assert "too long" in out["error"]
    assert out["prompts"] is None


def test_validation_accepts_prompt_within_budget():
    out = validate_node(_state("b" * 500, "m" * 150))
    assert out["error"] is None
    assert out["prompts"] is not None
    assert settings.comfyui_video_max_prompt_chars == 800
