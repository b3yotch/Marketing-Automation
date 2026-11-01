"""
Node functions for the Image Generation Agent.

Retry loop mirrors Agents 1-3's shape (generate -> validate -> bump_retry ->
generate) per your call to keep auto-retry despite no persistence existing
yet. One deliberate addition: each retry draws a NEW random seed rather than
reusing the same one.

Why: with a deterministic sampler, same seed + same prompt + same params
will reproduce the exact same output - including the exact same failure if
the failure was ComfyUI erroring on that specific generation (not a
transient network blip). Retrying with a fresh seed means a retry can
actually succeed differently; retrying with the same seed only helps for
failures that are pure infrastructure flakiness (timeout, connection reset),
and silently wastes a retry attempt on anything else. This is a real
decision, not an oversight - flagging it here the same way Agent 3 flagged
its model-choice decisions, in case you want same-seed retries instead for
a specific transient-failure debugging scenario.

State keys (prompts/retries/error/images) match the naming pattern
test_full_pipeline_live.py already uses for Agents 1-3 (research -> creative
-> prompts), so this agent slots into the same script without inventing a
different vocabulary.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from app.core.config import settings
from app.agents.image_generation.aspect_ratio import resolve_dimensions
from app.agents.image_generation.comfyui_client import ComfyUIClient, ComfyUIError
from app.agents.image_generation.utils import slugify_url, distribute_total, new_run_id
from app.agents.image_generation.schema import (
    GeneratedImage,
    ImageGenerationOutput,
    ThemeGenerationResult,
)
from app.agents.image_generation.state import ImageGenerationState
from observability import observability as obs

_client = ComfyUIClient()


def _classify_comfy_error(exc: Exception) -> str:
    """Observability label for a ComfyUIError. Derived from the message text
    (the client raises one exception type for everything), so it is a
    best-effort label only - nothing in the pipeline's control flow uses it.
    Unrecognised messages fall into 'comfy_error'."""
    msg = str(exc)
    if msg.startswith("Timed out"):
        return "timeout"
    if msg.startswith("Failed to queue") or "rejected the prompt" in msg:
        return "queue_error"
    if "no image output" in msg:
        return "no_output"
    return "comfy_error"


def _current_theme(state: ImageGenerationState):
    prompt_sets = state["prompts"].prompt_sets
    return prompt_sets[state.get("current_theme_index", 0)]


def start_node(state: ImageGenerationState) -> dict:
    """
    Runs once: verifies the checkpoint exists before spending any time on
    generation, generates a run_id to keep this run's images from mixing
    into a previous run's folders, and splits total_images_per_product
    across however many themes this product actually has (typically 2-3,
    not known until Agent 3's output arrives - so this can't be precomputed
    as a fixed per-theme constant).
    """
    _client.check_checkpoint(settings.comfyui_checkpoint)
    num_themes = len(state["prompts"].prompt_sets)
    return {
        "current_theme_index": 0,
        "completed_theme_results": [],
        "retries": 0,
        "error": None,
        "run_id": new_run_id(),
        "theme_image_counts": distribute_total(settings.total_images_per_product, num_themes),
    }


def _save_images_locally(
    *, source_url: str, run_id: str, theme_index: int, images: list[dict], width: int, height: int, seed: int
) -> list[GeneratedImage]:
    """
    Downloads each generated image from ComfyUI's /view endpoint and writes
    it under this project's own output directory, structured as
    <image_output_dir>/<slugified-product>/<run_id>/theme_<n>/<filename> -
    the run_id segment is what keeps repeated runs against the same product
    URL from piling into the same folder (each run gets its own timestamped
    subfolder instead of accumulating alongside every prior run's images).

    Runs inside generate_node rather than as a separate graph node: a
    partially-downloaded set of images isn't a meaningful state to pause at
    or retry independently of the generation call that produced them - if
    the copy fails, the whole attempt should be treated as failed and
    retried from generation, not resumed from "generation succeeded, copy
    didn't."
    """
    product_dir = Path(settings.image_output_dir) / slugify_url(source_url) / run_id / f"theme_{theme_index}"
    product_dir.mkdir(parents=True, exist_ok=True)

    saved: list[GeneratedImage] = []
    for img in images:
        content = _client.fetch_image_bytes(img["filename"], img["subfolder"])
        local_path = product_dir / img["filename"]
        local_path.write_bytes(content)

        saved.append(
            GeneratedImage(
                filename=img["filename"],
                subfolder=img["subfolder"],
                local_path=str(local_path),
                width=width,
                height=height,
                seed=seed,
            )
        )
    return saved


def generate_node(state: ImageGenerationState) -> dict:
    theme = _current_theme(state)
    theme_index = state.get("current_theme_index", 0)
    width, height = resolve_dimensions(theme.image_prompt.aspect_ratio)
    seed = state.get("current_seed") or random.randint(0, 2**32 - 1)
    batch_size = state["theme_image_counts"][theme_index]

    workflow = _client.build_image_workflow(
        positive_prompt=theme.image_prompt.positive_prompt,
        negative_prompt=theme.image_prompt.negative_prompt,
        width=width,
        height=height,
        seed=seed,
        batch_size=batch_size,
        filename_prefix=f"agent4_{theme_index}",
    )

    steps = settings.comfyui_steps
    start = None
    with obs.comfy_job(
        kind="image",
        name="comfyui.image_batch",
        model=settings.comfyui_checkpoint,
        input={
            "positive_prompt": theme.image_prompt.positive_prompt,
            "negative_prompt": theme.image_prompt.negative_prompt,
        },
        model_parameters={
            "steps": steps,
            "cfg": settings.comfyui_cfg,
            "sampler": settings.comfyui_sampler,
            "scheduler": settings.comfyui_scheduler,
            "width": width,
            "height": height,
            "batch_size": batch_size,
            "denoise": 1.0,
        },
        metadata={
            "theme_index": theme_index,
            "attempt": state.get("retries", 0),
            "seed": seed,
            "aspect_ratio": theme.image_prompt.aspect_ratio,
        },
        unit="images",
        requested=batch_size,
        steps=steps,
        width=width,
        height=height,
        ollama_host=settings.ollama_host,
    ) as job:
        job.set_probes(
            queue=_client.queue_snapshot,
            stats=_client.system_stats,
            history=_client.get_history_entry,
        )
        try:
            if job.active:
                job.probe_start()
                obs.reconcile_abandoned(_client.get_history_entry)
            start = time.monotonic()
            job.phase("queue")
            prompt_id = _client.queue_prompt(workflow)
            job.submitted(prompt_id)
            history_entry = _client.wait_for_completion(prompt_id)
            job.completed(history_entry)
            raw_images = _client.extract_images(history_entry)
            saved_images = _save_images_locally(
                source_url=state["prompts"].source_url,
                run_id=state["run_id"],
                theme_index=theme_index,
                images=raw_images,
                width=width,
                height=height,
                seed=seed,
            )
            job.outputs([img.filename for img in saved_images], len(saved_images))
            if len(saved_images) != batch_size:
                job.unusable("count_mismatch")  # validate_node will reject it; count the GPU time as waste
        except ComfyUIError as exc:
            # Caught here (not left to propagate) so validate_node has a
            # concrete error to act on immediately - same "write the failure
            # reason the moment it happens" pattern as Agents 1-3's
            # validate_node, not deferred until retries are exhausted.
            outcome = _classify_comfy_error(exc)
            job.fail(outcome, str(exc), abandoned=(outcome == "timeout"))
            return {"error": str(exc), "current_seed": None}
        except OSError as exc:
            # Local disk write failure (permissions, disk full) - distinct from
            # a ComfyUI-side failure, but still needs to fail this attempt and
            # retry, since a partially-saved theme isn't usable.
            # (Observability only: a requests exception during queue/wait also
            # lands here because requests' errors subclass OSError, so the
            # phase decides whether it is reported as a lost connection.)
            lost_connection = job.current_phase in ("queue", "wait")
            job.fail(
                "connection_error" if lost_connection else "save_error",
                f"{type(exc).__name__}: {exc}",
                abandoned=(job.current_phase == "wait"),
            )
            return {"error": f"Failed saving images locally: {exc}", "current_seed": None}

    elapsed = time.monotonic() - start

    result = ThemeGenerationResult(
        source_setting=theme.source_setting,
        aspect_ratio=theme.image_prompt.aspect_ratio,
        resolved_width=width,
        resolved_height=height,
        images=saved_images,
        generation_time_seconds=elapsed,
        retries_used=state.get("retries", 0),
    )

    return {"error": None, "_pending_result": result}


def validate_node(state: ImageGenerationState) -> dict:
    """
    Low bar, same philosophy as Agents 1-3: did generation actually produce
    the expected number of images, not whether they're any good - that's
    Agent 5's job once it exists.
    """
    if state.get("error"):
        return {}  # already has a failure reason from generate_node - nothing to add

    pending = state.get("_pending_result")
    theme_index = state.get("current_theme_index", 0)
    expected = state["theme_image_counts"][theme_index]
    if not pending or len(pending.images) != expected:
        got = len(pending.images) if pending else 0
        return {"error": f"Expected {expected} images, got {got}."}

    return {}


def bump_retry_node(state: ImageGenerationState) -> dict:
    return {"retries": state.get("retries", 0) + 1}


def advance_theme_node(state: ImageGenerationState) -> dict:
    """On success: files the completed theme result, resets per-attempt scratch state, moves to the next theme."""
    completed = list(state.get("completed_theme_results", []))
    pending = state.get("_pending_result")
    if pending:
        completed.append(pending)

    return {
        "completed_theme_results": completed,
        "current_theme_index": state.get("current_theme_index", 0) + 1,
        "retries": 0,
        "error": None,
        "current_seed": None,
        "_pending_result": None,
    }


def finalize_node(state: ImageGenerationState) -> dict:
    obs.reconcile_abandoned(_client.get_history_entry)  # observability only; no-op when untraced
    output = ImageGenerationOutput(
        source_url=state["prompts"].source_url,
        theme_results=state.get("completed_theme_results", []),
    )
    return {"images": output}