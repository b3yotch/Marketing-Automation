"""
Node functions for the Video Generation Agent (Agent 6).

Retry loop and fresh-seed-per-retry decision mirror Agent 4's nodes.py
exactly, same reasoning: a deterministic sampler means a same-seed retry
only helps for pure infrastructure flakiness, not a real generation
failure, so a fresh seed is drawn on every retry.

One new case Agent 4 never had to handle: a theme can arrive here with NO
usable source image at all. Two distinct upstream causes collapse into the
same outcome here: either the theme exhausted ITS retries in Agent 4 and
was skipped there (Image_generation.md Challenge 3, so it never appears in
Agent 5's output at all), or Agent 5 itself had nothing valid to select
from after its own deterministic filter and recorded an empty
selected_local_path (Image_selection.md). Either way, retrying a video job
with no source image can't possibly succeed - so it's handled as an
immediate, retry-free "skip" in generate_node itself, not left to burn
through max_video_gen_retries attempts for no reason.

No more _pick_source_image placeholder here - that was this file's own
"revisit once a critic exists" marker, and Agent 5 (Image Selection) is
that critic. This file now trusts Agent 5's selected_local_path directly
rather than reaching into a candidate list and taking images[0] itself.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from app.core.config import settings
from app.agents.image_generation.utils import slugify_url, new_run_id
from app.agents.image_selection_agent.schema import ImageSelectionOutput, ThemeSelectionResult
from app.agents.video_generation.video_comfyui_client import VideoComfyUIClient, ComfyUIError
from app.agents.video_generation.schema import GeneratedVideo, ThemeVideoResult, VideoGenerationOutput
from app.agents.video_generation.state import VideoGenerationState
from observability import observability as obs

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_client = VideoComfyUIClient()


def _truncate_words(text: str, limit: int) -> str:
    """Trim to <= limit chars on a word boundary. Never returns a trailing
    comma/space, and never raises on a zero/negative limit."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return cut or text[:limit].rstrip(" ,;:-")


def _fit_video_prompt(
    base_prompt: str,
    motion_description: str,
    max_chars: int | None = None,
) -> tuple[str, bool]:
    """
    Fit the combined CogVideoX prompt into its text-encoder budget.

    The wrapper hard-caps at 226 T5 tokens and raises "Prompt is too long"
    instead of truncating, which fails the whole theme after three GPU
    attempts. The scene description is the verbose part, so keep the motion
    instruction (it is what makes the shot move) and trim the scene to fit.
    Returns (prompt, was_clipped).
    """
    budget = max_chars or settings.comfyui_video_max_prompt_chars
    base = " ".join((base_prompt or "").split())
    motion = " ".join((motion_description or "").split())
    combined = f"{base} {motion}".strip()
    if len(combined) <= budget:
        return combined, False

    # Reserve a slice for the motion (usually short); give the rest to the
    # scene. A fifth of the budget is enough for the camera move itself.
    motion_budget = min(len(motion), max(40, budget // 5), max(0, budget))
    motion = _truncate_words(motion, motion_budget)
    base = _truncate_words(base, max(0, budget - len(motion) - 1))
    return f"{base} {motion}".strip(), True


def _classify_comfy_error(exc: Exception) -> str:
    """Observability label for a ComfyUIError, from its message text (the
    client raises one exception type for everything) - best-effort only,
    nothing in the pipeline's control flow uses it."""
    msg = str(exc)
    if msg.startswith("Timed out"):
        return "timeout"
    if msg.startswith("Failed to upload"):
        return "upload_error"
    if msg.startswith("Failed to queue") or "rejected the prompt" in msg:
        return "queue_error"
    if "no outputs" in msg or "no output files" in msg:
        return "no_output"
    if "Configured node id" in msg or "workflow" in msg.lower():
        return "workflow_config_error"
    return "comfy_error"


def _workflow_facts(template: dict | None) -> dict:
    """Facts that live inside the exported workflow JSON rather than in
    Settings (resolution, fps, model precision, offload), read for tracing.
    Found by class_type, so it does not depend on node ids. Best-effort: {}
    when the template has a different shape. `resize_keep_proportion=False`
    is worth watching - it means the source image is stretched to the
    resize node's fixed width x height."""
    facts: dict = {}
    try:
        for node in (template or {}).values():
            cls, inputs = node.get("class_type"), node.get("inputs", {})
            if cls == "ImageResizeKJ":
                facts.update(
                    width=inputs.get("width"),
                    height=inputs.get("height"),
                    resize_keep_proportion=inputs.get("keep_proportion"),
                )
            elif cls == "VHS_VideoCombine":
                facts["fps"] = inputs.get("frame_rate")
            elif cls == "DownloadAndLoadCogVideoModel":
                facts.update(
                    model=inputs.get("model"),
                    precision=inputs.get("precision"),
                    quantization=inputs.get("quantization"),
                    sequential_cpu_offload=inputs.get("enable_sequential_cpu_offload"),
                )
    except Exception:
        return {}
    return facts


def _current_theme(state: VideoGenerationState):
    prompt_sets = state["prompts"].prompt_sets
    return prompt_sets[state.get("current_theme_index", 0)]


def _next_pending_index(
    state: VideoGenerationState,
    start: int,
    completed: list[ThemeVideoResult] | None = None,
) -> int:
    """First theme at or after `start` that has no result yet.

    A run can start with results already seeded (a review decision reopened
    a previously finished video stage, keeping the still-approved videos),
    so "next index" is not simply `current + 1`: themes already done must be
    skipped, even when the done ones are not contiguous.
    """
    done = {
        result.source_setting
        for result in (
            completed
            if completed is not None
            else state.get("completed_theme_results") or []
        )
    }
    prompt_sets = state["prompts"].prompt_sets
    index = max(0, start)
    while index < len(prompt_sets) and prompt_sets[index].source_setting in done:
        index += 1
    return index


def _find_source_theme(state: VideoGenerationState, source_setting: str) -> ThemeSelectionResult | None:
    """
    Looks up this theme's already-judged selection from Agent 5's output,
    matched by source_setting - same matching convention used throughout
    this pipeline (Agent 4 -> Agent 5 -> Agent 6 all echo source_setting
    verbatim so results can be traced back to their origin theme without
    relying on list position staying aligned).

    Returns None only when the theme is missing from Agent 5's output
    entirely - which only happens if Agent 4 skipped it upstream (Agent 5
    loops over whatever Agent 4 produced, so a theme absent from Agent 4's
    output is absent from Agent 5's too). A theme Agent 5 DID process, but
    found nothing valid to select for, still shows up here with a
    ThemeSelectionResult - just one whose selected_local_path is empty
    (see Image_selection.md) - so that case is handled in generate_node,
    not here.
    """
    selection: ImageSelectionOutput = state["selection"]
    for theme_result in selection.theme_results:
        if theme_result.source_setting == source_setting:
            return theme_result
    return None  # that theme was skipped entirely upstream of Agent 5 (Agent 4 retries exhausted there)


def _is_theme_approved(allowlist: list[str] | None, source_setting: str) -> bool:
    """True when the reviewer's allowlist admits this theme.

    None means "no per-theme review happened" (a URL passed directly, or a
    legacy product-level approvals file), so every theme is allowed. A list
    means review.py wrote a decision for each theme it showed, and only the
    approved/replaced ones belong in this run.
    """
    return allowlist is None or source_setting in allowlist


def _resolve_source_image(
    state: VideoGenerationState,
    source_setting: str,
    overrides: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """
    The image Agent 6 should animate for this theme: the reviewer's manual
    replacement if there is one, otherwise Agent 5's pick.

    Returns (path, error). The override is checked for existence here
    because upload_image would otherwise fail as a generic OSError and get
    reported as "Failed saving video locally", which points at the wrong
    stage entirely. Review paths are stored the same way the rest of the
    pipeline stores them (relative to the project root), so both forms are
    accepted.
    """
    override = (overrides or {}).get(source_setting)
    if override:
        raw = Path(override)
        candidates = [raw] if raw.is_absolute() else [_PROJECT_ROOT / raw, raw]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate), None
        return None, (
            "The image approved in review no longer exists on disk: "
            f"{override} (re-run the review step, or generate it again)"
        )

    source_theme = _find_source_theme(state, source_setting)
    return (source_theme.selected_local_path if source_theme else None), None


def start_node(state: VideoGenerationState) -> dict:
    """
    Runs once: loads the video workflow template (fails loudly here if
    it's the wrong JSON format - see VideoComfyUIClient.
    load_video_workflow_template - rather than after wasting a generation
    attempt), and generates a run_id following Agent 4's exact convention
    (own scoped output folder, so repeated runs against the same product
    don't pile into the same directory).
    """
    template = _client.load_video_workflow_template()
    # Preserve results seeded by the caller (review reopened a finished
    # stage) and start at the first theme that still needs work.
    completed = list(state.get("completed_theme_results") or [])
    return {
        "current_theme_index": _next_pending_index(state, 0, completed),
        "completed_theme_results": completed,
        "retries": 0,
        "error": None,
        "run_id": new_run_id(),
        "video_workflow_template": template,
    }


def _save_video_locally(
    *, source_url: str, run_id: str, theme_index: int, file_info: dict
) -> tuple[str, str]:
    """
    Same run-scoped save pattern as Agent 4's _save_images_locally:
    <video_output_dir>/<slugified-product>/<run_id>/theme_<n>/<filename>.
    Returns (local_path, filename). Only ever one file per theme (see
    schema.py - no batching), unlike Agent 4's loop over N images.
    """
    product_dir = Path(settings.video_output_dir) / slugify_url(source_url) / run_id / f"theme_{theme_index}"
    product_dir.mkdir(parents=True, exist_ok=True)

    content = _client.fetch_image_bytes(file_info["filename"], file_info["subfolder"])
    local_path = product_dir / file_info["filename"]
    local_path.write_bytes(content)
    return str(local_path), file_info["filename"]


def generate_node(state: VideoGenerationState, config=None) -> dict:
    theme = _current_theme(state)
    theme_index = state.get("current_theme_index", 0)

    # Review decisions travel through the graph config, not the checkpointed
    # state: a resumed thread invokes with ainvoke(None), which would ignore
    # any fresh input, while config is re-read on every run - so a review
    # decision takes effect even when the theme loop resumes mid-product.
    configurable = (config or {}).get("configurable", {})
    allowlist = configurable.get("theme_allowlist")
    overrides = configurable.get("theme_overrides") or {}

    if not _is_theme_approved(allowlist, theme.source_setting):
        # A human rejected this theme's image in review.py while approving
        # others for the same product. Not a generation failure: skip it
        # without touching retries, exactly like the no-source case below.
        result = ThemeVideoResult(
            source_setting=theme.source_setting,
            video=None,
            generation_time_seconds=0.0,
            retries_used=0,
            status="skipped_not_approved",
        )
        return {"error": None, "_pending_result": result}

    source_image_path, source_error = _resolve_source_image(
        state, theme.source_setting, overrides
    )
    if source_error:
        return {"error": source_error, "current_seed": None}

    # Covers both causes described in the module docstring: source_theme
    # itself is None (Agent 4 skipped this theme, so it never reached
    # Agent 5), or Agent 5 processed the theme but selected_local_path is
    # "" because nothing passed its deterministic filter. Both are falsy,
    # so one check handles both without needing to distinguish them here -
    # neither is retry-worthy from this agent's side either way.
    if not source_image_path:
        # No retry-worthy failure here - there's nothing generation could
        # succeed at. Skip immediately, don't touch retries.
        result = ThemeVideoResult(
            source_setting=theme.source_setting,
            video=None,
            generation_time_seconds=0.0,
            retries_used=0,
            status="skipped_no_source_image",
        )
        return {"error": None, "_pending_result": result}

    seed = state.get("current_seed") or random.randint(0, 2**32 - 1)
    # CogVideoX's ComfyUI node takes a single "prompt" string - Agent 3's
    # schema deliberately keeps base_prompt (scene) and motion_description
    # (camera/subject motion) as separate fields for clarity/editability,
    # but they're combined here since there's only one text input to feed.
    # Fit it to the text encoder's budget first: a too-long prompt makes
    # ComfyUI raise "Prompt is too long" and burns every retry without ever
    # reaching the sampler (see _fit_video_prompt).
    raw_prompt = f"{theme.video_prompt.base_prompt} {theme.video_prompt.motion_description}".strip()
    combined_prompt, prompt_clipped = _fit_video_prompt(
        theme.video_prompt.base_prompt,
        theme.video_prompt.motion_description,
    )
    if prompt_clipped:
        print(
            f"[video] prompt clipped from {len(raw_prompt)} to {len(combined_prompt)} chars "
            f"to fit CogVideoX's 226-token limit (theme {theme_index})"
        )

    facts = _workflow_facts(state.get("video_workflow_template"))
    frames = settings.comfyui_video_num_frames
    steps = settings.comfyui_video_steps
    start = None
    with obs.comfy_job(
        kind="video",
        name="comfyui.video",
        model=facts.get("model") or "cogvideox-2b-img2vid",
        input={
            "prompt": combined_prompt,
            "negative_prompt": theme.video_prompt.negative_prompt,
            "source_image": source_image_path,
        },
        model_parameters={
            "num_frames": frames,
            "steps": steps,
            "cfg": settings.comfyui_video_cfg,
            "scheduler": settings.comfyui_video_scheduler,
            "denoise": settings.comfyui_video_denoise_strength,
            "width": facts.get("width"),
            "height": facts.get("height"),
            "fps": facts.get("fps"),
            "precision": facts.get("precision"),
            "quantization": facts.get("quantization"),
            "sequential_cpu_offload": facts.get("sequential_cpu_offload"),
            "resize_keep_proportion": facts.get("resize_keep_proportion"),
        },
        metadata={
            "theme_index": theme_index,
            "attempt": state.get("retries", 0),
            "seed": seed,
            "timeout_s": settings.comfyui_video_generation_timeout_seconds,
        },
        unit="videos",
        requested=1,
        steps=steps,
        frames=frames,
        width=facts.get("width"),
        height=facts.get("height"),
        ollama_host=settings.ollama_host,
    ) as job:
        job.set_probes(
            queue=_client.queue_snapshot,
            stats=_client.system_stats,
            history=_client.get_history_entry,
        )
        # Snapshot BEFORE the blanket interrupt below, so a job left running
        # by an earlier attempt or an interrupted run is visible in the trace.
        if job.active:
            job.probe_start()
            obs.reconcile_abandoned(_client.get_history_entry)

        # Defensive, not just reactive: clears any job left running from a
        # prior attempt that timed out client-side (see VideoComfyUIClient.
        # interrupt's docstring) before spending time trying to talk to a
        # server that may still be blocked on it. Cheap and a no-op if
        # ComfyUI is already idle.
        _client.interrupt()

        start = time.monotonic()
        try:
            job.phase("upload")
            uploaded_filename = _client.upload_image(source_image_path)
            workflow = _client.build_video_workflow(
                state["video_workflow_template"],
                image_filename=uploaded_filename,
                positive_prompt=combined_prompt,
                negative_prompt=theme.video_prompt.negative_prompt,
                seed=seed,
                filename_prefix=f"agent6_{theme_index}",
            )
            job.phase("queue")
            prompt_id = _client.queue_prompt(workflow)
            job.submitted(prompt_id)
            history_entry = _client.wait_for_completion(prompt_id)
            job.completed(history_entry)
            raw_outputs = _client.extract_video_outputs(history_entry)
            if not raw_outputs:
                raise ComfyUIError("Video job finished but no output files were found.")

            local_path, filename = _save_video_locally(
                source_url=state["prompts"].source_url,
                run_id=state["run_id"],
                theme_index=theme_index,
                file_info=raw_outputs[0],  # exactly one video per theme, no batching
            )
            job.outputs([filename], 1)
        except ComfyUIError as exc:
            outcome = _classify_comfy_error(exc)
            cancel = getattr(_client, "last_cancel", None) if outcome == "timeout" else None
            job.set_meta(cancel=cancel)
            job.fail(outcome, str(exc), abandoned=(outcome == "timeout" and cancel != "dequeued"))
            return {"error": str(exc), "current_seed": None}
        except OSError as exc:
            lost_connection = job.current_phase in ("upload", "queue", "wait")
            job.fail(
                "connection_error" if lost_connection else "save_error",
                f"{type(exc).__name__}: {exc}",
                abandoned=(job.current_phase == "wait"),
            )
            return {"error": f"Failed saving video locally: {exc}", "current_seed": None}

    elapsed = time.monotonic() - start

    video = GeneratedVideo(
        filename=filename,
        subfolder=raw_outputs[0].get("subfolder", ""),
        local_path=local_path,
        seed=seed,
        source_image_local_path=source_image_path,
    )
    result = ThemeVideoResult(
        source_setting=theme.source_setting,
        video=video,
        generation_time_seconds=elapsed,
        retries_used=state.get("retries", 0),
        status="success",
    )
    return {"error": None, "_pending_result": result}


def validate_node(state: VideoGenerationState) -> dict:
    """
    Low bar, same as every other agent's validate_node: did generate_node
    produce SOMETHING (a success or an explicit skip), not whether the
    video is any good - image quality was already judged by Agent 5 before
    this agent ever ran, but video quality itself has no judge yet (a
    separate video-quality review step, if one gets built, would be a
    later addition, not this node's job). Unlike Agent 4, there's no
    "expected count" to check against, since it's always exactly one video
    or an explicit skip - the binary presence of a valid _pending_result
    IS the check.
    """
    if state.get("error"):
        return {}  # already has a failure reason from generate_node

    if not state.get("_pending_result"):
        return {"error": "generate_node produced neither a result nor an error - this shouldn't happen."}

    return {}


def bump_retry_node(state: VideoGenerationState) -> dict:
    return {"retries": state.get("retries", 0) + 1}


def advance_theme_node(state: VideoGenerationState) -> dict:
    """
    On success or explicit skip: files the result, resets per-attempt
    scratch state, moves to the next theme. On retries-exhausted failure:
    pending is None here (generate_node's except-branches never set
    _pending_result), so nothing gets appended - same as Agent 4, the
    theme is simply absent from theme_results rather than recorded as a
    distinct "failed" entry. See schema.py's module docstring for why this
    asymmetry (skips ARE recorded, retry-exhausted failures aren't) is
    intentional, not an inconsistency.
    """
    completed = list(state.get("completed_theme_results", []))
    pending = state.get("_pending_result")
    if pending:
        completed.append(pending)

    return {
        "completed_theme_results": completed,
        "current_theme_index": _next_pending_index(
            state, state.get("current_theme_index", 0) + 1, completed
        ),
        "retries": 0,
        "error": None,
        "current_seed": None,
        "_pending_result": None,
    }


def finalize_node(state: VideoGenerationState) -> dict:
    obs.reconcile_abandoned(_client.get_history_entry)  # observability only; no-op when untraced
    output = VideoGenerationOutput(
        source_url=state["prompts"].source_url,
        theme_results=state.get("completed_theme_results", []),
    )
    return {"videos": output}