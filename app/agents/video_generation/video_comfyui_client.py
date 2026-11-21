"""
ComfyUI client for the Video Generation Agent (Agent 5).

Subclasses Agent 4's ComfyUIClient rather than duplicating or modifying it,
to reuse queue_prompt/fetch_image_bytes/check_checkpoint as-is - those are
generic to ComfyUI's HTTP API, not specific to image generation. Two
things are added deliberately as an override + new methods, not as edits
to Agent 4's file, so its already-tested code stays untouched:

1. wait_for_completion is overridden here, not patched in the base class.
   The base version (correctly, for images) checks each node's output for
   an "images" key. The video workflow's save node's output key isn't
   "images" and isn't a stable name across custom node packages -
   confirmed during hand-testing, not assumed - so this override accepts
   any non-empty outputs dict instead of a specific key, same as the
   hand-test script that was proven working before this became agent
   code.
2. upload_image, load_video_workflow_template, build_video_workflow, and
   extract_video_outputs are new. Agent 4 never needed an image-upload
   path (text-to-image has no image input), and the video workflow graph
   is structurally nothing like build_image_workflow's core-node graph -
   it's an externally-authored, custom-node-based workflow this project
   doesn't construct from scratch, only patches specific node IDs inside
   (see Settings' comfyui_video_*_node_id fields for what those IDs are
   pinned to and what breaks the pinning).
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import requests

from app.core.config import settings
from app.agents.image_generation.comfyui_client import (
    ComfyUIClient as _ImageComfyUIClient,
    ComfyUIError,
)


class VideoComfyUIClient(_ImageComfyUIClient):
    def upload_image(self, image_path: str) -> str:
        """
        POSTs a local file to /upload/image, placing it in ComfyUI's own
        input/ directory and returning the server-side filename to
        reference in the workflow's LoadImage-style node. Needed because
        the source frame is an Agent 4 output already on local disk - it
        has to round-trip back through ComfyUI before the video workflow
        can reference it.

        Wraps request failures as ComfyUIError explicitly rather than
        letting a raw requests exception propagate (requests.
        ConnectionError happens to subclass OSError, which generate_node
        also catches - but with a misleading "failed saving locally"
        message, since that except-branch was written for local disk
        failures, not upload failures). Caught a real instance of this:
        ComfyUI can't service ANY new connection - including a plain
        upload - while synchronously blocked on GPU sampling from a
        still-running prior job, which surfaces here as a connection
        reset, not a timeout.
        """
        path = Path(image_path)
        try:
            with open(path, "rb") as f:
                files = {"image": (path.name, f, "image/png")}
                r = requests.post(f"{self.server}/upload/image", files=files, timeout=30)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ComfyUIError(f"Failed to upload {path.name} to ComfyUI: {exc}") from exc
        return r.json()["name"]

    def interrupt(self) -> None:
        """
        Stops whatever job ComfyUI is currently executing. Necessary
        because a client-side timeout in wait_for_completion does NOT
        stop the job on ComfyUI's end - it keeps running on the GPU,
        blocking the server from servicing any subsequent request
        (including an unrelated theme's upload_image call) until it
        finishes or is explicitly interrupted. Best-effort: if this call
        itself fails, the next attempt is in the same position it would
        have been without this method, so there's nothing more useful to
        do than swallow the error and let the retry proceed.
        """
        try:
            requests.post(f"{self.server}/interrupt", timeout=10)
        except requests.RequestException:
            pass

    def load_video_workflow_template(self, path: str | None = None) -> dict:
        """
        Loads the API-format workflow JSON once (call from start_node, not
        per-theme - the template doesn't change across themes, only the
        per-call patches in build_video_workflow do).

        Guards against the single most likely setup mistake, caught during
        hand-testing: pointing this at ComfyUI's *UI*-format export
        (nodes/links graph for the canvas) instead of the *API*-format
        export /prompt actually needs (flat {node_id: {class_type,
        inputs}}). Fails loudly here rather than surfacing as a confusing
        error from ComfyUI later.
        """
        workflow_path = Path(path or settings.comfyui_video_workflow_json)
        if not workflow_path.exists():
            raise ComfyUIError(f"Video workflow JSON not found: {workflow_path}")
        with open(workflow_path) as f:
            workflow = json.load(f)
        if "nodes" in workflow and "links" in workflow:
            raise ComfyUIError(
                f"{workflow_path} is a UI-format ComfyUI workflow (has "
                "top-level 'nodes'/'links' keys), not the API-format "
                "export the /prompt endpoint needs. Re-export via "
                "ComfyUI's 'Save (API Format)' option (requires Dev Mode "
                "enabled in Settings)."
            )
        return workflow

    def build_video_workflow(
        self,
        template: dict,
        *,
        image_filename: str,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        filename_prefix: str,
    ) -> dict:
        """
        Patches a deep copy of the loaded template with per-call values.
        Always deep-copies: repeated calls (one per theme) must never
        mutate shared state between themes - a mistake that wouldn't raise
        anything, it would just silently corrupt the next theme's job with
        this one's values.

        Node IDs come from Settings, pinned to the specific workflow
        export that was hand-tested working - not discovered generically,
        since there's no portable class_type convention across CogVideoX
        custom node packages the way there is for core ComfyUI nodes.
        """
        workflow = copy.deepcopy(template)

        img_node = settings.comfyui_video_image_load_node_id
        pos_node = settings.comfyui_video_positive_prompt_node_id
        neg_node = settings.comfyui_video_negative_prompt_node_id
        sampler_node = settings.comfyui_video_sampler_node_id
        save_node = settings.comfyui_video_save_node_id

        for node_id in (img_node, pos_node, neg_node, sampler_node, save_node):
            if node_id not in workflow:
                raise ComfyUIError(
                    f"Configured node id {node_id!r} not found in the video "
                    f"workflow template. Available node ids: {list(workflow.keys())}"
                )

        workflow[img_node]["inputs"]["image"] = image_filename
        workflow[pos_node]["inputs"]["prompt"] = positive_prompt
        workflow[neg_node]["inputs"]["prompt"] = (
            negative_prompt or settings.comfyui_video_default_negative_prompt
        )

        sampler_inputs = workflow[sampler_node]["inputs"]
        sampler_inputs["seed"] = seed
        sampler_inputs["num_frames"] = settings.comfyui_video_num_frames
        sampler_inputs["steps"] = settings.comfyui_video_steps
        sampler_inputs["cfg"] = settings.comfyui_video_cfg
        sampler_inputs["scheduler"] = settings.comfyui_video_scheduler
        sampler_inputs["denoise_strength"] = settings.comfyui_video_denoise_strength

        save_inputs = workflow[save_node]["inputs"]
        save_inputs["filename_prefix"] = filename_prefix
        save_inputs["save_output"] = True

        return workflow

    @staticmethod
    def extract_video_outputs(history_entry: dict) -> list[dict]:
        """
        Deliberately generic, unlike the base class's extract_images: the
        video save node's output key isn't "images" (confirmed via
        hand-testing) and isn't guaranteed stable across custom node
        packages. Scans every node's outputs for anything shaped like a
        file reference instead of assuming a key name - same approach the
        hand-test script used successfully.
        """
        found = []
        for node_id, node_output in history_entry.get("outputs", {}).items():
            for key, value in node_output.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "filename" in item:
                            found.append({
                                "node_id": node_id,
                                "output_key": key,
                                "filename": item["filename"],
                                "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output"),
                            })
        return found

    def wait_for_completion(self, prompt_id: str, timeout_seconds: float | None = None) -> dict:
        """
        Overrides the base class's image-specific "images" key check with
        a generic "any outputs at all" check - see module docstring point
        1. Also uses a slower poll cadence and video's own (much longer)
        timeout default, rather than image's, since a multi-minute video
        job doesn't need /history hit every second.
        """
        timeout_seconds = timeout_seconds or settings.comfyui_video_generation_timeout_seconds
        deadline = time.monotonic() + timeout_seconds
        poll_interval = 2.0
        self.last_cancel = None

        while time.monotonic() < deadline:
            try:
                r = requests.get(f"{self.server}/history/{prompt_id}", timeout=20)
                r.raise_for_status()
                history = r.json()
            except requests.RequestException:
                # ComfyUI can't service ANY request - not just uploads,
                # this polling GET too - while synchronously blocked on a
                # GPU sampling step. A failed poll does NOT mean the job
                # failed; it means we asked at a bad moment. Retry the
                # poll rather than giving up - only an explicit
                # status_str == "error" or exceeding the full deadline
                # below should end this loop. Getting this wrong is what
                # killed two genuinely-still-running jobs last run (see
                # the interrupt() removal below).
                time.sleep(poll_interval)
                continue

            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})

                if status.get("status_str") == "error":
                    raise ComfyUIError(f"ComfyUI reported an error for prompt {prompt_id}: {status}")

                outputs = entry.get("outputs", {})
                if outputs:
                    return entry

                raise ComfyUIError(f"Prompt {prompt_id} finished with no outputs. Status: {status}")

            time.sleep(poll_interval)

        # Interrupt BEFORE raising - a timeout here means the job is still
        # running on the GPU with nothing else telling it to stop. Without
        # this, the next attempt (retry, or the next theme entirely) tries
        # to talk to a server that's still synchronously blocked on this
        # abandoned job and gets a connection reset on its very first
        # request, not a clean failure - see upload_image's docstring for
        # exactly what that looked like in practice.
        # (cancel_job() is that same interrupt, but aimed: it checks /queue first, so a job
        # that already finished is left alone and a still-pending one is dequeued instead
        # of interrupting some other running job; it falls back to the blanket interrupt
        # when /queue cannot be read.)
        self.last_cancel = self.cancel_job(prompt_id)
        raise ComfyUIError(f"Timed out after {timeout_seconds}s waiting for prompt {prompt_id}")