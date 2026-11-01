"""
Thin wrapper around ComfyUI's HTTP API, built directly on top of the
already-tested queue/poll flow. Two things added on top of the original
test script, both called out in review before this agent was written:

1. wait_for_completion now has a bound (generation_timeout_seconds). The
   original `while True: sleep(1)` had no exit if a job stalled or dropped
   from the queue silently - that's fatal for a retry loop, since a retry
   loop can only retry after something actually raises.
2. History is inspected for ComfyUI-reported errors, not just for the
   presence of an "images" key. A node failing mid-graph (bad checkpoint
   ref, OOM) still writes a history entry; the original script would have
   silently printed nothing rather than surfacing the failure.

Kept separate from nodes.py/graph.py (LangGraph-facing) so this stays a
plain, independently-testable HTTP client with no LangGraph dependency.
"""

from __future__ import annotations

import time
import uuid

import requests

from app.core.config import settings


class ComfyUIError(Exception):
    """Raised for any ComfyUI-side failure: bad checkpoint, node error, or timeout."""


class ComfyUIClient:
    def __init__(self, server: str | None = None):
        self.server = server or settings.comfyui_server
        self.client_id = str(uuid.uuid4())

    def check_checkpoint(self, checkpoint: str) -> None:
        r = requests.get(f"{self.server}/object_info/CheckpointLoaderSimple", timeout=10)
        r.raise_for_status()
        checkpoints = r.json()["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        if checkpoint not in checkpoints:
            raise ComfyUIError(
                f"Checkpoint {checkpoint!r} not found on ComfyUI server. "
                f"Available: {checkpoints}"
            )

    def build_image_workflow(
        self,
        *,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        batch_size: int,
        checkpoint: str | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        sampler: str | None = None,
        scheduler: str | None = None,
        filename_prefix: str = "agent4",
    ) -> dict:
        """Same graph shape as the original test script - checkpoint/sampler
        params now come from Settings (or an explicit override) instead of
        module-level constants, and batch_size/seed/dims are per-call rather
        than hardcoded."""
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint or settings.comfyui_checkpoint},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": positive_prompt, "clip": ["1", 1]},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["1", 1]},
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": batch_size},
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps or settings.comfyui_steps,
                    "cfg": cfg or settings.comfyui_cfg,
                    "sampler_name": sampler or settings.comfyui_sampler,
                    "scheduler": scheduler or settings.comfyui_scheduler,
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": filename_prefix, "images": ["6", 0]},
            },
        }

    def queue_prompt(self, workflow: dict) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        try:
            r = requests.post(f"{self.server}/prompt", json=payload, timeout=10)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ComfyUIError(f"Failed to queue prompt: {exc}") from exc

        body = r.json()
        if "error" in body:
            # ComfyUI can 200 with an embedded error (e.g. invalid node graph)
            # rather than a non-2xx status - checked explicitly rather than
            # assuming raise_for_status catches every failure mode.
            raise ComfyUIError(f"ComfyUI rejected the prompt: {body['error']}")
        return body["prompt_id"]

    def wait_for_completion(self, prompt_id: str, timeout_seconds: float | None = None) -> dict:
        """
        Polls /history until the job appears, an error surfaces, or
        timeout_seconds elapses - whichever comes first. Raises ComfyUIError
        on timeout or a reported node failure, rather than returning
        something the caller has to re-check for success.
        """
        timeout_seconds = timeout_seconds or settings.comfyui_generation_timeout_seconds
        deadline = time.monotonic() + timeout_seconds
        poll_interval = 1.0

        while time.monotonic() < deadline:
            r = requests.get(f"{self.server}/history/{prompt_id}", timeout=10)
            r.raise_for_status()
            history = r.json()

            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})

                if status.get("status_str") == "error":
                    raise ComfyUIError(f"ComfyUI reported an error for prompt {prompt_id}: {status}")

                outputs = entry.get("outputs", {})
                has_images = any("images" in node_out for node_out in outputs.values())
                if has_images:
                    return entry

                # Entry exists but no images and no explicit error status -
                # treat as a failure rather than looping forever on a
                # malformed/empty result.
                raise ComfyUIError(
                    f"Prompt {prompt_id} finished with no image output. Status: {status}"
                )

            time.sleep(poll_interval)

        raise ComfyUIError(f"Timed out after {timeout_seconds}s waiting for prompt {prompt_id}")

    def fetch_image_bytes(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """
        Pulls the raw image bytes back from ComfyUI via its /view endpoint.
        Needed because SaveImage only writes into ComfyUI's own output/ dir -
        this is what lets the agent copy that file into the project's own
        output structure instead of leaving every generated image stranded
        inside the ComfyUI install.
        """
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = requests.get(f"{self.server}/view", params=params, timeout=30)
        r.raise_for_status()
        return r.content

    @staticmethod
    def extract_images(history_entry: dict) -> list[dict]:
        """Returns a flat list of {filename, subfolder} dicts from a completed history entry."""
        images = []
        for node_output in history_entry.get("outputs", {}).values():
            for image in node_output.get("images", []):
                images.append({"filename": image["filename"], "subfolder": image.get("subfolder", "")})
        return images

    # ------------------------------------------------------------------
    # Read-only probes used by observability. Each one is best-effort:
    # short timeout, returns None on ANY failure, never raises - so a busy
    # or unreachable ComfyUI can cost a trace some detail, never a run.
    # They are only called when a job is actually being traced.
    # ------------------------------------------------------------------

    def queue_snapshot(self, timeout: float = 2.0) -> dict | None:
        """What ComfyUI has running/pending right now (GET /queue)."""
        try:
            r = requests.get(f"{self.server}/queue", timeout=timeout)
            r.raise_for_status()
            body = r.json()
            running = body.get("queue_running") or []
            pending = body.get("queue_pending") or []
            ids = [item[1] for item in running if isinstance(item, (list, tuple)) and len(item) > 1]
            return {"running": len(running), "pending": len(pending), "running_ids": ids}
        except Exception:
            return None

    def system_stats(self, timeout: float = 2.0) -> dict | None:
        """ComfyUI's own view of RAM/VRAM (GET /system_stats), in MiB."""
        try:
            r = requests.get(f"{self.server}/system_stats", timeout=timeout)
            r.raise_for_status()
            body = r.json()
            system = body.get("system") or {}
            device = (body.get("devices") or [{}])[0]

            def mib(value):
                return round(value / 1048576) if isinstance(value, (int, float)) else None

            return {
                "comfyui_version": system.get("comfyui_version"),
                "device": device.get("name"),
                "ram_free_mib": mib(system.get("ram_free")),
                "ram_total_mib": mib(system.get("ram_total")),
                "vram_free_mib": mib(device.get("vram_free")),
                "torch_vram_total_mib": mib(device.get("torch_vram_total")),
                "torch_vram_free_mib": mib(device.get("torch_vram_free")),
            }
        except Exception:
            return None

    def get_history_entry(self, prompt_id: str, timeout: float = 5.0) -> dict | None:
        """One job's /history entry, or None if absent (still queued or
        running) or unreachable."""
        try:
            r = requests.get(f"{self.server}/history/{prompt_id}", timeout=timeout)
            r.raise_for_status()
            return r.json().get(prompt_id)
        except Exception:
            return None