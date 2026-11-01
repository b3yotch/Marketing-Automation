"""
Langfuse tracing for the bulk pipeline - one trace per product per command
invocation, one span per stage, a few trace-level scores, and the trace_id
written into the results manifest so a later, separate process
(scripts/review.py) can attach a human-approval score to the right trace
without needing the original Python objects.

Written against the Langfuse Python SDK v5 (`langfuse==5.0.2`), which is
OpenTelemetry-based. Two facts about that API shaped this module:

- **There is no `trace` object.** A trace is just the tree under a root
  observation, so `TraceHandle` below holds the root span and exposes the
  few operations the pipeline needs. The root span is created with
  `start_observation()` (NOT `start_as_current_observation()`), so it is
  never installed as the ambient OpenTelemetry context - a sequential
  batch loop never nests one product's spans under the previous one's, and
  nothing here can interfere with any other code that happens to be
  OTel-instrumented.
- **Spans cannot be backdated.** `start_observation()` has no start_time
  parameter, so a span's duration is real wall-clock between its creation
  and `.end()`. Stages therefore open a span *before* running
  (`begin_stage`) and close it after (`log_stage`). A stage served from a
  checkpoint never calls `begin_stage`, so its span is opened and closed
  back to back - near-zero duration, which is accurate for a cache hit.

Everything is best-effort and degrades to a no-op:
- No `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` set, or the package isn't
  importable -> every function returns immediately and `start_trace()`
  returns None.
- Any SDK exception is caught and printed, never raised. A tracing bug or
  a down Langfuse server must cost you a trace, never a product.

Beyond stage spans, this module also records what happens *inside* a stage:

- **Generations** (`generation()`): one observation per LLM call - model,
  prompt, output, token usage, finish reason, latency, errors. The call
  sites live deep inside LangGraph nodes that know nothing about the
  trace, so the active stage span travels in a `ContextVar` set by
  `begin_stage()`; a generation attaches itself to whatever stage is
  running. (Verified: the value is visible inside async nodes,
  `asyncio.gather` children and sync nodes run in worker threads.)
  With no active stage - an agent run standalone, or tracing off - a
  generation is a silent no-op, so no orphan traces are created.
- **Tool spans** (`tool_span()`): the same idea for non-LLM external calls
  such as Tavily searches.
- **Resource sampling**: while a stage runs, a background thread polls
  GPU memory (`nvidia-smi`) and, if `psutil` is installed, system RAM,
  keeping the peak. Peaks land in the stage span's metadata and as
  per-stage trace scores (`gpu_peak_mib:<stage>`) so they can be charted.
  `ollama_memory()` adds what Ollama itself reports for a loaded model.

The failure mode that best-effort creates is *silent* misconfiguration
(wrong key, server not up, server too old for this SDK - v5 needs
self-hosted Langfuse >= 4.59.0), which is why `startup_check()` exists:
call it once at the top of a run and it tells you out loud.
"""

from __future__ import annotations

import contextvars
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

try:
    from langfuse import Langfuse, propagate_attributes

    _SDK_AVAILABLE = True
except ImportError:  # package not installed -> tracing silently off
    _SDK_AVAILABLE = False

# The project's other settings come from `.env` via pydantic-settings, which
# populates the Settings object but NOT os.environ - and this module reads
# os.environ (that's where the Langfuse SDK itself looks, too). So load the
# same `.env` here. python-dotenv is already installed as a dependency of
# pydantic-settings. Variables genuinely set in the shell win over the file.
try:
    from dotenv import find_dotenv, load_dotenv

    # usecwd=True: resolve `.env` from the current directory, exactly like
    # Settings(env_file=".env") does. The default searches upward from this
    # module's own location, which can find a different file (or none).
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

# Fallback credentials: the self-hosted Langfuse stack is bootstrapped by
# `langfuse.env` NEXT TO THIS MODULE - on first boot (empty database) the
# server creates an org/project whose API keys are exactly the
# LANGFUSE_INIT_PROJECT_* values chosen in that file, so those self-chosen
# values ARE valid SDK credentials. They are read via dotenv_values (NOT
# load_dotenv) so the LANGFUSE_INIT_* variables never leak into os.environ
# or into any other process's env resolution. Precedence: explicit
# LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in the shell or `.env` always
# win - that is the escape hatch once keys are rotated (or once the DB is
# no longer empty and bootstrap values no longer apply).
try:
    from dotenv import dotenv_values

    _INIT_FILE = Path(__file__).resolve().parent / "langfuse.env"
    _INIT_DEFAULTS = {k: v for k, v in (dotenv_values(_INIT_FILE) or {}).items() if v}
except ImportError:  # python-dotenv missing -> no bootstrap defaults, env vars only
    _INIT_DEFAULTS = {}


def _credential(name_init: str, name_env: str) -> str | None:
    """Effective credential: explicit env var wins, bootstrap file is the default."""
    if os.getenv(name_env):
        return os.environ[name_env]
    return _INIT_DEFAULTS.get(name_init)


def _public_key() -> str | None:
    return _credential("LANGFUSE_INIT_PROJECT_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY")


def _secret_key() -> str | None:
    return _credential("LANGFUSE_INIT_PROJECT_SECRET_KEY", "LANGFUSE_SECRET_KEY")

DEFAULT_BASE_URL = "http://localhost:3000"

# The stage span currently running, so code that doesn't hold a TraceHandle
# (llm.py, market_research.py) can attach child observations to it.
_active_parent: contextvars.ContextVar[Any] = contextvars.ContextVar("obs_active_parent", default=None)


def _max_chars() -> int:
    try:
        return int(os.getenv("OBS_MAX_TEXT_CHARS", "15000"))
    except ValueError:
        return 15000


def _clip(value: Any) -> Any:
    """Truncates long strings (recursively) so one scraped page or prompt
    can't turn into a multi-hundred-KB trace attribute."""
    limit = _max_chars()
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"... [truncated {len(value) - limit} chars]"
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clip(v) for v in value]
    return value


def enabled() -> bool:
    return _SDK_AVAILABLE and bool(_public_key()) and bool(_secret_key())


def _base_url() -> str:
    # LANGFUSE_HOST is the pre-v4 name; the SDK still honors it but marks
    # it deprecated. Accept either so an older .env keeps working.
    return os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or DEFAULT_BASE_URL


_client: Optional["Langfuse"] = None


def get_client() -> Optional["Langfuse"]:
    """Lazily builds one shared client. None when tracing isn't configured."""
    global _client
    if not enabled():
        return None
    if _client is None:
        try:
            _client = Langfuse(
                public_key=_public_key(),
                secret_key=_secret_key(),
                base_url=_base_url(),
            )
        except Exception as exc:
            print(f"[observability] Langfuse client init failed, tracing disabled: {exc}")
            return None
    return _client


def startup_check() -> str:
    """
    One human-readable line about tracing state, printed by the scripts at
    startup. `auth_check()` makes a real authenticated call, so this
    catches a wrong key or an unreachable/incompatible server immediately
    instead of after an 8-hour batch that silently traced nothing.
    """
    if not _SDK_AVAILABLE:
        return "Tracing: off (langfuse package not installed)"
    if not enabled():
        return "Tracing: off (no LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY env vars and no LANGFUSE_INIT_PROJECT_* keys in langfuse.env)"
    client = get_client()
    if client is None:
        return "Tracing: OFF - client failed to initialise (see message above)"
    try:
        if client.auth_check():
            return f"Tracing: on -> {_base_url()}\n{resource_status()}"
        return f"Tracing: OFF - Langfuse rejected the keys at {_base_url()}"
    except Exception as exc:
        return (
            f"Tracing: OFF - could not reach Langfuse at {_base_url()} "
            f"({type(exc).__name__}). Is the server up?"
        )


# --------------------------------------------------------------------------
# Resource sampling (GPU / RAM)
# --------------------------------------------------------------------------

try:  # optional: system-wide RAM. GPU sampling needs only nvidia-smi.
    import psutil
except ImportError:
    psutil = None

_SMI_FALLBACKS = (
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)
_smi_path: Any = "unset"


def _find_smi() -> str | None:
    global _smi_path
    if _smi_path == "unset":
        found = shutil.which("nvidia-smi")
        if not found:
            found = next((p for p in _SMI_FALLBACKS if os.path.exists(p)), None)
        _smi_path = found
    return _smi_path


def _gpu_mib() -> tuple[int, int] | None:
    """(used, total) MiB for GPU 0, or None. Whole-GPU, not per-process:
    ComfyUI and Ollama share the card, and on Windows (WDDM) nvidia-smi
    generally can't attribute memory to individual processes anyway."""
    smi = _find_smi()
    if not smi:
        return None
    try:
        kwargs: dict = {}
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        out = subprocess.run(
            [smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, **kwargs,
        ).stdout
        used, total = (int(x.strip()) for x in out.strip().splitlines()[0].split(","))
        return used, total
    except Exception:
        return None


def _ram_used_mib() -> int | None:
    if psutil is None:
        return None
    try:
        return round(psutil.virtual_memory().used / 1048576)
    except Exception:
        return None


def _sampling_enabled() -> bool:
    return os.getenv("OBS_RESOURCE_SAMPLING", "1") != "0" and (_find_smi() is not None or psutil is not None)


def resource_status() -> str:
    if os.getenv("OBS_RESOURCE_SAMPLING", "1") == "0":
        return "Resource sampling: off (OBS_RESOURCE_SAMPLING=0)"
    gpu = "GPU on" if _find_smi() else "GPU off (nvidia-smi not found)"
    ram = "RAM on" if psutil is not None else "RAM off (pip install psutil)"
    return f"Resource sampling: {gpu}, {ram}"


class _Sampler:
    """Polls memory on a daemon thread while one stage runs, keeping the
    baseline and the peak. Whole-GPU/whole-system figures, sampled every
    few seconds - a short spike between samples can be missed, so treat
    the peak as a floor on the true peak."""

    def __init__(self, interval: float):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.gpu_start = self.gpu_peak = self.gpu_total = None
        self.ram_start = self.ram_peak = None

    def _sample(self) -> None:
        gpu = _gpu_mib()
        if gpu is not None:
            used, total = gpu
            self.gpu_total = total
            if self.gpu_start is None:
                self.gpu_start = used
            self.gpu_peak = used if self.gpu_peak is None else max(self.gpu_peak, used)
        ram = _ram_used_mib()
        if ram is not None:
            if self.ram_start is None:
                self.ram_start = ram
            self.ram_peak = ram if self.ram_peak is None else max(self.ram_peak, ram)

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="obs-resource-sampler")
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._sample()
        values = {
            "gpu_mem_start_mib": self.gpu_start,
            "gpu_mem_peak_mib": self.gpu_peak,
            "gpu_mem_total_mib": self.gpu_total,
            "ram_used_start_mib": self.ram_start,
            "ram_used_peak_mib": self.ram_peak,
        }
        return {k: v for k, v in values.items() if v is not None}


def _sample_interval() -> float:
    try:
        return max(0.5, float(os.getenv("OBS_SAMPLE_INTERVAL_SECONDS", "3")))
    except ValueError:
        return 3.0


class _OpenStage:
    """A stage span that is currently running, plus its resource sampler."""

    def __init__(self, span: Any, sampler: _Sampler | None, usage: Any = None):
        self.span = span
        self.sampler = sampler
        self.usage = usage  # StageUsage: ComfyUI jobs run inside this stage



class TraceHandle:
    """
    What the pipeline holds for one product: the root span, its trace id,
    and the stage spans currently open (keyed by stage name, so a failure
    handler that only knows the stage's *name* can still close the right
    span).
    """

    def __init__(self, root: Any):
        self.root = root
        self.trace_id: str = root.trace_id
        self.open_spans: dict[str, Any] = {}
        self.usage_by_stage: dict[str, dict] = {}


def start_trace(
    url: str,
    run_label: str,
    session_id: str | None = None,
    product: str | None = None,
) -> Optional[TraceHandle]:
    """
    One trace per product per command invocation. `run_label` ("batch" |
    "video") becomes a tag and part of the trace name so the two commands'
    runs are filterable in the UI.

    `session_id` groups every product of one command invocation into a
    Langfuse *session* (the scripts pass the manifest's file stem, so a
    batch in Langfuse and its results CSV share a name); `product` is the
    URL slug, stored as trace metadata so traces can be filtered per product.

    `propagate_attributes` must wrap the root span's creation for the tags
    and name to land on the trace - it only affects spans created inside
    its context.
    """
    client = get_client()
    if client is None:
        return None
    try:
        name = f"product:{run_label}"
        trace_meta = {"run_label": run_label}
        if product:
            trace_meta["product"] = product[:200]
        attrs: dict[str, Any] = {"trace_name": name, "tags": [run_label], "metadata": trace_meta}
        if session_id:
            attrs["session_id"] = session_id[:200]
        with propagate_attributes(**attrs):
            root = client.start_observation(name=name, as_type="span", input={"url": url})
        return TraceHandle(root)
    except Exception as exc:
        print(f"[observability] failed to start trace: {exc}")
        return None


def trace_id(trace: Optional[TraceHandle]) -> str | None:
    return trace.trace_id if trace is not None else None


def begin_stage(trace: Optional[TraceHandle], stage: str, **input_data: Any) -> None:
    """
    Opens a span for a stage that is about to actually run, makes it the
    active parent for generations/tool spans created inside the stage, and
    starts sampling GPU/RAM for its duration.
    """
    if trace is None:
        return
    try:
        span = trace.root.start_observation(name=stage, as_type="span", input=input_data or None)
        sampler = None
        if _sampling_enabled():
            sampler = _Sampler(_sample_interval())
            sampler.start()
        usage = StageUsage()
        trace.open_spans[stage] = _OpenStage(span, sampler, usage)
        _active_parent.set(span)
        _active_usage.set(usage)
    except Exception as exc:
        print(f"[observability] failed to begin stage '{stage}': {exc}")


def log_stage(
    trace: Optional[TraceHandle],
    stage: str,
    *,
    ok: bool,
    elapsed_seconds: float,
    cached: bool = False,
    error: str | None = None,
    extra: dict | None = None,
) -> None:
    """
    Closes a stage's span. If `begin_stage` opened one, that span is
    ended (its real duration is the stage's real cost) and its resource
    sampler is stopped; if not - a checkpoint hit, or a failure before the
    stage started - a span is created and ended immediately.

    `elapsed_seconds` is recorded in metadata regardless, because for a
    checkpoint hit it's the *original* compute time pulled from the
    checkpoint's own metadata, which is more useful in a dashboard than
    the ~0s the cache lookup took.

    Peak GPU/RAM for a live stage go into the span's metadata and are also
    written as trace-level scores named `gpu_peak_mib:<stage>` /
    `ram_peak_mib:<stage>`: metadata is readable per trace, scores are the
    thing Langfuse can aggregate and chart.
    """
    if trace is None:
        return
    try:
        opened = trace.open_spans.pop(stage, None)
        span = opened.span if opened is not None else trace.root.start_observation(name=stage, as_type="span")
        resources = opened.sampler.stop() if opened is not None and opened.sampler is not None else {}
        usage_summary: dict = {}
        if opened is not None and getattr(opened, "usage", None) is not None:
            try:
                usage_summary = opened.usage.summary(float(elapsed_seconds))
            except Exception as exc:
                print(f"[observability] failed to summarise usage for '{stage}': {exc}")
        has_jobs = bool(usage_summary.get("jobs") or usage_summary.get("jobs_unreconciled"))

        span.update(
            output={"ok": ok, **(extra or {})},
            metadata={
                "cached": cached,
                "elapsed_seconds": round(float(elapsed_seconds), 2),
                **resources,
                **({"comfy": usage_summary} if has_jobs else {}),
            },
            level="DEFAULT" if ok else "ERROR",
            status_message=error,
        )
        span.end()
        if opened is not None:
            _active_parent.set(None)
            _active_usage.set(None)

        if has_jobs:
            trace.usage_by_stage[stage] = usage_summary
            _write_usage_scores(trace, stage, usage_summary)
        if "gpu_mem_peak_mib" in resources:
            score_trace(trace, f"gpu_peak_mib:{stage}", resources["gpu_mem_peak_mib"])
        if "ram_used_peak_mib" in resources:
            score_trace(trace, f"ram_peak_mib:{stage}", resources["ram_used_peak_mib"])
    except Exception as exc:
        print(f"[observability] failed to log stage '{stage}': {exc}")


def score_trace(trace: Optional[TraceHandle], name: str, value: float, comment: str | None = None) -> None:
    if trace is None:
        return
    try:
        trace.root.score_trace(name=name, value=float(value), comment=comment)
    except Exception as exc:
        print(f"[observability] failed to score trace ({name}): {exc}")


def score_trace_by_id(trace_id_: str, name: str, value: float, comment: str | None = None) -> bool:
    """
    For a second process (scripts/review.py) scoring a trace it didn't
    create. Langfuse scores by id, so only the id persisted in the manifest
    is needed. Returns whether the call was attempted - scoring is
    fire-and-forget (queued and sent by the SDK's background exporter).
    """
    client = get_client()
    if client is None or not trace_id_:
        return False
    try:
        client.create_score(trace_id=trace_id_, name=name, value=float(value), comment=comment)
        return True
    except Exception as exc:
        print(f"[observability] failed to score trace {trace_id_} ({name}): {exc}")
        return False


def finish_trace(
    trace: Optional[TraceHandle],
    *,
    status: str,
    stage_reached: str,
    selection_flag_count: int = 0,
    images_kept: int | None = None,
) -> None:
    """
    Closes out a product's trace: final output on the root span, summary
    scores, then ends the root. Spans still open at this point (a stage
    that raised something `_fail` didn't name) are ended so nothing is left
    dangling and unexported.
    """
    if trace is None:
        return
    try:
        for stage in list(trace.open_spans):
            log_stage(trace, stage, ok=False, elapsed_seconds=0.0, error="stage did not complete")

        trace.root.update(
            output={"status": status, "stage_reached": stage_reached},
            level="DEFAULT" if status == "ok" else "ERROR",
        )
        score_trace(trace, "completed", 1 if status == "ok" else 0)
        if selection_flag_count:
            score_trace(trace, "themes_below_threshold", selection_flag_count)
        if images_kept is not None:
            # Kept = what Agent 5 selected. Produced minus kept is the
            # by-design cost of choosing (candidate surplus), reported
            # separately from failed-attempt waste (gpu_wasted_s:images).
            score_trace(trace, "images_kept", images_kept)
            gen = trace.usage_by_stage.get("images")
            if gen:
                if gen.get("images_produced") is not None:
                    score_trace(trace, "selection_surplus", max(0, gen["images_produced"] - images_kept))
                if images_kept > 0 and gen.get("gpu_s") is not None:
                    score_trace(trace, "gpu_s_per_kept_image", round(gen["gpu_s"] / images_kept, 1))
        trace.root.end()
    except Exception as exc:
        print(f"[observability] failed to finalize trace: {exc}")
    finally:
        _active_parent.set(None)
        _active_usage.set(None)


def abort_trace(trace: Optional[TraceHandle], reason: str = "interrupted") -> None:
    """
    Closes a trace whose run was cancelled (Ctrl-C arrives as
    asyncio.CancelledError inside run_product, which `except Exception`
    does not catch). Without this the root and open stage span are never
    ended, so a 33-minute video stage that is interrupted exports nothing
    for the stage itself. Never raises; the caller re-raises the cancel.
    """
    if trace is None:
        return
    try:
        for stage in list(trace.open_spans):
            log_stage(trace, stage, ok=False, elapsed_seconds=0.0, error=reason)
        trace.root.update(output={"status": reason}, level="WARNING", status_message=reason)
        score_trace(trace, "completed", 0)
        trace.root.end()
    except Exception as exc:
        print(f"[observability] failed to abort trace: {exc}")
    finally:
        _active_parent.set(None)
        _active_usage.set(None)
    flush()


def flush() -> None:
    """
    Called once per product (not once per batch): the SDK exports in a
    background thread, and flushing after each product means a crash on
    product 30 doesn't lose products 1-29 still sitting in an unsent
    buffer.
    """
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            print(f"[observability] flush failed: {exc}")


# --------------------------------------------------------------------------
# Observations inside a stage: LLM generations and tool calls
# --------------------------------------------------------------------------


class _Recorder:
    """
    Accumulates what is learned about one call (output, usage, metadata)
    and writes it to the span when the call ends - on success, or on any
    exception, which is what guarantees a span is never left open.
    Every method is a no-op when there was no active stage to attach to.
    """

    def __init__(self, span: Any, kind: str):
        self.span = span
        self.kind = kind
        self._output: Any = None
        self._usage: dict[str, int] = {}
        self._meta: dict[str, Any] = {}
        self._warning: str | None = None

    @property
    def active(self) -> bool:
        """True when this call is actually being traced - lets callers skip
        extra work (like an Ollama /api/ps lookup) when nothing will use it."""
        return self.span is not None

    def record(
        self,
        *,
        output: Any = None,
        usage: dict | None = None,
        metadata: dict | None = None,
        warning: str | None = None,
    ) -> None:
        if output is not None:
            self._output = output
        if usage:
            self._usage.update({k: int(v) for k, v in usage.items() if isinstance(v, (int, float))})
        if metadata:
            self._meta.update({k: v for k, v in metadata.items() if v is not None})
        if warning:
            self._warning = warning

    def close(self, error: str | None = None) -> None:
        if self.span is None:
            return
        try:
            kwargs: dict[str, Any] = {
                "output": _clip(self._output) if self._output is not None else None,
                "metadata": self._meta or None,
                "level": "ERROR" if error else ("WARNING" if self._warning else "DEFAULT"),
                "status_message": error or self._warning,
            }
            if self.kind == "generation" and self._usage:
                kwargs["usage_details"] = self._usage
            self.span.update(**kwargs)
            self.span.end()
        except Exception as exc:
            print(f"[observability] failed to close {self.kind} span: {exc}")
        finally:
            self.span = None


@contextmanager
def _observation(kind: str, **start_kwargs: Any) -> Iterator[_Recorder]:
    parent = _active_parent.get()
    span = None
    if parent is not None:
        try:
            span = parent.start_observation(as_type=kind, **start_kwargs)
        except Exception as exc:
            print(f"[observability] failed to start {kind} '{start_kwargs.get('name')}': {exc}")
    rec = _Recorder(span, kind)
    try:
        yield rec
    except BaseException as exc:
        rec.close(error=f"{type(exc).__name__}: {str(exc)[:500]}")
        raise
    else:
        rec.close()


def generation(
    *,
    name: str,
    model: str,
    provider: str,
    input: Any,
    model_parameters: dict | None = None,
    metadata: dict | None = None,
):
    """
    Context manager around ONE LLM call. Attaches to the currently running
    stage; a no-op when there isn't one. Use `rec.record(...)` as results
    become available - whatever was recorded before an exception is still
    written, so a call that returned but failed schema validation keeps its
    raw output and token usage on the errored span.

        with obs.generation(name=..., model=..., provider="groq", input=msgs) as rec:
            response = await client.chat.completions.create(...)
            rec.record(output=text, usage={"input": 812, "output": 190})
    """
    return _observation(
        "generation",
        name=name,
        model=model,
        input=_clip(input),
        model_parameters={k: v for k, v in (model_parameters or {}).items() if v is not None} or None,
        metadata={"provider": provider, **(metadata or {})},
    )


def tool_span(*, name: str, input: Any = None, metadata: dict | None = None):
    """Context manager around a non-LLM external call (e.g. a Tavily search).
    Safe to use from concurrent tasks: each call's parent is captured when
    it starts."""
    return _observation("tool", name=name, input=_clip(input), metadata=metadata or None)


async def ollama_memory(host: str, model: str, timeout: float = 2.0) -> dict:
    """
    What Ollama itself reports about memory, from `GET /api/ps`: the
    model's total size, how much of it is resident in GPU memory, and the
    context length in effect. `ollama_on_gpu_pct` below 100 means part of
    the model is running on CPU - the thing to watch on a small GPU that is
    also shared with ComfyUI. Best-effort: {} on any failure.
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(host.rstrip("/") + "/api/ps")
            resp.raise_for_status()
            loaded = resp.json().get("models", [])
    except Exception:
        return {}

    out: dict[str, Any] = {"ollama_loaded_models": [m.get("name") for m in loaded]}
    match = next((m for m in loaded if model in (m.get("name"), m.get("model"))), None)
    if match:
        size = match.get("size") or 0
        vram = match.get("size_vram") or 0
        out["ollama_model_mib"] = round(size / 1048576)
        out["ollama_vram_mib"] = round(vram / 1048576)
        if size:
            out["ollama_on_gpu_pct"] = round(100 * vram / size)
        if match.get("context_length"):
            out["ollama_context_length"] = match["context_length"]
    return out


# --------------------------------------------------------------------------
# GPU job telemetry: what one local diffusion job cost
#
# Diffusion models have no tokens, so "usage" is defined as GPU time plus
# work units, and every ComfyUI attempt becomes one `generation`
# observation under the running stage:
#
#   usage_details (integers only - the SDK type is Dict[str, int])
#       gpu_ms          ComfyUI's own execution time (execution_start ->
#                       execution_success/error/interrupted from /history),
#                       NOT client wall time. 0 when unknown (a timed-out
#                       job; see reconcile_abandoned()).
#       queue_wait_ms   execution_start - extra_data.create_time (both are
#                       ComfyUI server timestamps - no client clock needed)
#       wait_ms/save_ms/wall_ms   client-side phases of the attempt
#       images_requested / images_produced      (image jobs)
#       videos_requested / videos_produced / frames   (video jobs)
#       sampler_steps   steps of a job that completed
#       energy_mwh      board energy integrated from nvidia-smi power.draw
#       total           = gpu_ms (set explicitly so the server does not
#                       invent a meaningless sum of unrelated keys)
#   metadata            per-job detail: outcome, prompt_id, seed, queue
#                       state at start, throughput, GPU utilisation / power
#                       / clocks / throttling, VRAM, Ollama and RAM context.
#
# A per-stage accumulator (StageUsage) sums the attempts so log_stage() can
# write the few derived trace scores that a dashboard cannot compute from
# the observations alone (waste, duty cycle, retries).
#
# Same contract as everything above: best-effort, never raises, and with no
# active stage (tracing off, or an agent run standalone) it is a no-op that
# makes NO extra HTTP or subprocess calls.
# --------------------------------------------------------------------------

_SMI_CORE = (
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "clocks.max.sm",
    "temperature.gpu",
    "memory.used",
    "memory.total",
    "pstate",
    "clocks_throttle_reasons.active",
)
# Cumulative microseconds spent throttled; a delta over a job is exact where
# sampling the bitmask can miss short events. Newer drivers only.
_SMI_COUNTERS = (
    "clocks_event_reasons_counters.sw_power_cap",
    "clocks_event_reasons_counters.sw_thermal_slowdown",
    "clocks_event_reasons_counters.hw_thermal_slowdown",
    "clocks_event_reasons_counters.hw_power_brake_slowdown",
)
_SMI_MINIMAL = ("utilization.gpu", "utilization.memory", "memory.used", "memory.total")
_SMI_TIERS = (_SMI_CORE + _SMI_COUNTERS, _SMI_CORE, _SMI_MINIMAL)
_smi_tier: int | None = None
_smi_tier_lock = threading.Lock()

# nvml.h clocks-throttle-reason bits (gpu_idle=0x1 and display_clock=0x100
# are deliberately not counted as throttling).
_THROTTLE_BITS = {
    0x02: "app_clocks",
    0x04: "sw_power_cap",
    0x08: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
}
_THROTTLE_PENALTY = ("sw_power_cap", "hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake")
_THROTTLE_COUNTER_NAMES = {
    "clocks_event_reasons_counters.sw_power_cap": "sw_power_cap",
    "clocks_event_reasons_counters.sw_thermal_slowdown": "sw_thermal",
    "clocks_event_reasons_counters.hw_thermal_slowdown": "hw_thermal",
    "clocks_event_reasons_counters.hw_power_brake_slowdown": "hw_power_brake",
}


def _parse_smi_value(raw: str) -> Any:
    v = raw.strip()
    if not v or v.startswith("[") or v.upper() in ("N/A", "NOT SUPPORTED", "UNKNOWN ERROR"):
        return None
    if v.lower().startswith("0x"):
        try:
            return int(v, 16)
        except ValueError:
            return None
    if len(v) >= 2 and v[0] in "Pp" and v[1:].isdigit():
        return int(v[1:])
    try:
        return int(v) if v.lstrip("-").isdigit() else float(v)
    except ValueError:
        return None


def _smi_run(fields: tuple[str, ...]) -> tuple[int | None, dict | None]:
    """(returncode, row). returncode is None when the call itself failed
    (timeout, exec error) - which is NOT evidence the fields are invalid."""
    smi = _find_smi()
    if not smi:
        return None, None
    try:
        kwargs: dict = {}
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.run(
            [smi, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, **kwargs,
        )
        if proc.returncode != 0:
            return proc.returncode, None
        lines = proc.stdout.strip().splitlines()
        if not lines:
            return 0, None
        parts = lines[0].split(",")
        if len(parts) != len(fields):
            return 0, None
        return 0, {f: _parse_smi_value(p) for f, p in zip(fields, parts)}
    except Exception:
        return None, None


def gpu_snapshot() -> dict | None:
    """One richer nvidia-smi reading for GPU 0 (utilisation, power, clocks,
    temperature, memory, throttle state). Field support differs by driver,
    so the first call probes three field sets from richest to minimal and
    latches the first that the installed nvidia-smi accepts. Fields a laptop
    GPU does not report come back None. Whole-card figures - see _gpu_mib."""
    global _smi_tier
    try:
        if _smi_tier is None:
            with _smi_tier_lock:
                if _smi_tier is None:
                    rejected = 0
                    for i, fields in enumerate(_SMI_TIERS):
                        rc, row = _smi_run(fields)
                        if row is not None:
                            _smi_tier = i
                            return row
                        if rc not in (None, 0):
                            rejected += 1
                    if rejected == len(_SMI_TIERS):  # every set refused: stop trying
                        _smi_tier = len(_SMI_TIERS)
                    return None
        if _smi_tier >= len(_SMI_TIERS):
            return None
        return _smi_run(_SMI_TIERS[_smi_tier])[1]
    except Exception:
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _summarize_gpu_samples(samples: list[dict], failures: int) -> dict:
    """Pure function (unit-tested): sampled nvidia-smi rows -> job metadata."""
    out: dict[str, Any] = {"gpu_samples": len(samples), "gpu_sample_failures": failures}
    if not samples:
        return out

    def col(key: str) -> list[tuple[float, float]]:
        return [(s["_t"], s[key]) for s in samples if s.get(key) is not None]

    util = [v for _, v in col("utilization.gpu")]
    if util:
        out["gpu_util_mean_pct"] = round(_mean(util), 1)
        out["gpu_util_low_sample_pct"] = round(100 * sum(1 for v in util if v < 50) / len(util), 1)
    mem_util = [v for _, v in col("utilization.memory")]
    if mem_util:
        out["gpu_mem_util_mean_pct"] = round(_mean(mem_util), 1)

    power = col("power.draw")
    if power:
        watts = [v for _, v in power]
        out["gpu_power_mean_w"] = round(_mean(watts), 1)
        out["gpu_power_max_w"] = round(max(watts), 1)
        if len(power) >= 2:
            joules = sum(
                (power[i][1] + power[i + 1][1]) / 2 * (power[i + 1][0] - power[i][0])
                for i in range(len(power) - 1)
            )
            out["gpu_energy_mwh"] = int(round(joules / 3.6))  # J -> mWh

    sm = [v for _, v in col("clocks.sm")]
    if sm:
        out["gpu_clock_sm_mean_mhz"] = round(_mean(sm))
        max_sm = next((v for _, v in col("clocks.max.sm")), None)
        if max_sm:
            out["gpu_clock_sm_max_mhz"] = round(max_sm)
            out["gpu_clock_ratio"] = round(_mean(sm) / max_sm, 3)
    mem_clk = [v for _, v in col("clocks.mem")]
    if mem_clk:
        out["gpu_clock_mem_mean_mhz"] = round(_mean(mem_clk))
    temp = [v for _, v in col("temperature.gpu")]
    if temp:
        out["gpu_temp_max_c"] = round(max(temp))

    used = [v for _, v in col("memory.used")]
    if used:
        out["gpu_mem_start_mib"] = round(used[0])
        out["gpu_mem_peak_mib"] = round(max(used))
    total = next((v for _, v in col("memory.total")), None)
    if total:
        out["gpu_mem_total_mib"] = round(total)

    masks = [int(v) for _, v in col("clocks_throttle_reasons.active")]
    if masks:
        pct: dict[str, float] = {}
        for bit, name in _THROTTLE_BITS.items():
            p = round(100 * sum(1 for m in masks if m & bit) / len(masks), 1)
            if p:
                pct[name] = p
        out["gpu_throttle_reasons_pct"] = pct
        penalty_bits = [b for b, n in _THROTTLE_BITS.items() if n in _THROTTLE_PENALTY]
        out["gpu_throttled_sample_pct"] = round(
            100 * sum(1 for m in masks if any(m & b for b in penalty_bits)) / len(masks), 1
        )

    deltas: dict[str, int] = {}
    for key, name in _THROTTLE_COUNTER_NAMES.items():
        vals = [v for _, v in col(key)]
        if len(vals) >= 2 and vals[-1] >= vals[0]:
            deltas[name] = int(round((vals[-1] - vals[0]) / 1000))  # us -> ms
    if deltas:
        out["gpu_throttle_ms"] = deltas
    return out


def _gpu_sampling_enabled() -> bool:
    return os.getenv("OBS_RESOURCE_SAMPLING", "1") != "0" and _find_smi() is not None


def _job_sample_interval() -> float:
    try:
        return max(0.5, float(os.getenv("OBS_JOB_SAMPLE_INTERVAL_SECONDS", "2")))
    except ValueError:
        return 2.0


class _JobSampler:
    """Polls gpu_snapshot() on a daemon thread for the duration of ONE job.
    nvidia-smi is independent of ComfyUI's HTTP server, which is why this
    works while ComfyUI is busy sampling and cannot answer requests."""

    def __init__(self, interval: float):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []
        self.failures = 0

    def _sample(self) -> None:
        try:
            row = gpu_snapshot()
        except Exception:
            row = None
        if row is None:
            self.failures += 1
            return
        row["_t"] = time.monotonic()
        self.samples.append(row)

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="obs-job-sampler")
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._sample()  # closing reading: end-of-job counters/clocks
        return _summarize_gpu_samples(self.samples, self.failures)


def ollama_resident_vram_mib(host: str, timeout: float = 1.5) -> dict:
    """Sync twin of ollama_memory() for code running in worker threads: total
    GPU memory held by whatever Ollama has loaded right now. Context for a
    slow ComfyUI job on a card shared with Ollama. {} on any failure."""
    try:
        import json as _json
        import urllib.request

        with urllib.request.urlopen(host.rstrip("/") + "/api/ps", timeout=timeout) as resp:
            loaded = _json.loads(resp.read().decode("utf-8")).get("models", [])
        return {
            "ollama_loaded_models": [m.get("name") for m in loaded],
            "ollama_vram_mib_start": round(sum((m.get("size_vram") or 0) for m in loaded) / 1048576),
        }
    except Exception:
        return {}


def comfy_job_timing(entry: dict | None) -> dict:
    """
    Pure parser for one ComfyUI /history entry (unit-tested against a real
    entry from ComfyUI 0.28.0). Everything is milliseconds on the SERVER's
    clock, so no client clock is involved:

        queue_wait_ms = execution_start.timestamp - prompt[3].create_time
        gpu_ms        = <final event>.timestamp  - execution_start.timestamp

    where <final event> is execution_success, execution_error or
    execution_interrupted. Missing pieces are simply absent from the result.
    """
    out: dict[str, Any] = {}
    try:
        if not isinstance(entry, dict):
            return out
        prompt = entry.get("prompt")
        create = None
        if isinstance(prompt, (list, tuple)) and len(prompt) > 3 and isinstance(prompt[3], dict):
            create = prompt[3].get("create_time")
        start = end = None
        for msg in (entry.get("status") or {}).get("messages") or []:
            if not (isinstance(msg, (list, tuple)) and len(msg) == 2 and isinstance(msg[1], dict)):
                continue
            event, data = msg
            if event == "execution_start":
                start = data.get("timestamp")
            elif event == "execution_cached":
                out["cached_nodes"] = len(data.get("nodes") or [])
            elif event in ("execution_success", "execution_error", "execution_interrupted"):
                end = data.get("timestamp")
                out["final_event"] = event
        if start is not None and end is not None and end >= start:
            out["gpu_ms"] = int(end - start)
        if start is not None and create is not None and start >= create:
            out["queue_wait_ms"] = int(start - create)
        count = 0
        for node_out in (entry.get("outputs") or {}).values():
            if isinstance(node_out, dict):
                for value in node_out.values():
                    if isinstance(value, list):
                        count += sum(1 for item in value if isinstance(item, dict) and "filename" in item)
        out["outputs_count"] = count
    except Exception:
        pass
    return out


class StageUsage:
    """Thread-safe accumulator of the ComfyUI jobs one stage ran in this
    process. Lives on the stage (a ContextVar, like the active span) and
    never in LangGraph state, so nothing here touches checkpoints or the
    msgpack allow-list."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.jobs: list[dict] = []
        self.abandoned: list[dict] = []

    def add_job(self, rec: dict) -> None:
        with self._lock:
            self.jobs.append(rec)

    def add_abandoned(self, item: dict) -> None:
        with self._lock:
            self.abandoned.append(item)

    def pending_abandoned(self) -> list[dict]:
        with self._lock:
            return [a for a in self.abandoned if a.get("gpu_ms") is None]

    def resolve(self, prompt_id: str, **fields: Any) -> None:
        with self._lock:
            for a in self.abandoned:
                if a["prompt_id"] == prompt_id:
                    a.update(fields)

    def summary(self, elapsed_seconds: float | None = None) -> dict:
        with self._lock:
            jobs = list(self.jobs)
            abandoned = list(self.abandoned)
        resolved = [a for a in abandoned if a.get("gpu_ms") is not None]
        ab_ms = sum(a["gpu_ms"] for a in resolved)
        total_ms = sum(j["gpu_ms"] for j in jobs) + ab_ms
        wasted_ms = sum(j["gpu_ms"] for j in jobs if j["outcome"] != "success") + ab_ms
        out: dict[str, Any] = {
            "jobs": len(jobs),
            "jobs_ok": sum(1 for j in jobs if j["outcome"] == "success"),
            "jobs_timed_out": sum(1 for j in jobs if j["outcome"] == "timeout"),
            "jobs_failed": sum(1 for j in jobs if j["outcome"] not in ("success", "timeout")),
            "gpu_s": round(total_ms / 1000, 1),
            "gpu_wasted_s": round(wasted_ms / 1000, 1),
            "wall_s": round(sum(j["wall_ms"] for j in jobs) / 1000, 1),
            "jobs_unreconciled": sum(1 for a in abandoned if a.get("gpu_ms") is None),
        }
        if total_ms > 0:
            out["waste_pct"] = round(100 * wasted_ms / total_ms, 1)
        themes = {j["theme_index"] for j in jobs if j.get("theme_index") is not None}
        if themes:
            out["retries"] = len(jobs) - len(themes)
        energy = [j["energy_mwh"] for j in jobs if j.get("energy_mwh") is not None]
        if energy:
            out["energy_wh"] = round(sum(energy) / 1000, 2)
        behind = [j for j in jobs if j.get("queue_running_at_start") is not None]
        if behind:
            out["jobs_started_behind_running"] = sum(1 for j in behind if j["queue_running_at_start"] > 0)
        for unit in ("images", "videos"):
            rows = [j for j in jobs if j.get("unit") == unit]
            if rows:
                out[f"{unit}_requested"] = sum(j["requested"] for j in rows)
                out[f"{unit}_produced"] = sum(j["produced"] for j in rows)
        weighted = [(j["gpu_ms"], j["throttled_pct"]) for j in jobs if j.get("throttled_pct") is not None and j["gpu_ms"] > 0]
        if weighted:
            out["gpu_throttled_pct"] = round(sum(w * p for w, p in weighted) / sum(w for w, _ in weighted), 1)
        if elapsed_seconds and elapsed_seconds > 0 and total_ms > 0:
            out["gpu_duty_pct"] = round(100 * (total_ms / 1000) / elapsed_seconds, 1)
        return out


# The accumulator of the stage currently running (see begin_stage).
_active_usage: contextvars.ContextVar[Any] = contextvars.ContextVar("obs_active_usage", default=None)


def _clean_params(params: dict | None) -> dict | None:
    return {k: v for k, v in (params or {}).items() if v is not None} or None


class ComfyJob:
    """Handle yielded by comfy_job(). Every method is a no-op when the job
    is not being traced and none of them can raise."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.span: Any = None
        self.prompt_id: str | None = None
        self.outcome: str | None = None
        self.message: str | None = None
        self._phase = "setup"
        self._t0 = time.monotonic()
        self._t_submit: float | None = None
        self._t_done: float | None = None
        self._timing: dict = {}
        self._meta: dict[str, Any] = {}
        self._output: Any = None
        self._produced = 0
        self._probes: dict[str, Callable] = {}
        self._sampler: _JobSampler | None = None
        self._queue_running: int | None = None

    # -- lifecycle ----------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.span is not None

    @property
    def current_phase(self) -> str:
        return self._phase

    def _open(self, parent: Any) -> None:
        spec = self.spec
        try:
            self.span = parent.start_observation(
                as_type="generation",
                name=spec["name"],
                model=spec.get("model"),
                input=_clip(spec.get("input")),
                model_parameters=_clean_params(spec.get("model_parameters")),
                metadata={"provider": "comfyui", "kind": spec["kind"], **(spec.get("metadata") or {})},
            )
        except Exception as exc:
            self.span = None
            print(f"[observability] failed to start comfy job '{spec.get('name')}': {exc}")
            return
        try:
            if _gpu_sampling_enabled():
                self._sampler = _JobSampler(_job_sample_interval())
                self._sampler.start()
        except Exception:
            self._sampler = None

    # -- called from the node -----------------------------------------------
    def set_probes(self, **probes: Callable) -> None:
        """queue=, stats=, history= : zero-arg-safe callables from the
        ComfyUI client. Only ever invoked when the job is active."""
        self._probes.update(probes)

    def probe_start(self) -> None:
        """Context at job start: what is already queued/running in ComfyUI
        (stale jobs from a previous timeout), ComfyUI's own VRAM/RAM view,
        and how much GPU memory Ollama is holding."""
        if not self.active:
            return
        try:
            queue = self._probes.get("queue")
            snap = queue() if queue else None
            if snap:
                self._queue_running = snap.get("running")
                self._meta["queue_running_at_start"] = snap.get("running")
                self._meta["queue_pending_at_start"] = snap.get("pending")
            stats = self._probes.get("stats")
            st = stats() if stats else None
            if st:
                for key in ("comfyui_version", "device", "vram_free_mib", "ram_free_mib", "ram_total_mib"):
                    if st.get(key) is not None:
                        self._meta[f"comfy_{key}_start" if key.endswith("mib") else f"comfy_{key}"] = st[key]
            host = self.spec.get("ollama_host")
            if host:
                self._meta.update(ollama_resident_vram_mib(host))
            ram = _ram_available_mib()
            if ram is not None:
                self._meta["ram_available_mib_start"] = ram
        except Exception:
            pass

    def phase(self, name: str) -> None:
        self._phase = name

    def set_meta(self, **meta: Any) -> None:
        self._meta.update({k: v for k, v in meta.items() if v is not None})

    def submitted(self, prompt_id: str) -> None:
        self.prompt_id = prompt_id
        self._t_submit = time.monotonic()
        self._phase = "wait"

    def completed(self, history_entry: dict | None) -> None:
        self._t_done = time.monotonic()
        self._timing = comfy_job_timing(history_entry)
        self._phase = "save"

    def outputs(self, names: list[str], produced: int) -> None:
        self._output = {"files": names}
        self._produced = produced

    def fail(self, outcome: str, message: str | None = None, *, abandoned: bool = False) -> None:
        """Record why the attempt failed. `abandoned=True` means the client
        gave up while ComfyUI may still be running the job (timeout, lost
        connection): its true GPU time is filled in later by
        reconcile_abandoned()."""
        self.outcome = outcome
        self.message = (message or outcome)[:500]
        if abandoned and self.prompt_id:
            usage = _active_usage.get()
            if usage is not None:
                usage.add_abandoned({
                    "prompt_id": self.prompt_id,
                    "kind": self.spec["kind"],
                    "model": self.spec.get("model"),
                    "theme_index": (self.spec.get("metadata") or {}).get("theme_index"),
                    "attempt": (self.spec.get("metadata") or {}).get("attempt"),
                    "reason": outcome,
                    "gpu_ms": None,
                })

    def unusable(self, outcome: str) -> None:
        """The job finished but its output is not usable (e.g. wrong image
        count): counted as waste even though ComfyUI reported success."""
        if self.outcome is None:
            self.outcome = outcome
            self.message = outcome

    # -- closing ------------------------------------------------------------
    def _close(self, exc: BaseException | None) -> None:
        if self.span is None:
            return
        span, self.span = self.span, None
        try:
            t_end = time.monotonic()
            resources = self._sampler.stop() if self._sampler is not None else {}
            outcome = self.outcome or ("exception" if exc is not None else "success")

            timing = self._timing
            history = self._probes.get("history")
            if not timing.get("gpu_ms") and self.prompt_id and history and outcome in ("comfy_error", "no_output"):
                timing = comfy_job_timing(history(self.prompt_id))
            end_stats = None
            stats = self._probes.get("stats")
            if stats and outcome not in ("timeout", "connection_error"):
                end_stats = stats()  # skipped when ComfyUI is likely still busy

            spec = self.spec
            unit = spec.get("unit", "images")
            requested = int(spec.get("requested", 1))
            produced = int(self._produced)
            gpu_ms = int(timing.get("gpu_ms") or 0)
            wall_ms = int((t_end - self._t0) * 1000)
            if self._t_submit is not None:
                wait_ms = int(((self._t_done or t_end) - self._t_submit) * 1000)
            else:
                wait_ms = 0
            save_ms = int((t_end - self._t_done) * 1000) if self._t_done is not None else 0
            ok = outcome == "success"

            usage: dict[str, int] = {
                "gpu_ms": gpu_ms,
                "queue_wait_ms": int(timing.get("queue_wait_ms") or 0),
                "wait_ms": wait_ms,
                "save_ms": save_ms,
                "wall_ms": wall_ms,
                f"{unit}_requested": requested,
                f"{unit}_produced": produced,
                "total": gpu_ms,
            }
            if ok and spec.get("steps"):
                usage["sampler_steps"] = int(spec["steps"])
            if ok and spec.get("frames"):
                usage["frames"] = int(spec["frames"])
            energy = resources.get("gpu_energy_mwh")
            if energy is not None:
                usage["energy_mwh"] = int(energy)

            meta: dict[str, Any] = {
                "outcome": outcome,
                "phase_at_end": self._phase,
                "prompt_id": self.prompt_id,
                "cached_nodes": timing.get("cached_nodes"),
                "warm": (timing.get("cached_nodes") or 0) > 0 if "cached_nodes" in timing else None,
                "comfy_final_event": timing.get("final_event"),
                "comfy_outputs_count": timing.get("outputs_count"),
            }
            gpu_s = gpu_ms / 1000
            if ok and gpu_s > 0:
                if unit == "images" and produced:
                    meta["sec_per_image"] = round(gpu_s / produced, 2)
                    meta["images_per_min"] = round(produced / gpu_s * 60, 2)
                    if spec.get("width") and spec.get("height"):
                        meta["megapixels_per_s"] = round(spec["width"] * spec["height"] * produced / 1e6 / gpu_s, 4)
                if unit == "videos" and spec.get("frames"):
                    meta["sec_per_frame"] = round(gpu_s / spec["frames"], 2)
                    meta["frames_per_min"] = round(spec["frames"] / gpu_s * 60, 2)
                if spec.get("steps"):
                    meta["sec_per_step"] = round(gpu_s / spec["steps"], 2)
            if energy is not None and produced and ok:
                meta["energy_mwh_per_unit"] = round(energy / produced, 1)
            meta.update(self._meta)
            meta.update(resources)
            if end_stats:
                for key in ("vram_free_mib", "torch_vram_total_mib", "torch_vram_free_mib", "ram_free_mib"):
                    if end_stats.get(key) is not None:
                        meta[f"comfy_{key}_end"] = end_stats[key]
            # Whole-card used-VRAM figures from the sampler are the reliable
            # ones here; ComfyUI's own torch_vram_* are recorded raw so you
            # can judge whether they move on your allocator (cudaMallocAsync).

            error = None
            if exc is not None:
                error = f"{type(exc).__name__}: {str(exc)[:500]}"
            elif not ok:
                error = self.message or outcome

            rec = _Recorder(span, "generation")
            rec.record(output=self._output, usage=usage, metadata=meta)
            rec.close(error=error)

            acc = _active_usage.get()
            if acc is not None:
                acc.add_job({
                    "outcome": outcome,
                    "gpu_ms": gpu_ms,
                    "wall_ms": wall_ms,
                    "unit": unit,
                    "requested": requested,
                    "produced": produced,
                    "energy_mwh": energy,
                    "theme_index": (spec.get("metadata") or {}).get("theme_index"),
                    "queue_running_at_start": self._queue_running,
                    "throttled_pct": resources.get("gpu_throttled_sample_pct"),
                })
        except Exception as exc2:
            print(f"[observability] failed to close comfy job: {exc2}")
            try:
                span.end()
            except Exception:
                pass


@contextmanager
def comfy_job(**spec: Any) -> Iterator[ComfyJob]:
    """
    Context manager around ONE ComfyUI attempt (queue -> wait -> download).
    Required spec: kind ("image"|"video"), name, model. Optional: input,
    model_parameters, metadata (theme_index/attempt/seed...), unit
    ("images"|"videos"), requested, steps, frames, width, height,
    ollama_host.

    Attaches to the running stage like generation() does; with no active
    stage it yields an inactive job whose methods all do nothing. A node
    that handles its own exceptions must call job.fail(...) before leaving
    the block; an exception that escapes is recorded as outcome
    "exception" and re-raised unchanged.
    """
    job = ComfyJob(spec)
    parent = _active_parent.get()
    if parent is not None:
        job._open(parent)
    try:
        yield job
    except BaseException as exc:
        job._close(exc)
        raise
    else:
        job._close(None)


def reconcile_abandoned(fetch_entry: Callable[[str], dict | None]) -> int:
    """
    Fills in the real GPU time of jobs the client gave up on (timeouts, lost
    connections). Such a job keeps running in ComfyUI, so at the moment of
    the timeout its cost is unknown; later its /history entry says exactly
    how long it ran. For each pending job whose entry is now final this
    emits a `comfyui.abandoned_job` generation (spans cannot be backdated,
    so it is a new, short observation carrying the usage) and adds the time
    to the stage's wasted GPU seconds. Jobs still running stay pending and
    are reported as `jobs_unreconciled` at stage end.

    Call at the start of each attempt and when the stage's graph finishes.
    No-op without an active stage. Returns the number resolved.
    """
    usage = _active_usage.get()
    if usage is None:
        return 0
    resolved = 0
    try:
        for item in usage.pending_abandoned():
            entry = fetch_entry(item["prompt_id"])
            timing = comfy_job_timing(entry)
            if timing.get("gpu_ms") is None:
                continue
            usage.resolve(item["prompt_id"], gpu_ms=int(timing["gpu_ms"]), final_event=timing.get("final_event"))
            resolved += 1
            with _observation(
                "generation",
                name="comfyui.abandoned_job",
                model=item.get("model"),
                input={"prompt_id": item["prompt_id"]},
                metadata={
                    "provider": "comfyui",
                    "kind": item["kind"],
                    "theme_index": item.get("theme_index"),
                    "attempt": item.get("attempt"),
                    "abandoned_because": item.get("reason"),
                    "comfy_final_event": timing.get("final_event"),
                    "comfy_outputs_count": timing.get("outputs_count"),
                    "produced_after_client_gave_up": bool(timing.get("outputs_count")),
                },
            ) as rec:
                rec.record(usage={
                    "gpu_ms": int(timing["gpu_ms"]),
                    "queue_wait_ms": int(timing.get("queue_wait_ms") or 0),
                    "total": int(timing["gpu_ms"]),
                })
    except Exception as exc:
        print(f"[observability] reconcile failed: {exc}")
    return resolved


def _ram_available_mib() -> int | None:
    if psutil is None:
        return None
    try:
        return round(psutil.virtual_memory().available / 1048576)
    except Exception:
        return None


def _write_usage_scores(trace: "TraceHandle", stage: str, s: dict) -> None:
    """The derived numbers a dashboard cannot compute from observations."""
    if not (s.get("jobs") or s.get("jobs_unreconciled")):
        return
    pairs = {
        f"gpu_s:{stage}": s.get("gpu_s"),
        f"gpu_wasted_s:{stage}": s.get("gpu_wasted_s"),
        f"waste_pct:{stage}": s.get("waste_pct"),
        f"jobs_timed_out:{stage}": s.get("jobs_timed_out"),
        f"jobs_failed:{stage}": s.get("jobs_failed"),
        f"retries:{stage}": s.get("retries"),
        f"gpu_duty_pct:{stage}": s.get("gpu_duty_pct"),
        f"energy_wh:{stage}": s.get("energy_wh"),
        f"gpu_throttled_pct:{stage}": s.get("gpu_throttled_pct"),
        "images_requested": s.get("images_requested"),
        "images_produced": s.get("images_produced"),
        "videos_requested": s.get("videos_requested"),
        "videos_produced": s.get("videos_produced"),
    }
    for name, value in pairs.items():
        if value is not None:
            score_trace(trace, name, value)