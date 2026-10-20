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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

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

    def __init__(self, span: Any, sampler: _Sampler | None):
        self.span = span
        self.sampler = sampler



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


def start_trace(url: str, run_label: str) -> Optional[TraceHandle]:
    """
    One trace per product per command invocation. `run_label` ("batch" |
    "video") becomes a tag and part of the trace name so the two commands'
    runs are filterable in the UI.

    `propagate_attributes` must wrap the root span's creation for the tags
    and name to land on the trace - it only affects spans created inside
    its context.
    """
    client = get_client()
    if client is None:
        return None
    try:
        name = f"product:{run_label}"
        with propagate_attributes(
            trace_name=name,
            tags=[run_label],
            metadata={"run_label": run_label},
        ):
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
        trace.open_spans[stage] = _OpenStage(span, sampler)
        _active_parent.set(span)
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

        span.update(
            output={"ok": ok, **(extra or {})},
            metadata={"cached": cached, "elapsed_seconds": round(float(elapsed_seconds), 2), **resources},
            level="DEFAULT" if ok else "ERROR",
            status_message=error,
        )
        span.end()
        if opened is not None:
            _active_parent.set(None)

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
        trace.root.end()
    except Exception as exc:
        print(f"[observability] failed to finalize trace: {exc}")
    finally:
        _active_parent.set(None)


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