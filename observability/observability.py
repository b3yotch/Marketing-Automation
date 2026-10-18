"""
Langfuse tracing for the bulk pipeline - one trace per product per command
invocation, one span per stage, a few trace-level scores, and the trace_id
written into the results manifest so a later, separate process
(scripts/review.py in this package) can attach a human-approval score to the right trace
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
- No credentials at all (neither `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
  set nor the `LANGFUSE_INIT_PROJECT_*` bootstrap keys in the `langfuse.env`
  next to this module), or the package isn't
  importable -> every function returns immediately and `start_trace()`
  returns None.
- Any SDK exception is caught and printed, never raised. A tracing bug or
  a down Langfuse server must cost you a trace, never a product.

The failure mode that best-effort creates is *silent* misconfiguration
(wrong key, server not up, server too old for this SDK - v5 needs
self-hosted Langfuse >= 4.59.0), which is why `startup_check()` exists:
call it once at the top of a run and it tells you out loud.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

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

# The self-hosted Langfuse stack is bootstrapped by `langfuse.env`, which now
# lives NEXT TO THIS MODULE: on first boot (empty database) the server creates
# an org/project whose API keys are exactly the LANGFUSE_INIT_PROJECT_* values
# chosen in that file. Those self-chosen keys ARE the SDK credentials, so read
# them straight out of the file as the default credentials. They are loaded
# via dotenv_values (NOT load_dotenv) so the LANGFUSE_INIT_* variables never
# leak into os.environ or into any other process's env resolution.
# Precedence: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY set explicitly in the
# shell or `.env` always win - that is the escape hatch once keys are rotated
# (or once the DB is no longer empty and the bootstrap values no longer apply).
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
            return f"Tracing: on -> {_base_url()}"
        return f"Tracing: OFF - Langfuse rejected the keys at {_base_url()}"
    except Exception as exc:
        return (
            f"Tracing: OFF - could not reach Langfuse at {_base_url()} "
            f"({type(exc).__name__}). Is the server up?"
        )


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
    """Opens a span for a stage that is about to actually run."""
    if trace is None:
        return
    try:
        trace.open_spans[stage] = trace.root.start_observation(
            name=stage, as_type="span", input=input_data or None
        )
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
    ended (its real duration is the stage's real cost); if not - a
    checkpoint hit, or a failure before the stage started - a span is
    created and ended immediately.

    `elapsed_seconds` is recorded in metadata regardless, because for a
    checkpoint hit it's the *original* compute time pulled from the
    checkpoint's own metadata, which is more useful in a dashboard than
    the ~0s the cache lookup took.
    """
    if trace is None:
        return
    try:
        span = trace.open_spans.pop(stage, None)
        if span is None:
            span = trace.root.start_observation(name=stage, as_type="span")
        span.update(
            output={"ok": ok, **(extra or {})},
            metadata={"cached": cached, "elapsed_seconds": round(float(elapsed_seconds), 2)},
            level="DEFAULT" if ok else "ERROR",
            status_message=error,
        )
        span.end()
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