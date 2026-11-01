# Observability: Langfuse Tracing + Human-in-the-Loop Review

## Purpose

Two concerns that share one mechanism, added after all six agents and the
bulk-processing layer were built:

1. **Tracing** - one trace per product per command run, one span per
   pipeline stage, recorded in a self-hosted Langfuse instance. Answers
   "what did the pipeline do for this product, how long did each stage
   take, and where did it fail" without re-reading terminal scrollback
   from an 8-hour batch.
2. **Human-in-the-loop (HITL) review** - a gate between Agent 5 (Image
   Selection) and Agent 6 (Video Generation) where a person looks at what
   Agent 5 picked and approves or rejects each product before ~33 GPU-
   minutes are spent on it. The approve/reject decision is recorded as a
   score on the *same trace* that recorded what the pipeline did, which is
   what turns a one-off approval log into the start of an eval dataset.

Both are **optional and best-effort**: the pipeline runs identically with
tracing completely off, and a down or misconfigured Langfuse can cost a
trace but never a product.

This document covers the server stack (what the six Docker containers do),
the client wrapper (`observability/observability.py`), how the pipeline and the
two scripts use it, the data model that ends up in Langfuse (including how
the local ComfyUI image/video models' *usage* is defined and recorded), and a
candid list of what is *not* observed.

## Architecture at a glance

```
                  scripts/run_batch.py          scripts/generate_videos.py
                          |                               |
                          +---------------+---------------+
                                          |
                                  run_product()            app/core/pipeline.py
                                          |
          obs.start_trace(url, label) --> root span  "product:batch" | "product:video"
                                          |
          per stage:  obs.begin_stage --> [agent runs] --> obs.log_stage     (child span)
                                          |
          obs.finish_trace  --> root output + scores ("completed", ...) --> root.end()
          obs.flush()
                                          |
                                          v
                        results manifest CSV  (trace_id column)       app/core/manifest.py
                                          |
                                          v
                              scripts/review.py
                 loads selection checkpoint -> shows/opens images -> human decides
                          |                                   |
                          v                                   v
                 approvals CSV                  obs.score_trace_by_id(trace_id,
                          |                                   "human_approval", 1|0)
                          v
                scripts/generate_videos.py --approvals   (only approved URLs run)


   SDK background exporter (OTLP/HTTP)
        |
        v
   langfuse-web :3000 --> MinIO (raw event persisted first) --> Redis (queue reference)
                                                                  |
                                      langfuse-worker <-----------+
                                            |
                                            v
                                       ClickHouse   (traces, spans, scores)

   Postgres: Langfuse's own transactional state (users, orgs, projects, API keys)
```

Two independent layers:

- **Infrastructure** (`docker-compose.langfuse.yml`, `langfuse.env`) - the
  Langfuse server. Nothing in the pipeline's code knows about its
  internals; it only needs a URL and a key pair.
- **Client** (`observability.py` + the hooks in `pipeline.py`,
  `review.py`) - everything this project owns.

## The server stack

`docker-compose.langfuse.yml` is the official self-host compose file from
the Langfuse repository, with one local change (see Challenge 5). It runs
six containers:

| Container | Role | Published port |
|---|---|---|
| `langfuse-web` | Serves the UI and the ingestion API | 3000 |
| `langfuse-worker` | Pulls queued events out of blob storage and writes them to ClickHouse | 3030 (localhost only) |
| `clickhouse` | Analytical (OLAP) store for traces, observations/spans and scores | 8123, 9000 (localhost only) |
| `postgres` | Transactional store for Langfuse's own state: users, organizations, projects, API keys | none (see below) |
| `redis` | Ingestion queue plus caches (API keys, prompts) | 6379 (localhost only) |
| `minio` | S3-compatible blob store: every incoming event is persisted here first | 9090 (S3 API), 9091 (console, localhost only) |

### How an event travels

1. The SDK's background exporter sends spans to `langfuse-web` over
   OTLP/HTTP (the SDK depends on the OpenTelemetry OTLP HTTP exporter).
2. `langfuse-web` persists the raw event to MinIO first, so an event can
   be recovered if later processing fails, and puts only a *reference* on
   the Redis queue.
3. `langfuse-worker` takes the reference off the queue, reads the event
   from MinIO, and ingests it into ClickHouse.
4. The UI reads traces and scores from ClickHouse, and accounts/projects/
   keys from Postgres. Redis additionally caches API keys so that
   authenticating each ingestion request doesn't hit Postgres.

### Why Postgres is not "where the logs go"

A common assumption, and a wrong one. Trace data is high-volume,
append-only, and queried with aggregations ("mean latency per stage",
"scores by tag"), which is why Langfuse moved it to ClickHouse and kept
Postgres for relational app state. The project's own pipeline spans,
stage timings and `human_approval` scores all live in ClickHouse. This is
also why an earlier, smaller compose file (a Postgres-only server) can't be
used any more - see Challenge 2.

### Initialisation and keys (`langfuse.env`)

`langfuse.env` sets the `LANGFUSE_INIT_*` variables so that Langfuse
creates the organization, project, user, **and the API key pair** on first
boot. There is no sign-up step and no third party involved: the key pair
is chosen in this file and copied into the project's `.env`. The init
values only take effect on an *empty* database; editing them later does
not rotate keys that already exist.

Run it with the env file passed explicitly, because plain `docker compose`
would otherwise read the project's own `.env`:

```bash
docker compose --env-file langfuse.env -f docker-compose.langfuse.yml up -d
```

Data persists in named Docker volumes. `docker compose down` keeps it;
`docker compose down -v` deletes it, along with every trace and score.

## The client wrapper: `observability/observability.py`

Written against the Langfuse Python SDK **v5** (`langfuse==5.0.2`), which
is built on OpenTelemetry. Two properties of that API shaped the module:

- **There is no trace object.** A trace is the tree under a root
  observation. `TraceHandle` wraps the root span and is what the pipeline
  carries around. The root is created with `start_observation()` rather
  than `start_as_current_observation()`, so it is *never* installed as the
  ambient OpenTelemetry context - sequential products can't nest under
  one another, and nothing else that is OTel-instrumented is affected.
- **Spans can't be backdated.** `start_observation()` has no start-time
  parameter, so a span's duration is the real wall-clock time between its
  creation and `.end()`. Stages therefore open a span *before* running and
  close it *after*.

### Design rules

- **No-op by default.** With no `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
  (or no `langfuse` package), every function returns immediately and
  `start_trace()` returns `None`. Call sites guard on that, so there is no
  stub class hierarchy.
- **Never raises.** Every function that touches the SDK catches its own
  exceptions and prints one `[observability]` line.
- **Loud at startup, quiet afterwards.** Best-effort tracing fails
  silently by nature (wrong key, server down, server too old), so
  `startup_check()` makes one authenticated call and the scripts print the
  result once, before any GPU time is spent.

### Public API

| Function | Called by | What it does |
|---|---|---|
| `enabled()` | internal | True only if the SDK imported and both keys are set |
| `get_client()` | internal | Lazily builds one shared `Langfuse` client; `None` if tracing is off or init fails |
| `startup_check()` | the three scripts | Returns one status line (see Verification) via `client.auth_check()` |
| `start_trace(url, run_label, session_id=None, product=None)` | `run_product` | Creates the root span `product:<run_label>`, grouped into a Langfuse session and tagged with the product slug; returns a `TraceHandle` or `None` |
| `trace_id(trace)` | `run_product` | The 32-hex trace id, stored on `ProductRunResult` |
| `begin_stage(trace, stage)` | `run_product` | Opens a child span for a stage that is about to genuinely run; remembers it in `trace.open_spans[stage]` |
| `log_stage(trace, stage, ok, ...)` | `run_product`, `_fail` | Closes the stage's span (or creates-and-closes one for a checkpoint hit); sets output, level, metadata |
| `score_trace(trace, name, value)` | `finish_trace` | Scores the live trace via the root span |
| `score_trace_by_id(id, name, value)` | `review.py` | Scores a trace from a *different process* using only its id |
| `finish_trace(trace, ..., images_kept=None)` | `_finish`, `_fail` | Closes dangling spans, writes the root's output, adds summary scores (incl. `images_kept`, `selection_surplus`, `gpu_s_per_kept_image`), ends the root |
| `abort_trace(trace, reason)` | `run_product` (on cancel) | Closes a cancelled run's open stage and root so Ctrl-C no longer loses the in-flight product's trace |
| `generation(...)`, `tool_span(...)` | `llm.py`, Tavily code | Child observations for LLM calls / external tool calls inside the running stage |
| `comfy_job(kind=, name=, model=, ...)` | Agent 4/6 `generate_node` | Context manager around ONE ComfyUI attempt; records it as a `generation` with GPU usage (see "ComfyUI job usage") |
| `reconcile_abandoned(fetch_entry)` | Agent 4/6 `generate_node` start, `finalize_node` | Fills in the real GPU time of jobs the client gave up on (timeout, lost connection) |
| `comfy_job_timing(entry)` | internal, tests | Pure parser for a ComfyUI `/history` entry: `gpu_ms`, `queue_wait_ms`, cached nodes, final event |
| `gpu_snapshot()` | internal | One rich `nvidia-smi` reading (utilisation, power, clocks, temperature, memory, throttle state) with driver-dependent field fallback |
| `ollama_resident_vram_mib(host)` | internal | Sync read of `/api/ps`: GPU memory Ollama holds right now |
| `flush()` | `_finish`, `_fail`, `review.py` | Forces the background exporter to send what it has |

### Configuration

| Variable | Purpose |
|---|---|
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | The key pair (from `langfuse.env`'s init values). Tracing is off if either is missing. |
| `LANGFUSE_BASE_URL` | Server URL, default `http://localhost:3000`. The older `LANGFUSE_HOST` name is still accepted. |
| `OBS_RESOURCE_SAMPLING` | `0` turns off all `nvidia-smi`/RAM sampling (stage-level and per-job). Default on. |
| `OBS_SAMPLE_INTERVAL_SECONDS` | Stage-level memory sampling interval, default 3. |
| `OBS_JOB_SAMPLE_INTERVAL_SECONDS` | Per-ComfyUI-job GPU sampling interval, default 2 (minimum 0.5). |
| `OBS_MAX_TEXT_CHARS` | Truncation limit for prompts/outputs stored on observations, default 15000. |

The module loads `.env` itself with `python-dotenv`. The project's other
settings come from `.env` through pydantic-settings, which fills the
`Settings` object but does **not** populate `os.environ` - and both this
module and the SDK read `os.environ`. The lookup uses the current
directory (`find_dotenv(usecwd=True)`), matching `Settings(env_file=".env")`;
variables genuinely set in the shell win over the file.

## What gets recorded

### Traces

One trace per product **per command invocation**.

| Field | Value |
|---|---|
| Name | `product:batch` (from `run_batch.py`) or `product:video` (from `generate_videos.py`) |
| Tags | `["batch"]` or `["video"]` |
| Metadata | `{"run_label": ..., "product": <url slug>}` |
| Session | The manifest's file stem (e.g. `batch_results_20261101_...`) - every product of one command invocation shares it, so a batch in Langfuse and its results CSV have the same name. `run_batch.py` and `generate_videos.py` pass it as `session_id=out_path.stem`. |
| Root input | `{"url": <product url>}` |
| Root output | `{"status": "ok" \| "failed", "stage_reached": ...}` |

### Spans (one per stage)

Names: `research`, `creative`, `prompts`, `images`, `selection`, `videos`.

| Field | Value |
|---|---|
| Duration | Real wall-clock for a stage that ran; ~0 for a checkpoint hit |
| Metadata | `cached` (bool), `elapsed_seconds` (the stage's compute time - for a cache hit, the *original* time from the checkpoint's metadata) |
| Output | `{"ok": bool, ...extras}` |
| Extras | `model_used` (creative: one model; prompts: comma-joined set of the per-theme models that succeeded); `total_images`, `themes` (images); `videos_generated` (videos) |
| `comfy` metadata | For `images`/`videos` stages that ran ComfyUI jobs in this process: the stage's usage summary (jobs, ok/timed-out/failed, GPU seconds, wasted GPU seconds, waste %, retries, energy, duty cycle, unreconciled jobs). Absent when the stage ran no jobs (e.g. a thread that was already complete), so zero jobs is not a failure. |
| Level | `DEFAULT` on success, `ERROR` on failure, with the error text as `status_message` |

### Scores

| Score | Written by | Meaning |
|---|---|---|
| `completed` | `finish_trace` | 1 if the product finished the requested stages, else 0 |
| `themes_below_threshold` | `finish_trace` | Count of themes Agent 5 selected below its confidence threshold (written only when > 0) |
| `human_approval` | `review.py` | 1 approved / 0 rejected, with the reviewer's reason as the comment |
| `gpu_s:<stage>`, `gpu_wasted_s:<stage>`, `waste_pct:<stage>` | `log_stage` (images, videos) | GPU seconds the stage spent in this run, how many went to attempts whose output was thrown away, and the percentage |
| `jobs_timed_out:<stage>`, `jobs_failed:<stage>`, `retries:<stage>` | `log_stage` | Attempt outcomes; retries = attempts minus distinct themes |
| `gpu_duty_pct:<stage>` | `log_stage` | Share of the stage's wall clock that ComfyUI was executing |
| `energy_wh:<stage>`, `gpu_throttled_pct:<stage>` | `log_stage` | Board energy integrated from `power.draw`; share of GPU time spent power/thermal throttled (job-weighted) |
| `images_requested`, `images_produced`, `videos_requested`, `videos_produced` | `log_stage` | Work requested vs produced across attempts |
| `images_kept`, `selection_surplus`, `gpu_s_per_kept_image` | `finish_trace` | Themes with a selected image; produced minus kept (by-design cost of choosing); GPU seconds per kept image (only when the images stage ran in this process) |

Scores exist only for numbers a dashboard cannot derive from the
observations themselves. Everything summable (GPU ms, images, energy) lives
in `usage_details` instead - see below.

### ComfyUI job usage (Agents 4 and 6)

Diffusion models have no tokens, so **usage is defined as GPU time plus work
units**. Every ComfyUI attempt (`generate_node` call) is one `generation`
observation under the running stage: `comfyui.image_batch`, `comfyui.video`,
plus `comfyui.abandoned_job` (below). It is a generation rather than a tool
span because only generations carry `usage_details`, which is what a
dashboard can sum.

`usage_details` (integers only - the SDK type is `Dict[str, int]`):

| Key | Meaning |
|---|---|
| `gpu_ms` | ComfyUI's own execution time: `execution_start` to `execution_success`/`_error`/`_interrupted` from `/history`, not client wall time. 0 when unknown (a timed-out job - see reconciliation). |
| `queue_wait_ms` | `execution_start` minus the entry's `extra_data.create_time` (both ComfyUI server timestamps, so no client clock is involved) |
| `wait_ms`, `save_ms`, `wall_ms` | Client-side phases: waiting for completion, downloading+saving, whole attempt |
| `images_requested`, `images_produced` | Image jobs: the latent batch size vs images actually saved |
| `videos_requested`, `videos_produced`, `frames` | Video jobs |
| `sampler_steps` | Steps of a job that completed |
| `energy_mwh` | Board energy over the attempt, integrated from `power.draw` samples (includes idle baseline) |
| `total` | Set to `gpu_ms` explicitly so the server does not invent a sum of unrelated keys |

`model` is the checkpoint (`juggernautXL_ragnarok.safetensors`, or the workflow's
CogVideoX model); `model_parameters` hold steps, cfg, sampler, scheduler,
width, height, batch size / frames, fps, precision, offload. Seed, theme index,
attempt, prompt id live in metadata (seed is for reproduction, not a chart dimension).

Per-job metadata:

- **Outcome**: `success`, `timeout`, `comfy_error`, `queue_error`, `no_output`,
  `upload_error`/`workflow_config_error` (video), `connection_error`,
  `save_error`, `count_mismatch` (job succeeded but produced the wrong
  number of images - counted as waste), `exception`. The `phase_at_end`
  (`queue`/`wait`/`save`) is what separates a lost ComfyUI connection from a
  local disk failure, because the clients' `requests` errors subclass `OSError`
  and `generate_node` reports both as "Failed saving ... locally". The error
  *string returned to the graph is unchanged*; only the trace label is corrected.
  Failed attempts are `level=ERROR`.
- **Throughput** (successful jobs): `sec_per_image`, `images_per_min`,
  `megapixels_per_s`, `sec_per_step`; video: `sec_per_frame`, `frames_per_min`.
  `warm` / `cached_nodes`: whether ComfyUI reused cached nodes (a cold job includes the checkpoint load).
- **GPU telemetry** from `nvidia-smi`, sampled every 2 s for the job
  (independent of ComfyUI's HTTP server, which cannot answer while sampling):
  `gpu_util_mean_pct`, `gpu_util_low_sample_pct`, `gpu_mem_util_mean_pct`,
  `gpu_power_mean_w`/`max_w`, `gpu_clock_sm_mean_mhz`, `gpu_clock_ratio`
  (mean SM clock / max), `gpu_temp_max_c`, `gpu_mem_start/peak/total_mib`,
  `gpu_throttle_reasons_pct`, `gpu_throttled_sample_pct`, and `gpu_throttle_ms`
  (cumulative microsecond counters differenced over the job, where the driver
  reports them). Fields a driver rejects or reports as `[N/A]` are simply
  absent: `gpu_snapshot()` probes three field sets (full, core, minimal) once
  and latches the first the installed `nvidia-smi` accepts.
  `nvidia-smi` utilisation only says a kernel was running, so it reads near
  100% for any job; it is useful for spotting stalls, not for judging speed.
  Power, clocks and throttle state are the better performance indicators on a
  laptop GPU.
- **Context that explains slow jobs**: `queue_running_at_start` /
  `queue_pending_at_start` (stale jobs from an earlier timeout), Ollama's
  resident VRAM at start, free system RAM, ComfyUI version and its own
  VRAM/RAM view at start and end. ComfyUI's `torch_vram_*` figures are
  recorded raw because this install uses `cudaMallocAsync`, where they are
  unlikely to reflect real usage (idle reading was 32 MiB); judge them from
  real traces before relying on them.
- **Video workflow facts** read from the exported workflow JSON by class type
  (not node id): resolution, fps, precision, quantization, offload, and
  `resize_keep_proportion`. When that is `false` the source image is
  stretched to the resize node's fixed width x height.

**Waste vs. selection surplus.** `gpu_wasted_s` counts only attempts whose
output was thrown away (timeouts, errors, count mismatches, and the late real
cost of abandoned jobs). Candidates that Agent 5 does not pick are *not*
waste: that is the by-design cost of choosing, reported as `selection_surplus`.
For video there is no post-generation judge (the review gate precedes it), so
"kept" means "produced".

**Abandoned jobs.** A client timeout does not stop the job in ComfyUI, so its
GPU cost is unknown when it times out. `reconcile_abandoned()` runs at the start
of each attempt and in each graph's `finalize_node`, looks the job up in
`/history`, and emits a `comfyui.abandoned_job` generation carrying the real
`gpu_ms` (spans cannot be backdated, so it is a new short observation). The
time is added to the stage's wasted GPU seconds. A job still running at stage
end stays in `jobs_unreconciled`. This also records, via
`produced_after_client_gave_up`, how often a timed-out job produced its images
anyway.

**Accumulation.** Per-stage totals are held in a `StageUsage` object carried in
a `ContextVar` next to the active span, never in LangGraph state, so nothing
here touches checkpoints or the msgpack allow-list. Usage covers the jobs that
ran *in this process*: a resumed stage reports only the remaining themes.

**Using it in Langfuse.** Chart `usage_details` types (`gpu_ms`, `energy_mwh`,
`images_produced`) by observation name or model, latency percentiles per
`comfyui.*` observation, and the scores above per trace. Group a batch with the
Sessions view. Whether the v4 server's dashboard builder offers custom usage
types as chartable measures has **not** been confirmed (see Verification).

### One product is more than one trace

`run_batch.py` and `generate_videos.py` each open their own trace, so a
product that goes through both has at least two traces, with different
ids. `review.py` reads the id from the **batch** manifest, so
`human_approval` attaches to the batch trace, not the video trace. The
products are joined by URL (the root input), not by trace. See "Deferred".

## How the pipeline uses it (`app/core/pipeline.py`)

`run_product(url, ..., run_label)` is the single place the six-stage
sequence runs, so it is the single place the hooks live.

```
trace = obs.start_trace(url, run_label, session_id=session_id, product=slug)   # before the try block
result.trace_id = obs.trace_id(trace)

for each stage:
    cached?  -> obs.log_stage(cached=True)               # span opened and closed at once
    else     -> obs.begin_stage(stage)                   # span opens
                ... agent runs ...
                success -> obs.log_stage(ok=True, ...)   # span closes with real duration
                failure -> _fail(...) -> obs.log_stage(ok=False, error=...)

_finish / _fail  ->  obs.finish_trace(...)  ->  obs.flush()
```

Details worth knowing:

- **`trace` is assigned before the `try`** so the outer `except` can still
  hand it to `_fail` even if something breaks before the first stage.
  `start_trace` never raises, so the assignment itself can't be the
  failure.
- **`_fail` closes the right span** because open spans are keyed by stage
  name inside the `TraceHandle`; the failure handler only needs the name.
- **`finish_trace` ends any span still open** (a stage that raised
  something `_fail` didn't name), so nothing is left unended and
  therefore unexported.
- **`themes_below_threshold` is correct on both paths** because
  `result.selection_flags` is computed after the cached/fresh branches
  merge, before `_finish` runs.
- **The `images` stage can't tell a fresh run from an already-complete
  thread** - `run_checkpointed` hides that - so its span's `cached` flag
  is always false; an already-complete thread simply shows a ~0s duration.
- **`flush()` runs once per product**, not once per batch, so a crash on
  product 30 doesn't lose products 1-29 sitting in an unsent buffer.
- **Ctrl-C now closes the trace.** Cancellation arrives inside `run_product`
  as `asyncio.CancelledError`, which `except Exception` does not catch. A
  dedicated handler calls `obs.abort_trace(trace, "interrupted")` and re-raises,
  so the interrupted stage and its finished ComfyUI jobs are exported (root
  `WARNING`, stage `ERROR`, `completed=0`). A hard kill of the process still
  loses the in-flight product, and the in-flight ComfyUI attempt's own span
  (it only ends when the worker thread returns).
- **`images_kept`** (themes with a non-empty `selected_local_path`) is set on
  `ProductRunResult` after Agent 5 and passed to `finish_trace` on both the
  success and failure paths.
- **`ALLOWED_MSGPACK_MODULES` now includes `ThemeVideoResult`.** The video
  state holds `ThemeVideoResult` objects directly (`completed_theme_results`,
  `_pending_result`), so they are serialized on their own; before this they
  came back from a checkpoint as plain dicts. Nested models (`GeneratedVideo`,
  `GeneratedImage`) are dumped inside their parent and need no entry. Verified
  against the installed serializer in the test sandbox only - see Verification.

## The manifest link (`app/core/manifest.py`)

The results manifest has a `trace_id` column, blank when tracing is off.
It is the only thing connecting the two processes: `run_batch.py` writes
it, and a later, separate `review.py` process reads it. Langfuse scores
by trace id, so the original Python objects never need to survive the gap
between a batch and its review - which could be hours or days.

## The HITL gate

### `scripts/review.py`

Walks every `status=ok` row of a batch manifest and asks for a decision.

1. **Loads Agent 5's checkpointed selection** for the URL
   (`load_stage(url, "selection", ImageSelectionOutput)`).
2. **Prints every theme** with its status and the path of the image it
   chose, plus any scalar score/rationale-like fields on the theme record.
   Those extras are found by field-*name pattern* (`score`, `rationale`,
   `reason`) rather than hard-coded names, so they survive schema renames.
3. **Opens images** in the default viewer (`os.startfile` on Windows,
   `open` on macOS, `xdg-open` elsewhere): by default only the themes
   Agent 5 selected **below its threshold** - the ones it wasn't sure
   about, where a human look is worth most. `--open-all` opens every
   selected image; `--no-open` opens none; `v` at the prompt opens all of
   the current product's images on demand.
4. **Prompts** `y` approve / `n` reject (with an optional reason) / `v`
   view all / `s` skip for now.
5. **Writes the decision immediately** to the approvals CSV, then calls
   `obs.score_trace_by_id(trace_id, "human_approval", 1|0, comment)`.
6. **Resumes.** Re-running on the same manifest keeps earlier approved/
   rejected decisions and asks only about unreviewed or skipped products.
   The approvals file is opened in append mode, and a product that was
   skipped and later decided has two rows - consumers treat the last row
   per URL as the decision.

If the selection checkpoint can't be loaded, the script says so loudly
rather than falling back to a bare prompt: reviewing without seeing
anything is exactly what the gate exists to prevent.

`--auto-approve-clean` skips the prompt for products with no flagged
themes. It weakens the gate, so it is opt-in.

### `scripts/generate_videos.py`

| Invocation | Gate applied |
|---|---|
| Positional URLs, or `--csv` | None - typing each URL is itself the human decision |
| `--from-results <manifest> --rows 3,7` | None - hand-picking rows is the human decision |
| `--from-results <manifest>` (no `--rows`) | **Refuses** unless `--approvals <file>` or `--force` is also given |
| Any of the above plus `--approvals <file>` | Rows not marked `approved` are dropped and listed as skipped |

The gate targets exactly one path: "animate everything that worked", the
one that could spend every GPU-hour in a batch on Agent 5's unaudited
judgment.

## End to end, for one product

1. `python scripts/run_batch.py urls.csv` - prints `Tracing: on -> ...`,
   then per product: a `product:batch` trace with up to five stage spans
   (research through selection), `completed` and possibly
   `themes_below_threshold` scores. The manifest row carries the trace id.
2. `python scripts/review.py outputs/batch_results_<ts>.csv` - flagged
   images open, the person decides, `human_approval` lands on the batch
   trace, the approvals CSV is written.
3. `python scripts/generate_videos.py --from-results <manifest>
   --approvals <approvals>` - only approved products run; each gets a
   `product:video` trace with its own `videos` span and its own trace id
   in the video manifest.
4. In Langfuse: filter by tag `batch` or `video`; sort traces by the
   `human_approval` or `themes_below_threshold` score; open a failed
   trace to see which stage's span carries the `ERROR` level and message.

## Verification status

**Verified**

- The wrapper's every SDK call (span creation, update, end, scoring by
  trace id, flush) against the real `langfuse==5.0.2` package, with an
  unreachable server: no exceptions, a valid 32-hex trace id, a dangling
  span closed by `finish_trace`, and a clean no-op when keys are absent.
- `.env`-only configuration enables tracing (after the cwd lookup fix).
- `startup_check()` against the live local server reported tracing on,
  which means the server was reachable and the keys were accepted.
- `review.py`, end to end in a stand-in project: flagged-only opening,
  `v`, `--open-all`, `--no-open`, a missing selection checkpoint, and
  resume with the last-row-wins approvals file.

**Verified for the ComfyUI usage work (sandbox: Linux, Python 3.12,
`langfuse==5.0.2`, `langgraph-checkpoint 4.3.1`; fake ComfyUI and fake
`nvidia-smi`, stubs for the project files that were not available)**

- `usage_details` with custom integer keys, `model_parameters`, metadata and
  `session_id` serialize as expected on the exported spans.
- `comfy_job_timing` against a real ComfyUI 0.28.0 `/history` entry captured on
  the target machine: 144366 ms for 3 images, queue wait 7 ms, nothing cached.
- **Differential tests**: the original and the instrumented `generate_node`
  (Agent 4 and Agent 6) return identical dicts across 11 scenarios (success,
  timeout, ComfyUI error, dropped connection, wrong image count, `/view` failure;
  video: upload failure, no outputs) in four modes (tracing off, on, Langfuse
  raising on every call, `nvidia-smi` failing). With tracing off the HTTP
  requests are identical (no extra traffic); with it on, only `/queue`,
  `/system_stats` and `/history/<id>` are added. Video `interrupt` counts match.
- Trace content per failure mode, the stage summary, the span tree
  (`product` -> `images` -> generations), session propagation, derived scores,
  reconciliation of a job that timed out and finished later, and the Ctrl-C path,
  through the real `run_product` with the real image graph.
- `nvidia-smi` field fallback (full / core / minimal / `[N/A]` / rejected) and
  value parsing, against a fake that mimics those cases. 40 tests pass.
- The allow-list: with the old list `ThemeVideoResult` came back as `dict`; with
  the new list it is a model instance.

**Not verified (needs your machine)**

- Real `nvidia-smi` output on the RTX 3050 Laptop GPU: which of the new fields
  report values rather than `[N/A]` (power draw and the throttle counters are
  the likely gaps), and the throttle bit values (from `nvml.h`, from memory).
- ComfyUI's `/history` for an *interrupted* or *errored* job: the parser expects
  `execution_interrupted`/`execution_error` messages with timestamps.
- That the v4 Langfuse UI lets you chart custom `usage_details` keys, and how it
  treats `total`.
- The allow-list on your installed `langgraph-checkpoint` version (the sandbox had
  4.3.1); `tests/obs/test_allowlist.py` prints the result in any environment.
- Python 3.14, Windows process/subprocess behaviour, and real timings.
- The stubs stand in for `aspect_ratio`, `utils`, the prompt/selection schemas and the
  other agents' graphs, so the tests prove the instrumentation, not those modules.

**Not yet confirmed**

- That the trace structure appears in the Langfuse UI as described above
  (name, tag, span tree, scores) after a real batch run. Tags and name
  are set through the SDK's attribute propagation, which is the part most
  worth eyeballing first.
- The full six-container stack under sustained load, and on the target
  machine's memory alongside Ollama and ComfyUI.
- `os.startfile` opening a real image viewer on Windows (the Linux code
  path was exercised with a stand-in).

## Troubleshooting

| Startup line | Meaning | Fix |
|---|---|---|
| `Tracing: off (... not set)` | Keys missing from the process environment | Add the three `LANGFUSE_*` lines to `.env`; run from the project root |
| `Tracing: off (langfuse package not installed)` | Package missing in this environment | `pip install langfuse==5.0.2` |
| `Tracing: OFF - could not reach Langfuse ...` | Server not up (or wrong URL) | Start Docker Desktop, then `docker compose ... up -d`; check `LANGFUSE_BASE_URL` |
| `Tracing: OFF - Langfuse rejected the keys` | Key pair doesn't match any project | Match `.env` to `langfuse.env`; init values only apply on an empty database |
| `Tracing: OFF - client failed to initialise` | SDK constructor error (message printed above it) | Read that message |

If the server is reachable but traces never appear, check the server
version: SDK v5 requires self-hosted Langfuse **4.59.0 or later**.

## Major challenges

### 1. The first wrapper was written against the wrong SDK

The first draft of `observability.py` used `client.trace()`,
`trace.span(start_time=...)` and `span.end()` - the Langfuse v2 API - because
that is what the code was written from. `pip show langfuse` reported
**5.0.2**, an OpenTelemetry-based SDK in which none of that exists. The
fix was to inspect the installed package's real signatures and rewrite,
then run the rewrite against the real SDK. Lesson: check the installed
version before writing against an API, and treat "written from memory"
as unverified until it has executed once.

### 2. SDK and server versions are coupled

SDK v5 requires a self-hosted server of 4.59.0 or later. The first
compose file pinned a minimal Postgres-only `langfuse:2` server, which
would have rejected the SDK's data. The replacement is the official
stack, which is why this document describes six containers rather than
two.

### 3. Spans can't be backdated

The first design recorded each stage *after* it finished, backdating a
span's start by the stage's elapsed time. v5 has no start-time parameter.
The design changed to open a span before the stage and close it after
(`begin_stage` / `log_stage`), which also gives true timestamps. Stages
served from a checkpoint never open a live span, so they get a
created-and-closed one with ~0 duration.

### 4. `.env` is not the process environment

Settings are read from `.env` by pydantic-settings into an object, not
into `os.environ`, so a wrapper reading `os.getenv` would have seen no
keys and reported tracing off. Loading `.env` with `python-dotenv` fixed
it - but `load_dotenv()` alone searches upward from the *module's*
location, not the current directory, which a test caught immediately.
`find_dotenv(usecwd=True)` matches how `Settings` resolves the file.

### 5. A reserved port on Windows blocked Postgres

`up -d` failed binding `127.0.0.1:5432` with "forbidden by its access
permissions" - the signature of a port inside a Windows-reserved range
(Hyper-V/WSL2), not of a port already in use. Postgres only needs to be
reachable by the other containers over the compose network, so the host
port mapping was removed. The file is otherwise the official one; the
change is documented in a comment next to it.

### 6. The review gate was blind, and reruns erased decisions

The first `review.py` showed only a flag string and a prompt, so a person
approved a product without ever seeing the image Agent 5 chose. It now
loads the selection and opens the flagged images. A second flaw sat next
to it: the approvals path is derived from the manifest's name, and the
writer truncated the file, so a second session would have deleted every
decision from the first. The file is now opened in append mode and earlier
decisions are honored.

### 7. Best-effort tracing fails silently

Because tracing must never break a run, a wrong key or a stopped server
produces no error at all - just a batch with no traces. `startup_check()`
exists to convert that silent failure into one visible line at the start
of every run.

## Deliberately deferred (not gaps, decisions)

- **What is still not recorded inside a stage.** LLM calls and Tavily
  searches are recorded as `generation`/`tool` observations, and ComfyUI jobs
  as above. Not recorded: per-node timing and live step progress inside a
  ComfyUI job (needs ComfyUI's websocket, which may stall like its HTTP server),
  and per-process GPU memory from the Windows `GPU Process Memory` performance
  counters (would show Ollama vs ComfyUI and spill into shared memory; needs a
  probe on your machine first).
- **Batch and video traces are grouped per command, not per product.**
  `session_id` is the manifest stem, so a batch and a later video run are two
  sessions and two traces per product, joined only by URL / `product` metadata.
- **Hard kills still lose the in-flight product's trace.** Ctrl-C is handled
  (see above); a killed process is not.
- **No monetary cost tracking.** GPU seconds and energy (Wh) are recorded for
  ComfyUI jobs; Groq spend and Tavily credits are not converted to money, and
  no price is attached to GPU time.
- **Timeout and cancellation behaviour of ComfyUI jobs is unchanged.** The
  image client still lets a timed-out job keep running (the video client
  interrupts it), and the 180 s image timeout is still shorter than a batch.
  This work only makes those visible (`jobs_timed_out`, `gpu_wasted_s`,
  `queue_running_at_start`, `comfyui.abandoned_job`). Fixing it - size the
  timeout to the batch from the recorded `gpu_ms`, and cancel a timed-out job
  properly (`POST /queue {"delete": [id]}` for pending, `/interrupt` for
  running) - is a separate step. Cancelling without raising the timeout would
  make every attempt fail with no output.
- **`ImageResizeKJ` stretches source frames.** The video workflow's resize
  node is fixed at 720x480 with `keep_proportion: false`; agent-4 images in
  other aspect ratios are distorted before animation. Now visible as
  `resize_keep_proportion=false` on each video job; not changed.
- **Datasets and experiments aren't used yet.** Scores are being
  collected, which is the prerequisite for the human-evaluation loop -
  a held-out product set scored against a rubric and re-run on every
  prompt or model change - but that loop itself is not built, and
  Agent 5's judgment quality remains unvalidated against human choices.
- **`--auto-approve-clean`'s threshold is Agent 5's own.** It trusts the
  same unaudited critic the gate exists to check; opt-in for that reason.
- **Per-product `flush()` costs a few seconds when the server is down**
  (the exporter retries before giving up). Negligible against minutes of
  generation per product.
- **Compose security defaults.** The official file ships placeholder
  passwords and publishes ports 3000 and 9090 on all interfaces, so they
  are reachable from the local network. Fine on a trusted home network;
  change the `# CHANGEME` secrets and restrict those ports otherwise.

## Design decisions worth remembering

- **Make the tracing layer incapable of failing a run.** Every SDK call is
  wrapped, every function no-ops without configuration, and a loud
  startup line replaces silent degradation. Observability that can take
  down the thing it observes is worse than none.
- **Persist a handle, not an object.** Storing the trace id in the
  manifest lets a separate process attach a human decision to the right
  trace days later. Anything that has to survive a process boundary
  should be reduced to an id that the other side can look up.
- **Put the gate on the path that needs it.** Hand-picked rows and typed
  URLs already are human decisions; only "everything that worked" is
  gated. A gate applied everywhere gets bypassed everywhere.
- **Show the reviewer the evidence.** An approval prompt without the
  image is a rubber stamp. Opening the flagged images makes the decision
  an actual look.
- **Check an API against the installed package, not memory.** Two of
  this layer's bugs (the SDK version and the server version) came from the
  same mistake.
- **Reuse the approach already in the project.** Flat, readable CSVs for
  the manifest and approvals, flushed per row, matching the checkpoint
  files' philosophy - with Langfuse added where the flat-file approach
  genuinely stops being enough (cross-run aggregation, scoring, a UI).

## Files

| File | Role |
|---|---|
| `observability/observability.py` | Langfuse v5 wrapper: `TraceHandle`, stage spans, scores, `startup_check`, no-op behavior |
| `app/core/pipeline.py` | `run_product()` hooks: trace per product (session + product slug), span per stage, `trace_id` and `images_kept` on the result, Ctrl-C handler, allow-list incl. `ThemeVideoResult` |
| `app/agents/image_generation/nodes.py`, `app/agents/video_generation/nodes.py` | `generate_node` wrapped in `obs.comfy_job`; outcome classification; `finalize_node` reconciles abandoned jobs |
| `app/agents/image_generation/comfyui_client.py` | Added read-only probes `queue_snapshot()`, `system_stats()`, `get_history_entry()` (best-effort, never raise; inherited by the video client). Existing methods unchanged. |
| `tests/obs/` | Fake ComfyUI, fake `nvidia-smi`, stub overlay, differential / trace-content / unit / integration / allow-list tests |
| `app/core/manifest.py` | `trace_id` column in the results manifest; `read_manifest_rows()` |
| `scripts/review.py` | HITL review: shows and opens selected images, writes approvals, scores the trace |
| `scripts/generate_videos.py` | Approvals gate on `--from-results` without `--rows` |
| `scripts/run_batch.py` | Prints the tracing status line; passes `run_label="batch"` and `session_id=<manifest stem>` |
| `docker-compose.langfuse.yml` | Official Langfuse v4 stack (Postgres host port removed) |
| `langfuse.env` | First-boot org/project/user and API key init values |