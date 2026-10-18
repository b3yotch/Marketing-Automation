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
the client wrapper (`app/core/observability.py`), how the pipeline and the
two scripts use it, the data model that ends up in Langfuse, and a
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

## The client wrapper: `app/core/observability.py`

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
| `start_trace(url, run_label)` | `run_product` | Creates the root span `product:<run_label>`; returns a `TraceHandle` or `None` |
| `trace_id(trace)` | `run_product` | The 32-hex trace id, stored on `ProductRunResult` |
| `begin_stage(trace, stage)` | `run_product` | Opens a child span for a stage that is about to genuinely run; remembers it in `trace.open_spans[stage]` |
| `log_stage(trace, stage, ok, ...)` | `run_product`, `_fail` | Closes the stage's span (or creates-and-closes one for a checkpoint hit); sets output, level, metadata |
| `score_trace(trace, name, value)` | `finish_trace` | Scores the live trace via the root span |
| `score_trace_by_id(id, name, value)` | `review.py` | Scores a trace from a *different process* using only its id |
| `finish_trace(trace, ...)` | `_finish`, `_fail` | Closes dangling spans, writes the root's output, adds summary scores, ends the root |
| `flush()` | `_finish`, `_fail`, `review.py` | Forces the background exporter to send what it has |

### Configuration

| Variable | Purpose |
|---|---|
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | The key pair (from `langfuse.env`'s init values). Tracing is off if either is missing. |
| `LANGFUSE_BASE_URL` | Server URL, default `http://localhost:3000`. The older `LANGFUSE_HOST` name is still accepted. |

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
| Metadata | `{"run_label": ...}` |
| Root input | `{"url": <product url>}` |
| Root output | `{"status": "ok" \| "failed", "stage_reached": ...}` |

### Spans (one per stage)

Names: `research`, `creative`, `prompts`, `images`, `selection`, `videos`.

| Field | Value |
|---|---|
| Duration | Real wall-clock for a stage that ran; ~0 for a checkpoint hit |
| Metadata | `cached` (bool), `elapsed_seconds` (the stage's compute time - for a cache hit, the *original* time from the checkpoint's metadata) |
| Output | `{"ok": bool, ...extras}` |
| Extras | `model_used` (creative, prompts); `total_images`, `themes` (images); `videos_generated` (videos) |
| Level | `DEFAULT` on success, `ERROR` on failure, with the error text as `status_message` |

### Scores

| Score | Written by | Meaning |
|---|---|---|
| `completed` | `finish_trace` | 1 if the product finished the requested stages, else 0 |
| `themes_below_threshold` | `finish_trace` | Count of themes Agent 5 selected below its confidence threshold (written only when > 0) |
| `human_approval` | `review.py` | 1 approved / 0 rejected, with the reviewer's reason as the comment |

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
trace = obs.start_trace(url, run_label)      # defined before the try block
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

- **Stage-level granularity only.** The spans mark the six stage
  boundaries. Inside a stage - Agent 2's model-fallback chain, a
  retry's `previous_error`, Agent 4/6's per-retry seed draws, individual
  LLM calls with prompts, token counts and cost - nothing is recorded.
  Langfuse's LangChain/LangGraph callback integration is the likely route,
  but note the interaction with a deliberate choice above: the root span
  is *not* made the ambient context, so callback-created traces would
  appear as separate traces rather than nested under a stage span, unless
  trace context is passed explicitly. Not attempted.
- **A product's batch and video traces aren't linked.** Grouping them
  under a shared session id is the natural fix; not implemented.
- **An interrupted run loses that product's trace.** `KeyboardInterrupt`
  is caught outside `run_product`, so `finish_trace` doesn't run for the
  product in flight, and an unended span is never exported. Completed
  products are unaffected (flushed per product).
- **No cost tracking.** Groq usage, Tavily credits and GPU-hours are still
  tracked, if at all, locally in each agent.
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
| `app/core/observability.py` | Langfuse v5 wrapper: `TraceHandle`, stage spans, scores, `startup_check`, no-op behavior |
| `app/core/pipeline.py` | `run_product()` hooks: trace per product, span per stage, `trace_id` on the result |
| `app/core/manifest.py` | `trace_id` column in the results manifest; `read_manifest_rows()` |
| `scripts/review.py` | HITL review: shows and opens selected images, writes approvals, scores the trace |
| `scripts/generate_videos.py` | Approvals gate on `--from-results` without `--rows` |
| `scripts/run_batch.py` | Prints the tracing status line; passes `run_label="batch"` |
| `docker-compose.langfuse.yml` | Official Langfuse v4 stack (Postgres host port removed) |
| `langfuse.env` | First-boot org/project/user and API key init values |