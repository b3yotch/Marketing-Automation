# AI Product Creative Generation Workflow

A multi-agent LangGraph pipeline for ecommerce brands: takes a product page
URL and produces ad creative direction plus ComfyUI-ready generation
prompts, on the way to generating marketing images/videos automatically.

## Pipeline

```
[URL] --> Agent 1: Product Research --> Agent 2: Creative Strategy --> Agent 3: Prompt Generation --> Agent 4: Image Generation (ComfyUI) --> Agent 5: Image Selection (Critic) --> Agent 6: Video Generation (ComfyUI)
              (done)                          (done)                        (done)                              (done)                                        (done)                                              (done)
```

Video generation was originally planned as part of a combined Agent 4
("Image/Video Generation"), but landed as its own agent once actually
built - see `Video_generation.md`'s intro for why that split was made
explicit rather than drifting into whichever agent the code happened to
get written into. That agent temporarily claimed the number 5 (the number
Agent 4's own schema docstring had already reserved for a future critic),
since it was built before the critic was. Once Image Selection was
actually designed and built, the pipeline was renumbered so agent number
matches graph position again: **Image Selection is Agent 5, Video
Generation is Agent 6** - see `Image_selection.md` Challenge 4 and
`Video_generation.md` Challenge 8 for the full mechanics of that swap.

A video-*quality* critic (judging Agent 6's finished output, as opposed to
Agent 5 judging Agent 4's candidate frames) was discussed during Agent 5's
design and deliberately deferred until Agent 5 proved itself in a real
run. It has - see "Known gaps to revisit" below.

Plus a bulk processing layer (built) for running many product URLs at
once - see "Bulk processing" below. It's two commands, not one: a batch
runner that takes every URL through Agent 5 (Image Selection) and stops,
and a separate video command that spends Agent 6's ~33 minutes/product
only against whichever subset you pick after reviewing what Agent 5
selected.

Each agent is a self-contained LangGraph subgraph living in its own
`app/agents/<name>/` folder (`schema.py`, `prompts.py`, `state.py`,
`nodes.py`, `graph.py`), sharing common infrastructure in `app/core/` -
except Agent 6, which deliberately owns its own `comfyui_client.py` rather
than sharing Agent 4's (see `Video_generation.md`), and Agent 5, which has
no `prompts.py`/ComfyUI client at all - its only external call is a vision
LLM judgment, no generation. Design rationale and hard-won debugging
lessons for each agent are written up in detail in `Product_research.md`,
`Creative_strategy.md`, `Prompt_generation.md`, `Image_generation.md`,
`Image_selection.md`, and `Video_generation.md` - this README covers
setup, running, and configuration; those files cover *why* things are
built the way they are.

## Agent 1: Product Research

Takes a product page URL, returns structured product data (title, price,
features, specs, review synthesis, brand positioning).

```
scrape --> extract --> validate --+--> END (valid, or retries exhausted)
             ^                    |
             |                    v
             +------ bump_retry <-+ (invalid, retries remain)
```

1. **scrape** (`scraper.py`, deterministic, no LLM) - Playwright renders the
   page, then we pull out: schema.org `Product` JSON-LD if present, the OG
   image, cleaned visible text (chrome-stripped - cookie banners,
   mega-menus, breadcrumbs, newsletter blocks, not just semantic
   nav/footer tags), and any review-section text found via heuristic CSS
   selectors, escalating to a scroll pass only when there's an actual
   signal it's worth it (see `Product_research.md` Challenge 4).
   Navigation itself tries several `wait_until` strategies in order
   (fastest/most-reliable first), each capped by its own
   `nav_attempt_timeout_ms` rather than sharing the full `scrape_timeout_ms`
   ceiling - an earlier version let a single strategy (`networkidle`, which
   many real ecommerce pages never cleanly reach due to persistent
   analytics/chat-widget traffic) burn the entire scrape budget before
   falling through to a faster one.
2. **extract** (`nodes.py` + `llm.py`) - qwen3.5:4b via Ollama. Does **not**
   use Ollama's native `format=` schema constraint - qwen3.5 has stacking
   bugs there (empty output, or the constraint silently ignored - see
   `Product_research.md` Challenge 1) - so thinking is disabled, the schema
   is described via a **filled example instance** in the prompt (not the
   raw JSON Schema, which a smaller model can echo back verbatim instead of
   returning real data), the response streams with **early-stop once the
   JSON object closes** (cuts wasted trailing-commentary generation), and
   is parsed with a tolerant brace-matching extractor.
3. **validate** - low bar sanity check (non-empty title). Deeper quality
   judgment belongs to Agent 5 later in the pipeline, for image output at
   least - see `Image_selection.md`.
4. **bump_retry** - increments the retry counter and loops back to
   `extract` if validation failed and retries remain
   (`max_extraction_retries`, default 2).

A 4B local model is viable here because this agent's job is *extraction*,
not *recall* - it's structuring text already in its context window, not
answering from training knowledge. The one genuinely interpretive field is
`brand_positioning`. The same local model (`qwen3.5:4b`) later turned out
to be natively multimodal too, which is what Agent 5 ended up running its
vision judgment on - see `Image_selection.md` Challenge 7.

## Agent 2: Creative Strategy

Takes Agent 1's `ProductResearch` object, returns ad creative direction:
hooks, audience targeting angles, visual themes, captions, messaging notes.

```
generate --> validate --+--> END (valid, or retries exhausted)
    ^                    |
    |                    v
    +----- bump_retry <--+ (invalid, retries remain)
```

1. **generate** (`nodes.py` + `llm.py` + `market_research.py`) - fetches
   market/trend and competitor context via two concurrent Tavily searches
   (`asyncio.gather`, not sequential - keeps the added latency close to
   the slower of the two calls rather than their sum: ~11s total in
   practice), then calls `structured_chat_groq` against a primary model
   (**gpt-oss-120b**), falling back to a secondary model (**gpt-oss-20b**)
   on any provider failure. Model choice came from a 3-way comparison
   (Llama 4 Scout / Llama 3.3 70B / gpt-oss-120b) against sample inputs
   before writing any agent code - see `Creative_strategy.md` Challenge 1.
   This agent needs genuine creative/marketing judgment, not extraction,
   which is why it uses a different (cloud, larger) model than Agent 1.
2. **validate** - same low-bar philosophy as Agent 1 (non-empty
   hooks/captions), but writes a *specific* failure reason the moment
   validation fails - not just once retries are exhausted - so the
   `previous_error` fed into the next attempt's prompt actually has
   something useful in it. (Agent 1's `validate_node` still has the older,
   weaker version of this pattern - worth backporting.)
3. **bump_retry** - same as Agent 1, `max_creative_retries` (default 2).

**Market research grounding (`market_research.py`):** the model's training
cutoff means it has no idea what ad creative is actually working right now
or what competitors are currently saying, so `fetch_market_context()` runs
a trend search (`topic="news"`, biases toward recent content) and a
competitor search (`search_depth="advanced"`) against Tavily. Three
decisions worth knowing before touching this:
- **Best-effort, not blocking** - every exception is caught and an empty
  string returned rather than propagated, so a slow or down search API
  never fails the whole creative call over what's an enrichment step, not
  a hard dependency. Agent 2 runs fine with no Tavily key configured at
  all.
- **Explicit anti-copying instruction** - search results surface actual
  competitor ad copy verbatim; the prompt is explicit that the model
  should synthesize the *insight*, never reuse a competitor's actual
  slogan or phrasing.
- **Fetched once per `graph.ainvoke()` call, cached in state, not
  per-retry** - `market_context` doesn't change based on what the model
  got wrong last attempt, unlike `previous_error`, so a failed-then-retried
  attempt still only pays for one Tavily fetch. This caching does *not*
  survive across separate process invocations, though - see Creative_
  strategy.md Challenge 6's "known nuance" and the matching entry under
  "Known gaps to revisit".

See `Creative_strategy.md` Challenge 6 for the full narrative, including
how this was confirmed to actually change the model's output (not just
plumbing that runs and gets ignored) on a real product page.

Two fields, `messaging_notes` and `ungrounded_claims_flagged`, are prompted
as a required pair rather than interchangeable alternatives - see
`Creative_strategy.md` Challenge 4 for why that distinction had to be made
explicit.

**Note on `groq_fallback_model`:** originally `llama-3.3-70b-versatile`,
which Groq deprecated for free/developer tier on June 17, 2026. Now
`openai/gpt-oss-20b` - see `Prompt_generation.md` Challenge 4 for how this
was caught (it hadn't bitten this agent yet only because its primary had
never needed to fall back).

## Agent 3: Prompt Generation

Takes Agent 2's `CreativeDirection` object, returns ComfyUI-ready image and
video generation prompts - one `ThemePromptSet` per input `visual_theme`
(typically 2-3 per product), each pairing an `image_prompt` and a
`video_prompt` with hooks/captions selected (verbatim, never invented) from
Agent 2's existing output.

```
generate --> validate --+--> END (valid, or retries exhausted)
    ^                    |
    |                    v
    +----- bump_retry <--+ (invalid, retries remain)
```

Same shape as Agent 2 - no scrape-equivalent stage.

1. **generate** (`nodes.py` + `llm.py`) - calls `structured_chat_groq`
   against `prompt_gen_primary_model` (**gpt-oss-120b**, an inherited
   placeholder - not yet validated with its own comparison test, see
   `Prompt_generation.md`), falling back to `prompt_gen_fallback_model`
   (**gpt-oss-20b**). Image and video prompts for *all* of a product's
   visual themes are produced in a single call, not split across multiple
   calls - keeps this agent's request footprint equal to Agent 2's rather
   than doubling it (see `Prompt_generation.md` Challenge 3).
2. **validate** - same low-bar philosophy: checks `prompt_sets` count
   matches the input's `visual_themes` count and that no prompt field came
   back empty, not prompt *quality*.
3. **bump_retry** - same pattern, `max_prompt_gen_retries` (default 2).

Sampler/workflow parameters (seed, steps, cfg_scale, checkpoint/LoRA
selection) are deliberately **not** modeled in this agent's schema - those
are deterministic ComfyUI workflow configuration that belongs to Agent 4,
not something an LLM should be guessing at. See `Prompt_generation.md`
Challenge 1.

**Video prompts are now shaped around a confirmed target model:**
image-to-video generation via `NimVideo/cogvideox-2b-img2vid`, a community
fine-tune of CogVideoX-2B (there's no official THUDM CogVideoX-2B
image-to-video release - only 5B-I2V exists upstream). Two schema
consequences followed directly from that: `VideoGenerationPrompt` no longer
has a `duration_seconds` field, and `base_prompt` is now explicitly
prompted to stay dense and verbose - CogVideoX was trained on long,
detailed captions, not short ones - with a length check added to
`validate_node` to catch a prompt likely to exceed the text encoder's
~226-token ceiling before it silently truncates at generation time.

**Correction, discovered while building Agent 6 (then numbered Agent 5):**
the `duration_seconds` removal was originally justified as "this
checkpoint's output length is fixed by the checkpoint itself, not a real
parameter." That's wrong - `num_frames` turned out to be a plain editable
input on the ComfyUI sampler node once the actual workflow was inspected
(see `Video_generation.md` Challenge 3). The schema decision itself still
stands (an LLM shouldn't be choosing this), just for a different reason:
it's deterministic pipeline configuration that belongs to Agent 6's
`Settings`, the same category as seed/steps/cfg - not something fixed and
therefore moot.

## Agent 4: Image Generation

Takes Agent 3's `PromptGenerationOutput`, turns each `ThemePromptSet`'s
`image_prompt` into generated images via a local ComfyUI instance
(SDXL-family checkpoint, core ComfyUI nodes - no custom node package
needed here, unlike Agent 6).

```
start --> generate --> validate --+--> [more themes?] --+--> advance_theme --> generate (next theme)
             ^                    |                      |
             |                    v                      +--> finalize --> END (all themes done)
             +----- bump_retry <--+ (invalid, retries remain)
                                  |
                                  +--> advance_theme (retries exhausted - skip this theme)
```

Two loops layered on top of Agents 1-3's single retry loop: a retry loop
(same shape as before) and a theme loop, since this agent iterates a
variable number of visual themes (typically 2-3) within one product,
something Agents 1-3 never needed since they each produced exactly one
output per product.

1. **start** - verifies the ComfyUI checkpoint exists on the server,
   generates a `run_id`, and splits `total_images_per_product` across
   however many themes this product has via `distribute_total()` (e.g. 5
   images / 2 themes -> `[3, 2]`).
2. **generate** (`nodes.py` + `comfyui_client.py`) - resolves
   `aspect_ratio` to pixel dimensions (`aspect_ratio.py` - Agent 4 owns
   this mapping, not Agent 3, since it owns the checkpoint), builds and
   queues a ComfyUI workflow, polls to completion, and copies the
   resulting images into this project's own output directory
   (`<image_output_dir>/<slug>/<run_id>/theme_<n>/`).
3. **validate** - low bar: did generation produce the expected image
   count for this theme, not whether the images are any good.
4. **bump_retry** - draws a **fresh random seed** on every retry, not the
   same one - a deterministic sampler would just reproduce the exact same
   failure otherwise.
5. **advance_theme** - files whatever result exists, resets scratch
   state, moves to the next theme. A theme that exhausts retries is
   skipped entirely (absent from `theme_results`) rather than failing the
   whole product.
6. **finalize** - assembles `ImageGenerationOutput` once every theme has
   been attempted.

No `prompts.py` here - generation is a deterministic API call, not an LLM
call, so there's no prompt to construct. Candidate selection ("which of
the N images per theme is best") is deliberately **not** modeled here -
that's Agent 5's job (see `Image_selection.md`); this agent generates
candidates and confirms they're real, nothing more. See
`Image_generation.md` for the full narrative, including a state-schema bug
(an undeclared LangGraph state key silently dropping generation results
between nodes) that's worth reading before touching a new agent's
`state.py`, since the same class of mistake was guarded against
preemptively in both Agent 5's and Agent 6's state.

**Checkpointing:** this agent's `graph.py` accepts an optional
`checkpointer` at compile time, used by the pipeline runner for real
mid-loop resume (a crash on theme 3 of 5 doesn't mean redoing themes 1-2)
- see "Checkpointing" below.

## Agent 5: Image Selection (Critic)

Takes Agent 4's `ImageGenerationOutput`, picks exactly one candidate image
per theme - the source frame Agent 6 will animate - judged by a vision LLM
against the theme's creative brief.

```
start --> select --+--> [more themes?] --+--> select (next theme)
                    |                      |
                    v                      +--> finalize --> END (all themes done)
                    (always produces a result)
```

Simpler loop than Agents 4/6 - one node handles both retry and
theme-advance, since this agent never leaves a theme without a result
(see `Image_selection.md` Challenge 3).

1. **start** - generates a `run_id`. No ComfyUI involvement at all - this
   agent's only external call is a vision LLM judgment.
2. **select** (`nodes.py` + `llm.py`'s `structured_chat_vision`) - a
   deterministic pre-filter (`PIL.Image.verify()`) drops corrupt
   candidates before spending a model call; survivors are shown together
   to a vision LLM alongside the creative brief, which scores and ranks
   them comparatively rather than independently. Below-threshold or
   failed judgments still produce a selection, flagged via `status:
   "selected_below_threshold"` - Agent 6 always gets exactly one frame per
   non-skipped theme.
3. **finalize** - assembles `ImageSelectionOutput` once every theme has
   been attempted.

Runs locally via Ollama (`qwen3.5:4b` - the same model Agent 1 uses for
extraction, confirmed separately to be vision-capable), not on a hosted
model. That wasn't the first choice - see `Image_selection.md` Challenges
5-7 for two real hosted-model failures (a rate limit, then a strict
schema-validation rejection) that led here, and for why the switch
actually resolves the failure category by construction rather than just
changing providers.

Uses the flat JSON checkpoint (`app/core/checkpoint.py`), not LangGraph's
native checkpointer - unlike Agents 4/6, a partial loss here is cheap
(fast, single-shot vision calls, not GPU generation) - see "Checkpointing"
below.

## Agent 6: Video Generation

Takes Agent 3's `PromptGenerationOutput` (for `video_prompt` per theme)
**and** Agent 5's `ImageSelectionOutput` (for the already-chosen source
frame per theme) - the only agent in this pipeline that consumes two
upstream outputs at once, since image-to-video generation needs both a
prompt and a starting image. Runs `NimVideo/cogvideox-2b-img2vid` through a
custom ComfyUI node package (not core nodes, unlike Agent 4).

```
start --> generate --> validate --+--> [more themes?] --+--> advance_theme --> generate (next theme)
             ^                    |                      |
             |                    v                      +--> finalize --> END (all themes done)
             +----- bump_retry <--+ (invalid, retries remain)
                                  |
                                  +--> advance_theme (retries exhausted - skip this theme)
```

Same two-loop shape as Agent 4, deliberately - one video per theme, no
batching (video's per-generation cost doesn't reward it the way cheap
image batches did).

1. **start** - loads the API-format ComfyUI workflow JSON once (fails
   loudly if it's the wrong export format - see `Video_generation.md`
   Challenge 1) and generates a `run_id`, same convention as Agent 4.
2. **generate** (`nodes.py` + `video_comfyui_client.py`, which subclasses
   Agent 4's `ComfyUIClient` rather than duplicating or editing it) -
   looks up Agent 5's already-judged selection for the current theme and
   reads its `selected_local_path` directly (no picking logic lives here
   anymore - see `Video_generation.md` Challenge 6), uploads it back to
   ComfyUI, builds and queues the video workflow, polls to completion, and
   saves the result under `<video_output_dir>/<slug>/<run_id>/theme_<n>/`.
   A theme with no usable source frame is recorded as an explicit
   `skipped_no_source_image` result rather than burning a retry on
   something retrying can't fix.
3. **validate** - lowest bar of any agent so far: did `generate_node`
   produce a result at all (success or explicit skip) - there's no
   "count" to check, it's always exactly one video or nothing.
4. **bump_retry** - same fresh-seed-per-retry logic as Agent 4, but
   defaults to **0** retries, not 2 - a real generation attempt on
   6GB VRAM costs minutes, not seconds, so a "free" retry isn't free here.
5. **advance_theme / finalize** - same shape as Agent 4.

Building this agent surfaced two real bugs in how ComfyUI was being
talked to, not just new agent logic: ComfyUI's HTTP server can't service
*any* request while synchronously blocked on a GPU sampling step (broke
both image uploads and this agent's own polling loop), and a defensive
"clear any stuck job" call ended up killing jobs that were still
legitimately running and about to succeed. Both are written up in full in
`Video_generation.md` Challenges 4 and 5 - worth reading before assuming
similar defensive cleanup calls are safe elsewhere in this pipeline.

**Checkpointing:** same as Agent 4 - a `checkpointer`-accepting `graph.py`
for real mid-loop resume, given this agent's real measured cost (up to
~4500s per attempt; ~1964s for an actual 2-video run) - see
"Checkpointing" below.

## Checkpointing

Two different mechanisms, used for different stages on purpose - they
answer different questions and neither is a strict upgrade of the other:

- **`app/core/checkpoint.py`** - flat JSON, one file per stage
  (`outputs/checkpoints/<slug>/<stage>.json`), written after a stage fully
  succeeds. Used for Agents 1, 2, 3, and 5 - fast, single-shot stages
  where the only question worth asking is "did this whole stage already
  finish." A crash mid-stage means redoing that stage from zero, which is
  cheap for these four.
- **`app/core/langgraph_checkpoint.py`** - wraps LangGraph's own
  checkpointer (`AsyncSqliteSaver`), used for Agents 4 and 6 specifically,
  the two stages where a partial loss is actually expensive (image
  batches; up to ~4500s per video attempt). This persists state after
  every node, not just at the end, so a crash after video 1 of 2 succeeded
  resumes at video 2, not video 0 - a distinction the flat JSON mechanism
  can't express. `run_checkpointed()` picks between three outcomes (never
  started / partway through / already finished) by reading
  `graph.aget_state(config)`, not just two.

Both are keyed by the product URL (slugified), not a run ID, so re-running
the same command against the same URL always resumes rather than
restarting. `--fresh` forces a clean run through both mechanisms;
`--clear-checkpoints` wipes everything for a URL without running anything.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium

cp .env.example .env
```

`.env` needs:

```bash
# Agent 1 (extraction) & Agent 5 (vision judgment) - Ollama (local)
OLLAMA_HOST=http://localhost:11434
OLLAMA_RESEARCH_MODEL=qwen3.5:4b

# Agents 2 & 3 - Groq (cloud)
GROQ_API_KEY=your-key-here

# Agent 2 - Tavily (cloud, market/competitor research - optional, best-effort)
TAVILY_API_KEY=your-key-here

# Agents 4 & 6 - ComfyUI (local)
COMFYUI_SERVER=http://127.0.0.1:8188
```

```bash
ollama pull qwen3.5:4b
```

Confirm vision support before relying on Agent 5 - it's not automatic just
because the model family is multimodal upstream (see `Image_selection.md`
Challenge 7):

```bash
ollama show qwen3.5:4b   # look for "vision" under Capabilities
```

**Agent 4** needs a local ComfyUI instance running with the
`juggernautXL_ragnarok.safetensors` checkpoint (or whatever
`comfyui_checkpoint` points at) installed.

**Agent 6** needs, on top of that:
- `NimVideo/cogvideox-2b-img2vid` downloaded (e.g. via `huggingface-cli`)
  and the matching custom ComfyUI node package cloned - this checkpoint
  does **not** run on core ComfyUI nodes, unlike Agent 4's.
- The workflow exported in **API format**, not the UI-canvas format most
  ComfyUI workflow downloads ship as (Settings -> enable Dev Mode -> load
  the workflow -> "Save (API Format)", a different button from plain
  "Save"). Loading the wrong format fails loudly at Agent 6's `start_node`
  rather than silently misbehaving.
- That exported JSON placed at whatever `comfyui_video_workflow_json`
  points at (default `workflows/cogvideox-2b-img2vid-workflow-API.json`,
  relative to wherever the process is run from).
- A GPU with enough VRAM to actually finish a generation before
  `comfyui_video_generation_timeout_seconds` elapses - see
  `Video_generation.md` Challenge 2 for real measured numbers on a 6GB
  card at different frame counts. This checkpoint is meaningfully heavier
  than Agent 4's SDXL-family checkpoint.

**Checkpointing** (Agents 4 & 6) needs two extra packages beyond
`requirements.txt`'s base set:

```bash
pip install aiosqlite langgraph-checkpoint-sqlite
```

Groq's free tier is enough to develop against (1K requests/day per model as
of writing - see `Creative_strategy.md` Challenge 1 for why that's not
actually the binding constraint it looks like at first, and
`Prompt_generation.md` Challenge 3 for how Agent 3 avoids doubling its
request footprint despite generating two prompt types per theme). Worth
periodically checking your configured model strings (`groq_primary_model`,
`groq_fallback_model`, `prompt_gen_primary_model`,
`prompt_gen_fallback_model`) against Groq's current model catalog -
deprecated models fail silently into the fallback chain rather than raising
an obvious error; see `Prompt_generation.md` Challenge 4. The same lesson
bit harder on a *hosted* model chosen for Agent 5 originally - see
`Image_selection.md` Challenges 5-6 for a real rate limit and a real
strict-schema rejection that ultimately motivated moving that agent local.

## Running

```bash
# Agent 1 only, prints the validated ProductResearch object as JSON
python main.py https://example-store.com/products/some-widget
```

```bash
# Agent 1 with step-by-step debugging output (scrape results, JSON-LD
# presence, review-extraction method used, retry counts)
python scripts/test_live.py https://example-store.com/products/some-widget
python scripts/test_live.py https://example-store.com/products/some-widget --scrape-only
python scripts/test_live.py https://example-store.com/products/some-widget --save
```

```bash
# Agents 1 + 2 chained - tests the research-to-creative handoff
python scripts/test_pipeline_live.py https://example-store.com/products/some-widget
python scripts/test_pipeline_live.py https://example-store.com/products/some-widget --research-only
python scripts/test_pipeline_live.py https://example-store.com/products/some-widget --save
```

```bash
# All six agents chained - tests the full research-to-video handoff
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget --fresh
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget --clear-checkpoints
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget --research-only
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget --creative-only
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget --prompts-only
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget --images-only
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget --selection-only
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget --save
```

All scripts exit non-zero with a stderr message on failure (scrape failure,
or exhausted retries at whichever stage failed). Each stage prints its
elapsed time and retry count separately - useful for figuring out which
stage actually owns a slow run rather than guessing (see
`Prompt_generation.md` Challenge 5, where scrape latency was initially
mis-attributed to Agent 1's local model). `--images-only` and
`--selection-only` are worth reaching for while iterating on Agent 6
specifically - video generation is by a wide margin the slowest stage in
this pipeline (see `Video_generation.md` Challenge 2 for how much slower,
in real measured numbers), and re-running everything upstream every time
is pure waste while debugging Agent 6 alone. Since checkpointing is the
default, re-running the same command without `--fresh` already skips
whatever finished last time - these flags are for stopping *early*, not
for avoiding recomputation, which happens automatically.

Real end-to-end run (all six agents, no early-exit flag), against
`bananaclub.co.in/products/lion_embroidered_black_patent_loafers`:

```
research:   55.67s
creative:    5.75s
prompts:     4.99s
images:    472.92s
selection:  26.36s
videos:   1963.87s
─────────────────
total:    2529.68s  (~42 minutes)
```

Video Generation dominates the total by a wide margin - this is why it's
one of the two stages on LangGraph's native checkpointer rather than the
flat JSON mechanism (see "Checkpointing" above).

## Bulk processing

Two commands, not one - because Video Generation alone is ~78% of a full
run's wall clock (2529.68s of it, from the measured run above), and it's
the one output you'd most want a human to approve before an overnight
batch spends GPU-hours on it.

```bash
# Agents 1-5 for every URL in a CSV (one 'url' column, or a bare URL list),
# or for a single URL passed directly. Always stops before Video
# Generation. Writes a timestamped results manifest as it goes.
python scripts/run_batch.py urls.csv
python scripts/run_batch.py https://example-store.com/products/some-widget
python scripts/run_batch.py urls.csv --out outputs/october_batch.csv
python scripts/run_batch.py urls.csv --stop-after prompts
python scripts/run_batch.py urls.csv --fresh
```

```bash
# Agent 6 for a chosen subset - reuses whatever's already checkpointed
# (a selection-complete URL costs ~33 min; a never-run URL runs the full
# pipeline first, ~42 min). Asks for confirmation before starting, since a
# mistake here costs hours, not seconds.
python scripts/generate_videos.py https://example-store.com/products/some-widget
python scripts/generate_videos.py --csv picked.csv
python scripts/generate_videos.py --from-results outputs/batch_results_<ts>.csv --rows 3,7,12-15
python scripts/generate_videos.py --from-results outputs/batch_results_<ts>.csv --yes
```

**Why the split, not a `--with-video` flag on one command:** a batch of 50
is ~7.9 hours through Agent 5 alone; adding video unconditionally would
make it ~35 hours, almost all of it GPU-serialized and therefore not
meaningfully parallelizable across products on one GPU (see Image_
generation.md and Video_generation.md's ComfyUI-concurrency challenges for
why running two products' GPU stages side by side is actively harmful, not
just unhelpful). Stopping at Agent 5 turns the batch into a same-day
reviewable artifact - selected images plus a flag for any theme Agent 5
selected below its confidence threshold - and `generate_videos.py` spends
the expensive stage only where that review says it's worth it.

**Failure isolation is per-row, with no separate retry mechanism**, on
purpose: `run_product()` (`app/core/pipeline.py`) never raises past a
stage boundary and never exits, so one bad URL costs that row and nothing
else. Re-running the same command re-processes the whole file, but every
row that already finished is a checkpoint hit - this is the existing
per-product checkpointing (see "Checkpointing" above) doing double duty as
the batch-retry mechanism, not a second one layered on top.

**The results manifest** (`app/core/manifest.py`, `outputs/
{batch,video}_results_<UTC timestamp>.csv`) is written and flushed after
each product, not buffered to the end - an interrupted 8-hour batch still
leaves a readable record of everything that finished. Columns: `url`,
`status`, `stage_reached`, `error`, `themes_completed`, `total_images`,
`selection_flags` (which themes came back `selected_below_threshold` -
the signal worth sorting by before deciding what to animate),
`videos_generated` (blank on a batch-only run), `elapsed_seconds`,
`finished_at`.

**Sequential, deliberately, with no concurrency across products.** Through
Agent 5, roughly 97% of a product's time is GPU/local-model-bound (Agents
1 and 5 share the same Ollama model as Agent 4's ComfyUI instance shares
the GPU); the LLM-bound stages (Agents 2, 3, and now Agent 2's Tavily
calls) are a few percent of the total, so overlapping them across products
buys single-digit-percent speedup for real implementation complexity.
Sequential isn't the "simple but slow" option here - it's within a few
percent of the achievable optimum on one machine with one GPU.

## Configuration notes

Settings live in `app/core/config.py`, env-driven, shared across agents.
Worth understanding rather than just accepting the defaults:

| Setting | Default | Why it matters |
|---|---|---|
| `ollama_num_ctx` | 16384 | Must comfortably fit system+schema prompt + visible_text + review_text or output gets silently truncated - unrelated to the model's advertised max context, which is a separate, larger number. |
| `ollama_num_predict` | 2048 | Worst-case output token ceiling for Agent 1. Real `ProductResearch` output is a few hundred tokens; this just bounds a rambling response. |
| `max_visible_text_chars` / `max_review_text_chars` | 20000 / 8000 | Hard caps on how much scraped text reaches the LLM - keeps prompts (and Ollama latency) bounded regardless of how bloated a page's HTML is. |
| `min_review_text_chars` | 200 | Below this, review text extracted from the static page counts as "not really found," which is what triggers the scroll-escalation pass in the scraper. |
| `nav_attempt_timeout_ms` | 10000 | Per-strategy navigation timeout, separate from `scrape_timeout_ms`. Without this, a single slow strategy (`networkidle` on a page with persistent background traffic) could burn the entire scrape budget before falling through to a faster fallback strategy. |
| `review_wait_timeout_ms` | 4000 | How long to wait for a review selector to appear after the scroll-trigger pass, before giving up and calling it `review_extraction_method: "none"`. |
| `max_extraction_retries` | 2 | Agent 1's retry budget. |
| `groq_primary_model` / `groq_fallback_model` | `openai/gpt-oss-120b` / `openai/gpt-oss-20b` | Agent 2's model chain - see `Creative_strategy.md` Challenge 1 for the comparison that produced the primary choice. |
| `groq_temperature` | 0.7 | Higher than Agent 1's 0.2 - Agent 2 is a creative task, not extraction, so more variance is wanted rather than suppressed. |
| `max_creative_retries` | 2 | Agent 2's retry budget. |
| `prompt_gen_primary_model` / `prompt_gen_fallback_model` | `openai/gpt-oss-120b` / `openai/gpt-oss-20b` | Agent 3's model chain - kept as its own settings block, separate from Agent 2's, because prompt-writing for generation models is a different skill from ad copywriting. The primary is an inherited placeholder, not yet validated with its own comparison test - see `Prompt_generation.md`. |
| `prompt_gen_temperature` | 0.6 | Slightly lower than Agent 2's 0.7 - prompt-writing benefits from more precision/consistency than open-ended ad copy, but still needs some variation across 2-3 distinct visual themes. |
| `max_prompt_gen_retries` | 2 | Agent 3's retry budget. |
| `comfyui_checkpoint` | `juggernautXL_ragnarok.safetensors` | Agent 4's SDXL-family checkpoint. |
| `total_images_per_product` / `max_image_gen_retries` | 5 / 2 | Agent 4's total candidate images (split across themes via `distribute_total()`) and retry budget. |
| `comfyui_video_num_frames` | 41 | Real runtime parameter, not fixed by the checkpoint - see `Video_generation.md` Challenge 3. Must be `4n+1` for CogVideoX's temporal VAE. |
| `comfyui_video_generation_timeout_seconds` | 4500.0 | Set from real measured per-attempt cost, not a guess - see `Video_generation.md` Challenge 2. |
| `max_video_gen_retries` | 0 | A cost decision, not a resilience default - a "free" retry costs another ~60 minutes on this hardware. See `Video_generation.md` Challenge 2. |
| `image_selection_model` | `qwen3.5:4b` (Ollama, local) | Agent 5's vision judgment model. Originally a hosted Groq preview model - moved local after two real failures (rate limit, then strict-schema rejection). See `Image_selection.md` Challenges 5-7. |
| `image_selection_score_threshold` | 55 | Below this, Agent 5 still picks a candidate (Agent 6 needs one regardless) but flags it `selected_below_threshold`. |
| `image_selection_max_images_per_call` | 5 | No longer a hard API ceiling (that was Groq-specific) - kept as a soft cap for latency/focus even on local inference. |
| `max_image_selection_retries` | 2 | Agent 5's retry budget - cheap to raise, since these are fast local calls, not GPU generation. |

## Files

| File | Responsibility |
|---|---|
| `app/core/config.py` | Env-driven settings for all six agents |
| `app/core/llm.py` | `structured_chat` (Ollama text), `structured_chat_vision` (Ollama + images, used by Agent 5), `structured_chat_groq` (Groq text, used by Agents 2/3) |
| `app/core/checkpoint.py` | Flat JSON stage checkpoint (Agents 1, 2, 3, 5) |
| `app/core/langgraph_checkpoint.py` | LangGraph-native resume helper (`run_checkpointed`), used by Agents 4 and 6 |
| `app/core/pipeline.py` | `run_product()` - the shared stage runner both bulk-processing entry points call; never raises past a stage boundary, returns a `ProductRunResult` instead |
| `app/core/manifest.py` | `ManifestWriter` (per-row-flushed results CSV) + `read_urls()` (accepts a bare URL or a CSV) |
| `app/agents/research_agent/schema.py` | `ProductResearch` output contract + `ScrapedProductData` intermediate shape |
| `app/agents/research_agent/scraper.py` | Playwright fetch + JSON-LD/OG/text extraction, cost-aware review-scroll escalation |
| `app/agents/research_agent/prompts.py` | System + user prompt for the extraction call |
| `app/agents/research_agent/state.py` | LangGraph state TypedDict |
| `app/agents/research_agent/nodes.py` | Node functions + retry/validation routing |
| `app/agents/research_agent/graph.py` | Graph assembly |
| `app/agents/creative_strategy_agent/schema.py` | `CreativeDirection` output contract + `AudienceAngle`/`VisualTheme` sub-models |
| `app/agents/creative_strategy_agent/market_research.py` | `fetch_market_context()` - concurrent Tavily trend + competitor searches, best-effort (never raises, empty string on any failure) |
| `app/agents/creative_strategy_agent/prompts.py` | System + user prompt for the creative-generation call |
| `app/agents/creative_strategy_agent/state.py` | LangGraph state TypedDict |
| `app/agents/creative_strategy_agent/nodes.py` | Node functions, model-fallback chain, retry/validation routing |
| `app/agents/creative_strategy_agent/graph.py` | Graph assembly |
| `app/agents/prompt_gen_agent/schema.py` | `PromptGenerationOutput` output contract + `ImageGenerationPrompt`/`VideoGenerationPrompt`/`ThemePromptSet` sub-models |
| `app/agents/prompt_gen_agent/prompts.py` | System + user prompt for the prompt-generation call |
| `app/agents/prompt_gen_agent/state.py` | LangGraph state TypedDict |
| `app/agents/prompt_gen_agent/nodes.py` | Node functions, model-fallback chain, retry/validation routing |
| `app/agents/prompt_gen_agent/graph.py` | Graph assembly |
| `app/agents/image_generation/schema.py` | `ImageGenerationOutput` output contract + `GeneratedImage`/`ThemeGenerationResult` |
| `app/agents/image_generation/state.py` | LangGraph state TypedDict |
| `app/agents/image_generation/nodes.py` | Node functions, ComfyUI calls, retry/validation routing |
| `app/agents/image_generation/graph.py` | Graph assembly - accepts an optional `checkpointer` |
| `app/agents/image_generation/comfyui_client.py` | Generic ComfyUI HTTP client (queue/poll/fetch), base class for Agent 6's client |
| `app/agents/image_generation/aspect_ratio.py` | `aspect_ratio` string -> pixel dimensions |
| `app/agents/image_selection_agent/schema.py` | `ImageSelectionOutput` output contract + `ImageCandidateScore`/`ImageCritiqueResponse`/`ThemeSelectionResult` |
| `app/agents/image_selection_agent/prompts.py` | Critic system prompt (local-checkpoint-calibrated rubric) + user prompt builder |
| `app/agents/image_selection_agent/state.py` | LangGraph state TypedDict |
| `app/agents/image_selection_agent/nodes.py` | Node functions - deterministic pre-filter, vision judgment call, always-produces-a-result failure handling |
| `app/agents/image_selection_agent/graph.py` | Graph assembly |
| `app/agents/video_generation/schema.py` | `VideoGenerationOutput` output contract + `GeneratedVideo`/`ThemeVideoResult` |
| `app/agents/video_generation/state.py` | LangGraph state TypedDict - consumes Agent 5's `ImageSelectionOutput` |
| `app/agents/video_generation/nodes.py` | Node functions, ComfyUI calls, retry/validation routing |
| `app/agents/video_generation/graph.py` | Graph assembly - accepts an optional `checkpointer` |
| `app/agents/video_generation/video_comfyui_client.py` | Subclasses Agent 4's `ComfyUIClient`, overrides `wait_for_completion` |
| `scripts/test_live.py` | Standalone step-by-step debugging runner for Agent 1 |
| `scripts/test_pipeline_live.py` | Chains Agent 1 into Agent 2 for two-stage testing |
| `scripts/test_full_pipeline_live.py` | Chains all six agents for end-to-end testing, with per-stage early-exit flags and checkpointing |
| `scripts/run_batch.py` | Bulk (or single-URL) entry point - Agents 1-5, always stops before Video Generation, writes the results manifest |
| `scripts/generate_videos.py` | Video generation for a chosen subset of URLs - reuses whatever's already checkpointed, prompts for confirmation before starting |
| `main.py` | CLI runner for Agent 1, standalone |
| `Product_research.md` | Agent 1 design narrative - challenges hit, decisions made, why |
| `Creative_strategy.md` | Agent 2 design narrative - same format |
| `Prompt_generation.md` | Agent 3 design narrative - same format |
| `Image_generation.md` | Agent 4 design narrative - same format |
| `Image_selection.md` | Agent 5 design narrative - same format |
| `Video_generation.md` | Agent 6 design narrative - same format |

## Known gaps to revisit

- **Review pagination** (Agent 1): the current review selector grabs
  whatever's server-rendered/loaded within the wait window. Sites with
  "load more" review pagination will only yield a partial set.
- **Anti-bot pages** (Agent 1): mitigated for basic fingerprinting (see
  `Product_research.md` Challenge 3) but not hardened enterprise anti-bot
  systems (Akamai, PerimeterX, Cloudflare Bot Management, DataDome).
  `scrape_node` will still "succeed" against a challenge page, just with
  junk `visible_text` - worth a minimum-text-length sanity check before
  extraction.
- **Cross-run market-context persistence** (Agent 2): `market_context` is
  cached in state for the duration of one `graph.ainvoke()` call, so
  retries within a single run only pay for one Tavily fetch - but two
  separate process invocations (e.g. a batch row re-run after a crash
  before the creative-stage checkpoint was written) each start from
  scratch and each pay the full Tavily cost again. Acceptable for now -
  a few credits per re-fetch, and cheap even at bulk-CSV volume - but
  worth persisting `market_context` itself in the flat-JSON checkpoint
  rather than only the final `CreativeDirection` if Tavily volume ever
  becomes a real constraint. See `Creative_strategy.md` Challenge 6.
- **Typographic character normalization** (Agent 2): live output uses smart
  quotes/non-breaking hyphens/emoji (a gpt-oss-120b style tendency).
  Harmless in JSON, could matter later for ad-platform APIs or strict
  character-set contexts. Not fixed - no evidence yet it's caused a real
  problem.
- **Provider-level rate-limit-aware retry/backoff** (Agents 2 & 3): the
  fallback chain treats any primary-model failure the same way (fall
  through to secondary immediately) rather than distinguishing a
  rate-limit error (worth a backoff-and-retry-same-model) from a genuine
  failure. Not implemented for these two agents - Agent 5 did end up
  needing exactly this distinction in practice (`LLMRateLimitError`, see
  `Image_selection.md` Challenge 5) before moving off the hosted model
  that needed it; worth reconsidering whether Agents 2/3 are still safe
  without it or just haven't hit the same volume yet.
- **Generalizing lazy-load escalation beyond reviews** (Agent 1): the
  scroll-escalation heuristic is review-specific. The more general version
  would tie escalation to whatever the LLM's `missing_fields` output flags
  as absent, and have the retry trigger a re-scrape - not yet implemented,
  no second content type has needed it yet.
- **Prompt Generation model choice unvalidated** (Agent 3):
  `prompt_gen_primary_model` is an inherited placeholder from Agent 2, not
  chosen via a real 2-3 candidate comparison against sample
  `CreativeDirection` inputs. Worth running before trusting it - see
  `Prompt_generation.md`.
- **Image/video aspect ratio mismatch** (Agents 3 & 6): `ImageGenerationPrompt.
  aspect_ratio` is chosen per-theme for social framing (e.g. `4:5`,
  `9:16`), but the confirmed video target
  (`NimVideo/cogvideox-2b-img2vid`) has a fixed 720x480 landscape output.
  Whichever image Agent 5 selects as a video's source frame still needs
  reconciling with that fixed resolution - resize/letterbox vs. crop - not
  yet decided.
- **Cross-agent regeneration loop** (Agent 5): if every candidate for a
  theme scores below threshold, Agent 5 currently falls back to the
  best-available candidate rather than asking Agent 4 to regenerate. No
  such loop exists yet - see `Image_selection.md` Challenge 3. Worth
  building only if real failure rates justify it.
- **Local vs. hosted judgment quality unvalidated** (Agent 5): the switch
  to a local model was driven by reliability, not a head-to-head quality
  comparison against a working hosted baseline that never actually
  existed (the hosted path failed most of the time it was tried). Worth a
  real comparison once there's a fuller picture of end-to-end output
  quality to judge against.
- **Video-quality critic** (new agent, not yet built): a second critic,
  judging Agent 6's *finished* videos rather than Agent 4's candidate
  frames, was deliberately deferred until Agent 5 proved itself in
  practice. It has - this is the natural next agent, with a
  flag-and-pass-through failure mode (not auto-regeneration, matching the
  reasoning behind `max_video_gen_retries: 0`).
- **LangGraph HITL / streaming-to-UI**: not implemented at any agent's
  level. Persistence, however, now partially is - Agents 4 and 6 use
  LangGraph's own node-level checkpointer (see "Checkpointing" above);
  Agents 1, 2, 3, and 5 use a simpler flat-JSON, whole-stage mechanism.
  Both were added at the orchestration/pipeline-runner level, not inside
  any individual agent's own logic. HITL still belongs as a gate before
  the expensive image/video generation stages, not inside
  extraction/strategy/prompt-generation/selection themselves - not yet
  built.