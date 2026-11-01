# Agent 3: Prompt Generation Agent

## Purpose

Takes the `CreativeDirection` object the Creative Strategy Agent produces
and translates each `visual_theme` into a concrete, ComfyUI-ready image
prompt and video prompt. Third stage of the AI Product Creative Generation
Workflow - its output is what the Image/Video Generation Agent (ComfyUI)
will consume next.

## How it works

```
generate --> validate --+--> END (valid, or retries exhausted)
    ^                    |
    |                    v
    +----- bump_retry <--+ (invalid, retries remain)
```

Same shape as Agent 2 - no scrape-equivalent stage, since input is already a
validated `CreativeDirection` object, not something that can fail to fetch.

1. **generate** (`nodes.py` + `llm.py`) - `generate_node` makes one
   `structured_chat_groq` call per input `visual_theme`, all concurrently,
   each asking for a single `ThemePromptSet`. Every theme independently
   walks `prompt_gen_primary_model` -> `prompt_gen_fallback_model`, so a
   provider failure on one theme neither fails the others nor consumes a
   retry. Same fallback-vs-retry separation as Agent 2: a provider error
   isn't a signal the *approach* needs correcting, so it doesn't consume a
   retry.
2. **validate** - per-theme, same low bar as Agents 1 and 2: catches
   outright failures (empty `positive_prompt`, `base_prompt` or
   `motion_description`, a `base_prompt` over the length proxy), not prompt
   *quality*. Deeper judgment belongs to the Review/Critic agent, once it
   can actually look at what Agent 4 generates from these prompts. The
   number of `prompt_sets` is **not** checked here anymore - it is
   guaranteed by `generate_node`'s loop over `creative.visual_themes`, not
   by the model. `source_setting` is overwritten in code with the real
   theme setting instead of trusted from the model, and the product-level
   `extraction_confidence` is computed as the minimum of the per-theme
   values.
3. **bump_retry** - increments the retry counter and loops back to
   `generate` if any theme is still missing and retries remain
   (`max_prompt_gen_retries`, default 2). A failing theme is removed from
   `theme_sets` and its specific reason recorded in `theme_feedback`, so
   the next pass re-calls only that theme and appends the reason to its
   prompt. Provider failures are reported in the final error but never fed
   back - an API error has nothing actionable to tell the model.

## Testing

The per-theme routing is covered by a mocked test that runs the real graph
(`prompt_gen_graph.ainvoke`) with `structured_chat_groq` patched out - no
Groq calls. It covers: 3 themes producing 3 sets; only the failed theme
being re-called, with its validation reason fed back; a provider failure
retried with no reason fed back; the fallback model being used when the
primary raises; a theme that never validates giving up after
`max_prompt_gen_retries`; and zero themes failing cleanly. What it can't
prove is real model behavior, so one live 2-3 theme product is still worth
running before trusting the request-cost change in a batch.

Error messages number themes 1-based (`theme 1`, `theme 2`, ...), matching
the "theme N of M" wording in the prompt. The internal `theme_sets` /
`theme_models` / `theme_errors` / `theme_feedback` maps stay 0-based, keyed
by index into `creative.visual_themes`.

## Why this agent has its own model settings, not Agent 2's

`prompt_gen_primary_model` / `prompt_gen_fallback_model` are deliberately a
separate settings block from Agent 2's `groq_primary_model` /
`groq_fallback_model`, even though they started out pointing at the same
values. Writing a prompt a diffusion/video model actually responds well to
(checkpoint conventions, cinematography vocabulary for motion) is a
different skill from ad copywriting - "best at hooks and captions" isn't
guaranteed to be "best at generation prompts." **This hasn't been validated
with a real comparison test yet** (see Deliberately Deferred) - the current
primary is an inherited placeholder, not a chosen model.

One caveat on this settings block: `prompt_gen_max_tokens` is currently not
read anywhere. `structured_chat_groq` sends `groq_max_tokens` as
`max_tokens` for every Groq caller, and both happen to be 2048 today, so
nothing looks broken - but editing the prompt-gen-specific value has no
effect until that function accepts a per-call cap.

## Major challenges

### 1. The generation backend had to be settled before schema design

An early draft of this agent assumed a generic "prompt string" output
without knowing what would consume it. That's backwards - the schema shape
depends entirely on the target backend. Settled on ComfyUI, needing both
image *and* video prompts (not just images). That decision shaped the
schema toward checkpoint-agnostic natural-language prompts (positive/negative
text) rather than provider-specific parameters, and explicitly excludes
ComfyUI workflow config (seed, steps, cfg_scale, checkpoint/LoRA selection)
from the schema entirely - those are deterministic pipeline settings that
belong with Agent 4's workflow JSON, the same way `scrape_timeout_ms`
belongs to `Settings` rather than `ProductResearch`. An LLM asked to invent
a seed value would just be guessing.

### 2. Variant count is keyed to `visual_themes`, not `hooks`

Considered generating one prompt per hook instead of per visual theme.
Rejected: hooks and captions are *text copy* meant to pair with a visual,
not each drive their own separate image/video - and diffusion models render
legible text unreliably, so hook text was never going to be baked into the
generated image anyway (the system prompt explicitly tells the model not to
describe text/logos/typography in the visual). Settled on one
`ThemePromptSet` per input `visual_theme` (typically 2-3 per product), with
`suggested_hooks`/`suggested_captions` selected **verbatim** from Agent 2's
existing output rather than invented - avoids creating a second, drifting
source of creative copy alongside `CreativeDirection`.

### 3. One call per visual theme, not one per product - reversing the original request-budget decision

The first version asked for one response containing exactly N
`prompt_sets`, one per input `visual_theme`, specifically to keep this
agent's Groq request count equal to Agent 2's (one request per attempt
rather than N). That was wrong in practice: on a real product, three
consecutive attempts returned 2 of the 3 expected sets, and validation gave
up with `expected 3 prompt_sets (one per visual_theme), got 2 (gave up
after 2 retries)`. A count mismatch on every attempt is systematic, not a
fluke - and the output alone couldn't say whether a theme had been merged
into another, dropped, or cut off by the output-token budget.

`generate_node` now makes one call per theme, concurrently. That addresses
all three possible causes at once: the count is a property of the loop over
`creative.visual_themes` rather than something the model has to get right,
each response is roughly a third the size (less pressure on the output
budget), and a retry re-calls only the theme that failed instead of
re-rolling the whole product. Each theme still bundles its image and video
prompt into one `ThemePromptSet`, so the original bundling rationale still
holds - this reversal is about splitting by *item*, not by prompt kind.

The trade is real: a 3-theme product now costs 3 requests per attempt
instead of 1, plus one per retried theme, which matters against Groq's
free-tier requests/day ceiling. That is acceptable here because the
previous design's failure mode was a hard pipeline stop, not just extra
latency, and Agent 2 already costs one request per attempt. If the request
budget becomes the binding constraint again, revisit the bundling - but
correctness first while the failure is systematic.

### 4. A deprecated fallback model was silently masking itself

Live-tested with `prompt_gen_fallback_model` initially set to
`llama-3.3-70b-versatile` - inherited as a placeholder from Agent 2's
config. Groq deprecated that exact model (and `llama-3.1-8b-instant`) for
free/developer tier on June 17, 2026, recommending `gpt-oss-20b` /
`gpt-oss-120b` / `qwen/qwen3.6-27b` as replacements. The bug was invisible
under normal operation: the primary model was succeeding, so the dead
fallback never got exercised, and nothing in `validate_node` would have
caught it even if it had. It surfaced by accident during a manual config
swap that put the deprecated model in the *primary* slot - every run's
`model_used` field showed the fallback model instead of the configured
primary, which is the tell that the primary call is failing outright on
every single attempt, not intermittently. With per-theme calls, `model_used`
is now the comma-joined set of models that actually produced valid themes
(e.g. `openai/gpt-oss-120b, openai/gpt-oss-20b`), so the same tell appears
as the fallback name showing up alongside the primary rather than replacing
it wholesale - a healthy primary should be the only name in that field.

The same dead model string was sitting in Agent 2's `groq_fallback_model`
too, undetected for the same reason - its primary had simply never needed
to fall back yet. Fixed by swapping both fallback slots to
`openai/gpt-oss-20b` (Groq's own recommended replacement for the deprecated
8B-instant tier) rather than mirroring the 120b primary as the fallback too
- a fallback that matches the primary's cost/speed profile doesn't buy
anything extra.

**Lesson:** `model_used` (already logged per-stage as an observability
field) is what made this catchable at all. A configured-primary vs.
observed-`model_used` mismatch is worth actively watching for, not just
trusting the fallback chain to represent healthy degradation - a fallback
chain can mask a *permanently* broken primary exactly as quietly as it
handles a transient one, which is the whole point of it existing, and also
its blind spot.

### 5. Apparent latency was mis-attributed until it was actually measured

Early live runs showed a ~74s Stage 1 time and the local model (qwen3.5:4b)
was the first suspect. It wasn't the main one - the scraper's
`_goto_with_retries` applied the same `scrape_timeout_ms` (30s) ceiling to
*each* navigation strategy independently, including `networkidle`, which
many real ecommerce pages never cleanly reach (persistent analytics/chat-
widget/ad-pixel traffic keeps the network "busy" indefinitely). That could
burn the entire 30s budget on the first strategy alone before falling
through to a faster one. Fixed at the scraper level (not in this agent) by
reordering navigation strategies fast-and-reliable-first and adding a
separate `nav_attempt_timeout_ms` distinct from the overall
`scrape_timeout_ms` ceiling - cut Stage 1 time from ~74s to ~43s on the same
URL, no model changes involved. Noted here because it's a useful reminder
for evaluating *this* agent's latency too: don't assume which stage owns a
slow pipeline run without checking the per-stage timing the test scripts
already log.

### 6. Confirming the actual video model retroactively invalidated a schema field

`VideoGenerationPrompt.duration_seconds` was designed against a hypothetical
generic video model, before a real one was pinned down. Once the target was
confirmed as `NimVideo/cogvideox-2b-img2vid` - a community image-to-video
fine-tune of CogVideoX-2B (there's no official THUDM CogVideoX-2B I2V
release; only 5B-I2V exists upstream) - that field turned out to be asking
the LLM to choose something it can't actually influence: this checkpoint's
output is fixed at 6 seconds, 8fps, 48 frames, 720x480, baked into the
checkpoint itself, not a runtime parameter. Removed the field entirely
rather than leaving it in as a value nothing downstream would read - same
reasoning Challenge 1 already applied to sampler settings, just discovered
a step later, after the consuming model was actually known instead of
assumed.

Confirming the real model also *resolved* a question in the opposite
direction: whether `base_prompt` should stay a full dense scene description
or shrink toward motion-only, since diffusion/video prompt conventions
aren't uniform across models. CogVideoX was specifically trained on long,
verbose captions, not short ones - so the dense-description approach
already in use was right, and got reinforced in the system prompt rather
than second-guessed, with a text-length check added to `validate_node`
(catching anything likely to exceed the ~226-token text encoder ceiling)
so an overlong prompt fails validation instead of silently truncating at
generation time.

**Lesson, same shape as Challenge 1 but worth restating:** a schema
designed against an assumed downstream consumer needs re-auditing once the
real one is confirmed - not just extended. The safe assumption when the
consumer is unknown (leave a parameter open, let the LLM decide) can become
the wrong one once it's known (the parameter isn't real, remove it) or get
retroactively validated (the approach was right, reinforce it) - it won't
be obvious which without actually checking.

## Deliberately deferred (not gaps, decisions)

- **Model choice not yet validated**: `prompt_gen_primary_model` is still
  an inherited placeholder (`gpt-oss-120b`), not chosen via Agent 2's 3-way
  comparison methodology. Worth running before trusting it - especially
  since Groq now hosts `qwen/qwen3.6-27b`, a plausible candidate given this
  agent's task sits closer to structured translation than Agent 2's more
  open-ended creative writing.
- **Live confirmation of the per-theme path**: the mocked routing test
  covers control flow (including provider fallback and the give-up path),
  but the rework has not had a real Groq run. One 2-3 theme product end to
  end is worth doing before a batch, and watch the requests/day budget now
  that each product costs N calls instead of one.
- **`aspect_ratio` and hook-pairing differentiation across themes**:
  promising in live testing (different ratios and different suggested hooks
  per theme, matching composition rather than defaulting) but only observed
  across two live runs - not yet confirmed to hold at scale.
- **Image/video aspect ratio mismatch**: `ImageGenerationPrompt.aspect_ratio`
  is chosen per-theme for social framing (`4:5`, `9:16`, etc.), but the
  confirmed video target has a fixed 720x480 landscape output. Whichever
  generated image ends up selected as a video's source frame will need
  reconciling with that fixed resolution - resize/letterbox vs. crop - not
  yet decided. Belongs to the video generation agent, not this one, but
  noted here since it's a direct consequence of Challenge 6.
- **Deeper prompt-quality grading**: same low-bar `validate_node` philosophy
  as Agents 1 and 2 - schema completeness only, not whether a prompt would
  actually generate something good. That's the Review/Critic agent's job,
  once it can look at Agent 4's actual output.

## Design decisions worth remembering for later agents

- **Settle the downstream consumer's actual interface before schema
  design, not after.** The schema shape (checkpoint-agnostic prompt text
  vs. provider-specific parameters) depends entirely on what's on the other
  end - guessing here would have meant redesigning the schema anyway.
- **Bundle by output type when the constraint is request budget; split by
  item when the constraint is cardinality.** Image and video prompts stay
  bundled into one `ThemePromptSet` per theme because the request budget
  doesn't care about that distinction - but the themes themselves are now
  separate calls, because asking one response to return exactly N sets let
  the model merge or drop one. The split point follows the failure mode,
  not a blanket "fewer requests is better" rule.
- **A "which path actually executed" field (`model_used`) needs to be
  watched, not just logged.** It's the only thing that surfaced the
  deprecated-model bug - the fallback chain itself was working exactly as
  designed, which is precisely why the failure was invisible without it.
- **Provider model catalogs change without warning in code.** A hardcoded
  model string that works today can become dead weight with no runtime
  error, only a behavior mismatch (primary configured vs. fallback always
  used). Worth periodically checking configured model strings against the
  provider's current catalog, not just when something visibly breaks.