"""
Shared stage runner for one product URL - the single place the six-stage
sequence is expressed, called by both entry points (`run_batch.py`,
`generate_videos.py`).

Factored out of scripts/test_full_pipeline_live.py for one specific reason
rather than general tidiness: that script is built to be a *terminal*
process. It prints every stage's full JSON, and it calls `sys.exit(1)` the
moment any stage fails. Both behaviors are correct for a one-URL live test
and both are fatal for a batch - row 12 failing to scrape would kill rows
13-50, and fifty full JSON dumps is not output anyone reads.

So this module is the same sequence with those two properties inverted:

- **Returns instead of exiting.** Every failure path produces a
  `ProductRunResult` with `status="failed"` and the stage that failed,
  handed back to the caller to record and move on. Nothing here calls
  `sys.exit`, and nothing raises past `run_product` except genuinely
  unexpected errors (which are caught at the top level and recorded the
  same way, so a caller looping over 50 URLs never has to defend itself).
- **Quiet by default.** One compact line per stage rather than the full
  model dump; `verbose=True` restores the per-stage JSON for single-URL
  debugging.

It deliberately adds NO persistence of its own. Both existing checkpoint
mechanisms are used exactly as `test_full_pipeline_live.py` uses them
(flat JSON via `app.core.checkpoint` for stages 1/2/3/5, LangGraph's own
checkpointer for 4/6 via `run_checkpointed`), which is what makes
`stop_after` cheap: a URL already taken through Agent 5 by a batch run and
then handed to `generate_videos.py` hits five checkpoints and only pays
for Agent 6. Resume is not a feature this module implements - it's a
property it inherits by not getting in the way of one that already exists.

`stop_after` defaults to "selection", not "videos", because that is now
the normal end of the pipeline: video generation is a separate, opt-in
second pass (~33 min/product on the target hardware) run against a chosen
subset, not something a bulk run should ever decide to spend on by itself.

The `checkpointer` parameter exists so a batch can open ONE AsyncSqliteSaver
connection across all rows instead of opening and closing one per product.
Passing None keeps the original behavior (open one for this product,
close it on the way out) and is what a single-URL caller should do.
"""

from __future__ import annotations

import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.research_agent.graph import research_graph
from app.agents.research_agent.schema import ProductResearch
from app.agents.creative_strategy_agent.graph import creative_strategy_graph
from app.agents.creative_strategy_agent.schema import CreativeDirection
from app.agents.prompt_gen_agent.graph import prompt_gen_graph
from app.agents.prompt_gen_agent.schema import PromptGenerationOutput
from app.agents.image_generation.graph import build_graph as build_image_generation_graph
from app.agents.image_selection_agent.graph import image_selection_graph
from app.agents.image_selection_agent.schema import ImageSelectionOutput
from app.agents.video_generation.graph import build_graph as build_video_generation_graph
from app.agents.image_generation.utils import slugify_url
from app.core.checkpoint import clear_checkpoints, load_stage, save_stage
from app.core.langgraph_checkpoint import run_checkpointed
from app.core import observability as obs

LANGGRAPH_CHECKPOINT_DB = "outputs/checkpoints/langgraph_checkpoints.db"

# Every Pydantic schema that a LangGraph state dict hands to the
# checkpointer for msgpack serialization. Without this, AsyncSqliteSaver
# deserializes them anyway today but warns that a future LangGraph version
# will refuse to - at which point every existing images/videos checkpoint
# becomes unreadable until this list exists. Add a new state-carried schema
# here the same day it's introduced, not when the warning reappears.
ALLOWED_MSGPACK_MODULES = [
    ("app.agents.prompt_gen_agent.schema", "PromptGenerationOutput"),
    ("app.agents.image_generation.schema", "ThemeGenerationResult"),
    ("app.agents.image_generation.schema", "ImageGenerationOutput"),
    ("app.agents.image_selection_agent.schema", "ImageSelectionOutput"),
    ("app.agents.video_generation.schema", "VideoGenerationOutput"),
]


def build_checkpoint_serde() -> JsonPlusSerializer:
    """Shared serde so run_batch.py, generate_videos.py, and this module's
    own fallback checkpointer all allow-list the same set of types."""
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)


@asynccontextmanager
async def open_checkpointer(db_path: str) -> Any:
    """
    Equivalent to `AsyncSqliteSaver.from_conn_string(db_path)`, except the
    installed langgraph-checkpoint-sqlite (3.1.1)'s `from_conn_string` does
    not forward a `serde` argument to the saver it constructs - it always
    builds one with the default serializer, silently discarding any serde
    you'd want to pass. The constructor itself (`AsyncSqliteSaver(conn,
    serde=...)`) takes one fine; `from_conn_string` just doesn't plumb it
    through. So this opens the same aiosqlite connection `from_conn_string`
    would and constructs the saver directly. No explicit `.setup()` call is
    needed - every read/write method calls it lazily on first use, which is
    also how `from_conn_string`'s own callers get away without one.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as conn:
        yield AsyncSqliteSaver(conn, serde=build_checkpoint_serde())

# Ordered so `stop_after` can be compared positionally rather than with a
# chain of if/elif - also what tells us whether a LangGraph checkpointer
# connection is needed at all for this run (stages 4 and 6 use it; 1/2/3/5
# don't, and opening a db connection for a --stop-after prompts run would
# be holding something open that never gets touched).
STAGE_ORDER = ["research", "creative", "prompts", "images", "selection", "videos"]


@dataclass
class ProductRunResult:
    """
    Everything a caller needs to write one manifest row, plus the stage
    payloads themselves for callers that want to print or re-serialize them.

    `stage_reached` is the last stage that COMPLETED, which on a failed run
    is the stage before the one that broke - `failed_stage` carries the one
    that actually broke. On a successful run `failed_stage` is None and
    `stage_reached` equals whatever `stop_after` asked for.
    """

    url: str
    status: str = "ok"  # "ok" | "failed"
    stage_reached: str = "-"
    failed_stage: str | None = None
    error: str | None = None
    trace_id: str | None = None  # Langfuse trace id - None when tracing isn't configured

    elapsed: dict[str, float] = field(default_factory=dict)
    total_elapsed_seconds: float = 0.0

    themes_expected: int | None = None
    themes_completed: int | None = None
    total_images: int = 0
    selection_flags: list[str] = field(default_factory=list)
    videos_generated: int | None = None

    research: Any = None
    creative: Any = None
    prompts: Any = None
    images: Any = None
    selection: Any = None
    videos: Any = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def themes_fraction(self) -> str:
        """'2/3' for the manifest - partial theme failures without opening any JSON."""
        if self.themes_completed is None or self.themes_expected is None:
            return "-"
        return f"{self.themes_completed}/{self.themes_expected}"


def _needs_langgraph(stop_after: str) -> bool:
    return STAGE_ORDER.index(stop_after) >= STAGE_ORDER.index("images")


def _should_run(stage: str, stop_after: str) -> bool:
    return STAGE_ORDER.index(stage) <= STAGE_ORDER.index(stop_after)


def _fail(result: ProductRunResult, stage: str, error: Any, t_start: float) -> ProductRunResult:
    result.status = "failed"
    result.failed_stage = stage
    result.error = str(error) if error else f"{stage} produced no output and no error message"
    result.total_elapsed_seconds = time.perf_counter() - t_start
    return result


def _finish(result: ProductRunResult, t_start: float) -> ProductRunResult:
    result.total_elapsed_seconds = time.perf_counter() - t_start
    return result


def _try_load(url: str, stage: str, schema, resume: bool):
    """Flat-JSON checkpoint load for stages 1/2/3/5. Returns (data, metadata) or None."""
    if not resume:
        return None
    loaded = load_stage(url, stage, schema)
    if loaded is None:
        return None
    return loaded


async def run_product(
    url: str,
    *,
    stop_after: str = "selection",
    fresh: bool = False,
    checkpointer: Any = None,
    verbose: bool = False,
    prefix: str = "",
) -> ProductRunResult:
    """
    Runs stages 1..`stop_after` for one URL and returns what happened.

    Never raises for a stage failure and never calls sys.exit - see the
    module docstring. `prefix` is prepended to every progress line so a
    batch can tag output with '[12/50] '.
    """
    if stop_after not in STAGE_ORDER:
        raise ValueError(f"stop_after must be one of {STAGE_ORDER}, got {stop_after!r}")

    result = ProductRunResult(url=url)
    resume = not fresh
    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"{prefix}{msg}", flush=True)

    def dump(obj) -> None:
        if verbose:
            print(obj.model_dump_json(indent=2), flush=True)

    try:
        if fresh:
            clear_checkpoints(url)

        async with AsyncExitStack() as stack:
            lg_checkpointer = checkpointer
            if lg_checkpointer is None and _needs_langgraph(stop_after):
                lg_checkpointer = await stack.enter_async_context(
                    open_checkpointer(LANGGRAPH_CHECKPOINT_DB)
                )

            if fresh and lg_checkpointer is not None:
                await lg_checkpointer.adelete_thread(f"images:{slugify_url(url)}")
                await lg_checkpointer.adelete_thread(f"videos:{slugify_url(url)}")

            # ---- Stage 1: Product Research ----------------------------------
            cached = _try_load(url, "research", ProductResearch, resume)
            if cached:
                research, meta = cached
                result.elapsed["research"] = meta.get("elapsed_seconds", 0.0)
                log("  research   ↺ checkpoint")
            else:
                t = time.perf_counter()
                try:
                    r = await research_graph.ainvoke({"url": url, "retries": 0})
                except Exception as exc:
                    return _fail(result, "research", exc, t_start, trace)
                result.elapsed["research"] = time.perf_counter() - t

                if r.get("error") or not r.get("research"):
                    return _fail(result, "research", r.get("error"), t_start, trace)

                research = r["research"]
                save_stage(
                    url, "research", research,
                    elapsed_seconds=result.elapsed["research"],
                    retries=r.get("retries", 0),
                )
                log(f"  research   ✓ {result.elapsed['research']:.1f}s")

            result.research = research
            result.stage_reached = "research"
            dump(research)
            if not _should_run("creative", stop_after):
                return _finish(result, t_start, trace)

            # ---- Stage 2: Creative Strategy ---------------------------------
            cached = _try_load(url, "creative", CreativeDirection, resume)
            if cached:
                creative, meta = cached
                result.elapsed["creative"] = meta.get("elapsed_seconds", 0.0)
                log("  creative   ↺ checkpoint")
            else:
                t = time.perf_counter()
                try:
                    c = await creative_strategy_graph.ainvoke({"research": research, "retries": 0})
                except Exception as exc:
                    return _fail(result, "creative", exc, t_start, trace)
                result.elapsed["creative"] = time.perf_counter() - t

                if c.get("error") or not c.get("creative"):
                    return _fail(result, "creative", c.get("error"), t_start, trace)

                creative = c["creative"]
                save_stage(
                    url, "creative", creative,
                    elapsed_seconds=result.elapsed["creative"],
                    retries=c.get("retries", 0),
                    model_used=c.get("model_used"),
                )
                log(f"  creative   ✓ {result.elapsed['creative']:.1f}s")

            result.creative = creative
            result.stage_reached = "creative"
            dump(creative)
            if not _should_run("prompts", stop_after):
                return _finish(result, t_start, trace)

            # ---- Stage 3: Prompt Generation ---------------------------------
            cached = _try_load(url, "prompts", PromptGenerationOutput, resume)
            if cached:
                prompts, meta = cached
                result.elapsed["prompts"] = meta.get("elapsed_seconds", 0.0)
                log("  prompts    ↺ checkpoint")
            else:
                t = time.perf_counter()
                try:
                    p = await prompt_gen_graph.ainvoke({"creative": creative, "retries": 0})
                except Exception as exc:
                    return _fail(result, "prompts", exc, t_start, trace)
                result.elapsed["prompts"] = time.perf_counter() - t

                if p.get("error") or not p.get("prompts"):
                    return _fail(result, "prompts", p.get("error"), t_start, trace)

                prompts = p["prompts"]
                save_stage(
                    url, "prompts", prompts,
                    elapsed_seconds=result.elapsed["prompts"],
                    retries=p.get("retries", 0),
                    model_used=p.get("model_used"),
                )
                log(f"  prompts    ✓ {result.elapsed['prompts']:.1f}s")

            result.prompts = prompts
            result.stage_reached = "prompts"
            result.themes_expected = len(prompts.prompt_sets)
            dump(prompts)
            if not _should_run("images", stop_after):
                return _finish(result, t_start, trace)

            # ---- Stage 4: Image Generation ----------------------------------
            image_graph = build_image_generation_graph(checkpointer=lg_checkpointer)
            image_config = {"configurable": {"thread_id": f"images:{slugify_url(url)}"}}

            t = time.perf_counter()
            try:
                img = await run_checkpointed(
                    image_graph, image_config, {"prompts": prompts, "retries": 0}, "images"
                )
            except Exception as exc:
                return _fail(result, "images", exc, t_start, trace)
            result.elapsed["images"] = time.perf_counter() - t

            if img.get("error") and not img.get("images"):
                return _fail(result, "images", img.get("error"), t_start, trace)
            if not img.get("images"):
                return _fail(result, "images", None, t_start, trace)

            images = img["images"]
            result.images = images
            result.stage_reached = "images"
            result.themes_completed = len(images.theme_results)
            result.total_images = sum(len(th.images) for th in images.theme_results)
            log(
                f"  images     ✓ {result.elapsed['images']:.1f}s "
                f"({result.total_images} images, themes {result.themes_fraction})"
            )
            dump(images)
            if not _should_run("selection", stop_after):
                return _finish(result, t_start, trace)

            # ---- Stage 5: Image Selection (Critic) --------------------------
            cached = _try_load(url, "selection", ImageSelectionOutput, resume)
            if cached:
                selection, meta = cached
                result.elapsed["selection"] = meta.get("elapsed_seconds", 0.0)
                log("  selection  ↺ checkpoint")
            else:
                t = time.perf_counter()
                try:
                    s = await image_selection_graph.ainvoke({"images": images, "retries": 0})
                except Exception as exc:
                    return _fail(result, "selection", exc, t_start, trace)
                result.elapsed["selection"] = time.perf_counter() - t

                if s.get("error") or not s.get("selection"):
                    return _fail(result, "selection", s.get("error"), t_start, trace)

                selection = s["selection"]
                save_stage(url, "selection", selection, elapsed_seconds=result.elapsed["selection"])
                log(f"  selection  ✓ {result.elapsed['selection']:.1f}s")

            result.selection = selection
            result.stage_reached = "selection"
            result.selection_flags = [
                th.source_setting
                for th in selection.theme_results
                if th.status == "selected_below_threshold"
            ]
            if result.selection_flags:
                log(f"  ⚠ {len(result.selection_flags)} theme(s) selected below threshold")
            dump(selection)
            if not _should_run("videos", stop_after):
                return _finish(result, t_start, trace)

            # ---- Stage 6: Video Generation ----------------------------------
            video_graph = build_video_generation_graph(checkpointer=lg_checkpointer)
            video_config = {"configurable": {"thread_id": f"videos:{slugify_url(url)}"}}

            t = time.perf_counter()
            try:
                v = await run_checkpointed(
                    video_graph,
                    video_config,
                    {"prompts": prompts, "selection": selection, "retries": 0},
                    "videos",
                )
            except Exception as exc:
                return _fail(result, "videos", exc, t_start, trace)
            result.elapsed["videos"] = time.perf_counter() - t

            if v.get("error") and not v.get("videos"):
                return _fail(result, "videos", v.get("error"), t_start, trace)
            if not v.get("videos"):
                return _fail(result, "videos", None, t_start, trace)

            videos = v["videos"]
            result.videos = videos
            result.stage_reached = "videos"
            result.videos_generated = len(
                [th for th in videos.theme_results if th.status == "success"]
            )
            log(
                f"  videos     ✓ {result.elapsed['videos']:.1f}s "
                f"({result.videos_generated} video(s))"
            )
            dump(videos)
            return _finish(result, t_start, trace)

    except Exception as exc:
        # Backstop for anything outside a stage guard (checkpointer setup,
        # a checkpoint file that won't deserialize, disk full mid-save).
        # A batch caller must never have to wrap this call in its own
        # try/except to survive one bad row.
        return _fail(result, result.failed_stage or "unexpected", exc, t_start, trace)


async def clear_all_checkpoints(url: str) -> None:
    """Clears both mechanisms for a URL - JSON checkpoints and both LangGraph threads."""
    clear_checkpoints(url)
    async with open_checkpointer(LANGGRAPH_CHECKPOINT_DB) as cp:
        await cp.adelete_thread(f"images:{slugify_url(url)}")
        await cp.adelete_thread(f"videos:{slugify_url(url)}")