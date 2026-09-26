"""
Bulk (and single-URL) runner: Research -> Creative Strategy -> Prompt
Generation -> Image Generation -> Image Selection. Stops there, always.

Video generation is NOT part of this command by design, not by omission.
On the target hardware Agent 6 is ~33 minutes per product against ~9.4
minutes for everything upstream of it combined - 78% of a full run's wall
clock for the one output you'd most want a human to approve before paying
for. So this command produces the reviewable artifact (selected images per
theme, plus a manifest saying which selections were made below Agent 5's
confidence threshold) and `generate_videos.py` spends the GPU hours on
whichever subset survives that review.

Single URL and CSV are the same code path - a single URL is just a
one-row batch. There is no separate "full pipeline" mode any more.

Usage:
    python scripts/run_batch.py urls.csv
    python scripts/run_batch.py https://store.com/products/widget
    python scripts/run_batch.py urls.csv --out outputs/october_batch.csv
    python scripts/run_batch.py urls.csv --fresh
    python scripts/run_batch.py urls.csv --stop-after prompts
    python scripts/run_batch.py urls.csv --verbose

--stop-after {research,creative,prompts,images,selection} stops earlier
still; `selection` is the default and `videos` is deliberately not an
accepted value here (use generate_videos.py, which exists precisely so
that spending ~33 min/product is always an explicit act).

Concurrency: none, deliberately. ~88% of a run through Agent 5 is
GPU-serialized (ComfyUI for Agent 4, Ollama for Agents 1 and 5, one
RTX 3050 between them), so overlapping products cannot use hardware that
isn't there - and ComfyUI's single-threaded server actively misbehaves
under concurrent clients (see Video_generation.md Challenges 4 and 5: a
poll landing mid-step fails, and a timeout-triggered interrupt() can kill
another product's still-running job). Sequential isn't the slow-but-simple
option here; it's within a few percent of the achievable optimum.

Failure isolation: per-row. `run_product` never raises for a stage failure
and never exits, so a bad URL costs that row and nothing else. No
batch-level retry mechanism exists on top, and that's the point of already
having per-product checkpointing - re-running this exact command reruns
only what didn't finish, at checkpoint-hit cost for everything that did.
Fixing three bad rows means fixing the CSV and re-running the whole file.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.manifest import ManifestWriter, default_manifest_path, read_urls  # noqa: E402
from app.core.pipeline import (  # noqa: E402
    LANGGRAPH_CHECKPOINT_DB,
    STAGE_ORDER,
    ProductRunResult,
    open_checkpointer,
    run_product,
)


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _print_summary(results: list[ProductRunResult], total_urls: int, elapsed: float, out_path: Path) -> None:
    print(f"\n{'=' * 70}\nBATCH SUMMARY\n{'=' * 70}")

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    not_reached = total_urls - len(results)

    print(f"Processed:   {len(results)}/{total_urls}")
    print(f"Succeeded:   {len(ok)}")
    print(f"Failed:      {len(failed)}")
    if not_reached:
        print(f"Not reached: {not_reached} (interrupted)")
    print(f"Wall clock:  {_fmt_duration(elapsed)}")
    if ok:
        print(f"Mean/product: {_fmt_duration(sum(r.total_elapsed_seconds for r in ok) / len(ok))}")
    print(f"Images:      {sum(r.total_images for r in results)}")

    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  [{r.failed_stage}] {r.url}")
            print(f"      {(r.error or '')[:160]}")

    flagged = [r for r in ok if r.selection_flags]
    if flagged:
        print(
            f"\n⚠ {len(flagged)} product(s) had at least one theme selected below "
            f"Agent 5's score threshold - worth eyeballing before committing "
            f"video time to them:"
        )
        for r in flagged:
            print(f"  {r.url}  ({len(r.selection_flags)} theme(s))")

    print(f"\nManifest: {out_path}")
    if ok:
        print(
            "Next: review the selected images, then\n"
            "  python scripts/generate_videos.py --from-results "
            f"{out_path} --rows <n,n,n>"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run products through Agents 1-5 (stops before Video Generation)"
    )
    parser.add_argument(
        "source",
        help="Path to a CSV of product URLs (header with a 'url' column, or one URL per line), or a single URL",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Manifest CSV path (default: outputs/batch_results_<timestamp>.csv)",
    )
    parser.add_argument(
        "--stop-after",
        default="selection",
        choices=[s for s in STAGE_ORDER if s != "videos"],
        help="Stop each product after this stage (default: selection)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoints for every URL in the batch and recompute from scratch",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each stage's full JSON output (noisy - intended for a one-URL run)",
    )
    args = parser.parse_args()

    urls = read_urls(args.source, PROJECT_ROOT)
    if not urls:
        print(f"No URLs found in {args.source}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else default_manifest_path(PROJECT_ROOT, "batch")

    print(f"{len(urls)} product(s) to process, through stage '{args.stop_after}'.")
    print(f"Manifest: {out_path}\n")

    results: list[ProductRunResult] = []
    batch_start = time.perf_counter()

    # One AsyncSqliteSaver for the whole batch rather than one per product:
    # every row would otherwise open and close the same sqlite file, and
    # the connection is cheap to hold for the duration of a run that is
    # sequential anyway.
    try:
        with ManifestWriter(out_path) as manifest:
            async with open_checkpointer(LANGGRAPH_CHECKPOINT_DB) as checkpointer:
                for i, url in enumerate(urls, start=1):
                    prefix = f"[{i}/{len(urls)}] "
                    print(f"{prefix}{url}", flush=True)

                    result = await run_product(
                        url,
                        stop_after=args.stop_after,
                        fresh=args.fresh,
                        checkpointer=checkpointer,
                        verbose=args.verbose,
                        prefix=prefix,
                    )

                    results.append(result)
                    manifest.write(result)

                    if result.ok:
                        print(
                            f"{prefix}✓ {result.stage_reached} in "
                            f"{_fmt_duration(result.total_elapsed_seconds)}\n",
                            flush=True,
                        )
                    else:
                        print(
                            f"{prefix}✗ failed at {result.failed_stage}: "
                            f"{(result.error or '')[:120]}\n",
                            flush=True,
                        )
    except KeyboardInterrupt:
        # Everything finished so far is already on disk (checkpoints) and
        # in the manifest (flushed per row), so an interrupt is a pause,
        # not a loss - re-running the same command picks up where this
        # stopped at checkpoint-hit cost for the rows already done.
        print("\n\nInterrupted. Completed rows are checkpointed and in the manifest.", flush=True)

    _print_summary(results, len(urls), time.perf_counter() - batch_start, out_path)


if __name__ == "__main__":
    asyncio.run(main())