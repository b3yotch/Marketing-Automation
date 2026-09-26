"""
Video generation for a chosen subset of products - the opt-in second pass
that `run_batch.py` deliberately doesn't do.

The important thing about this script is how little it does. It calls the
same `run_product()` the batch runner calls, with `stop_after="videos"`
and nothing else special. There is no "resume from stage 5" logic here,
because there doesn't need to be: for a URL the batch already took through
Agent 5, stages 1-5 are all checkpoint hits and Agent 6 is the only thing
that actually executes (~33 min). For a URL that was never run, or that
failed partway, the same call simply computes whatever is missing first
and then does the video - which is exactly the behavior you'd want from
"generate a video for this URL" and is inherited free from checkpointing
rather than implemented here.

So the cost of a row varies by how much of it already exists, and that's
fine and invisible:
    selection-complete URL  -> ~33 min  (Agent 6 only)
    never-run URL           -> ~42 min  (full pipeline)
    failed-at-scrape URL    -> fails again, fast, and the batch continues

Three ways to say which products:
    python scripts/generate_videos.py https://store.com/products/widget
    python scripts/generate_videos.py --csv picked.csv
    python scripts/generate_videos.py --from-results outputs/batch_results_X.csv --rows 3,7,12-15

`--rows` is 1-based over the *data* rows of the manifest (row 1 = first
product, not the header), and accepts ranges. `--from-results` without
`--rows` takes every row whose status is ok, which is the "animate
everything that worked" case.

Because a mistake here costs hours rather than seconds, the script prints
the selection and a time estimate and asks for confirmation before
starting. `--yes` skips that, for when this is driven from another script
rather than typed.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.manifest import ManifestWriter, default_manifest_path, read_urls  # noqa: E402
from app.core.pipeline import (  # noqa: E402
    LANGGRAPH_CHECKPOINT_DB,
    ProductRunResult,
    open_checkpointer,
    run_product,
)

# Measured on the target hardware (RTX 3050 6GB, 41 frames, 20 steps):
# ~1964s for 2 videos in a real full-pipeline run. Used only to print an
# estimate before committing - nothing depends on it being exact.
SECONDS_PER_PRODUCT_ESTIMATE = 2000.0


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _parse_rows(spec: str) -> list[int]:
    """'3,7,12-15' -> [3, 7, 12, 13, 14, 15], 1-based, de-duplicated, ordered."""
    picked: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, _, end = chunk.partition("-")
            for n in range(int(start), int(end) + 1):
                if n not in picked:
                    picked.append(n)
        else:
            n = int(chunk)
            if n not in picked:
                picked.append(n)
    return picked


def _read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _select_from_manifest(path: Path, rows_spec: str | None) -> list[str]:
    rows = _read_manifest(path)
    if not rows:
        raise ValueError(f"{path} has no data rows")

    if rows_spec is None:
        selected = [r for r in rows if (r.get("status") or "").strip() == "ok"]
        if not selected:
            raise ValueError(f"No rows in {path} have status 'ok'")
        return [r["url"].strip() for r in selected]

    picked = _parse_rows(rows_spec)
    out: list[str] = []
    for n in picked:
        if n < 1 or n > len(rows):
            raise ValueError(f"--rows {n} is out of range (manifest has {len(rows)} data rows)")
        url = rows[n - 1]["url"].strip()
        if url not in out:
            out.append(url)
    return out


def _print_summary(results: list[ProductRunResult], total: int, elapsed: float, out_path: Path) -> None:
    print(f"\n{'=' * 70}\nVIDEO RUN SUMMARY\n{'=' * 70}")

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print(f"Processed:  {len(results)}/{total}")
    print(f"Succeeded:  {len(ok)}")
    print(f"Failed:     {len(failed)}")
    print(f"Videos:     {sum(r.videos_generated or 0 for r in results)}")
    print(f"Wall clock: {_fmt_duration(elapsed)}")

    # A product can succeed as a run while producing zero videos - every
    # theme skipped for want of a source image, or retries exhausted (and
    # max_video_gen_retries is 0 by design). That is not a failure of the
    # run, but it IS the thing you'd want to notice.
    empty = [r for r in ok if not r.videos_generated]
    if empty:
        print(f"\n⚠ {len(empty)} product(s) completed with zero videos:")
        for r in empty:
            print(f"  {r.url}")

    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  [{r.failed_stage}] {r.url}")
            print(f"      {(r.error or '')[:160]}")

    print(f"\nManifest: {out_path}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate videos (Agent 6) for selected products, resuming from existing checkpoints"
    )
    parser.add_argument("urls", nargs="*", help="One or more product URLs")
    parser.add_argument("--csv", default=None, help="CSV of URLs to animate (same shape as the input urls.csv)")
    parser.add_argument("--from-results", default=None, help="A batch manifest CSV to pick rows out of")
    parser.add_argument(
        "--rows",
        default=None,
        help="With --from-results: 1-based data rows to animate, e.g. '3,7,12-15'. Omit to take every ok row.",
    )
    parser.add_argument("--out", default=None, help="Manifest CSV path (default: outputs/video_results_<timestamp>.csv)")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and recompute every stage")
    parser.add_argument("--verbose", action="store_true", help="Print each stage's full JSON output")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    if args.rows and not args.from_results:
        parser.error("--rows only makes sense with --from-results")

    urls: list[str] = []
    for url in args.urls:
        if url not in urls:
            urls.append(url)
    if args.csv:
        for url in read_urls(args.csv, PROJECT_ROOT):
            if url not in urls:
                urls.append(url)
    if args.from_results:
        for url in _select_from_manifest(Path(args.from_results), args.rows):
            if url not in urls:
                urls.append(url)

    if not urls:
        parser.error("No URLs given - pass them positionally, or use --csv / --from-results")

    estimate = len(urls) * SECONDS_PER_PRODUCT_ESTIMATE
    print(f"{len(urls)} product(s) selected for video generation:")
    for i, url in enumerate(urls, start=1):
        print(f"  {i:>3}. {url}")
    print(f"\nEstimated: ~{_fmt_duration(estimate)} (~{_fmt_duration(SECONDS_PER_PRODUCT_ESTIMATE)} each)")
    print("Products missing upstream stages will run those first and take longer.")

    if not args.yes:
        answer = input("\nProceed? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    out_path = Path(args.out) if args.out else default_manifest_path(PROJECT_ROOT, "video")
    print(f"\nManifest: {out_path}\n")

    results: list[ProductRunResult] = []
    start = time.perf_counter()

    try:
        with ManifestWriter(out_path) as manifest:
            async with open_checkpointer(LANGGRAPH_CHECKPOINT_DB) as checkpointer:
                for i, url in enumerate(urls, start=1):
                    prefix = f"[{i}/{len(urls)}] "
                    print(f"{prefix}{url}", flush=True)

                    result = await run_product(
                        url,
                        stop_after="videos",
                        fresh=args.fresh,
                        checkpointer=checkpointer,
                        verbose=args.verbose,
                        prefix=prefix,
                    )

                    results.append(result)
                    manifest.write(result)

                    if result.ok:
                        print(
                            f"{prefix}✓ {result.videos_generated} video(s) in "
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
        # Agent 6 is on LangGraph's own checkpointer, so an interrupt
        # between themes doesn't lose a finished video - re-running this
        # command resumes mid-theme-loop rather than restarting the
        # product. Note that Ctrl-C does not tell ComfyUI to stop the job
        # it's currently running (Video_generation.md Challenge 4); give
        # the GPU a minute before starting anything else.
        print("\n\nInterrupted. Finished videos are checkpointed.", flush=True)

    _print_summary(results, len(urls), time.perf_counter() - start, out_path)


if __name__ == "__main__":
    asyncio.run(main())