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
product, not the header), and accepts ranges.

Because a mistake here costs hours rather than seconds, the script prints
the selection and a time estimate and asks for confirmation before
starting. `--yes` skips that, for when this is driven from another script
rather than typed.

## Human-in-the-loop gate

Typing specific row numbers (`--rows`) or specific URLs (positional args,
`--csv`) IS the human decision - no further gate is applied to those,
because choosing them one by one already required a person to look at
each one.

`--from-results <manifest>` with **no** `--rows`, though, means "animate
everything that came back ok" - the one path that can silently commit
every GPU-hour in the batch to Agent 5's judgment with no human ever
having looked at a single selected image. That path refuses to run unless
you also pass either:

    --approvals outputs/approvals_<ts>.csv   (written by scripts/review.py)
    --force                                  (explicit, deliberate bypass)

`scripts/review.py` is what produces an approvals file: it walks every
`status=ok` row in a results manifest, shows you the selection-threshold
flags Agent 5 raised, and asks approve/reject per product. When
`--approvals` is given, rows marked anything other than "approved" are
dropped from the run (and printed as skipped) even if they'd otherwise
have been included - this applies on top of `--rows` too, if you pass
both, so an approvals file can also work as a safety net on a hand-picked
list.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from observability import observability as obs  # noqa: E402
from app.core.manifest import (  # noqa: E402
    ManifestWriter,
    default_manifest_path,
    read_manifest_rows,
    read_urls,
)
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


def _select_from_manifest(path: Path, rows_spec: str | None) -> list[str]:
    rows = read_manifest_rows(path)
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


def _load_approvals(path: Path) -> dict[str, str]:
    """url -> decision ('approved' | 'rejected' | whatever scripts/review.py wrote),
    last entry wins if a URL appears more than once in the file."""
    decisions: dict[str, str] = {}
    for row in read_manifest_rows(path):
        url = (row.get("url") or "").strip()
        decision = (row.get("decision") or "").strip().lower()
        if url:
            decisions[url] = decision
    return decisions


def _apply_approvals(urls: list[str], approvals_path: Path) -> list[str]:
    decisions = _load_approvals(approvals_path)
    approved, skipped = [], []
    for url in urls:
        decision = decisions.get(url)
        if decision == "approved":
            approved.append(url)
        else:
            skipped.append((url, decision or "not reviewed"))

    if skipped:
        print(f"\n⚠ {len(skipped)} product(s) skipped by the approvals gate:")
        for url, reason in skipped:
            print(f"  [{reason}] {url}")

    return approved


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
    parser.add_argument(
        "--approvals",
        default=None,
        help="An approvals CSV from scripts/review.py - rows not marked 'approved' are dropped from the run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the HITL approvals requirement for --from-results with no --rows (use deliberately)",
    )
    parser.add_argument("--out", default=None, help="Manifest CSV path (default: outputs/video_results_<timestamp>.csv)")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and recompute every stage")
    parser.add_argument("--verbose", action="store_true", help="Print each stage's full JSON output")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    if args.rows and not args.from_results:
        parser.error("--rows only makes sense with --from-results")

    # The HITL gate: "--from-results with no --rows" is the one path that
    # can commit every GPU-hour in a batch to Agent 5's judgment with no
    # human ever having looked at a selected image. Everything else
    # (typing URLs, --csv, or hand-picked --rows) already required a
    # person to choose each one, so it isn't gated further here.
    if args.from_results and not args.rows and not args.approvals and not args.force:
        parser.error(
            "--from-results with no --rows selects every successful product in the "
            "manifest with no review step. Pass one of:\n"
            "  --rows <n,n,n>              review and pick specific rows yourself\n"
            "  --approvals <csv>           from `python scripts/review.py <manifest>`\n"
            "  --force                     explicit bypass - no review at all"
        )

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

    if args.approvals:
        urls = _apply_approvals(urls, Path(args.approvals))
        if not urls:
            print("\nNo products remain after the approvals gate. Nothing to do.")
            return

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
    print(f"\n{obs.startup_check()}")
    print(f"Manifest: {out_path}\n")

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
                        run_label="video",
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