"""
The human-in-the-loop gate between Agent 5 (Image Selection) and
Agent 6 (Video Generation).

Walks every `status=ok` row in a batch results manifest, shows what's
worth knowing before committing ~33 GPU-minutes to it (which themes Agent
5 flagged `selected_below_threshold`, how many images/themes completed),
and asks approve/reject per product. Writes an approvals CSV that
`generate_videos.py --approvals <this file>` reads to decide which
products actually run.

Decisions are written and flushed **per product**, same reasoning as
`ManifestWriter`: reviewing 50 products is itself a long task you might
not finish in one sitting, and a half-finished review session should
leave a usable partial approvals file behind, not nothing.

If a trace_id is present for a row (Langfuse configured when the batch
ran), the approve/reject decision is also pushed to that trace as a
`human_approval` score via `trace_id` - no need for the original trace
object, since Langfuse scores by id. This is what actually closes the
loop between "a human looked at this" and the observability data: the
same trace that recorded what the pipeline did now also records what a
person decided about it, which is the beginning of a real eval dataset,
not just a one-off approval log.

Usage:
    python scripts/review.py outputs/batch_results_<ts>.csv
    python scripts/review.py outputs/batch_results_<ts>.csv --out outputs/approvals_oct.csv
    python scripts/review.py outputs/batch_results_<ts>.csv --auto-approve-clean

--auto-approve-clean skips the prompt (and records an automatic
"approved" decision) for any row with zero selection_flags - i.e. every
theme cleared Agent 5's confidence threshold on its own. It still prompts
for every flagged row. This is a reasonable default once you trust Agent 5
on the easy cases and want your attention spent only on the ones it was
unsure about - but it IS weakening the gate, so it's opt-in, not default.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.manifest import read_manifest_rows  # noqa: E402
from observability import observability as obs  # noqa: E402

APPROVAL_COLUMNS = ["url", "decision", "note", "reviewed_at", "trace_id"]


class ApprovalsWriter:
    """Same per-row-flush shape as ManifestWriter, different columns -
    not reusing ManifestWriter itself since its schema is pipeline results,
    not review decisions, and forcing one CSV writer class to cover both
    would mean a column set that's half-empty no matter which it's used for."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self._writer = None

    def __enter__(self) -> "ApprovalsWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=APPROVAL_COLUMNS)
        self._writer.writeheader()
        self._fh.flush()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._fh is not None:
            self._fh.close()

    def write(self, *, url: str, decision: str, note: str, trace_id: str) -> None:
        self._writer.writerow(
            {
                "url": url,
                "decision": decision,
                "note": note,
                "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "trace_id": trace_id,
            }
        )
        self._fh.flush()


def _default_approvals_path(manifest_path: Path) -> Path:
    stem = manifest_path.stem.replace("batch_results_", "").replace("results_", "")
    return manifest_path.parent / f"approvals_{stem}.csv"


def _prompt_decision(url: str, row: dict) -> tuple[str, str]:
    """Returns (decision, note). Loops until a recognized answer is given."""
    flags = (row.get("selection_flags") or "").strip()
    themes = row.get("themes_completed") or "-"
    images = row.get("total_images") or "0"

    print(f"\n{url}")
    print(f"  themes: {themes}   images: {images}")
    if flags:
        print(f"  ⚠ below-threshold themes: {flags}")
    else:
        print("  all themes cleared Agent 5's threshold")

    while True:
        answer = input("  Approve for video generation? [y/n/s=skip for now]: ").strip().lower()
        if answer in ("y", "yes"):
            return "approved", ""
        if answer in ("n", "no"):
            note = input("  Reason (optional): ").strip()
            return "rejected", note
        if answer in ("s", "skip", ""):
            return "skipped", ""
        print("  Please answer y, n, or s.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human review gate: approve/reject products from a batch manifest before video generation"
    )
    parser.add_argument("manifest", help="A results manifest CSV from scripts/run_batch.py")
    parser.add_argument("--out", default=None, help="Approvals CSV path (default: outputs/approvals_<ts>.csv)")
    parser.add_argument(
        "--auto-approve-clean",
        action="store_true",
        help="Skip the prompt for rows with no selection_flags - still prompts for every flagged row",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"No such manifest: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    rows = [r for r in read_manifest_rows(manifest_path) if (r.get("status") or "").strip() == "ok"]
    if not rows:
        print(f"No successful rows in {manifest_path} to review.")
        return

    out_path = Path(args.out) if args.out else _default_approvals_path(manifest_path)
    print(f"{len(rows)} product(s) to review. Writing decisions to {out_path}")
    print(obs.startup_check() + "\n")

    approved = rejected = skipped = auto = 0

    with ApprovalsWriter(out_path) as writer:
        for row in rows:
            url = (row.get("url") or "").strip()
            trace_id = (row.get("trace_id") or "").strip()
            flags = (row.get("selection_flags") or "").strip()

            if args.auto_approve_clean and not flags:
                decision, note = "approved", "auto-approved (no selection_flags)"
                print(f"\n{url}\n  all themes cleared threshold - auto-approved")
                auto += 1
            else:
                decision, note = _prompt_decision(url, row)
                if decision == "approved":
                    approved += 1
                elif decision == "rejected":
                    rejected += 1
                else:
                    skipped += 1

            writer.write(url=url, decision=decision, note=note, trace_id=trace_id)

            if trace_id and decision in ("approved", "rejected"):
                obs.score_trace_by_id(
                    trace_id,
                    "human_approval",
                    1 if decision == "approved" else 0,
                    comment=note or None,
                )

    obs.flush()

    print(f"\n{'=' * 60}")
    print(f"Approved: {approved} (+{auto} auto)   Rejected: {rejected}   Skipped: {skipped}")
    print(f"Approvals file: {out_path}")
    if approved or auto:
        print(
            "\nNext:\n"
            f"  python scripts/generate_videos.py --from-results {manifest_path} "
            f"--approvals {out_path}"
        )


if __name__ == "__main__":
    main()