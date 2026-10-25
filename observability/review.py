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
    python scripts/review.py outputs/batch_results_<ts>.csv --open-all
    python scripts/review.py outputs/batch_results_<ts>.csv --no-open

For each product the script loads Agent 5's checkpointed selection and
prints every theme with the path of the image it chose. Themes selected
below the confidence threshold have their images opened automatically in
your default viewer - those are the ones Agent 5 itself wasn't sure about,
so they're where a human look is worth the most. `--open-all` opens every
selected image instead, `--no-open` opens none, and `v` at the prompt
opens everything for the current product on demand.

Re-running on the same manifest resumes: products already approved or
rejected keep their decision, and only unreviewed/skipped ones are asked.

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
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.image_selection_agent.schema import ImageSelectionOutput  # noqa: E402
from observability import observability as obs  # noqa: E402
from app.core.checkpoint import load_stage  # noqa: E402
from app.core.manifest import read_manifest_rows  # noqa: E402

APPROVAL_COLUMNS = ["url", "decision", "note", "reviewed_at", "trace_id"]


class ApprovalsWriter:
    """Same per-row-flush shape as ManifestWriter, different columns -
    not reusing ManifestWriter itself since its schema is pipeline results,
    not review decisions, and forcing one CSV writer class to cover both
    would mean a column set that's half-empty no matter which it's used for.

    Opens in APPEND mode when the file already has content (header written
    only the first time). The default approvals path is derived from the
    manifest's name, so a second review session on the same manifest lands
    on the same file - truncating it would silently erase every decision
    from the first session. Appending is safe because
    `generate_videos.py`'s approvals loader already treats the last row for
    a URL as the decision."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self._writer = None

    def __enter__(self) -> "ApprovalsWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.path.exists() and self.path.stat().st_size > 0
        self._fh = self.path.open("a" if existing else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=APPROVAL_COLUMNS)
        if not existing:
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


def _load_selection(url: str):
    """Agent 5's checkpointed output for this URL. Returns (selection, error)."""
    try:
        loaded = load_stage(url, "selection", ImageSelectionOutput)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if loaded is None:
        return None, "no selection checkpoint found for this URL"
    return loaded[0], None


def _resolve_image(raw: str | None) -> Path | None:
    """selected_local_path may be relative to the project root or the cwd."""
    if not raw:
        return None
    p = Path(raw)
    for candidate in ([p] if p.is_absolute() else [PROJECT_ROOT / p, p]):
        if candidate.exists():
            return candidate.resolve()
    return None


def _theme_views(selection) -> list[dict]:
    """
    One dict per theme: setting, status, flagged, the chosen image's path,
    and any scalar score/rationale-like fields Agent 5 recorded. Those
    extras are found by field *name pattern* rather than hard-coded
    names, so this keeps working whatever ThemeSelectionResult calls them.
    """
    views = []
    for theme in selection.theme_results:
        d = theme.model_dump()
        raw = d.get("selected_local_path")
        extras = {}
        for key, value in d.items():
            if isinstance(value, (dict, list)) or value in (None, ""):
                continue
            if any(word in key for word in ("score", "rationale", "reason")):
                extras[key] = value if not isinstance(value, str) else (value[:200] + ("..." if len(value) > 200 else ""))
        views.append(
            {
                "setting": d.get("source_setting") or "?",
                "status": d.get("status") or "?",
                "flagged": d.get("status") == "selected_below_threshold",
                "raw_path": raw,
                "path": _resolve_image(raw),
                "extras": extras,
            }
        )
    return views


def _open_image(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606 - opens in the default image viewer
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        print(f"    (couldn't open {path.name}: {exc})")


def _open_themes(views: list[dict], *, only_flagged: bool) -> int:
    opened = 0
    for v in views:
        if only_flagged and not v["flagged"]:
            continue
        if v["path"] is not None:
            _open_image(v["path"])
            opened += 1
    return opened


def _prompt_decision(url: str, row: dict, open_mode: str) -> tuple[str, str]:
    """
    Returns (decision, note). `open_mode` is "flagged" (default: open only
    the images Agent 5 selected below its threshold), "all", or "none".
    Loops until a recognized answer is given; `v` re-opens every selected
    image, flagged or not.
    """
    selection, load_error = _load_selection(url)
    views = _theme_views(selection) if selection is not None else None

    print(f"\n{url}")
    print(f"  themes: {row.get('themes_completed') or '-'}   images: {row.get('total_images') or '0'}")

    if views is None:
        # Reviewing without being able to see anything is exactly the thing
        # this gate exists to prevent, so say so loudly rather than quietly
        # falling back to a bare flag string.
        print(f"  ⚠ couldn't load Agent 5's selection ({load_error}) - no images to show")
        flags = (row.get("selection_flags") or "").strip()
        if flags:
            print(f"  ⚠ below-threshold themes (from manifest): {flags}")
    else:
        for v in views:
            mark = "⚠ BELOW THRESHOLD" if v["flagged"] else "ok"
            print(f"  [{mark}] {v['setting']}")
            if v["path"] is not None:
                print(f"      image: {v['path']}")
            elif v["raw_path"]:
                print(f"      image: {v['raw_path']}  (file not found on disk)")
            else:
                print("      no image selected for this theme")
            for key, value in v["extras"].items():
                print(f"      {key}: {value}")

        if open_mode != "none":
            n = _open_themes(views, only_flagged=(open_mode == "flagged"))
            if n:
                label = "flagged" if open_mode == "flagged" else "selected"
                print(f"  (opened {n} {label} image(s) in your default viewer)")
            elif open_mode == "flagged" and not any(v["flagged"] for v in views):
                print("  (no themes flagged - nothing auto-opened; type v to view them anyway)")

    while True:
        answer = input("  Approve for video generation? [y/n/v=view all images/s=skip for now]: ").strip().lower()
        if answer in ("y", "yes"):
            return "approved", ""
        if answer in ("n", "no"):
            note = input("  Reason (optional): ").strip()
            return "rejected", note
        if answer in ("v", "view"):
            if views is None:
                print("  No selection data to open.")
            elif _open_themes(views, only_flagged=False) == 0:
                print("  None of the selected image files could be found on disk.")
            continue
        if answer in ("s", "skip", ""):
            return "skipped", ""
        print("  Please answer y, n, v, or s.")


def _load_existing_decisions(path: Path) -> dict[str, str]:
    """url -> last recorded decision from an earlier session (empty if none)."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    decisions: dict[str, str] = {}
    for row in read_manifest_rows(path):
        url = (row.get("url") or "").strip()
        if url:
            decisions[url] = (row.get("decision") or "").strip().lower()
    return decisions


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
    parser.add_argument(
        "--open-all",
        action="store_true",
        help="Auto-open every selected image, not just the below-threshold ones",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Never auto-open images (e.g. over SSH). `v` at the prompt still tries to.",
    )
    args = parser.parse_args()
    open_mode = "none" if args.no_open else "all" if args.open_all else "flagged"

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

    approved = rejected = skipped = auto = already = 0
    earlier = _load_existing_decisions(out_path)

    with ApprovalsWriter(out_path) as writer:
        for row in rows:
            url = (row.get("url") or "").strip()
            trace_id = (row.get("trace_id") or "").strip()
            flags = (row.get("selection_flags") or "").strip()

            # Resume: a decision made in an earlier session stands. Products
            # previously *skipped* are asked again - that's what skip means.
            if earlier.get(url) in ("approved", "rejected"):
                print(f"\n{url}\n  already {earlier[url]} in an earlier session - keeping that decision")
                already += 1
                continue

            if args.auto_approve_clean and not flags:
                decision, note = "approved", "auto-approved (no selection_flags)"
                print(f"\n{url}\n  all themes cleared threshold - auto-approved")
                auto += 1
            else:
                decision, note = _prompt_decision(url, row, open_mode)
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
    print(f"Approved: {approved} (+{auto} auto)   Rejected: {rejected}   Skipped: {skipped}"
          + (f"   Kept from earlier session: {already}" if already else ""))
    print(f"Approvals file: {out_path}")
    if approved or auto or already:
        print(
            "\nNext:\n"
            f"  python scripts/generate_videos.py --from-results {manifest_path} "
            f"--approvals {out_path}"
        )


if __name__ == "__main__":
    main()