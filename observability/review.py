"""
The human-in-the-loop gate between Agent 5 (Image Selection) and
Agent 6 (Video Generation).

Walks every `status=ok` row in a batch results manifest, shows the image
Agent 5 picked for each theme, and records a decision **per theme**:

    y  approve  -> animate Agent 5's pick
    n  replace  -> show every other candidate Agent 4 generated for that
                   theme and let you choose the frame to animate instead
    s  skip     -> no video for this theme (the other themes still run)
    a  approve all remaining themes of this product

Rejecting one image therefore never drops the rest of the product: the
approvals CSV carries the approved set, and `generate_videos.py` passes it
into Agent 6, which skips only the themes you skipped and uses your chosen
image where you replaced Agent 5's.

Writes an approvals CSV that `generate_videos.py --approvals <this file>`
reads. Columns are `url,theme,decision,note,image_path,reviewed_at,trace_id`;
`theme` is the theme's `source_setting`, not a list index, because Agents 4/5
omit themes they skipped and positions do not stay aligned. An empty `theme`
is a product-level decision, which is how old (pre-per-theme) approvals files
are migrated in place - those still work, and are read as "all themes".

Decisions are written and flushed **per theme**, same reasoning as
`ManifestWriter`: reviewing 50 products is itself a long task you might not
finish in one sitting, and a half-finished review session should leave a
usable partial approvals file behind, not nothing.

If a trace_id is present for a row (Langfuse configured when the batch
ran), one `human_approval` score is pushed to that trace per product - 1 if
at least one theme was approved/replaced, 0 otherwise. This is what actually
closes the loop between "a human looked at this" and the observability data:
the same trace that recorded what the pipeline did now also records what a
person decided about it.

Usage:
    python scripts/review.py outputs/batch_results_<ts>.csv
    python scripts/review.py outputs/batch_results_<ts>.csv --out outputs/approvals_oct.csv
    python scripts/review.py outputs/batch_results_<ts>.csv --auto-approve-clean
    python scripts/review.py outputs/batch_results_<ts>.csv --open-all
    python scripts/review.py outputs/batch_results_<ts>.csv --no-open

`--open-all` opens every selected image as each product comes up,
`--no-open` opens none, and `v` at a theme prompt re-opens that theme's
selected image. When you choose "replace", every candidate for that theme
is printed (and opened, unless `--no-open`) so you can pick by number.

Re-running on the same manifest resumes: themes already decided keep their
decision, and only undecided ones are asked again.

--auto-approve-clean skips the prompt (and records an automatic "approved"
decision) for any theme Agent 5 selected cleanly - i.e. it did not flag the
theme as below threshold. It still prompts for every flagged theme. This
weakens the gate, so it's opt-in, not default.
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

from app.agents.image_generation.schema import ImageGenerationOutput  # noqa: E402
from app.agents.image_selection_agent.schema import ImageSelectionOutput  # noqa: E402
from observability import observability as obs  # noqa: E402
from app.core.checkpoint import load_stage  # noqa: E402
from app.core.manifest import read_manifest_rows  # noqa: E402

APPROVAL_COLUMNS = [
    "url",
    "theme",
    "decision",
    "note",
    "image_path",
    "reviewed_at",
    "trace_id",
]

# Decisions that mean "this theme gets a video".
_ACCEPTED = ("approved", "replaced")


class ApprovalsWriter:
    """Same per-row-flush shape as ManifestWriter, different columns.

    Opens in APPEND mode when the file already has content (header written
    only the first time). The default approvals path is derived from the
    manifest's name, so a second review session on the same manifest lands
    on the same file - truncating it would silently erase every decision
    from the first session. Appending is safe because
    `generate_videos.py`'s approvals loader already treats the last row for
    a URL+theme as the decision.

    Old product-level files (no `theme` column) are migrated in place first,
    so one file can hold both old and new decisions without ambiguity.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self._writer = None

    def __enter__(self) -> "ApprovalsWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _migrate_legacy_file(self.path)
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

    def write(
        self,
        *,
        url: str,
        theme: str,
        decision: str,
        note: str,
        image_path: str,
        trace_id: str,
    ) -> None:
        self._writer.writerow(
            {
                "url": url,
                "theme": theme,
                "decision": decision,
                "note": note,
                "image_path": image_path,
                "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "trace_id": trace_id,
            }
        )
        self._fh.flush()


def _migrate_legacy_file(path: Path) -> None:
    """Rewrite a pre-per-theme approvals file (url,decision,...) in the new
    schema with theme="" - a product-level decision covering every theme."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if "theme" in fields:
            return
        rows = list(reader)

    tmp = path.with_name(path.name + ".migrating")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=APPROVAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: (row.get(column) or "") for column in APPROVAL_COLUMNS}
            )
    tmp.replace(path)


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


def _load_images(url: str):
    """Agent 4's checkpointed candidates for this URL. Returns (images, error)."""
    try:
        loaded = load_stage(url, "images", ImageGenerationOutput)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if loaded is None:
        return None, "no images checkpoint found for this URL"
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
                extras[key] = (
                    value
                    if not isinstance(value, str)
                    else (value[:200] + ("..." if len(value) > 200 else ""))
                )
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


def _candidates_for(images, view: dict) -> list[dict]:
    """Every image Agent 4 generated for one theme, in generation order.

    Preferred source is Agent 4's checkpoint, which carries the exact list.
    That stage is checkpointed by LangGraph, not by the JSON checkpoint the
    other stages use, so `load_stage(..., "images")` usually returns None -
    in which case the selected image's own folder is the source of truth:
    Agent 4 saves all candidates for a theme into one `theme_N` directory,
    so its siblings ARE the candidate set.
    """
    found: list[dict] = []
    seen: set[Path] = set()

    def add(raw: str, filename: str = "") -> None:
        resolved = _resolve_image(raw)
        key = resolved or Path(raw)
        if key in seen:
            return
        seen.add(key)
        found.append(
            {
                "raw": raw,
                "path": resolved,
                "filename": filename or Path(raw).name,
            }
        )

    if images is not None:
        for theme in images.theme_results:
            if theme.source_setting != view["setting"]:
                continue
            for img in theme.images:
                add(img.local_path, img.filename)
            if found:
                return found

    selected = view.get("path")
    if selected is not None and selected.parent.is_dir():
        for candidate in sorted(selected.parent.iterdir()):
            if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                add(str(candidate))

    return found


def _same_image(a: dict, b: dict) -> bool:
    """Compare by resolved path, since one side may be relative and the
    other absolute (checkpoint path vs. disk listing)."""
    return a.get("path") is not None and a.get("path") == b.get("path")


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


def _choose_candidate(view: dict, candidates: list[dict], *, open_all: bool) -> dict:
    """Show every Agent 4 candidate for one theme and let the reviewer pick."""
    if not candidates:
        print("    Agent 4's candidates for this theme aren't on disk - only approve or skip")
        while True:
            answer = input("    [y=keep Agent 5's pick, s=skip theme]: ").strip().lower()
            if answer in ("y", "yes", ""):
                return {"decision": "approved", "image_path": "", "note": ""}
            if answer in ("s", "skip"):
                return {"decision": "rejected", "image_path": "", "note": "no candidate images available"}
            print("    Please answer y or s.")

    print("    candidates:")
    for i, candidate in enumerate(candidates, start=1):
        mark = "   <- Agent 5's pick" if _same_image(candidate, view) else ""
        shown = candidate["path"] or f"{candidate['raw']}  (missing on disk)"
        print(f"      [{i}] {shown}{mark}")

    if open_all:
        for candidate in candidates:
            if candidate["path"] is not None:
                _open_image(candidate["path"])

    while True:
        answer = input(
            f"    Pick 1-{len(candidates)} [b=keep Agent 5's pick, s=skip theme]: "
        ).strip().lower()
        if answer in ("b", "back", ""):
            return {"decision": "approved", "image_path": "", "note": ""}
        if answer in ("s", "skip"):
            return {"decision": "rejected", "image_path": "", "note": "rejected at review"}
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            chosen = candidates[int(answer) - 1]
            if chosen["path"] is None:
                print("    that file is missing on disk - pick another")
                continue
            if _same_image(chosen, view):
                return {"decision": "approved", "image_path": "", "note": ""}
            return {
                "decision": "replaced",
                "image_path": chosen["raw"],
                "note": f"manually chose candidate {answer}",
            }
        print(f"    enter 1-{len(candidates)}, b, or s")


def _review_theme(
    index: int,
    total: int,
    view: dict,
    candidates: list[dict],
    *,
    open_all: bool,
    auto_approve_clean: bool,
) -> dict:
    """One theme's decision: approved, replaced, rejected, or approve_all."""
    mark = "⚠ BELOW THRESHOLD" if view["flagged"] else "ok"
    print(f"\n  theme {index}/{total}: {view['setting']}")
    print(f"    [{mark}]")
    if view["path"] is not None:
        print(f"    Agent 5 picked: {view['path']}")
    elif view["raw_path"]:
        print(f"    Agent 5 picked: {view['raw_path']}  (file not found on disk)")
    else:
        print("    Agent 5 selected no image for this theme")
    for key, value in view["extras"].items():
        print(f"    {key}: {value}")

    if auto_approve_clean and not view["flagged"]:
        print("    cleared threshold - auto-approved")
        return {
            "decision": "approved",
            "image_path": "",
            "note": "auto-approved (not flagged)",
        }

    while True:
        answer = input(
            "    [y=approve, n=choose another image, v=view pick, s=skip theme, a=approve all remaining]: "
        ).strip().lower()
        if answer in ("y", "yes", ""):
            return {"decision": "approved", "image_path": "", "note": ""}
        if answer in ("n", "no"):
            return _choose_candidate(view, candidates, open_all=open_all)
        if answer in ("v", "view"):
            if view["path"] is not None:
                _open_image(view["path"])
            else:
                print("    Agent 5's image isn't available on disk")
            continue
        if answer in ("s", "skip"):
            return {
                "decision": "rejected",
                "image_path": "",
                "note": "skipped at review",
            }
        if answer in ("a", "all"):
            return {"decision": "approve_all", "image_path": "", "note": ""}
        print("    Please answer y, n, v, s, or a.")


def _load_existing_decisions(path: Path) -> dict[tuple[str, str], str]:
    """(url, theme) -> last recorded decision. theme "" is product-level."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    decisions: dict[tuple[str, str], str] = {}
    for row in read_manifest_rows(path):
        url = (row.get("url") or "").strip()
        theme = (row.get("theme") or "").strip()
        if url:
            decisions[(url, theme)] = (row.get("decision") or "").strip().lower()
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human review gate: approve/replace/skip each theme's selected image before video generation"
    )
    parser.add_argument("manifest", help="A results manifest CSV from scripts/run_batch.py")
    parser.add_argument("--out", default=None, help="Approvals CSV path (default: outputs/approvals_<ts>.csv)")
    parser.add_argument(
        "--auto-approve-clean",
        action="store_true",
        help="Skip the prompt for themes Agent 5 did not flag - still prompts for every flagged theme",
    )
    parser.add_argument(
        "--open-all",
        action="store_true",
        help="Auto-open every selected image (and every candidate when replacing)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Never auto-open images (e.g. over SSH). Paths are still printed.",
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
    _migrate_legacy_file(out_path)
    print(f"{len(rows)} product(s) to review. Writing decisions to {out_path}")
    print(obs.startup_check() + "\n")

    approved = replaced = rejected = auto = already = 0
    earlier = _load_existing_decisions(out_path)

    with ApprovalsWriter(out_path) as writer:
        for row in rows:
            url = (row.get("url") or "").strip()
            trace_id = (row.get("trace_id") or "").strip()

            # A product-level decision from an old session covers every theme.
            product_decision = earlier.get((url, ""))
            if product_decision in ("approved", "rejected"):
                print(
                    f"\n{url}\n  already {product_decision} in an earlier session "
                    "(product-level) - keeping that decision"
                )
                already += 1
                continue

            selection, load_error = _load_selection(url)
            if selection is None:
                # Can't enumerate themes without Agent 5's output. Fall back
                # to the old product-level question rather than guessing.
                print(f"\n{url}")
                print(f"  ⚠ couldn't load Agent 5's selection ({load_error}) - no images to show")
                flags = (row.get("selection_flags") or "").strip()
                if flags:
                    print(f"  ⚠ below-threshold themes (from manifest): {flags}")
                while True:
                    answer = input(
                        "  Approve this product for video generation? [y/n/s=skip for now]: "
                    ).strip().lower()
                    if answer in ("y", "yes"):
                        decision, note = "approved", "approved without selection data"
                        break
                    if answer in ("n", "no"):
                        note = input("  Reason (optional): ").strip()
                        decision = "rejected"
                        break
                    if answer in ("s", "skip", ""):
                        decision, note = "skipped", ""
                        break
                    print("  Please answer y, n, or s.")
                if decision != "skipped":
                    writer.write(
                        url=url,
                        theme="",
                        decision=decision,
                        note=note,
                        image_path="",
                        trace_id=trace_id,
                    )
                    if decision == "approved":
                        approved += 1
                    else:
                        rejected += 1
                    if trace_id:
                        obs.score_trace_by_id(
                            trace_id,
                            "human_approval",
                            1 if decision == "approved" else 0,
                            comment=note or None,
                        )
                continue

            views = _theme_views(selection)
            # Agent 4's stage is LangGraph-checkpointed, not JSON-checkpointed,
            # so this is usually None. _candidates_for falls back to the
            # selected image's folder, which holds every candidate anyway.
            images, _ = _load_images(url)

            undecided = [
                v
                for v in views
                if earlier.get((url, v["setting"])) not in ("approved", "replaced", "rejected")
            ]
            if not undecided:
                print(f"\n{url}\n  every theme already reviewed in an earlier session - keeping those decisions")
                already += 1
                continue

            print(f"\n{url}")
            print(f"  themes: {row.get('themes_completed') or '-'}   images: {row.get('total_images') or '0'}")
            if open_mode != "none":
                n = _open_themes(views, only_flagged=(open_mode == "flagged"))
                if n:
                    label = "flagged" if open_mode == "flagged" else "selected"
                    print(f"  (opened {n} {label} image(s) in your default viewer)")
                elif open_mode == "flagged" and not any(v["flagged"] for v in views):
                    print("  (no themes flagged - nothing auto-opened; choose n to see all candidates)")

            approve_rest = False
            product_accepted = False

            for index, view in enumerate(views, start=1):
                setting = view["setting"]
                previous = earlier.get((url, setting))
                if previous in ("approved", "replaced", "rejected"):
                    print(f"\n  theme {index}/{len(views)}: {setting}\n    already {previous} - keeping that decision")
                    if previous in _ACCEPTED:
                        product_accepted = True
                    continue

                if approve_rest:
                    outcome = {
                        "decision": "approved",
                        "image_path": "",
                        "note": "approved with the rest of the product",
                    }
                    print(f"\n  theme {index}/{len(views)}: {setting}\n    approved with the rest")
                else:
                    try:
                        outcome = _review_theme(
                            index,
                            len(views),
                            view,
                            _candidates_for(images, view),
                            open_all=(open_mode != "none"),
                            auto_approve_clean=args.auto_approve_clean,
                        )
                    except EOFError:
                        print("\n  Input closed - stopping. Decisions so far are saved.")
                        obs.flush()
                        return

                    if outcome["decision"] == "approve_all":
                        approve_rest = True
                        outcome = {
                            "decision": "approved",
                            "image_path": "",
                            "note": "approved with the rest of the product",
                        }

                writer.write(
                    url=url,
                    theme=setting,
                    decision=outcome["decision"],
                    note=outcome["note"],
                    image_path=outcome["image_path"],
                    trace_id=trace_id,
                )
                if outcome["decision"] == "approved":
                    if outcome["note"].startswith("auto-approved"):
                        auto += 1
                    else:
                        approved += 1
                    product_accepted = True
                elif outcome["decision"] == "replaced":
                    replaced += 1
                    product_accepted = True
                else:
                    rejected += 1

            if trace_id:
                obs.score_trace_by_id(
                    trace_id,
                    "human_approval",
                    1 if product_accepted else 0,
                    comment=None if product_accepted else "no theme approved",
                )

    obs.flush()

    print(f"\n{'=' * 60}")
    print(
        f"Approved: {approved} (+{auto} auto)   Replaced: {replaced}   "
        f"Rejected: {rejected}   Kept from earlier session: {already}"
    )
    print(f"Approvals file: {out_path}")
    if approved or replaced or auto or already:
        print(
            "\nNext:\n"
            f"  python scripts/generate_videos.py --from-results {manifest_path} "
            f"--approvals {out_path}"
        )


if __name__ == "__main__":
    main()
