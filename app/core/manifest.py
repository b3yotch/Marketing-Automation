"""
The results manifest - one CSV row per product, written by both entry
points.

Deliberately one schema for both `run_batch.py` (which stops at Agent 5)
and `generate_videos.py` (which goes all the way through Agent 6), rather
than two shapes. `videos_generated` is simply blank on a run that never
attempted video, which keeps the two files diffable/concatenable in pandas
without reconciling columns. The cost is one mostly-empty column in the
batch manifest; the benefit is that "what happened to this URL" has one
answer shape no matter which command produced it.

Rows are written and flushed **as each product finishes**, not buffered
until the end. A 50-row batch is ~8 hours of wall clock and will
occasionally be interrupted (Ctrl-C, a reboot, a power cut); buffering
would mean an interrupt at row 47 loses the record of rows 1-46 even
though all their work is safely checkpointed on disk. Flushing per row
also means you can `tail -f` or open the CSV mid-run to see progress
without attaching to the terminal, which is most of what question 3 (see
the design discussion) actually wanted.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from app.core.pipeline import ProductRunResult

COLUMNS = [
    "url",
    "status",           # ok | failed
    "stage_reached",    # last stage that completed
    "error",            # populated only on failure
    "themes_completed", # "2/3"
    "total_images",
    "selection_flags",  # themes that came back selected_below_threshold
    "videos_generated", # blank when video was never attempted
    "elapsed_seconds",
    "finished_at",      # UTC, so a long batch's row ordering is auditable
]


class ManifestWriter:
    """
    Append-mode CSV writer that flushes after every row.

    Used as a context manager so the file handle closes cleanly on
    KeyboardInterrupt as well as on normal completion.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self._writer = None

    def __enter__(self) -> "ManifestWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._fh.flush()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._fh is not None:
            self._fh.close()

    def write(self, result: ProductRunResult) -> None:
        self._writer.writerow(
            {
                "url": result.url,
                "status": result.status,
                "stage_reached": result.failed_stage or result.stage_reached,
                "error": (result.error or "").replace("\n", " ")[:500],
                "themes_completed": result.themes_fraction,
                "total_images": result.total_images,
                # "; " not "," so the field stays readable even though the
                # csv module would quote a comma-separated one correctly.
                "selection_flags": "; ".join(result.selection_flags),
                "videos_generated": (
                    "" if result.videos_generated is None else result.videos_generated
                ),
                "elapsed_seconds": f"{result.total_elapsed_seconds:.1f}",
                "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        self._fh.flush()


def default_manifest_path(project_root: Path, kind: str) -> Path:
    """
    outputs/<kind>_results_<utc timestamp>.csv - timestamped rather than a
    fixed filename so a second batch run doesn't silently overwrite the
    manifest of the first. Pass --out to pin it if you'd rather it did.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return project_root / "outputs" / f"{kind}_results_{stamp}.csv"


def read_urls(source: str, project_root: Path) -> list[str]:
    """
    Accepts either a bare URL or a path to a CSV of them.

    The CSV may have a header row with a `url` column (extra columns are
    ignored, so you can keep notes/SKUs alongside), or be a plain
    single-column list with no header at all - both shapes turn up in
    practice and sniffing for a literal "url" header is enough to tell
    them apart. Blank lines and `#` comments are skipped, and duplicates
    are dropped while preserving first-seen order (re-running a duplicate
    would be a checkpoint hit and therefore harmless, just noise in the
    manifest).
    """
    if source.startswith("http://") or source.startswith("https://"):
        return [source]

    path = Path(source)
    if not path.is_absolute():
        path = (project_root / source) if not path.exists() else path
    if not path.exists():
        raise FileNotFoundError(f"No such URL source: {source}")

    urls: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if r and any(cell.strip() for cell in r)]

    if not rows:
        return []

    header = [c.strip().lower() for c in rows[0]]
    if "url" in header:
        idx = header.index("url")
        body = rows[1:]
    else:
        idx = 0
        body = rows

    for row in body:
        if idx >= len(row):
            continue
        value = row[idx].strip()
        if not value or value.startswith("#"):
            continue
        if value not in urls:
            urls.append(value)

    return urls