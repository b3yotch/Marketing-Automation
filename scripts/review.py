"""Compatibility entry point for the human review gate.

The implementation lives in `observability/review.py` next to the rest of
the observability code, but every doc, error message and `run_batch.py`
summary refers to `python scripts/review.py <manifest>`. Rather than
duplicating the script (or leaving that documented command broken), this
wrapper runs the real module as `__main__` so both paths behave identically.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

runpy.run_module("observability.review", run_name="__main__")
