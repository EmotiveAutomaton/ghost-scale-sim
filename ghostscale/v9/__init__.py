"""V9 — the minimal-model programme, and the two experiments the literature asked for.

Version 8 asked: keep the shape, throw the settings away, does the finding survive? For two of
three headlines it did, every time -- so those findings come from the SHAPE, and no further
parameter work changes that.

The complementary question has never been asked and it is the one that discriminates: **keep the
settings and strip the shape.** Remove one structural commitment at a time and see which removal
kills the finding. What survives every ablation was never the theory's. What dies to a specific
removal tells you exactly which commitment is load-bearing.

After this the remaining questions are human-subject questions, and this apparatus cannot answer
them.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V9_DIR = REPO_ROOT / "results" / "v9"
PREREG_PATH = V9_DIR / "preregistration_v9.json"

SEED_OFFSET = int(os.environ.get("GHOSTSCALE_V9_SEED_OFFSET", "0"))


def v9_dir(sub: str | None = None) -> Path:
    p = V9_DIR if sub is None else V9_DIR / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
