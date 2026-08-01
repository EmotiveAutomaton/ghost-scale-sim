"""V8 — the reader gets a mind of its own.

Every version until now modelled a reader that INFERS a maker without ever BEING one. It has no
hierarchy of its own, nothing it does costs it anything except looking, and nothing it learns ever
fades. V8 gives it all three, and then asks the question those three make possible for the first
time: whether reading and making are the same machinery, so that appreciating something installs
the capability to produce it.

It also runs the severity check the programme has been missing, builds a maker that can lie, and
takes the readymade seriously.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V8_DIR = REPO_ROOT / "results" / "v8"
PREREG_PATH = V8_DIR / "preregistration_v8.json"

SEED_OFFSET = int(os.environ.get("GHOSTSCALE_V8_SEED_OFFSET", "0"))


def v8_dir(sub: str | None = None) -> Path:
    p = V8_DIR if sub is None else V8_DIR / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
