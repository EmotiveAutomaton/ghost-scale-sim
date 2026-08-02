"""V10 — the reader as a defence, and what rides in anyway.

Every version to nine asked what happens to a reader. This one asks whether reading intent is
itself a DEFENCE, and then whether the defence leaks.

The motivating case is documented: coordinated networks publishing at industrial scale specifically
to be absorbed by models rather than read by people. The structural fact that makes it this
project's object is that those sites attract almost no genuine human traffic -- the artifacts have a
maker whose intent was never to be read by a person.

And it defeats the standard defence by construction, because surface-quality filtering measures
exactly what grooming optimises. That is E40 -- pay more, get less -- run against a data pipeline
instead of a reader.

WHAT THIS PACKAGE MAY NOT CLAIM. The contamination is documented; the claim that observed model
value-drift was caused by it is not, and is not made anywhere here. V10 demonstrates a MECHANISM.
It attributes nothing to anyone.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V10_DIR = REPO_ROOT / "results" / "v10"
PREREG_PATH = V10_DIR / "preregistration_v10.json"

SEED_OFFSET = int(os.environ.get("GHOSTSCALE_V10_SEED_OFFSET", "0"))


def v10_dir(sub: str | None = None) -> Path:
    p = V10_DIR if sub is None else V10_DIR / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
