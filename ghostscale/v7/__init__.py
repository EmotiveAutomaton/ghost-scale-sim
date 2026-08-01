"""V7 — closing what version 6 held back, and attacking E21 on the right axis.

TWO JOBS.

Version 6's walkthrough deliberately left four findings undrawn, because a picture makes a claim
hard to qualify and each of those four carried an open question. Holding them was right. Leaving
them held is not, so C-1 through C-4 close them.

And E21 gets attacked properly. E21 is the experiment that made this project withdraw a claim, and
the objection to it is specific: **you use your own architecture to simulate the maker's, which
cheats the solution space, and that is not something small-sample overfitting does.**

That objection is correct about what E21 did not test. E21 asked whether a reader with no
maker-model can produce the confident-and-contradictory SIGNATURE. It can. It never asked what the
maker-model BUYS -- and "not necessary to produce a signature" is a much weaker claim than "not
necessary", which is how the result has been stated.

If simulation is an efficiency device, and an active-inference account says it must be because
nature does not pay for machinery that buys nothing, then its advantage was never going to show up
in the signature. It shows up in how much evidence you need, and in whether you can read something
you have never seen before. Neither had been measured.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V7_DIR = REPO_ROOT / "results" / "v7"
PREREG_PATH = V7_DIR / "preregistration_v7.json"

SEED_OFFSET = int(os.environ.get("GHOSTSCALE_V7_SEED_OFFSET", "0"))


def v7_dir(sub: str | None = None) -> Path:
    p = V7_DIR if sub is None else V7_DIR / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
