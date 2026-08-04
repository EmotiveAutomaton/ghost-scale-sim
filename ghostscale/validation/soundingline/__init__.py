"""Tests run in this simulation on behalf of another project.

WHAT THIS DIRECTORY IS, AND WHY IT IS NOT A VERSION.

Sounding Line is a separate project that reads real text. It cannot construct ground truth: the
only thing it knows about a corpus is a label somebody guessed. This simulation can construct
ground truth, so Sounding Line sends it questions about MECHANISM and this package answers them.

These are therefore **not Ghost Scale hypotheses**. Nothing here is pre-registered against this
project's prediction card, nothing here decides a Ghost Scale claim, and none of it should be
quoted as a result about readers. S-1 asks whether a statistic in somebody else's pipeline is
broken. That is a legitimate thing to use a simulator for and it is not a finding about people.

**THIS IS THE MOST LIVING PART OF THIS REPOSITORY.** Everything in `ghostscale/v1` through `v10`
is closed: pre-registered, run, reported, and left alone. This directory is the opposite. It gets
new modules whenever the next project needs one, its questions arrive from outside, and it is
expected to churn. Read anything here as work in progress rather than as a settled result.

TWO RULES, BOTH LEARNED THE HARD WAY IN THIS REPOSITORY.

  1  NEVER CALL A VERSIONED ``run()``. The V10 severity pass re-ran real experiments to audit them
     and those experiments wrote their verdicts to the canonical paths, so the pass whose entire
     job was checking the headlines was overwriting the headlines. Modules here reimplement the
     inner loop they need and write only under ``results/validation/soundingline/``.

  2  IF YOU CLAIM TO USE AN EXPERIMENT'S ROLLOUTS, REPRODUCE ITS NUMBER FIRST. V-11a compared a
     control against E36 and only became trustworthy once its uncontrolled arm returned E36's
     committed gain to every digit. A harness that re-randomises is not running a control, it is
     running a different experiment.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "validation" / "soundingline"


def sl_dir(sub: str | None = None) -> Path:
    """The only directory anything in this package may write to."""
    d = OUT if sub is None else OUT / sub
    d.mkdir(parents=True, exist_ok=True)
    return d
