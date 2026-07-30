"""Diagnostics on the apparatus. Nothing here asks a question about the world.

WHAT SEPARATES THIS PACKAGE FROM `validation/`. The validation pass asked whether the answers
already recorded can be trusted. These diagnostics ask a narrower and more mechanical question:
**what are the instruments actually measuring, and over what range can they measure it at all.**
Neither fixes anything. Both gate the repair work that comes after.

The programme is P-1 and P-2 from `DIAGNOSTICS_P1_P2_SPEC.md`, plus five checks (D-1 to D-6, with
D-2 promoted out of P-2) that came out of reading the code alongside the validation output. The five
run first and cost almost nothing, because two of them change what the expensive sweeps should
sweep:

  D-1  channel accounting          arithmetic on A. No simulation. Locates the crossover at which
                                   the label stops beating the content, which predicts several
                                   parameter-sweep outcomes before any parameter is swept.
  D-2  the uptake response curve   is uptake monotone in goal recovery? If it is U-shaped, any
                                   manipulation spanning the trough returns a null for reasons
                                   having nothing to do with the manipulation.
  D-3  the disagreement estimator  a plug-in entropy of modal-goal counts, so it carries a known
                                   N-dependent bias and cannot tell disagreement from argmax noise
                                   on near-flat posteriors.
  D-4  solver coverage             how much of the body of work the exact solver can reach, and
                                   how far the approximation drifts where it can be measured.
  D-5  criterion power             how many independent units each pre-registered criterion is
                                   actually computed over.
  D-6  seed independence           whether the per-observer seed function is what it claims.

`DIAGNOSTICS.md` is generated from the verdict files, never from the expectations in the spec.
Nothing outside `results/diagnostics/` is written by anything in here.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = REPO_ROOT / "results" / "diagnostics"
FIG_DIR = REPO_ROOT / "figures" / "diagnostics"


def diagnostics_dir(sub: str | None = None) -> Path:
    p = DIAG_DIR if sub is None else DIAG_DIR / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR
