"""The validation pass — V-1 through V-9 of `docs/specs/SPEC_VALIDATION.md`.

WHAT THIS PACKAGE IS FOR, AND WHAT IT IS NOT FOR.

Nothing in here asks a new question about the world. Every module asks whether an answer already
recorded in `results/` can be trusted, and each one is built so it CAN RETURN THE UNWELCOME
ANSWER. That is not a slogan: the pass/fail wording of each check was written before its run,
the criteria live in `criteria.py` rather than in the code that scores them, and a check whose
threshold had to be chosen after seeing a measurement says so in its own verdict file.

The thing being defended against is specific. This is exploratory modelling and all of it is
confirmatory by construction: every prediction came from one prior theory, and the simulations
formalise that theory and test whether its parts fit together. Agreement between model and theory
is therefore the EXPECTED outcome rather than evidence for it, and a model can reproduce its own
assumptions while being indistinguishable, from outside, from a model that discovered something.

The project's own history says the risk is live rather than theoretical. Seven times an instrument
was answering a different question than the one being asked, and every time the wrong answer
looked completely reasonable. Six of those were caught by checks written for other reasons. This
package makes the checking systematic instead of lucky.

Each module writes one verdict JSON into `results/validation/` and nothing else. `VALIDATION.md`
is written FROM those files, never from the expectations in the spec.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = REPO_ROOT / "results" / "validation"


def validation_dir() -> Path:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    return VALIDATION_DIR
