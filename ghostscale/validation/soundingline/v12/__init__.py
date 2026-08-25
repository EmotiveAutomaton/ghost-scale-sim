"""V12 — The Other Model. Self-based maker inversion, projection correction, active
interrogation, and selective uptake, run as a manifest-driven card program.

Spec: V12_SPEC.md at the repository root (immutable handoff) and
docs/versions/v12-the-other-model/SPEC.md (implementation translation). Nothing here calls a
versioned run(); V11 primitives are imported as ordinary functions from ghostscale.v11.maker and
reproduced (card I01) before any V12 science is scored.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
V12_RESULTS = REPO / "results" / "v12"
V12_VERDICTS = REPO / "results" / "validation" / "soundingline" / "v12"
SEED_OFFSET_V12 = 1_200_000


def v12_dir(sub: str | None = None) -> Path:
    d = V12_RESULTS if sub is None else V12_RESULTS / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def verdict_dir(sub: str | None = None) -> Path:
    d = V12_VERDICTS if sub is None else V12_VERDICTS / sub
    d.mkdir(parents=True, exist_ok=True)
    return d
