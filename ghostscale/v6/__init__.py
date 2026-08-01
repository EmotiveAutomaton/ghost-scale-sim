"""V6 — aligning the simulation to the Intent Extraction Limit.

Every previous version asked a new question about the world. This one asks whether the
simulation and the theory it claims to implement are the same object, and then builds the six
extensions and two corrections that came out of reading one against the other.

THE GOVERNING DESIGN RULE, because V6 adds more at once than any previous version. Every
addition is INDEPENDENTLY SWITCHABLE AND OFF BY DEFAULT. With every switch off a V6 run
reproduces V5 elementwise (null N23), so any single addition is attributable on its own and the
pre-mortem's "too many changes to attribute anything" failure is closed by construction rather
than argued about afterwards.

Output goes to ``results/v6/`` and nowhere else, exactly as the three audit passes did, so a V6
run can never overwrite or be mistaken for the committed V1-V5 record.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V6_DIR = REPO_ROOT / "results" / "v6"
FIG_DIR = REPO_ROOT / "figures" / "v6"
PREREG_PATH = V6_DIR / "preregistration_v6.json"


def v6_dir(sub: str | None = None) -> Path:
    p = V6_DIR if sub is None else V6_DIR / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


# A whole-programme seed offset, for the robustness check. Every V6 experiment derives its per
# observer seeds from a literal base plus this, so setting it re-randomises every reader in every
# cell at once and nothing else. That is the V6 analogue of the validation pass's disjoint seed
# block, and it exists because "the result held on the seeds we happened to pick" is not a result.
SEED_OFFSET = int(os.environ.get("GHOSTSCALE_V6_SEED_OFFSET", "0"))
