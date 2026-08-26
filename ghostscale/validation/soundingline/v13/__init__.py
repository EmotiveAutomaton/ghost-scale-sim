"""V13 — Common Ground. Nested maker–reader priors, attention allocation, opportunity and cost,
projection correction, communicative goals, trust gates, and hierarchical production, run as a
manifest-driven, tier-calibrated, multi-day card program.

Spec: V13_SPEC.md at the repository root (immutable handoff; filed beside this version's pages
when the version closes) and docs/versions/v13-common-ground/SPEC.md (implementation
translation). Nothing here calls a versioned run(); V12 objects are imported as ordinary
functions only inside the anchor-reproduction card (I01) and the comparator audit (I02).

Verdict layout (results/validation/soundingline/v13/):
    <CARD>.json                 discovery-lane verdict of record
    transfer/<CARD>.json        transfer-lane verdict (fresh families, domains, ecologies)
    attacks/<XNN>.json          adversarial-matrix verdicts (run on the transfer lineage)
    confirmation/<CARD>.json    confirmation-lane verdict of a promoted card
Pilot and smoke verdicts never enter this tree: the pilot is quarantined under
results/v13/pilot_quarantine/ (ignored by git) and smoke passes go to a scratch directory.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
V13_RESULTS = REPO / "results" / "v13"
V13_VERDICTS = REPO / "results" / "validation" / "soundingline" / "v13"
SEED_OFFSET_V13 = 1_300_000

#: Where each lane's verdicts go. ``pilot`` and ``smoke`` are overridden by the runner to
#: quarantine or scratch locations; the defaults below keep them out of the committed tree too.
LANE_SUBDIR = {"discovery": "", "transfer": "transfer", "attack": "attacks",
               "confirmation": "confirmation"}
PILOT_QUARANTINE = V13_RESULTS / "pilot_quarantine" / "verdicts"
SMOKE_DIR = V13_RESULTS / "smoke" / "verdicts"


def v13_dir(sub: str | None = None) -> Path:
    d = V13_RESULTS if sub is None else V13_RESULTS / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def verdict_dir(lane: str = "discovery") -> Path:
    if lane == "pilot":
        d = PILOT_QUARANTINE
    elif lane == "smoke":
        d = SMOKE_DIR
    else:
        sub = LANE_SUBDIR[lane]
        d = V13_VERDICTS / sub if sub else V13_VERDICTS
    d.mkdir(parents=True, exist_ok=True)
    return d
