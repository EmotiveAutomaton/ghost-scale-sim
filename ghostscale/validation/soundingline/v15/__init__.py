"""V15 — The Boundary Map.

Coupling thresholds, model-space expansion, non-oracle learning history, foreground control,
persistent change, strategic sources, robust routing, and change-aware epistemic foraging, run as
a manifest-driven, work-conserving 168-hour card program.

Spec: ``V15_SPEC.md`` at the repository root (immutable handoff; filed beside this version's pages
when the version closes). Nothing here calls a versioned ``run()``. V14 is imported only as
committed numbers read from its ledger (card I01) and as an additive erratum that never overwrites
a V14 file (card I04).

The one question:

    Under which combinations of latent coupling, evidence overlap, scarcity, history, context and
    model misspecification does a reader need a coupled and expandable maker model to predict what
    happens next -- and when is a cheaper factorized reader equally good?

Verdict layout (``results/validation/soundingline/v15/``)::

    <CARD>.json                 discovery-lane verdict of record
    transfer/<CARD>.json        transfer-lane verdict (fresh families, vocabularies, ecologies)
    attacks/<XNN>.json          adversarial-matrix verdicts (run on the transfer lineage)
    confirmation/<CARD>.json    confirmation-lane verdict of a frozen promoted card
    coverage/<CELL>.json        balanced coverage-stream cells (machine-readable only)

Pilot and smoke verdicts never enter that tree: the pilot is quarantined under
``results/v15/pilot_quarantine/`` (gitignored) and smoke passes go to a scratch directory.

House rules that this package enforces mechanically, each of them paid for by an earlier version:

* seeds derive from ``zlib.crc32`` of a named string, never ``hash()``;
* a lineage name always contains its lane, so no object generated in one lane shares an ancestor
  with another lane;
* a gate records and a test fails -- gates never raise;
* **a gate bar is never a criterion bar.** A live/positive/prediction/oracle gate asks whether an
  effect exists at all (bar 0); the pre-registered magnitude lives only in the criterion. V14 lost
  three cards to that conflation and had to record an instrument repair mid-window;
* ``DONE`` is not a state.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
V15_RESULTS = REPO / "results" / "v15"
V15_VERDICTS = REPO / "results" / "validation" / "soundingline" / "v15"
SEED_OFFSET_V15 = 1_500_000

#: Where each lane's verdicts go. ``pilot`` and ``smoke`` are overridden by the runner.
LANE_SUBDIR = {"discovery": "", "transfer": "transfer", "attack": "attacks",
               "confirmation": "confirmation", "coverage": "coverage"}
PILOT_QUARANTINE = V15_RESULTS / "pilot_quarantine" / "verdicts"
SMOKE_DIR = V15_RESULTS / "smoke" / "verdicts"


def v15_dir(sub: str | None = None) -> Path:
    d = V15_RESULTS if sub is None else V15_RESULTS / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def verdict_dir(lane: str = "discovery") -> Path:
    if lane == "pilot":
        d = PILOT_QUARANTINE
    elif lane == "smoke":
        d = SMOKE_DIR
    else:
        sub = LANE_SUBDIR[lane]
        d = V15_VERDICTS / sub if sub else V15_VERDICTS
    d.mkdir(parents=True, exist_ok=True)
    return d
