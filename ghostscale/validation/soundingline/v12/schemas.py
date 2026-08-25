"""Card, manifest, and verdict schemas for the V12 program (spec §5, §21.2).

A card is the unit of the queue. Its state machine is the spec's: PLANNED, BUILT,
INSTRUMENT_VALID, RUNNING, LANDED, INSTRUMENT_FAILED, SCIENTIFIC_CLOSED, RESOURCE_BLOCKED. DONE is
forbidden and the validator rejects it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

STATES = ("PLANNED", "BUILT", "INSTRUMENT_VALID", "RUNNING", "LANDED",
          "INSTRUMENT_FAILED", "SCIENTIFIC_CLOSED", "RESOURCE_BLOCKED")
RESOLVED = ("LANDED", "INSTRUMENT_FAILED", "SCIENTIFIC_CLOSED", "RESOURCE_BLOCKED")
CLAIM_CEILINGS = ("METHOD", "CONSTRUCTED_MECHANISM", "BOUNDARY", "INSTRUMENT_FAILURE", "VOID")
PURSUIT = ("OPENED", "PROMISING", "STALLED", "EXHAUSTED", "PROMOTE")
WARRANT = ("DESCRIPTIVE_ONLY", "ANALOGUE_EVIDENCE", "MECHANISM_CANDIDATE",
           "CONFIRMATORY_SUPPORT", "BOUNDARY_ESTABLISHED", "INSTRUMENT_FAILED", "VOID")

# Floors from spec §5.1. A card may raise them; the validator refuses a card that lowers them
# without a curator amendment recorded in the manifest.
FLOORS = {
    "seeds_per_condition": 3,
    "discovery_worlds": 12,
    "confirmation_worlds": 12,
    "makers_per_comparison": 60,
    "artifact_prefixes": [1, 2, 4, 8, 12, 20, 50],
    "artifact_lengths": [2, 4, 8, 12, 24],
    "domains": 2,
}


@dataclass
class Card:
    id: str
    trunk: str
    wave: int
    question: str
    construction: str
    target: str
    independent_unit: str
    primary_estimand: str
    null_expectation: str
    alternative_expectation: str
    strongest_rival: str
    solver_paths: list = field(default_factory=lambda: ["exact"])
    gates_required: list = field(default_factory=lambda: ["live", "positive", "placebo"])
    factors: dict = field(default_factory=dict)
    floors: dict = field(default_factory=lambda: dict(FLOORS))
    discovery_worlds: list = field(default_factory=list)
    confirmation_worlds: list = field(default_factory=list)
    depends_on: list = field(default_factory=list)
    status: str = "PLANNED"
    module: str = ""
    output: str = ""
    completion_marker: str = ""
    estimated_cpu_minutes: float = 0.0
    actual_cpu_minutes: float | None = None
    closure_reason: str = ""
    repairs_used: int = 0
    amendments: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def card_from_dict(d: dict) -> Card:
    return Card(**{k: v for k, v in d.items() if k in Card.__dataclass_fields__})


def new_verdict(card: Card, lane: str, hypothesis: str, claim_ceiling: str) -> dict:
    """The skeleton every V12 verdict must fill (spec §21.2)."""
    assert claim_ceiling in CLAIM_CEILINGS, claim_ceiling
    assert lane in ("discovery", "confirmation", "both"), lane
    return {
        "card": card.id, "trunk": card.trunk, "wave": card.wave,
        "question": card.question, "hypothesis": hypothesis,
        "claim_ceiling": claim_ceiling, "lane": lane,
        "construction_realization": {}, "results": {}, "exact": {}, "pymdp": {}, "baseline": {},
        "cell_matrix": {}, "independent_unit": card.independent_unit, "effective_n": {},
        "coverage": {}, "pursuit": "OPENED", "warrant": "DESCRIPTIVE_ONLY",
        "strongest_rival": card.strongest_rival,
        "what_must_hold_outside_the_simulation": "",
        "deviations": [], "repairs": [], "runtime_seconds": None,
    }
