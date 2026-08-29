"""Card, verdict, ledger, tier, and completion schemas for the V14 program (spec §4, §5, §8, §10).

The card is the unit of the queue and the unit of accounting. Its state machine is the spec's:
PLANNED, BUILT, INSTRUMENT_VALID, RUNNING, LANDED, INSTRUMENT_FAILED, SCIENTIFIC_CLOSED, VOID,
RESOURCE_BLOCKED. DONE is forbidden and the validator rejects it. RESOURCE_BLOCKED never counts
as a scientific negative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

STATES = ("PLANNED", "BUILT", "INSTRUMENT_VALID", "RUNNING", "LANDED",
          "INSTRUMENT_FAILED", "SCIENTIFIC_CLOSED", "VOID", "RESOURCE_BLOCKED")
RESOLVED = ("LANDED", "INSTRUMENT_FAILED", "SCIENTIFIC_CLOSED", "VOID", "RESOURCE_BLOCKED")
CLAIM_CEILINGS = ("METHOD", "CONSTRUCTED_MECHANISM", "BOUNDARY", "INSTRUMENT_FAILURE", "VOID",
                  "RESOURCE_BLOCKED")
PURSUIT = ("OPENED", "PROMISING", "STALLED", "EXHAUSTED", "PROMOTE")
WARRANT = ("DESCRIPTIVE_ONLY", "ANALOGUE_EVIDENCE", "MECHANISM_CANDIDATE", "CONFIRMATORY_SUPPORT",
           "BOUNDARY_ESTABLISHED", "INSTRUMENT_FAILED", "VOID", "RESOURCE_BLOCKED")
LANES = ("pilot", "discovery", "transfer", "attack", "confirmation")
ROUTES = ("action", "semantic", "context", "forensic")
GATE_KINDS_CAUSAL = ("live", "placebo", "positive", "surface", "oracle", "prediction", "calibration")

#: Spec §8.3. ``makers`` is the makers-and-readers count per applicable world.
TIERS = {
    "T0": {"discovery_worlds": 32, "transfer_worlds": 16, "confirmation_worlds": 32, "repeats": 2, "makers": 32},
    "T1": {"discovery_worlds": 64, "transfer_worlds": 32, "confirmation_worlds": 64, "repeats": 3, "makers": 48},
    "T2": {"discovery_worlds": 128, "transfer_worlds": 64, "confirmation_worlds": 96, "repeats": 3, "makers": 64},
    "T3": {"discovery_worlds": 256, "transfer_worlds": 128, "confirmation_worlds": 128, "repeats": 4, "makers": 96},
}
TIER_ORDER = ("T0", "T1", "T2", "T3")
#: Spec §8.3: the smallest tier forecast to occupy 20-21 hours of the 24-hour window.
ENVELOPE_HOURS = (20.0, 21.0)
WINDOW_HOURS = 24.0
FREEZE_HOUR = 20.0

#: Spec §8.3, frozen order of admissible expansion if T3 finishes early.
EXPANSIONS = [
    {"id": "E1", "what": "independent world families and makers", "change": {"discovery_worlds_x": 2, "makers_x": 1.5}},
    {"id": "E2", "what": "transfer action vocabularies and competence ecologies", "change": {"transfer_worlds_x": 2, "extra_vocabularies": 2}},
    {"id": "E3", "what": "stronger surface collisions and equifinal histories", "change": {"equifinal_x": 2}},
    {"id": "E4", "what": "additional evidence-dose and history-decay points", "change": {"dose_points_x": 2}},
    {"id": "E5", "what": "additional foraging ecologies and conflict magnitudes", "change": {"extra_ecologies": 2}},
    {"id": "E6", "what": "confirmation worlds allocated before discovery", "change": {"confirmation_worlds_x": 2}},
]


@dataclass
class Card:
    id: str
    trunk: str
    wave: int
    question: str
    construction: str
    target: str
    estimand: str
    null_expectation: str
    alternative_expectation: str
    strongest_rival: str
    claim_ceiling: str = "CONSTRUCTED_MECHANISM"      # registered ceiling; a verdict may lower it
    solver_paths: list = field(default_factory=lambda: ["exact"])
    depends_on: list = field(default_factory=list)
    closure: str = ""
    causal: bool = True                              # seven gates required (spec §5.2)
    gates_required: list = field(default_factory=lambda: list(GATE_KINDS_CAUSAL))
    factors: dict = field(default_factory=dict)      # factor -> levels, every one a cell axis
    crossings: list = field(default_factory=list)
    domains: int = 2
    world_families: int = 4
    prior_routes: list = field(default_factory=list)
    attention_policies: list = field(default_factory=list)
    cost_ecologies: list = field(default_factory=list)
    reader_policies: list = field(default_factory=list)
    lanes: list = field(default_factory=lambda: ["discovery"])
    independent_unit: str = "world"
    min_effective_n: int = 24
    min_rows_per_unit: int = 1          # cells accumulate per-unit means: one row per cell per unit is the design floor; cards declaring more say so
    unit_kind: str = "world"                         # "world": one unit per (world, repeat); "single"
    work_weight: float = 1.0                         # cost relative to the trunk's pilot card
    pilot: bool = False
    module: str = ""
    output: str = ""
    checkpoint_key: str = ""
    completion_key: str = ""
    hashes: dict = field(default_factory=dict)
    estimated_wall_s: float = 0.0
    actual: dict = field(default_factory=dict)
    status: str = "PLANNED"
    pursuit: str = "OPENED"
    warrant: str = "DESCRIPTIVE_ONLY"
    closure_reason: str = ""
    repairs_used: int = 0
    amendments: list = field(default_factory=list)
    upstream_oracle: bool = False                    # set when a dependency failed and oracle input is used

    def to_dict(self) -> dict:
        return asdict(self)


def card_from_dict(d: dict) -> Card:
    return Card(**{k: v for k, v in d.items() if k in Card.__dataclass_fields__})


def expected_cells(card: Card, tier: dict, lane: str = "discovery") -> dict:
    """The cell matrix a card must realize under a tier: every declared factor level crossed, per
    independent unit, times units. ``levels`` is the product of factor levels; ``units`` the
    number of (world, repeat) pairs in the lane."""
    levels = 1
    for k, v in card.factors.items():
        levels *= max(1, len(v))
    if card.unit_kind == "single":
        units = 1
    else:
        key = {"discovery": "discovery_worlds", "transfer": "transfer_worlds",
               "confirmation": "confirmation_worlds", "attack": "transfer_worlds"}.get(lane, "discovery_worlds")
        units = int(tier[key]) * int(tier["repeats"])
    return {"levels": int(levels), "units": int(units), "cells": int(levels * units),
            "min_rows_per_unit": int(card.min_rows_per_unit), "factors": dict(card.factors)}


def new_verdict(card: Card, lane: str, hypothesis: str, claim_ceiling: str) -> dict:
    """The skeleton every V14 verdict must fill. The plain-language record (spec §5.5) comes
    first and is filled by the card's ``narrative``; metrics follow."""
    assert claim_ceiling in CLAIM_CEILINGS, claim_ceiling
    assert lane in LANES + ("smoke",), lane
    return {
        "card": card.id, "trunk": card.trunk, "wave": card.wave, "lane": lane,
        "record": {"question": card.question, "what_happened": "",
                   "what_changed_in_the_project_world": "", "what_remains_a_rival": card.strongest_rival,
                   "claim_ceiling": claim_ceiling},
        "hypothesis": hypothesis, "claim_ceiling": claim_ceiling,
        "construction_realization": {}, "results": {}, "conditional_matrix": {}, "exact": {},
        "pymdp": {}, "baselines": {}, "cells": {}, "expected_cell_receipt": {},
        "independent_unit": card.independent_unit, "effective_n": {}, "matching_residuals": {},
        "pursuit": "OPENED", "warrant": "DESCRIPTIVE_ONLY", "strongest_rival": card.strongest_rival,
        "what_must_hold_outside_the_simulation": "", "deviations": [], "repairs": [],
        "upstream_oracle": bool(card.upstream_oracle), "runtime": {}, "runtime_seconds": None,
    }


VERDICT_REQUIRED = ("record", "hypothesis", "claim_ceiling", "results", "gates", "produced_by",
                    "environment", "runtime_seconds", "state", "cells", "expected_cell_receipt",
                    "effective_n", "pursuit", "warrant")


def completion_entry(card_id: str, lane: str, verdict_path: str, verdict_sha: str, source_sha: str,
                     receipt: dict, state: str, timestamp: str) -> dict:
    return {"card": card_id, "lane": lane, "verdict_path": verdict_path, "verdict_sha256": verdict_sha,
            "source_sha256": source_sha, "expected_cell_receipt": receipt, "state": state,
            "timestamp": timestamp}
