"""Card, verdict, ledger, tier, claim-class and publication schemas for V15 (spec §1.2, §6, §10).

The card is the unit of the queue and the unit of accounting. Its state machine is::

    PLANNED, BUILT, INSTRUMENT_VALID, RUNNING, LANDED, INSTRUMENT_FAILED,
    SCIENTIFIC_CLOSED, VOID, RESOURCE_BLOCKED

``DONE`` is forbidden and the validator rejects it. ``RESOURCE_BLOCKED`` never counts as a
scientific negative.

Two fields that V14 proved must be separate and are separate here by type, not by convention:

``state``
    record completion only. ``LANDED`` means "this card produced a valid verdict", nothing more.
``criterion_status``
    whether the pre-registered criterion held: ``HELD``, ``FAILED``, ``NOT_APPLICABLE`` or
    ``UNEVALUATED``. A reader of the ledger who sees ``LANDED`` alone learns nothing about the
    science, and the reporting code refuses to print one without the other.

Spec §1.2's claim classes are a third, orthogonal axis: what *kind* of thing the result is.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

STATES = ("PLANNED", "BUILT", "INSTRUMENT_VALID", "RUNNING", "LANDED",
          "INSTRUMENT_FAILED", "SCIENTIFIC_CLOSED", "VOID", "RESOURCE_BLOCKED")
RESOLVED = ("LANDED", "INSTRUMENT_FAILED", "SCIENTIFIC_CLOSED", "VOID", "RESOURCE_BLOCKED")

#: Spec §1.2. Every result carries exactly one primary class.
CLAIM_CLASSES = ("CONSTRUCTION_IDENTITY", "METHOD", "BOUNDARY", "SIMULATOR_DISCOVERY",
                 "BRIDGE_CANDIDATE", "INSTRUMENT_FAILURE", "VOID")

#: Whether the pre-registered criterion held. Orthogonal to ``state`` (spec §1.2, §6).
CRITERION_STATUS = ("HELD", "FAILED", "NOT_APPLICABLE", "UNEVALUATED")

PURSUIT = ("OPENED", "PROMISING", "STALLED", "EXHAUSTED", "PROMOTE")
WARRANT = ("DESCRIPTIVE_ONLY", "ANALOGUE_EVIDENCE", "MECHANISM_CANDIDATE", "CONFIRMATORY_SUPPORT",
           "BOUNDARY_ESTABLISHED", "INSTRUMENT_FAILED", "VOID", "RESOURCE_BLOCKED")
LANES = ("pilot", "discovery", "transfer", "attack", "confirmation", "coverage")

#: Spec §5. Three independently coded generator families.
FAMILIES = ("chain", "composition", "communication")

#: Spec §3.4. Reader architectures. ``oracle_state`` is an upper bound and is never promotable.
ARCHITECTURES = ("surface", "label_only", "independent", "staged", "joint_exact", "factor_graph",
                 "particle", "expand", "direct_predictor", "oracle_model_space", "oracle_state")
NON_PROMOTABLE = ("oracle_state",)

#: Spec §4.2. Gate kinds a causal card must carry.
GATE_KINDS_CAUSAL = ("live", "placebo", "positive", "no_label_leak", "surface", "prediction",
                     "calibration")

#: Spec §4.1. Prospective endpoints. Retrospective latent accuracy is never primary.
ENDPOINTS = (
    # the eight the spec enumerates
    "next_action", "next_edit", "stop_or_continue", "next_episode_first_choice",
    "changed_context_choice", "next_evidence_selection", "next_intervention_issuer",
    "realized_gain_per_cost",
    # hidden events the spec's card table names in prose: each is a quantity withheld
    # during inference and scored afterwards, not a retrospective latent label
    "held_out_history", "hidden_error_location", "relearning_curve", "transfer_breadth",
    "switch_time", "cross_goal_dependency", "deviation_continuation", "method_change",
    "cost_owner", "feasible_set", "change_point", "source_motive", "selection_policy",
    "team_topology", "collision_residual", "abstention_rate", "expansion_decision",
    "route_weighting", "held_out_gain",
)

#: Spec §8.3 / §12. Publication-relevance ledger fields (spec §1.3).
PUBLICATION_FIELDS = ("established_component", "project_specific_delta", "evidence_grade",
                      "strongest_missing_rival", "independent_generator_count",
                      "external_validation_needed", "paper_shape", "maturity")
EVIDENCE_GRADES = ("identity", "method", "boundary", "simulator_discovery", "model_evidence",
                   "human_evidence")
PAPER_SHAPES = ("none", "benchmark_or_resource", "methods_note", "simulation_study",
                "model_study", "human_study")
MATURITY = ("context_only", "seed", "hardened_seed", "submission_ready")

# --------------------------------------------------------------------------- #
# Tiers. ``makers`` is the makers-per-world count; ``steps`` the episode length.
# Spec §9.2 lets the discarded pilot choose batch size, replicate floor, coverage length and
# worker count -- and nothing else. Tier membership is therefore a workload choice, and the
# hypotheses, criteria, factors and estimator membership above it are structural.
# --------------------------------------------------------------------------- #
TIERS = {
    # ``coverage_worlds`` and ``coverage_makers`` size one cell of the balanced stream. They
    # scale with the tier so that a bigger tier buys a bigger sample per cell rather than the
    # same sample computed more times. They are also sized so that a week of executed blocks
    # stays INSIDE the 20,736 distinct secondary settings: past that point the Sobol sequence
    # revisits design points, and replication at fresh seeds adds power rather than coverage.
    "T0": {"discovery_worlds": 24, "transfer_worlds": 12, "confirmation_worlds": 24,
           "repeats": 2, "makers": 24, "steps": 8, "episodes": 4,
           "coverage_worlds": 4, "coverage_makers": 12},
    "T1": {"discovery_worlds": 48, "transfer_worlds": 24, "confirmation_worlds": 48,
           "repeats": 2, "makers": 32, "steps": 10, "episodes": 5,
           "coverage_worlds": 6, "coverage_makers": 20},
    "T2": {"discovery_worlds": 96, "transfer_worlds": 48, "confirmation_worlds": 64,
           "repeats": 3, "makers": 48, "steps": 12, "episodes": 6,
           "coverage_worlds": 16, "coverage_makers": 48},
    "T3": {"discovery_worlds": 160, "transfer_worlds": 80, "confirmation_worlds": 96,
           "repeats": 3, "makers": 64, "steps": 12, "episodes": 6,
           "coverage_worlds": 24, "coverage_makers": 64},
}
TIER_ORDER = ("T0", "T1", "T2", "T3")

# --------------------------------------------------------------------------- #
# The 168-hour runtime contract (spec §9).
# --------------------------------------------------------------------------- #
WINDOW_HOURS = 168.0
FREEZE_HOUR = 150.0            # discovery/transfer/attacks/coverage end; flights freeze
CONFIRMATION_END_HOUR = 166.0  # confirmation, boundary replication, confirmation controls
INTEGRITY_END_HOUR = 168.0     # clean clone, aggregate regeneration, reconciliation
#: §9.3 guard: conservative upper forecast for the mandatory core, in wall hours at W_safe.
CORE_UPPER_FORECAST_MAX_H = 150.0
#: §9.3 guard: conservative lower forecast for core plus locked coverage must exceed this.
CORE_PLUS_COVERAGE_LOWER_FORECAST_MIN_H = 252.0
#: §9.3 guard: forecast worker-hours that must exist for confirmation and integrity alone.
CONFIRMATION_INTEGRITY_WORKER_HOURS_MIN = 24.0
#: §9.3 guard: the machine the guard must survive being faster than, relative to pilot median.
FAST_MACHINE_FACTOR = 3.0
#: §9.4: fraction of the safe-worker-capacity integral that must be scientific work.
OCCUPANCY_TARGET = 0.80
#: §9.4: any gap longer than this must carry a machine-readable reason.
GAP_REASON_SECONDS = 300.0


@dataclass
class Card:
    """One literal card. Every field that enters the structural lock is declared here."""

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

    #: Spec §1.2: the class this card's result may claim at most.
    claim_class: str = "BOUNDARY"
    #: Spec §8.2: the smallest effect of interest, and where its scale came from.
    sesoi: float = 0.0
    sesoi_basis: str = ""
    #: Spec §4.1: which hidden events this card scores. At least one for a substantive card.
    endpoints: list = field(default_factory=list)
    #: Spec §3.4: which reader architectures compete on this card.
    architectures: list = field(default_factory=list)
    #: Spec §5: which generator families this card runs in.
    families: list = field(default_factory=lambda: ["chain"])
    #: A claim that is explicitly family-bound skips the two-family promotion requirement.
    family_bound: bool = False

    depends_on: list = field(default_factory=list)
    closure: str = ""
    causal: bool = True
    gates_required: list = field(default_factory=lambda: list(GATE_KINDS_CAUSAL))
    factors: dict = field(default_factory=dict)
    crossings: list = field(default_factory=list)
    lanes: list = field(default_factory=lambda: ["discovery"])

    independent_unit: str = "world"      # world | maker | history | source | team
    min_effective_n: int = 24
    min_rows_per_unit: int = 1
    unit_kind: str = "world"             # world: one unit per (world, repeat); single; list
    work_weight: float = 1.0
    pilot: bool = False

    module: str = ""
    output: str = ""
    hashes: dict = field(default_factory=dict)
    estimated_wall_s: float = 0.0
    actual: dict = field(default_factory=dict)

    status: str = "PLANNED"
    criterion_status: str = "UNEVALUATED"
    pursuit: str = "OPENED"
    warrant: str = "DESCRIPTIVE_ONLY"
    closure_reason: str = ""
    amendments: list = field(default_factory=list)
    upstream_oracle: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def card_from_dict(d: dict) -> Card:
    return Card(**{k: v for k, v in d.items() if k in Card.__dataclass_fields__})


def expected_cells(card: Card, tier: dict, lane: str = "discovery") -> dict:
    """The cell matrix a card must realize: every declared factor level crossed, per independent
    unit, times units.

    ``units_required`` follows the sparsest-cell rule that V14's I01 had to learn twice: for a
    ``list`` card each declared cell lives in exactly one unit, so requiring every cell in every
    unit is arithmetically impossible and the receipt must ask for one.
    """
    levels = 1
    for v in card.factors.values():
        levels *= max(1, len(v))
    if card.unit_kind == "single":
        units = 1
    elif card.unit_kind == "list":
        units = 1
    else:
        key = {"discovery": "discovery_worlds", "transfer": "transfer_worlds",
               "confirmation": "confirmation_worlds", "attack": "transfer_worlds",
               "coverage": "transfer_worlds"}.get(lane, "discovery_worlds")
        units = int(tier[key]) * int(tier["repeats"])
    return {"levels": int(levels), "units": int(units), "cells": int(levels * units),
            "min_rows_per_unit": int(card.min_rows_per_unit), "factors": dict(card.factors)}


def new_verdict(card: Card, lane: str, hypothesis: str, claim_class: str) -> dict:
    """The skeleton every V15 verdict fills. Plain-language record first, metrics after."""
    assert claim_class in CLAIM_CLASSES, claim_class
    assert lane in LANES + ("smoke",), lane
    return {
        "card": card.id, "trunk": card.trunk, "wave": card.wave, "lane": lane,
        "record": {"question": card.question, "what_happened": "",
                   "what_changed_in_the_project_world": "",
                   "what_remains_a_rival": card.strongest_rival,
                   "claim_class": claim_class},
        "hypothesis": hypothesis,
        "claim_class": claim_class,
        "criterion_status": "UNEVALUATED",
        "sesoi": {"value": card.sesoi, "basis": card.sesoi_basis},
        "construction_realization": {},
        "results": {},
        "conditional_matrix": {},
        "phase": {},
        "budgets": {},
        "families": {},
        "equivalence": {},
        "trajectories": {},
        "causal_distance": {},
        "publication": {},
        "cells": {}, "expected_cell_receipt": {},
        "independent_unit": card.independent_unit, "effective_n": {},
        "pursuit": "OPENED", "warrant": "DESCRIPTIVE_ONLY",
        "strongest_rival": card.strongest_rival,
        "what_must_hold_outside_the_simulation": "",
        "deviations": [], "repairs": [],
        "upstream_oracle": bool(card.upstream_oracle),
        "runtime": {}, "runtime_seconds": None,
    }


VERDICT_REQUIRED = ("record", "hypothesis", "claim_class", "criterion_status", "results", "gates",
                    "produced_by", "environment", "runtime_seconds", "state", "cells",
                    "expected_cell_receipt", "effective_n", "pursuit", "warrant")


def completion_entry(card_id: str, lane: str, verdict_path: str, verdict_sha: str, source_sha: str,
                     receipt: dict, state: str, criterion_status: str, timestamp: str) -> dict:
    return {"card": card_id, "lane": lane, "verdict_path": verdict_path,
            "verdict_sha256": verdict_sha, "source_sha256": source_sha,
            "expected_cell_receipt": receipt, "state": state,
            "criterion_status": criterion_status, "timestamp": timestamp}


def publication_row(**kw) -> dict:
    """A ``PUBLICATION_MAP.json`` row (spec §1.3). Missing fields are recorded as unfilled rather
    than quietly omitted, so the novelty audit cannot pass by silence."""
    row = {k: kw.get(k) for k in PUBLICATION_FIELDS}
    row["unfilled"] = [k for k, v in row.items() if v in (None, "")]
    return row
