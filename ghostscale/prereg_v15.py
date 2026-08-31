"""Pre-registration for V15 — The Boundary Map. Internal pre-specification, hash-locked before
scientific execution; NOT external preregistration (spec §10.1).

Three locks, in order:

1. **STRUCTURAL LOCK**, before the discarded pilot: this module, the card set, every criterion,
   flight, attack target, the expected-cell template, the source lineages, the generator families,
   the architecture budgets, the construction graph, the world generators and the report interface.
2. **WORKLOAD LOCK**, after the pilot and before discovery: the selected tier, the instantiated
   expected-cell matrix, the frozen balanced coverage sequence, the resource governor and the
   forecast.
3. **SCIENTIFIC LOCK**, before discovery: the structural and workload hashes together.

Spec §9.2 is explicit about what the pilot may and may not touch. It may choose batch size,
replicate floor, coverage length and safe worker count. It may not touch a hypothesis, a criterion,
a factor or an estimator's membership -- all of which are above the workload lock and inside the
structural one.

Where the numeric bars came from
--------------------------------
Spec §8.2 forbids recycling V14's 0.02-nat bar and asks for a fraction of a live positive control
on the same score. Before anything was registered, the construction's own spans were measured: the
distance from the cheapest reader to the state oracle is **0.30 nats** at the atlas's reference
settings and **0.54 nats** under scarcity. The architecture bar is 0.015 -- five per cent of the
smaller span. Accuracy bars are fractions above the construction's own chance floor. Every bar is
recorded with its basis in ``SESOI`` and repeated in each card's verdict.

Instrument choices made before the lock, and recorded here because they are choices
-----------------------------------------------------------------------------------
Each of these was settled while validating an instrument against a known answer, never against an
outcome. They are listed so that a reader can see what was decided rather than derived.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .validation.soundingline.v15 import REPO, v15_dir

STRUCTURAL_PATH = v15_dir() / "prereg_v15_structural_lock.json"
WORKLOAD_PATH = v15_dir() / "WORKLOAD_LOCK.json"
PREREG_PATH = v15_dir() / "prereg_v15_lock.json"

_V15 = REPO / "ghostscale" / "validation" / "soundingline" / "v15"
#: Hash-locked generators. Editing one halts the program until the lock is amended on record.
GENERATOR_FILES = [
    "ontology.py", "world_chain.py", "world_composition.py", "world_communication.py",
    "exact.py", "particles.py", "expansion.py", "architectures.py", "learning_history.py",
    "foreground.py", "persistent.py", "strategic_source.py", "routes.py", "foraging.py",
    "hierarchy.py", "causal_distance.py", "coverage.py", "schemas.py", "common.py",
]
REPORT_INTERFACE = REPO / "runners" / "report_v15.py"

#: The construction's measured spans, from which every magnitude bar is a declared fraction.
SPANS = {
    "oracle_minus_surface_reference": 0.30,
    "oracle_minus_surface_scarce": 0.54,
    "measured_when": "before the structural lock, on the reference atlas settings",
    "how": ("mean held-out log score of oracle_state minus surface on the hidden next action, "
            "14 worlds x 16 makers per cell"),
}

SESOI = {
    "architecture_advantage": {
        "value": 0.015,
        "basis": "5% of the 0.30-nat oracle-minus-surface span at reference settings",
        "applies_to": "every card whose estimand is a paired log-score difference in nats",
        "note": ("V14's 0.02-nat bar is NOT reused. It was derived from a different construction "
                 "whose span this program does not share (spec 8.2)."),
    },
    "accuracy_above_chance": {
        "value": "0.08-0.20, stated per card",
        "basis": "a fraction above the construction's own chance floor, which differs by card",
    },
    "class_coverage": {"value": 0.85, "basis": "mass a reader must keep on a true equivalence class"},
    "abstention_rate": {"value": 0.70, "basis": "rate required in a null ecology"},
    "identity": {"value": 1e-9, "basis": "floating-point identity for exactness cards"},
}

#: Instrument constants chosen before the lock. Each is a construction choice, not a fitted value.
INSTRUMENT_CHOICES = {
    "coupling_construction": (
        "a marginal-preserving mixture with the weight solved to a target mutual information. A "
        "log-linear tilt was tried first and is wrong: mutual information is not monotone in the "
        "tilt, so a bisection returns degenerate worlds."),
    "coupling_axis": (
        "the phase-diagram axis is REALIZED coupling measured from the world's own prior, not the "
        "nominal knob, because the three families reach different ceilings."),
    "overlap_measurement": (
        "measured against a uniform reference prior, so that the coupling knob cannot leak into "
        "the overlap receipt."),
    "independent_rival": (
        "marginals of the factorized-prior posterior. The evidence is used once. Assigning home "
        "routes per component while leaving the shared action channel in each one triples the "
        "policy evidence and puts the exact posterior behind an approximation."),
    "pid_definition": "Williams-Beer I_min, exact, two sources only",
    "shapley_definition": "exact Shapley over all 2^k subsets, three components",
    "practice_outcome_visibility": (
        "0.40. At zero, self-directed practice cannot improve past its own first guesses and "
        "cannot be skill-matched to an instructed history, which is E01's precondition."),
    "foraging_selection_sharpness": (
        "2.5. Proportional selection on raw value collapses every controller onto the random "
        "floor; greedy selection makes the reported avoidance an artifact of the tie-break."),
    "changepoint_discount": "0.30 of the accumulated counts survive a detected change",
    "source_surface_profiles": (
        "sincere and strategic share one surface profile, mixed and contrarian the other. The "
        "artifact therefore recovers the collision class and is at chance inside it, and the "
        "belief prior is a property of the profile so the private channel cannot leak the motive."),
    "collision_fixtures_by_rejection": (
        "G01's surface collision is built by drawing worlds until the residual clears the "
        "tolerance, because switching's action marginal is a mixture of softmaxes and "
        "simultaneous control's is a softmax of a blend -- different families, so no parameter "
        "search closes the gap in an arbitrary world."),
}

#: Flights: the promotable groupings a confirmation packet may be frozen from.
FLIGHTS = {
    "coupling_access_atlas": ["C02", "C03", "C04", "C05", "C09", "C14"],
    "architecture_tournament": ["M02", "M03", "M06", "M07", "M08", "M12"],
    "endogenous_expertise": ["E02", "E03", "E07", "E08", "E12"],
    "foreground_control": ["G02", "G04", "G07", "G09", "G10"],
    "persistent_value": ["V01", "V04", "V06", "V10"],
    "strategic_sources": ["S03", "S06", "S07", "S09", "S10"],
    "robust_routing": ["R01", "R03", "R06", "R07"],
    "change_aware_foraging": ["F02", "F03", "F06", "F09", "F10"],
    "hierarchy_topology": ["H04", "H06", "H07", "H08"],
    "prospective_synthesis": ["P01", "P02", "P03", "P04", "P05"],
}
#: The card a flight is confirmed on when the freeze cap binds (spec §9.1, one per flight).
FLIGHT_PRIMARY_CARD = {
    "coupling_access_atlas": "C04", "architecture_tournament": "M02",
    "endogenous_expertise": "E03", "foreground_control": "G04", "persistent_value": "V01",
    "strategic_sources": "S03", "robust_routing": "R01", "change_aware_foraging": "F09",
    "hierarchy_topology": "H07", "prospective_synthesis": "P01",
}
#: Which attacks are relevant to which flight. An attack outside its flight's list records
#: NOT_APPLICABLE with this table as the schema-validated reason (spec §7).
ATTACK_RELEVANCE = {
    "coupling_access_atlas": ["X01", "X02", "X03", "X04", "X08", "X09", "X10", "X11", "X21"],
    "architecture_tournament": ["X01", "X05", "X06", "X07", "X10", "X22"],
    "endogenous_expertise": ["X12", "X13", "X01"],
    "foreground_control": ["X15", "X09", "X01"],
    "persistent_value": ["X14", "X08", "X11"],
    "strategic_sources": ["X16", "X17", "X18", "X04"],
    "robust_routing": ["X02", "X03", "X04", "X05"],
    "change_aware_foraging": ["X19", "X20"],
    "hierarchy_topology": ["X01", "X10"],
    "prospective_synthesis": ["X01", "X06", "X08", "X21"],
}
#: Every flight also faces the three integrity attacks.
UNIVERSAL_ATTACKS = ["X21", "X23", "X24"]

#: The confirmation freeze rule (spec §9.1). At the freeze hour, at most this many candidates.
CONFIRMATION_CAP = 6
FREEZE_RULE = ("at the freeze hour, freeze at most CONFIRMATION_CAP candidates, at most one per "
               "flight, preferring the flight's primary card, taking flights in the order "
               "declared in FLIGHTS")


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hj(obj) -> str:
    return _h(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def structural_payload() -> dict:
    from .validation.soundingline.v15.manifest import (ARCHITECTURE_BUDGETS, CELLS_TEMPLATE,
                                                       CONSTRUCTION_GRAPH, GENERATOR_FAMILIES,
                                                       SOURCE_LINEAGES, build_cards)
    cards = [c.to_dict() for c in build_cards()]
    gen = {f: _h((_V15 / f).read_bytes()) for f in GENERATOR_FILES if (_V15 / f).exists()}
    files = {p.name: (_h(p.read_bytes()) if p.exists() else None)
             for p in (CELLS_TEMPLATE, CONSTRUCTION_GRAPH, GENERATOR_FAMILIES,
                       ARCHITECTURE_BUDGETS, SOURCE_LINEAGES, REPORT_INTERFACE)}
    return {"module_sha256": _h(Path(__file__).read_bytes()),
            "cards_sha256": _hj(cards), "sesoi_sha256": _hj(SESOI), "spans_sha256": _hj(SPANS),
            "flights_sha256": _hj(FLIGHTS),
            "attack_relevance_sha256": _hj(ATTACK_RELEVANCE),
            "instrument_choices_sha256": _hj(INSTRUMENT_CHOICES),
            "generators": gen, "files": files, "n_cards": len(cards)}


def write_structural_lock() -> dict:
    lock = {"program": "v15", "kind": "structural", "locked": True,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "internal_prespecification_not_external_preregistration": True,
            **structural_payload(),
            "spans": SPANS, "sesoi": SESOI, "instrument_choices": INSTRUMENT_CHOICES,
            "flights": FLIGHTS, "flight_primary_card": FLIGHT_PRIMARY_CARD,
            "attack_relevance": ATTACK_RELEVANCE, "universal_attacks": UNIVERSAL_ATTACKS,
            "confirmation_cap": CONFIRMATION_CAP, "freeze_rule": FREEZE_RULE}
    if STRUCTURAL_PATH.exists():
        try:
            old = json.loads(STRUCTURAL_PATH.read_text(encoding="utf-8"))
            if {k: v for k, v in old.items() if k != "written"} == \
                    {k: v for k, v in lock.items() if k != "written"}:
                return old                              # identical payload keeps identical bytes
        except (json.JSONDecodeError, OSError):
            pass
    STRUCTURAL_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8", newline="\n")
    return lock


def write_workload_lock(tier_name: str, tier: dict, forecast: dict, pilot_summary: dict,
                        coverage_definition: dict, governor: dict) -> dict:
    from .validation.soundingline.v15.manifest import CELLS
    lock = {"program": "v15", "kind": "workload",
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "tier": tier_name,
            "tier_config": tier, "forecast": forecast,
            "pilot_summary_sha256": _hj(pilot_summary), "pilot_is_non_scientific": True,
            "pilot_seeds_excluded_from_science": True,
            "cells_sha256": _h(CELLS.read_bytes()) if CELLS.exists() else None,
            "coverage_definition": coverage_definition,
            "coverage_sha256": _hj(coverage_definition),
            "resource_governor": governor,
            "what_the_pilot_was_allowed_to_choose": [
                "batch size", "replicate floor", "coverage sequence length", "safe worker count"],
            "what_the_pilot_could_not_touch": [
                "hypotheses", "criteria", "factors", "estimator membership"]}
    WORKLOAD_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8", newline="\n")
    return lock


def write_scientific_lock() -> dict:
    st = json.loads(STRUCTURAL_PATH.read_text(encoding="utf-8"))
    wl = json.loads(WORKLOAD_PATH.read_text(encoding="utf-8"))
    lock = {"program": "v15", "kind": "scientific", "locked": True,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "internal_prespecification_not_external_preregistration": True,
            "structural_sha256": _h(STRUCTURAL_PATH.read_bytes()),
            "workload_sha256": _h(WORKLOAD_PATH.read_bytes()),
            "structural": {k: st[k] for k in ("module_sha256", "cards_sha256", "sesoi_sha256",
                                              "generators", "files")},
            "tier": wl["tier"], "coverage_sha256": wl.get("coverage_sha256"),
            "amended_after_data": []}
    PREREG_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8", newline="\n")
    return lock


def lock_status() -> dict:
    if not STRUCTURAL_PATH.exists():
        return {"locked": False, "stage": "none",
                "reason": "no structural lock; run python -m ghostscale.prereg_v15"}
    st = json.loads(STRUCTURAL_PATH.read_text(encoding="utf-8"))
    now = structural_payload()
    keys = ("module_sha256", "cards_sha256", "sesoi_sha256", "spans_sha256", "flights_sha256",
            "attack_relevance_sha256", "instrument_choices_sha256", "generators")
    structural_ok = all(st.get(k) == now[k] for k in keys)
    out = {"structural_locked": bool(structural_ok),
           "internal_prespecification_not_external_preregistration": True}
    if not structural_ok:
        out["changed"] = [k for k in keys if st.get(k) != now[k]]
        if "generators" in out["changed"]:
            out["changed_generators"] = [f for f, hsh in now["generators"].items()
                                         if st.get("generators", {}).get(f) != hsh]
    if not PREREG_PATH.exists():
        out.update({"locked": False, "stage": "structural",
                    "reason": "no scientific lock yet (pilot not run)"})
        return out
    sci = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    sci_ok = (structural_ok
              and sci.get("structural_sha256") == _h(STRUCTURAL_PATH.read_bytes())
              and WORKLOAD_PATH.exists()
              and sci.get("workload_sha256") == _h(WORKLOAD_PATH.read_bytes()))
    out.update({"locked": bool(sci_ok), "stage": "scientific", "tier": sci.get("tier")})
    if not sci_ok:
        out["reason"] = "structural, workload or scientific lock changed after locking"
    return out


if __name__ == "__main__":
    lock = write_structural_lock()
    print(f"structural lock -> {STRUCTURAL_PATH}")
    print(f"  cards {lock['cards_sha256'][:12]}  sesoi {lock['sesoi_sha256'][:12]}  "
          f"generators {len(lock['generators'])}")
