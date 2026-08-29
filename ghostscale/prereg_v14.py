"""Pre-registration for V14 — The Routed Reader. Internal pre-specification, hash-locked before
scientific execution; NOT external preregistration (spec §10).

Three locks, in order:

1. STRUCTURAL LOCK (before the runtime pilot): this module, the card set, every criterion,
   closure, flight, attack-relevance table and repair rule, the expected-cell template, the source
   lineages, the construction identities, the world generators and the report interface.
2. WORKLOAD LOCK (after the discarded pilot, before discovery): the selected tier, the expansion
   packets instantiated, the instantiated expected-cell matrix, the route-information table, the
   attack matrix, and the forecast.
3. SCIENTIFIC LOCK (before discovery): the structural and workload hashes together.

Confirmatory language is refused when any relevant hash changes. Amendments after data keep the
original beside the replacement in results/v14/AMENDMENTS.json. Numeric bars are set before the
run from exact chance, construction-derived floors, or the smallest effect that would alter
pursuit; each entry names its primary estimand's decision rule.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .validation.soundingline.v14 import REPO, v14_dir

STRUCTURAL_PATH = v14_dir() / "prereg_v14_structural_lock.json"
WORKLOAD_PATH = v14_dir() / "WORKLOAD_LOCK.json"
PREREG_PATH = v14_dir() / "prereg_v14_lock.json"
SOURCE_LINEAGES = v14_dir() / "SOURCE_LINEAGES.json"
CONSTRUCTION_IDENTITIES = v14_dir() / "CONSTRUCTION_IDENTITIES.json"
ROUTE_INFORMATION = v14_dir() / "ROUTE_INFORMATION.json"
ATTACK_MATRIX = v14_dir() / "ATTACK_MATRIX.json"

_V14 = REPO / "ghostscale" / "validation" / "soundingline" / "v14"
GENERATOR_FILES = ["world.py", "joint.py", "routes.py", "history_skill.py", "communication.py", "hierarchy.py",
                   "foraging.py", "schemas.py", "common.py"]
REPORT_INTERFACE = REPO / "runners" / "report_v14.py"

CRITERIA = {
    # ---- I ---------------------------------------------------------------------------------- #
    "I01": {"hash_exact": True, "cited_tolerance": 0.005, "rule": "every committed V13 verdict V14 imports hashes to its V13 ledger entry exactly, and every number V14 cites from it matches the file within half a unit of the cited precision; a mismatch blocks inheritance only"},
    "I02": {"rule": "the manifest enumerates exactly 64 mandatory cards and 12 attacks with factors, lanes and floors; the recursive validator passes"},
    "I03": {"identity_tol": 1e-9, "rule": "the joint enumerator normalizes, matches brute-force enumeration in a tiny exhaustive world, and is invariant to label and order relabelling"},
    "I04": {"min_divergence": 0.05, "min_information": 0.05, "max_null_route": 0.05, "rule": "routes are pairwise divergent and each carries information about some latent; a shuffled route gains at most the bar on the true latent (mean over eight shuffles)"},
    "I05": {"max_leak": 0.10, "rule": "surface collisions hash equal, equifinal histories produce identical non-forensic likelihoods, factors leak at most the bar into each other's measures over 24 episodes, lineages disjoint"},
    "I06": {"floor_share": 0.5, "rule": "REPAIR of V13 C03: the positive gate is the instrument seeing the planted prior-level gain (the construction floor); the CRITERION is within-common beating all-family at the first artifact by half that floor; a failed criterion closes the common-substrate mechanism as a null, not an instrument failure"},
    "I07": {"min_gain": 0.05, "convergence_tol": 0.05, "rule": "REPAIR of V13 C05: self beats within-common for near readers by the bar and not for anti-similar readers; the two routes' posteriors agree (Jensen-Shannon) within tolerance at the first dose, probed to 64, where both fall under 0.3 nats of entropy; no sufficient dose by 64 closes the mechanism"},
    "I08": {"max_half_life": 8, "max_residual": 0.10, "rate_margin": 0.05, "max_ece": 0.15, "min_slope": 0.2, "rule": "REPAIR of V13 P01: half-life and residual within bars per route; near makers' correction RATE (share of initial error removed) at least far makers' minus the margin; the calibration GATE is that endpoint confidence predicts correctness (slope above the bar); the endpoint ECE is the criterion's science"},
    # ---- J ---------------------------------------------------------------------------------- #
    "J01": {"min_class_mass": 0.8, "max_single_equivalent": 0.7, "rule": "with goal and preference supplied, process class mass above the bar; on process-equivalent histories no single state exceeds the split bar"},
    "J02": {"min_gain": 0.05, "rule": "with process and preference supplied, the hidden next action within the episode is predicted above the prior by the bar"},
    "J03": {"min_gain": 0.05, "rule": "with process and goal supplied across episodes, the next episode's first action is predicted above the prior by the bar after the local goal changed"},
    "J04": {"min_gain": 0.02, "max_ece_penalty": 0.05, "rule": "joint beats independent on held-out next-action log score by the bar at matched evidence and compute, with calibration no worse than the penalty; the cross-latent ablation is the independent estimator"},
    "J05": {"min_margin": 0.02, "rule": "order x regime interaction reported before any pooled number; a universally best order is claimed only if it wins in every access regime by the margin"},
    "J06": {"max_dose": 8, "rule": "each latent first improves prospective prediction over its baseline by the maximum dose; the dose trajectory is reported"},
    "J07": {"min_revision": 0.10, "max_false_revision": 0.10, "rule": "diagnostic contradiction revises every affected latent by the bar (Jensen-Shannon); a consistent continuation revises by at most the false-revision bar"},
    "J08": {"max_single_equifinal": 0.6, "min_class_equifinal": 0.8, "min_single_resolved": 0.8, "rule": "under exact equifinality the joint reader spreads mass within the class; after resolving evidence it contracts"},
    "J09": {"min_detect": 0.7, "max_confusion": 0.3, "rule": "a changed episode goal and a changed standing preference are each detected above the bar with cross-confusion below the bar, judged on held-out future choices"},
    "J10": {"min_gain": 0.02, "rule": "the frozen joint estimator keeps its advantage over independent on fresh vocabularies; domain-bound effects named"},
    # ---- R ---------------------------------------------------------------------------------- #
    "R01": {"min_information": 0.05, "rule": "the route with the most target information per latent is reported per access regime; exact conditional information agrees with the prediction ruler on the dominant route"},
    "R02": {"min_gain": 0.02, "rule": "learned route weighting beats equal, random and fixed weighting on held-out prediction by the bar with no target label at test"},
    "R03": {"max_ease_effect": 0.02, "min_bias": 0.2, "rule": "with accuracy equal, planted ease moves the ease-driven reader's route weight by the bias bar and its score, while the learned reader's score moves by at most the ease bar"},
    "R04": {"max_ease_effect": 0.02, "rule": "with ease equal, the learned reader follows accuracy: its score under the accurate-but-hard route is within the bar of the accurate-and-easy case"},
    "R05": {"min_gain": 0.05, "max_false_positive": 0.15, "rule": "expanding the latent set to a strategic source gains the bar in strategic worlds and flags at most the bar of consistent worlds"},
    "R06": {"margin": 0.01, "rule": "the exact information-per-cost purchase rule realizes at least the best of random, always-buy and never-buy minus the margin"},
    "R07": {"max_dup_rise": 0.02, "min_naive_rise": 0.05, "rule": "shared-cause fusion raises confidence by at most the bar under duplicated evidence where naive fusion rises by at least the naive bar; calibration reported"},
    "R08": {"margin": 0.02, "rule": "under a domain shift with changed reliabilities, reset or partial transfer is within the margin of the best; full transfer's loss reported"},
    # ---- E ---------------------------------------------------------------------------------- #
    "E01": {"min_move": 0.05, "max_leak": 0.02, "rule": "competence moves process accuracy and history moves early relevance by the bar; each leaks into the other's measure by at most the bar"},
    "E02": {"min_initial_bias": 0.05, "min_correction_share": 0.5, "rule": "with competence matched, a stale route history biases initial route weights by the bar and target evidence closes at least half the score gap"},
    "E03": {"min_gain": 0.05, "rule": "with history matched, higher competence improves process reconstruction by the bar"},
    "E04": {"max_residual_share": 0.5, "rule": "a learned attention bias decays to at most half after eight episodes past its reward reversal; its current-utility cost is reported"},
    "E05": {"min_bias_reduction": 0.5, "max_skill_loss": 0.02, "rule": "correction removes at least half the stale-history bias while process accuracy falls by at most the bar"},
    "E06": {"min_early_gain": 0.05, "rule": "competence improves early relevance detection at the first dose by the bar and the gap shrinks with dose; no generic intelligence interpretation"},
    "E07": {"rule": "cross-domain conditional matrix reported for history and competence; whichever transfers more narrowly is named"},
    "E08": {"min_signature": 0.7, "max_skill_gap": 0.05, "rule": "at equal competence, acquisition paths leave held-out signatures classified above the bar while process accuracy differs by at most the gap; prospective only"},
    "E09": {"margin": 0.02, "rule": "likelihood intersection is within the margin of the best member and at least the naive average; the average's overconfidence is reported"},
    "E10": {"min_gain": 0.02, "rule": "the object that best predicts the maker's next novel choice beats the last-choice baseline by the bar; the tournament is reported"},
    # ---- A ---------------------------------------------------------------------------------- #
    "A01": {"min_own": 0.10, "max_leak": 0.02, "rule": "each owner moves its own posterior by the bar and any other's by at most the leak"},
    "A02": {"min_gain": 0.05, "rule": "the reader's own induced response is a useful prior for intended effect when the reader is similar to the audience and not otherwise; the projection cost is reported"},
    "A03": {"min_intended": 0.6, "min_uncertainty": 0.5, "rule": "intended effect recovered above the bar while the maker-appraisal posterior keeps at least the bar of entropy"},
    "A04": {"min_owner": 0.6, "rule": "under owner swap the maker's appraisal and its private action are recovered above the bar"},
    "A05": {"min_separated": 0.6, "max_artifact_only": 0.4, "rule": "the four derived regions separate above the bar with counterfactual evidence and at most the artifact-only bar without it"},
    "A06": {"max_artifact_only": 0.6, "min_with_probes": 0.8, "max_abstain_mass": 0.6, "rule": "fanatic and propagandist are at chance from the artifact, separated by belief, private action and correction probes, and abstained on when no discriminator is observed"},
    "A07": {"min_interaction": 0.05, "rule": "an audience-aware reader gains by the bar only when the maker models the audience"},
    "A08": {"min_discrimination": 0.6, "min_true_uptake": 0.5, "rule": "influence awareness improves true/false discrimination above the bar while retaining true uptake above the bar; blanket suppression is the rival"},
    "A09": {"max_habituated_share": 0.5, "rule": "acute response at eight exposures is at most half of the first while cumulative uptake rises"},
    "A10": {"min_gate_effect": 0.2, "max_side_effect": 0.02, "rule": "factored trust gates policy uptake by the bar while content belief and inferred goal move by at most the side-effect bar"},
    # ---- H ---------------------------------------------------------------------------------- #
    "H01": {"min_boundary": 0.7, "min_gain": 0.05, "max_spurious": 0.3, "rule": "subtask boundaries recovered above the bar and next subtask predicted above baseline in hierarchical worlds; spurious boundaries in flat worlds below the bar"},
    "H02": {"max_shared": 0.65, "min_distinct": 0.8, "rule": "identical local actions under different higher goals keep top-goal mass at most the bar; distinct windows reach the distinct bar"},
    "H03": {"max_observational": 0.6, "min_intervened": 0.8, "rule": "policy-equivalent rewards are not distinguished observationally and are after the resolving intervention"},
    "H04": {"min_gain": 0.05, "rule": "across changed incentives the preference model wins for preference-driven makers and the habit model for habitual ones, each by the bar"},
    "H05": {"min_gain": 0.05, "rule": "after a history reversal the current preference predicts the hidden future choice better than the residue by the bar"},
    "H06": {"max_ece": 0.15, "min_compression": 0.05, "rule": "a higher coordinating level compresses multi-episode evidence by the bar with calibrated level uncertainty; no terminal horizon required"},
    "H07": {"min_interaction": 0.9, "max_coherence": 0.6, "min_next": 0.6, "rule": "with full records the interaction reader separates central control from the exact shared-brief twin above the bar, coherence stays below the bar, and the hidden next intervention is predicted above the bar"},
    "H08": {"margin": 0.02, "rule": "the selected hierarchy level predicts the next changed-context action at least as well as flat value and last goal plus the margin"},
    # ---- F ---------------------------------------------------------------------------------- #
    "F01": {"min_move": 0.10, "max_corr": 0.3, "rule": "each foraging factor varied alone moves its own measure by the bar; pairwise correlations across items stay below the bar"},
    "F02": {"min_share": 0.6, "rule": "learning-progress and gain-per-cost policies prefer the familiar unresolved structure after the first look by the bar; novelty prefers the explained novelty"},
    "F03": {"min_share": 0.6, "rule": "the compressible complex item is preferred over the simpler unresolved one by the bar"},
    "F04": {"max_noise_share": 0.2, "min_gain_margin": 0.10, "rule": "unlearnable noise draws at most the bar of a learning-progress policy's picks and its realized gain exceeds the surprise policy's by the margin"},
    "F05": {"min_gain": 0.05, "rule": "expected learning progress realizes more held-out gain than raw current error in a nonstationary curriculum by the bar"},
    "F06": {"margin": 0.02, "rule": "information gain per cost realizes at least each rival's held-out gain per cost minus the margin"},
    "F07": {"min_pursuit": 0.4, "max_warrant": 0.5, "rule": "queries go to the hoped-for hypothesis above the bar while its posterior stays below the warrant bar"},
    "F08": {"max_regret": 0.10, "min_abstain": 0.8, "rule": "the selector transfers with regret at most the bar and abstains on null probes at least the bar of the time"},
    # ---- B ---------------------------------------------------------------------------------- #
    "B01": {"rule": "one row per candidate ruler: access, construction gate, cheap rival, endpoint, shape, ceiling; a failed instrument licenses nothing"},
    "B02": {"rule": "final pursuit and warrant ledger, runtime audit and recommendation; no automatic V15"},
    "X": {"survival": "sign kept and at least half the magnitude retained", "rule": "an attacked effect survives, narrows or dies; dying under a causal-variable-preserving attack makes the result a shortcut; applicability is explicit, never a silent not-applicable"},
}

CLOSURES = {
    "I": "an identity or lineage failure stops dependent claims; a repaired V13 instrument that fails its named gate again closes its mechanism with no second repair",
    "J": "close the joint advantage if no matched estimator beats independent marginals on any prospective target; retain equivalence-class uncertainty as a boundary",
    "R": "close reliable routing if learned reliability never beats equal weighting or aliases with ease; retain the divergence and no-information rulers",
    "E": "close the dissociation if competence and history cannot be manipulated independently or leave no distinct prospective signature",
    "A": "close the factorization if an owner cannot be varied alone; close the fanatic/propagandist boundary if no counterfactual discriminator exists; retain abstention",
    "H": "retain equivalence classes wherever alternatives are policy-equivalent; never force a unique hierarchy; the records-dominant boundary stands",
    "F": "close a curiosity ruler that unlearnable noise can win; retain learning progress only where it realizes held-out gain",
    "B": "an unlicensed ruler is not exported",
}

FLIGHTS = {
    "joint_reconstruction_advantage": ["J04 joint beats matched estimators", "J02 J03 prospective", "J08 equifinality preserved", "J10 transfer", "X03 X04 X11 relevant"],
    "reliable_routing": ["R02 learned beats ease", "R03 R04 ease/accuracy crossing", "R07 duplicates", "R08 domain shift", "X02 X04 X05 relevant"],
    "competence_history_dissociation": ["E01 independence", "E02 E03 signatures", "E04 E05 correction", "E08 prospective", "X06 relevant"],
    "affect_source_factorization": ["A01 owners", "A05 A06 fanatic/propagandist boundary", "A08 A10 uptake", "X07 X08 relevant"],
    "learning_progress_foraging": ["F04 noise trap", "F05 F06 progress and gain per cost", "F08 transfer", "X10 relevant"],
}
FLIGHT_PRIMARY_CARD = {"joint_reconstruction_advantage": "J04", "reliable_routing": "R02",
                       "competence_history_dissociation": "E01", "affect_source_factorization": "A06",
                       "learning_progress_foraging": "F04"}
FLIGHT_CARDS = {"joint_reconstruction_advantage": ["J02", "J03", "J04", "J08", "J10"],
                "reliable_routing": ["R02", "R03", "R04", "R07", "R08"],
                "competence_history_dissociation": ["E01", "E02", "E03", "E04", "E05", "E08"],
                "affect_source_factorization": ["A01", "A05", "A06", "A08", "A10"],
                "learning_progress_foraging": ["F04", "F05", "F06", "F08"]}
ATTACK_RELEVANCE = {
    "joint_reconstruction_advantage": ["X01", "X03", "X04", "X05", "X11", "X12"],
    "reliable_routing": ["X01", "X02", "X04", "X05", "X11", "X12"],
    "competence_history_dissociation": ["X01", "X05", "X06", "X11", "X12"],
    "affect_source_factorization": ["X01", "X07", "X08", "X11", "X12"],
    "learning_progress_foraging": ["X01", "X10", "X11", "X12"],
}
#: The one allowed repair of each V13 debt: the named gate, the preserved target, and the rule.
REPAIRS = {
    "I06": {"v13_card": "C03", "failed_gate": "positive:within_common_above_floor", "preserves": "within-common versus all-family prior, the dose axis and the target",
            "repair": "the positive floor becomes the exact expected gain of the true within-common prior over the all-family prior at one artifact, computed from the construction; the gate passes at half that floor"},
    "I07": {"v13_card": "C05", "failed_gate": "placebo:routes_converge_at_sixteen", "preserves": "reader-type interaction and the convergence placebo",
            "repair": "the convergence dose is derived from the construction (the dose at which both routes' posteriors fall under 0.3 nats of entropy on the planted world) instead of a fixed sixteen"},
    "I08": {"v13_card": "P01", "failed_gate": "positive:near_makers_need_no_more_correction_than_far; calibration:final_confidence_calibrated", "preserves": "similarity bins, routes, the prospective endpoint and the failed calibration record",
            "repair": "the positive gate compares correction RATE per bin (share of the initial error removed per artifact) rather than residual; calibration is scored at the prospective endpoint with a unit-level ECE"},
}
CONFIRMATION_CAP = 4
FREEZE_HOUR = 20.0


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hj(obj) -> str:
    return _h(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def structural_payload() -> dict:
    from .validation.soundingline.v14.manifest import CELLS_TEMPLATE, build_cards
    cards = [c.to_dict() for c in build_cards()]
    gen = {f: _h((_V14 / f).read_bytes()) for f in GENERATOR_FILES if (_V14 / f).exists()}
    return {"module_sha256": _h(Path(__file__).read_bytes()), "cards_sha256": _hj(cards),
            "criteria_sha256": _hj(CRITERIA), "closures_sha256": _hj(CLOSURES), "flights_sha256": _hj(FLIGHTS),
            "attack_relevance_sha256": _hj(ATTACK_RELEVANCE), "repairs_sha256": _hj(REPAIRS), "generators": gen,
            "cells_template_sha256": _h(CELLS_TEMPLATE.read_bytes()) if CELLS_TEMPLATE.exists() else None,
            "source_lineages_sha256": _h(SOURCE_LINEAGES.read_bytes()) if SOURCE_LINEAGES.exists() else None,
            "construction_identities_sha256": _h(CONSTRUCTION_IDENTITIES.read_bytes()) if CONSTRUCTION_IDENTITIES.exists() else None,
            "report_interface_sha256": _h(REPORT_INTERFACE.read_bytes()) if REPORT_INTERFACE.exists() else None,
            "n_cards": len(cards)}


def write_structural_lock() -> dict:
    lock = {"program": "v14", "kind": "structural", "locked": True, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "internal_prespecification_not_external_preregistration": True, **structural_payload(),
            "criteria": CRITERIA, "closures": CLOSURES, "flights": FLIGHTS, "attack_relevance": ATTACK_RELEVANCE, "repairs": REPAIRS}
    if STRUCTURAL_PATH.exists():
        try:
            old = json.loads(STRUCTURAL_PATH.read_text(encoding="utf-8"))
            if {k: v for k, v in old.items() if k != "written"} == {k: v for k, v in lock.items() if k != "written"}:
                return old                                   # identical payload keeps identical bytes
        except (json.JSONDecodeError, OSError):
            pass
    STRUCTURAL_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8", newline="\n")
    return lock


def write_workload_lock(tier_name: str, tier: dict, expansions: list, forecast: dict, pilot_summary: dict) -> dict:
    from .validation.soundingline.v14.manifest import CELLS
    lock = {"program": "v14", "kind": "workload", "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "tier": tier_name,
            "tier_config": tier, "expansions_instantiated": expansions, "forecast": forecast,
            "pilot_summary_sha256": _hj(pilot_summary), "pilot_is_non_scientific": True,
            "cells_sha256": _h(CELLS.read_bytes()) if CELLS.exists() else None,
            "route_information_sha256": _h(ROUTE_INFORMATION.read_bytes()) if ROUTE_INFORMATION.exists() else None,
            "attack_matrix_sha256": _h(ATTACK_MATRIX.read_bytes()) if ATTACK_MATRIX.exists() else None}
    WORKLOAD_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8", newline="\n")
    return lock


def write_scientific_lock() -> dict:
    st = json.loads(STRUCTURAL_PATH.read_text(encoding="utf-8"))
    wl = json.loads(WORKLOAD_PATH.read_text(encoding="utf-8"))
    lock = {"program": "v14", "kind": "scientific", "locked": True, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "internal_prespecification_not_external_preregistration": True,
            "structural_sha256": _h(STRUCTURAL_PATH.read_bytes()), "workload_sha256": _h(WORKLOAD_PATH.read_bytes()),
            "structural": {k: st[k] for k in ("module_sha256", "cards_sha256", "criteria_sha256", "generators", "cells_template_sha256")},
            "tier": wl["tier"], "cells_sha256": wl.get("cells_sha256"), "amended_after_data": []}
    PREREG_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8", newline="\n")
    return lock


def lock_status() -> dict:
    if not STRUCTURAL_PATH.exists():
        return {"locked": False, "stage": "none", "reason": "no structural lock; run python -m ghostscale.prereg_v14"}
    st = json.loads(STRUCTURAL_PATH.read_text(encoding="utf-8"))
    now = structural_payload()
    keys = ("module_sha256", "cards_sha256", "criteria_sha256", "closures_sha256", "flights_sha256", "attack_relevance_sha256",
            "repairs_sha256", "generators")
    structural_ok = all(st.get(k) == now[k] for k in keys)
    out = {"structural_locked": bool(structural_ok), "internal_prespecification_not_external_preregistration": True}
    if not structural_ok:
        out["changed"] = [k for k in keys if st.get(k) != now[k]]
    if not PREREG_PATH.exists():
        out.update({"locked": False, "stage": "structural", "reason": "no scientific lock yet (pilot not run)"})
        return out
    sci = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    sci_ok = structural_ok and sci.get("structural_sha256") == _h(STRUCTURAL_PATH.read_bytes()) \
        and WORKLOAD_PATH.exists() and sci.get("workload_sha256") == _h(WORKLOAD_PATH.read_bytes())
    out.update({"locked": bool(sci_ok), "stage": "scientific", "tier": sci.get("tier")})
    if not sci_ok:
        out["reason"] = "structural, workload or scientific lock changed after locking"
    return out


if __name__ == "__main__":
    write_structural_lock()
    print(f"structural lock -> {STRUCTURAL_PATH}")
