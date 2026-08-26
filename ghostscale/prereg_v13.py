"""Pre-registration for V13 — Common Ground. Internal pre-specification, hash-locked before
scientific execution; NOT external preregistration (spec §20.1).

Three locks, in order:

1. STRUCTURAL LOCK (before the runtime pilot): this module, the card set, every criterion,
   closure, flight and attack-relevance table, the expected-cell template, the world generators
   and the report interface.
2. WORKLOAD LOCK (after the discarded pilot, before discovery): the selected tier, the expansion
   packets instantiated, the instantiated expected-cell matrix, and the forecast.
3. SCIENTIFIC LOCK (before discovery): the structural and workload hashes together.

Confirmatory language is refused when any relevant hash changes. Amendments after data keep the
original beside the replacement in results/v13/AMENDMENTS.json.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .validation.soundingline.v13 import REPO, v13_dir

STRUCTURAL_PATH = v13_dir() / "prereg_v13_structural_lock.json"
WORKLOAD_PATH = v13_dir() / "WORKLOAD_LOCK.json"
PREREG_PATH = v13_dir() / "prereg_v13_lock.json"

_V13 = REPO / "ghostscale" / "validation" / "soundingline" / "v13"
GENERATOR_FILES = ["world.py", "priors.py", "exact.py", "attention.py", "costs.py", "goals_trust.py", "hierarchy.py",
                   "projection.py", "pymdp_reader.py", "schemas.py", "common.py"]
REPORT_INTERFACE = REPO / "runners" / "report_v13.py"

# Numeric bars are set before the run from exact chance, known-positive gaps, or the smallest
# effect that would alter pursuit. Each entry names its primary estimand's decision rule.
CRITERIA = {
    "I01": {"anchor_tolerance": 1e-9, "rule": "every reconstructed anchor field equals the committed V12 field within tolerance"},
    "I02": {"rule": "report entropy, expected-divergence, parameter and coordinate imbalance of V12's generic prior, and the share of S04's near gain that a distance-matched rebuild removes"},
    "I03": {"entropy_tol": 1e-6, "divergence_tol": 0.10, "rule": "every matched route within entropy tolerance and expected-divergence tolerance; otherwise trunk C is instrument-failed"},
    "I04": {"min_js": 0.01, "max_leak": 0.02, "rule": "every factor moves its emission by at least min_js and moves the protected nuisance statistics by at most max_leak"},
    "I05": {"identity_tol": 1e-12, "max_null_gain": 0.02, "rule": "neutral weights reproduce the plain posterior bit for bit; no policy gains in a no-information world"},
    "I06": {"min_js": 0.01, "leak_margin": 0.10, "rule": "each dimension alone shifts the choice distribution; matched totals do not identify the dimension above chance"},
    "I07": {"chance_margin": 0.10, "min_oracle": 0.8, "rule": "surface classifier within margin of 1/7; oracle goal reader above min_oracle"},
    "I08": {"off_edge_tol": 1e-9, "rule": "changing one input moves only the posteriors that declare it"},
    "I09": {"chance_margin": 0.10, "rule": "artifact-only classifier within margin of 0.5; coherence, counts, quality, surface and final goals matched"},
    "I10": {"identity_tol": 1e-6, "rule": "exact and PyMDP agree in the independent world; the discrepancy surface and any confidently-wrong cell are reported"},
    "I11": {"chance_margin": 0.10, "max_ece": 0.10, "rule": "every null returns chance mass on the truth and calibrated uncertainty"},
    "I12": {"rule": "scientific fields identical across two clones and two process orders; lane ancestry disjoint; completion ledger validates"},
    "I13": {"rule": "forecast and tier written before the first discovery verdict; pilot ids absent from every scientific lane; child CPU recorded"},
    "I14": {"invariance_tol": 1e-9, "rule": "invariant scores unchanged under relabelling; antisymmetric quantities reverse sign"},
    "C01": {"min_gain": 0.05, "rule": "held-out continuation of the self model beats pooled frequency by the bar in both domains"},
    "C02": {"min_spearman": 0.8, "rule": "each level's measured distance recovers its planted ordering at rank correlation at least the bar"},
    "C03": {"min_gain": 0.05, "rule": "within-common beats all-family by the bar at the first artifact"},
    "C04": {"min_selective_gain": 0.05, "rule": "self privilege: self minus equal-local at least the bar in the near bin and at most zero far; a tie everywhere is locality"},
    "C05": {"min_gain": 0.05, "rule": "self beats within-common for near readers by the bar and not for anti-similar readers"},
    "C06": {"min_spearman": 0.3, "rule": "self gain rises with reader typicality; compared against the optimally weighted common mixture"},
    "C07": {"min_gap": 0.05, "rule": "gain under a true convention match exceeds gain under a claimed-only match by the bar"},
    "C08": {"min_gap": 0.05, "rule": "practitioners beat novices on the method target by the bar; separated from profile similarity"},
    "C09": {"min_gain": 0.05, "max_shortcut": 0.02, "rule": "eight artifacts of target history gain the bar; the relabelled-source control gains no more than max_shortcut beyond it"},
    "C10": {"min_partial_r2": 0.05, "rule": "at least one axis retains partial R2 above the bar for each target; the map is reported"},
    "C11": {"rule": "report whether confidence or self weight rises without accuracy under nuisance match; correction tested in P"},
    "C12": {"min_gain": 0.05, "rule": "the difference-from-self model beats all-family by the bar at sixteen artifacts"},
    "C13": {"min_gain": 0.03, "rule": "common focus beats uniform only in the common-causal world and never in the nuisance world"},
    "C14": {"min_gain": 0.03, "rule": "near-bin continuation gain at least the bar with far-bin gain at most zero, reported before pooling"},
    "C15": {"min_gain": 0.02, "rule": "the independent ensemble beats the best single reader by the bar; the correlated ensemble does not"},
    "C16": {"rule": "the C04 near/far interaction keeps its sign on fresh families"},
    "A01": {"identity_tol": 1e-12, "rule": "identity at neutral settings; distinct budget curves"},
    "A02": {"min_gain": 0.02, "rule": "oracle and learned information per cost beat random by the bar; salience does not beat learned"},
    "A03": {"min_gain": 0.02, "rule": "learned precision beats uniform by the bar; wrong precision scores below uniform"},
    "A04": {"min_gain": 0.03, "rule": "common focus beats uniform in the common-causal world only"},
    "A05": {"min_gain": 0.03, "rule": "matching helps only when maker attention controls emissions"},
    "A06": {"rule": "report entry-point gains by expertise; final posteriors converge within 0.05"},
    "A07": {"min_gain": 0.03, "rule": "anomaly attention gains only where handling differs"},
    "A08": {"min_gain": 0.05, "max_null": 0.02, "rule": "menu attention gains when the menu is relevant and not otherwise"},
    "A09": {"min_content": 0.6, "rule": "with strong artifact evidence, a false history leaves the content posterior on the truth above the bar"},
    "A10": {"min_gain": 0.05, "rule": "the adaptive reader's residual projection is below the static local reader's by the bar"},
    "A11": {"min_gap": 0.10, "rule": "narrow monitoring's post-change error exceeds broad monitoring's by the bar"},
    "A12": {"min_gap": 0.10, "rule": "under adversarial salience the salience policy scores below learned by the bar"},
    "A13": {"max_gain": 0.02, "max_inflation": 0.05, "rule": "no proper-score gain and no confidence inflation in any null"},
    "A14": {"rule": "report the policy by ecology surface; no universal policy claim"},
    "O01": {"rule": "constrained inversion beats partialling and the strong-cost shift exceeds the near-tie shift, as in V12"},
    "O02": {"min_gain": 0.03, "rule": "the factored reader beats the total-cost reader on held-out choices by the bar"},
    "O03": {"min_gain": 0.03, "rule": "full-menu beats size-only by the bar in at least two compositions"},
    "O04": {"rule": "posterior shift is larger under attractive and costly-chosen alternatives than under a near tie"},
    "O05": {"min_gain": 0.02, "rule": "at least one dimension's weight transfers across domains by the bar"},
    "O06": {"min_gain": 0.10, "rule": "joint accuracy beats effort-only by the bar"},
    "O07": {"max_imposed": 0.02, "rule": "imposed work supplies at most the bar in preference evidence to the record reader"},
    "O08": {"min_spearman": 0.8, "rule": "inferred goal strength is monotone in planted motivation"},
    "O09": {"min_acc": 0.5, "rule": "four-way epistemic-state accuracy above the bar (chance 0.25)"},
    "O10": {"rule": "the decision-time reader weights anticipated cost above sunk and late-discovered cost; hindsight error reported"},
    "O11": {"min_acc": 0.6, "rule": "planted social and risk tradeoffs recovered above the bar or abstained; no virtue label"},
    "O12": {"rule": "the mimic's size sensitivity is positive and below the exact reader's"},
    "O13": {"min_rise": 0.02, "rule": "sensitivity rises under salience conditions by the bar without changing the choice data"},
    "O14": {"equivalence_delta": 0.01, "rule": "report cross-validated equivalence classes; no declared law"},
    "O15": {"rule": "the hybrid is at least the mimic on held-out choices and no worse calibrated under missing menus"},
    "O16": {"min_ece_gain": 0.02, "rule": "the calibrated reader's ECE under missing or false menus is below the naive reader's by the bar"},
    "O17": {"rule": "proper-score gain over frequency on all five prospective targets"},
    "O18": {"rule": "claimed effort moves the novice's quality judgement more than the expert's; motivation inference is unmoved"},
    "P01": {"max_half_life": 8, "max_residual": 0.10, "max_order": 0.05, "rule": "half-life, residual bias and order effect within bars per route"},
    "P02": {"rule": "correction follows diagnostic validity: behaviour and process records correct more than a false biography"},
    "P03": {"min_gap": 0.10, "rule": "self weight is lower when the decision-relevant mapping differs, at high cue reliability"},
    "P04": {"min_ece_gain": 0.02, "rule": "accurate feedback lowers ECE across targets by the bar"},
    "P05": {"rule": "same-target transfer gains; new-target transfer does not become group certainty"},
    "P06": {"min_gap": 0.05, "rule": "the confidence-rewarded reader's residual projection exceeds the deliberative reader's by the bar"},
    "P07": {"rule": "graded surfaces reported; strong conflict at high reliability corrects more than weak at low"},
    "P08": {"min_gain": 0.02, "rule": "the robust reader beats reset and anchor on outlier and regime-change scenarios by the bar"},
    "P09": {"rule": "prediction and construction cost by route reported; retain the cheap-locality claim on a tie"},
    "P10": {"min_gain": 0.02, "rule": "posterior exchange among four readers beats the single reader by the bar"},
    "P11": {"rule": "correlated ensembles are more overconfident than independent ones; the penalty is reported"},
    "P12": {"min_gain": 0.05, "rule": "partial correspondence yields a target gain over the stereotype at sixteen artifacts"},
    "P13": {"min_gain": 0.02, "rule": "the corrected route beats uncorrected self and reset on passive and active targets"},
    "P14": {"max_before": 0.6, "min_after": 0.8, "min_div": 0.1, "rule": "top mass at most the bar before separation and at least the bar after, among pairs whose alternative method is distinguishable (mean JS of the method pair at least min_div); pairs whose methods coincide (JS under 0.01) must stay at abstention, and the band between is reported unjudged"},
    "G01": {"min_acc": 0.6, "rule": "goal accuracy above the bar at twelve artifacts with the surface classifier at chance"},
    "G02": {"min_gain": 0.02, "rule": "joint beats independent for conflicting pairs by the bar and ties for compatible pairs"},
    "G03": {"min_gain": 0.02, "rule": "the multi-goal model beats three-class stance on held-out token selection by the bar"},
    "G04": {"max_err": 0.15, "rule": "reliability posterior within the bar of planted reliability for every source kind"},
    "G05": {"min_move": 0.05, "rule": "content and source posteriors each move by the bar in their own cells"},
    "G06": {"rule": "belief update follows truth; preference movement follows alignment"},
    "G07": {"rule": "efficiency-vulnerability frontier reported; no personality label"},
    "G08": {"rule": "Brier by model and history reported; no universal dynamics claim"},
    "G09": {"min_gain": 0.05, "max_irrelevant": 0.02, "rule": "reinterpretation gains where relevant and not otherwise"},
    "G10": {"min_gap": 0.10, "rule": "costly action and repeated reliability restore trust more than an apology-like claim by the bar"},
    "G11": {"rule": "belief update is unchanged across conflicts while imitation and coordination drop"},
    "G12": {"min_gain": 0.05, "max_shortcut": 0.02, "rule": "same-domain transfer gains; shared-label transfer does not"},
    "G13": {"max_influence": 0.10, "rule": "a false note's influence with strong contradicting evidence is at most the bar"},
    "G14": {"min_gain": 0.05, "rule": "challenge beats static polish by the bar"},
    "G15": {"max_off": 0.02, "rule": "each channel responds only to its declared factors"},
    "G16": {"rule": "proper score, calibration and recovery transfer to fresh sources; failure regions reported"},
    "H01": {"identity_tol": 1e-9, "rule": "role-relative queries answer identically for the same goal; no single absolute level"},
    "H02": {"min_gain": 0.02, "rule": "the role-relative model beats project-only and actor-only by the bar"},
    "H03": {"chance_margin": 0.10, "min_interaction": 0.75, "rule": "artifact-only at floor; the interaction-aware reader above the bar"},
    "H04": {"min_gain": 0.10, "rule": "the graph reader beats coherence on next-control prediction by the bar"},
    "H05": {"rule": "role priors improve calibration and cannot replace target evidence"},
    "H06": {"min_acc": 0.75, "rule": "suppress and amplify routes classified above the bar"},
    "H07": {"min_acc": 0.7, "rule": "attribution follows relation and intervention in crossed cells above the bar"},
    "H08": {"min_acc": 0.7, "rule": "attribution follows role after reassignment above the bar"},
    "H09": {"min_acc": 0.6, "max_top_without": 0.5, "rule": "private goals recovered with evidence; abstained without"},
    "H10": {"min_gain": 0.05, "rule": "the sequential handling model beats frequency on downstream response by the bar"},
    "H11": {"rule": "project reach exceeds local reach; actor identity from reach alone is at chance"},
    "H12": {"min_survival": 0.7, "rule": "survival by trace class reported; goal structure survives global rewrite above the bar"},
    "H13": {"max_top": 0.6, "rule": "artifact-only top mass at most the bar on identical artifacts"},
    "H14": {"min_acc": 0.75, "rule": "the minimal sufficient record set is the first level reaching the bar"},
    "H15": {"min_gain": 0.05, "rule": "the graph model beats the best baseline by the bar"},
    "H16": {"rule": "conditional transfer reported; schema against artifact"},
    "Q01": {"min_agreement": 0.8, "rule": "PyMDP agrees with the exact best probe at least the bar of decisions"},
    "Q02": {"divergence_floor": 0.02, "min_gain": 0.05, "rule": "live gate on pairwise divergence, then exact selection beats random by the bar"},
    "Q03": {"min_rate": 0.6, "rule": "the query buys the highest decision-value field at least the bar of the time"},
    "Q04": {"min_rate": 0.6, "min_gain": 0.02, "rule": "the probe targets the most confused pair and improves held-out prediction"},
    "Q05": {"rule": "entry point changes with expertise; purpose-first reported as the cheap baseline"},
    "Q06": {"min_gain": 0.02, "rule": "attention-informed inspection (reading a channel the inferred allocation did not sharpen) beats attention-blind sampling by the bar"},
    "Q07": {"min_gain": 0.05, "rule": "goal-aware challenge beats passive by the bar when the adversary does not anticipate"},
    "Q08": {"divergence_floor": 0.02, "min_discriminativeness": 0.01, "rule": "every probe set above both floors"},
    "Q09": {"max_regret": 0.10, "rule": "regret against exact at most the bar at mid cost; premature and unnecessary probes reported"},
    "Q10": {"min_gain": 0.05, "rule": "robust or challenge beats naive salience by the bar under a decoy"},
    "Q11": {"rule": "conditional information-per-cost surface; no pooled self headline"},
    "Q12": {"min_agreement": 0.7, "rule": "agreement above the bar under independent coupling; the boundary mapped"},
    "X": {"survival": "sign kept and at least half the magnitude retained", "rule": "an attacked effect survives, narrows, or dies; dying under a causal-variable-preserving attack makes the result a shortcut"},
}
for _l in range(1, 13):
    CRITERIA[f"L{_l:02d}"] = {"rule": "deliverable names the licensing card, transfer gap, target, records, baseline, positive gate, failure meaning and human boundary"}

CLOSURES = {
    "I": "any failed identity or lineage gate stops dependent claims, not unrelated implementation",
    "C": "close self privilege if self never beats a truly equal local prior; close the common-prior mechanism only if neither common nor local priors improve efficiency, calibration or conditional prediction after one repair",
    "A": "retain attention as an entry-efficiency heuristic if final inference is unchanged; close the constructed wing if it saves neither compute nor inspection",
    "O": "return an equivalence class and close the affected interpretation if the cost vector cannot identify its causes even with records; retain a predictive instrument without psychological naming",
    "P": "close a correction mechanism that cannot distinguish diagnostic target evidence from vivid false context after one repair; retain non-identifiability where no separating observation exists",
    "G": "close the artifact-only claim if stance is readable only through unmatched features; retain the simpler reader if factored trust never beats a scalar after one repair",
    "H": "retain a records-dominant boundary if only records separate matched histories; never weaken the shared-brief rival",
    "Q": "close the world as lacking useful action if exact selection cannot beat passive baselines; close the present PyMDP reader if exact succeeds and PyMDP fails after one repair",
    "L": "an unlicensed shape is not exported",
}

FLIGHTS = {
    "nested_common_ground": ["C04 or C05 conditional gain", "P01 correction", "C14 continuation", "C16 transfer", "X04 X06 survive"],
    "cost_aware_maker_inference": ["O02 O06 O07 separation", "O17 prospective", "O13 salience response", "O16 calibration", "X09 X10 X11 survive"],
    "attention_as_safe_allocation": ["A02 or A03 gain", "A10 reallocation", "A12 adversarial salience", "A13 nulls", "A14 transfer", "X08 survives"],
    "factored_epistemic_vigilance": ["G01 stance", "G04 G05 separation", "G13 false context", "G15 channels", "G16 transfer", "X07 X15 survive"],
    "readable_interaction_hand": ["H03 director vs brief", "H15 next intervention", "H07 crossed styles", "H12 rewrite", "H13 abstention", "X13 survives"],
}

ATTACK_RELEVANCE = {
    "nested_common_ground": ["X01", "X02", "X03", "X04", "X05", "X06", "X14", "X16", "X17", "X18", "X19", "X20"],
    "cost_aware_maker_inference": ["X01", "X02", "X09", "X10", "X11", "X12", "X16", "X17", "X18", "X19", "X20"],
    "attention_as_safe_allocation": ["X01", "X03", "X08", "X15", "X16", "X17", "X18", "X19", "X20"],
    "factored_epistemic_vigilance": ["X01", "X02", "X07", "X12", "X14", "X15", "X16", "X17", "X18", "X19", "X20"],
    "readable_interaction_hand": ["X01", "X02", "X03", "X12", "X13", "X16", "X17", "X18", "X19", "X20"],
}
FLIGHT_PRIMARY_CARD = {"nested_common_ground": "C04", "cost_aware_maker_inference": "O02",
                       "attention_as_safe_allocation": "A03", "factored_epistemic_vigilance": "G01",
                       "readable_interaction_hand": "H03"}


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hj(obj) -> str:
    return _h(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def structural_payload() -> dict:
    from .validation.soundingline.v13.manifest import CELLS_TEMPLATE, build_cards
    cards = [c.to_dict() for c in build_cards()]
    gen = {f: _h((_V13 / f).read_bytes()) for f in GENERATOR_FILES if (_V13 / f).exists()}
    return {"module_sha256": _h(Path(__file__).read_bytes()), "cards_sha256": _hj(cards),
            "criteria_sha256": _hj(CRITERIA), "closures_sha256": _hj(CLOSURES), "flights_sha256": _hj(FLIGHTS),
            "attack_relevance_sha256": _hj(ATTACK_RELEVANCE), "generators": gen,
            "cells_template_sha256": _h(CELLS_TEMPLATE.read_bytes()) if CELLS_TEMPLATE.exists() else None,
            "report_interface_sha256": _h(REPORT_INTERFACE.read_bytes()) if REPORT_INTERFACE.exists() else None,
            "n_cards": len(cards)}


def write_structural_lock() -> dict:
    lock = {"program": "v13", "kind": "structural", "locked": True, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "internal_prespecification_not_external_preregistration": True, **structural_payload(),
            "criteria": CRITERIA, "closures": CLOSURES, "flights": FLIGHTS, "attack_relevance": ATTACK_RELEVANCE}
    STRUCTURAL_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8", newline="
")
    return lock


def write_workload_lock(tier_name: str, tier: dict, expansions: list, forecast: dict, pilot_summary: dict) -> dict:
    from .validation.soundingline.v13.manifest import CELLS
    lock = {"program": "v13", "kind": "workload", "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "tier": tier_name,
            "tier_config": tier, "expansions_instantiated": expansions, "forecast": forecast,
            "pilot_summary_sha256": _hj(pilot_summary), "pilot_is_non_scientific": True,
            "cells_sha256": _h(CELLS.read_bytes()) if CELLS.exists() else None}
    WORKLOAD_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8", newline="
")
    return lock


def write_scientific_lock() -> dict:
    st = json.loads(STRUCTURAL_PATH.read_text(encoding="utf-8"))
    wl = json.loads(WORKLOAD_PATH.read_text(encoding="utf-8"))
    lock = {"program": "v13", "kind": "scientific", "locked": True, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "internal_prespecification_not_external_preregistration": True,
            "structural_sha256": _h(STRUCTURAL_PATH.read_bytes()), "workload_sha256": _h(WORKLOAD_PATH.read_bytes()),
            "structural": {k: st[k] for k in ("module_sha256", "cards_sha256", "criteria_sha256", "generators", "cells_template_sha256")},
            "tier": wl["tier"], "cells_sha256": wl.get("cells_sha256"), "amended_after_data": []}
    PREREG_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8", newline="
")
    return lock


def lock_status() -> dict:
    if not STRUCTURAL_PATH.exists():
        return {"locked": False, "stage": "none", "reason": "no structural lock; run python -m ghostscale.prereg_v13"}
    st = json.loads(STRUCTURAL_PATH.read_text(encoding="utf-8"))
    now = structural_payload()
    keys = ("module_sha256", "cards_sha256", "criteria_sha256", "closures_sha256", "flights_sha256", "attack_relevance_sha256", "generators")
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
