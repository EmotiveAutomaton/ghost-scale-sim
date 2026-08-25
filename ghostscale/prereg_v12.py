"""Pre-registration for V12 — The Other Model. Internal pre-specification, hash-locked before
scientific execution; NOT external preregistration (spec section 21.1).

The lock hashes this module and the card manifest. Cards amended after data are identified by
the manifest's amendment ledger and retain their original hash beside the replacement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .validation.soundingline.v12 import v12_dir

PREREG_PATH = v12_dir() / "prereg_v12_lock.json"

# Numeric bars are derived before the full run from exact chance, the known-positive gap, or the
# smallest effect that would alter pursuit. They may not be moved after a result is seen.
CRITERIA = {
    "I01": {"anchor_tolerance": 1e-9,
            "rule": "every scientific field of S-15/S-14 reproduced within tolerance; provenance may differ"},
    "I02": {"min_draws": 50, "rule": "report reproduction surfaces for S-14 C5/C6 and S-15 C1/C3; "
                                      "Morris then Sobol on the surviving parameters"},
    "I03": {"identity_tol": 1e-6, "rule": "solvers agree in the factor-independent world; report the "
                                          "coupling at which mean-field max deviation exceeds 0.10"},
    "I04": {"chance_margin": 0.10, "rule": "no-information, shuffled, and permuted-correspondence "
                                           "posteriors sit within margin of chance"},
    "I05": {"min_js": 0.01, "rule": "every manipulation moves the emission by at least min_js; "
                                    "otherwise the card is VOID"},
    "I06": {"rule": "seed stability, disjoint lineages, points naming, hashes present"},
    "S01": {"min_gain_over_frequency": 0.05, "rule": "held-out self log score beats frequency by the bar"},
    "S02": {"min_spearman": 0.8, "rule": "each axis recovers its planted ordering at rank correlation >= bar"},
    "S03": {"rule": "report expert-minus-corrupted gap only in cells where the expert is in [0.15, 0.90] "
                    "accuracy; the V11 C2 failure stands"},
    "S04": {"min_selective_gain": 0.05, "rule": "self-first minus information-matched generic >= bar at the "
                                               "nearest distance AND <= 0 at the farthest, on log score"},
    "S05": {"rule": "surface reported; self-directed error must fall with evidence dose (Spearman <= -0.5)"},
    "S06": {"max_residual_bias": 0.10, "rule": "after decisive conflict, residual self-directed error <= bar; "
                                              "order effect on final posterior <= 0.05 in top-1 probability"},
    "S07": {"min_gain": 0.03, "rule": "self-first minus generic on hidden-continuation log score >= bar"},
    "S08": {"rule": "the frozen route's S04 gain on fresh worlds and the dialect domain retains sign"},
    "S09": {"rule": "risk at matched 0.6 coverage lower for self-first than generic where S04 gain held"},
    "S10": {"rule": "policy and process axes retain partial R2 >= 0.05 after surface and source"},
    "Q01": {"min_agreement": 0.8, "rule": "PyMDP probe agrees with the exact EIG-best probe at >= bar of decisions"},
    "Q02": {"min_gain": 0.05, "rule": "chosen commission's realized information gain beats random by bar"},
    "Q03": {"rule": "self-first selection log score >= uncertainty sampling; report vs oracle pair"},
    "Q04": {"rule": "purchases correlate with expected discrimination, not with polish (partial r)"},
    "Q05": {"rule": "regret vs exact optimum reported across cost; premature and unnecessary rates"},
    "Q06": {"rule": "active gain under adversary reported vs uncertainty sampling; closure if never better"},
    "B01": {"max_entropy_gap": 1e-9, "rule": "regimes matched on pair mass and entropy to tolerance"},
    "B02": {"rule": "cooperative reader gains on bards AND loses on concealers relative to the neutral reader"},
    "B03": {"rule": "regime posterior recovers the true regime after a switch within 12 artifacts"},
    "B04": {"rule": "regime-aware active selection >= uncertainty sampling on regime log score"},
    "B05": {"rule": "readability by concealment type reported; confidently-wrong rate per type"},
    "B06": {"rule": "own payoff with accurate adversary model >= without; adoption not required"},
    "U01": {"rule": "zero-weight bit identity; oracle moves toward maker; uniform posterior no direction"},
    "U02": {"rule": "wrong-and-confident cell has the highest wrong-direction rate; accuracy gates movement"},
    "U03": {"rule": "uncertainty-aware representations have lower catastrophic movement than MAP under equifinality"},
    "U04": {"rule": "each factor's effect estimated with the others held; no substitution"},
    "U05": {"rule": "process channel moves more than preference channel at matched exposure"},
    "U06": {"rule": "competent-divergent maker worsens own task under unconditional uptake"},
    "U07": {"rule": "reliable counterevidence reverses harmful movement while process gain is retained"},
    "U08": {"rule": "cumulative movement and reversal reported; constructed accumulation analogue only"},
    "R01": {"rule": "every latent has conditional entropy given the others above 0 (no coarsening)"},
    "R02": {"rule": "constrained inversion held-out log score >= partialling by 0.02"},
    "R03": {"rule": "partialling recovery drops under interacting habits more than constrained inversion"},
    "R04": {"rule": "cross-domain held-out log score beats identity-only baseline"},
    "R05": {"rule": "posterior shift under large opposing cost exceeds shift under near tie"},
    "R06": {"rule": "joint recovery of goal and profile above marginal recovery"},
    "R07": {"rule": "prospective log score gain over frequency on all four targets"},
    "R08": {"rule": "artifact-only reader abstains on equifinal pairs; records separate them"},
    "T01": {"rule": "exact ledger reported; deterministic edges flagged"},
    "T02": {"rule": "full supply matrix reported"},
    "T03": {"rule": "profile-supplied process gain reported on the same scale as process-supplied goal gain"},
    "T04": {"rule": "mechanic-supplied gains by mechanic type"},
    "T05": {"rule": "shapes reported only from off-ceiling cells"},
    "T06": {"rule": "topology equivalence class reported"},
    "D01": {"rule": "ecologies matched on quality, counts, style within tolerance"},
    "D02": {"rule": "director intervention reach > local intervention reach"},
    "D03": {"rule": "artifact-only accuracy vs coherence baseline reported"},
    "D04": {"rule": "per-level attribution accuracy reported; token share baseline"},
    "D05": {"rule": "survival by rewrite strength reported"},
    "D06": {"rule": "prospective log score vs baseline"},
    "D07": {"rule": "abstention on identical artifacts; separation when later artifacts differ"},
    "F01": {"rule": "bit difference under manipulation"},
    "F02": {"rule": "marginals matched to tolerance"},
    "F03": {"rule": "AUC by ruler on known worlds; validated on null worlds"},
    "F04": {"rule": "three-way accuracy"},
    "F05": {"rule": "survival under rewrite and template"},
}

CLOSURES = {
    "S": "close if self-first never beats the matched generic, does not interact with similarity, "
         "and worsens correction or calibration after one repair",
    "Q": "close if active selection never beats uncertainty sampling or recovers its cost",
    "B": "artifact-only VOID if regime cannot be varied without an unmatched surface channel",
    "U": "close if policy moves only when preference is directly written into C_AIF",
    "R": "close any estimator failing held-out cross-context prediction after one repair",
    "T": "no topology claim if determined by factorization or ceiling",
    "D": "close artifact-only attribution if logs decide while artifacts sit at floor",
    "F": "artifact-underdetermined if no validated statistic separates matched worlds",
}

FLIGHTS = {
    "self_to_other": ["S04 selective gain", "X01/X02/X05 survive", "S06 correction", "S07 or Q gain",
                      "S08 fresh domain", "exact/PyMDP agree"],
    "selective_uptake": ["calibration", "U04 separation", "U05 process not preference", "U03 gating",
                         "U07 reversal", "fresh-domain benefit"],
    "hierarchical_hand": ["D02 reach", "D03 above baseline", "D06 prospective", "D05 survives rewrite",
                          "D04 calibrated"],
}


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def lock_payload() -> dict:
    from .validation.soundingline.v12.manifest import MANIFEST, build_cards
    cards = [c.to_dict() for c in build_cards()]
    return {"module_sha256": _hash_bytes(Path(__file__).read_bytes()),
            "cards_sha256": _hash_bytes(json.dumps(cards, sort_keys=True).encode("utf-8")),
            "criteria_sha256": _hash_bytes(json.dumps(CRITERIA, sort_keys=True).encode("utf-8")),
            "n_cards": len(cards), "manifest_path": str(MANIFEST)}


def write_lock() -> dict:
    lock = {"program": "v12", "locked": True, "internal_prespecification_not_external_preregistration": True,
            **lock_payload(), "criteria": CRITERIA, "closures": CLOSURES, "flights": FLIGHTS,
            "amended_after_data": []}
    PREREG_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def lock_status() -> dict:
    if not PREREG_PATH.exists():
        return {"locked": False, "reason": "no lock; run python -m ghostscale.prereg_v12"}
    lock = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    now = lock_payload()
    current = all(lock.get(k) == now[k] for k in ("module_sha256", "cards_sha256", "criteria_sha256"))
    return {"locked": bool(current), "internal_prespecification_not_external_preregistration": True,
            **({} if current else {"reason": "prereg module, criteria, or card set changed after lock"})}


if __name__ == "__main__":
    write_lock()
    print(f"locked -> {PREREG_PATH}")
