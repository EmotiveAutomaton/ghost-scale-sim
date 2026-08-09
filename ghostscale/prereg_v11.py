"""Pre-registration for V11 — The Maker. Hash-locked, written from SPEC before any V11 run.

Criteria are executable predicates over the verdicts the modules will emit, so the written
criterion and the applied criterion are the same object. ``python -m ghostscale.prereg_v11``
writes the lock; the modules refuse to treat their criteria as pre-registered if the lock is
absent or stale.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .v11 import PREREG_PATH

CARD = {
    "version": "v11",
    "title": "The Maker — a persistent value profile, and what a bounded-family reader recovers",
    "written_before_code": True,
    "world_constants": {
        "profile_family": {
            "uniform": [0.25, 0.25, 0.25, 0.25],
            "peaked_k": "0.70 on k, 0.10 elsewhere, k in 0..3",
            "bimodal": [0.40, 0.40, 0.10, 0.10],
        },
        "n_timesteps": 24,
        "s14": {"epsilon_unused": 0.02, "amplification": 4.0, "compliance_lambda": 0.5},
        "s15": {"n_makers": 60, "n_artifacts": 50,
                "n_grid": [1, 2, 3, 5, 8, 12, 20, 30, 50],
                "unbounded_family_size": 64, "wrong_expertise_d": 0.5,
                "noisy_tier": "CURATOR", "noisy_reader_d": 0.25},
        "s12": {"n_units": 200, "n_positions": 30, "n_reps": 40},
    },
    "criteria": {
        "C1_convergence": ("s15 bounded_clean: accuracy(n=1) <= 0.60 AND accuracy(n=50) >= 0.95 "
                           "AND spearman(accuracy, n) >= 0.90"),
        "C2_assumption_price": ("s15 L1 at n=50: bounded_clean beats unbounded_family by >= 0.05 "
                                "AND beats wrong_expertise by >= 0.05"),
        "C3_construction_gap": "s15 accuracy at n=1: construction_B minus bounded_clean >= 0.25",
        "C4_corpus_price": "s15 bounded_noisy: report smallest n with accuracy >= 0.90. No bar",
        "C5_aperture": ("s14: commissioned masked-vs-unused accuracy at n=12 >= 0.85 AND "
                        "commissioned minus spontaneous accuracy >= 0.20"),
        "C6_how_channel": "s14: with lambda=1 (pure compliance) commissioned accuracy <= 0.60",
        "C7_smear": "s12: three-locus mean profile reads unimodal in >= 80% of repetitions",
        "C8_separation": "s12: early-late residual statistic separates the worlds at AUC >= 0.80",
    },
    "authors_recorded_priors": {
        "C3": ("The sibling's real-text record — single-artifact values attempts all failed, "
               "within-maker multi-work designs all worked, no acceleration at stacked "
               "motivations — sides with construction A. If B wins the discrimination the "
               "record's reading of those results must be revisited."),
        "C7": ("If the three-locus world does NOT smear, the field's mid-peak consensus cannot "
               "be explained this way and the trimodal architecture answers to the data as "
               "published."),
    },
    "queued_not_run": ["T-11 off-ceiling T-1 restatement", "T-12 supply arms (G47, G56)",
                       "T-13 residue duel + domains", "T-14 bard/concealer",
                       "T-15 flattened intent", "AL-6 anti-capture", "SV-T severity mini-pass"],
}


def evaluate_c1(acc_by_n: dict) -> dict:
    from scipy.stats import spearmanr
    ns = sorted(int(k) for k in acc_by_n)
    accs = [float(acc_by_n[str(n)] if str(n) in acc_by_n else acc_by_n[n]) for n in ns]
    rho = float(spearmanr(ns, accs).statistic)
    return {"acc_n1": accs[0], "acc_n50": accs[-1], "spearman": rho,
            "passed": bool(accs[0] <= 0.60 and accs[-1] >= 0.95 and rho >= 0.90)}


def evaluate_c2(l1_clean: float, l1_unbounded: float, l1_wrong: float) -> dict:
    return {"l1_clean": l1_clean, "l1_unbounded": l1_unbounded, "l1_wrong": l1_wrong,
            "margin_unbounded": l1_unbounded - l1_clean,
            "margin_wrong": l1_wrong - l1_clean,
            "passed": bool(l1_unbounded - l1_clean >= 0.05 and l1_wrong - l1_clean >= 0.05)}


def evaluate_c3(acc_b_n1: float, acc_a_n1: float) -> dict:
    return {"acc_B_n1": acc_b_n1, "acc_A_n1": acc_a_n1, "gap": acc_b_n1 - acc_a_n1,
            "passed": bool(acc_b_n1 - acc_a_n1 >= 0.25)}


def evaluate_c5(acc_commissioned: float, acc_spontaneous: float) -> dict:
    return {"commissioned": acc_commissioned, "spontaneous": acc_spontaneous,
            "gap": acc_commissioned - acc_spontaneous,
            "passed": bool(acc_commissioned >= 0.85
                           and acc_commissioned - acc_spontaneous >= 0.20)}


def evaluate_c6(acc_lambda1: float) -> dict:
    return {"commissioned_lambda1": acc_lambda1, "passed": bool(acc_lambda1 <= 0.60)}


def evaluate_c7(unimodal_fraction: float) -> dict:
    return {"unimodal_fraction": unimodal_fraction, "passed": bool(unimodal_fraction >= 0.80)}


def evaluate_c8(auc: float) -> dict:
    return {"auc": auc, "passed": bool(auc >= 0.80)}


def card_hash() -> str:
    src = Path(__file__).read_bytes()
    return hashlib.sha256(src).hexdigest()


def write_lock() -> dict:
    lock = {"card": CARD, "sha256_of_this_module": card_hash(), "locked": True}
    PREREG_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def lock_status() -> dict:
    """The modules stamp this into their verdicts: locked-and-current, or not."""
    if not PREREG_PATH.exists():
        return {"locked": False, "reason": "no lock file; run python -m ghostscale.prereg_v11"}
    lock = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    current = lock.get("sha256_of_this_module") == card_hash()
    return {"locked": bool(current), "sha256": card_hash(),
            **({} if current else {"reason": "prereg module edited after lock"})}


if __name__ == "__main__":
    write_lock()
    print(f"locked -> {PREREG_PATH}")
