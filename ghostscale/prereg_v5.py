"""V5 pre-registration — criteria as executable code, content-hash locked before any run.

Same machinery as ``prereg_v4_5.py`` and for the same reason: the written criterion and the
applied criterion are ONE OBJECT, so a criterion cannot drift from the thing the experiment
actually computes. The experiments and the tests both call these functions.

-----------------------------------------------------------------------------------------
WHAT IS LOCKED HERE AND WHEN IT WAS LOCKED, because V5's own history makes the question fair.

N21's second clause was restated after the first version failed (V5 deviation 1, logged in
RESULTS_V5.md and in ``n21_depth_not_effort``'s docstring). That restatement happened while
NOTHING WAS PRE-REGISTERED — this file did not exist. It was written immediately afterwards,
before N21 ran at scale and before a single line of E30, and it locks BOTH clauses: the one
that decides and the one that was replaced and is retained beside it.

That is the mitigation and it is not a defence. An operationalisation changed after seeing a
measurement is an operationalisation changed after seeing a measurement, whatever was locked
afterwards, and V5's results section says so in those words.

E30's criteria are locked here too, in the same payload, before E30 was built. So they were
fixed before N21 reported at scale — which is the property that matters, since a criterion for
E30 chosen after seeing whether its own gate passed would be worthless.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import Config

# --------------------------------------------------------------------------- #
# N21 — the mu/beta dissociation null. The gate on E30.
# --------------------------------------------------------------------------- #
N21_DOMINANCE = 3.0
N21_CELLS = (("committed_shallow", 1.00, 1), ("committed_deep", 1.00, 3),
             ("offhand_shallow", 0.25, 1), ("offhand_deep", 0.25, 3))

# --------------------------------------------------------------------------- #
# E30 — mu replaces beta. A NEW DESIGN, not a re-run of E28 (V5 decision 8).
# --------------------------------------------------------------------------- #
E30_MU_GRID = (1, 2, 3)
E30_DELTA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)   # depth CONTRAST, the continuous axis

# Spearman(true mu, recovered mu) at or above this. Three grid points, so this is a
# monotonicity requirement and is stated as one rather than dressed as a correlation result.
E30_RECOVERY_RHO = 0.99

# THE PREDICTION (V5 §E30): update magnitude scales with recovered depth.
E30_UPDATE_RHO = 0.99

# ...WHILE GOAL ACCURACY HOLDS ACROSS A WIDER RANGE THAN BETA ACHIEVED. This is the clause
# that makes E30 more than E28 with a new variable, and it is stated against E28's OWN
# measured number rather than against a bare threshold: E28 held accuracy at 0.904 at the
# midpoint of its range and 0.257 at the bottom. mu must hold accuracy at EVERY level of its
# grid, including the shallowest, because depth is constructed so that a shallow artifact is
# still perfectly legible -- that is the whole content of "depth is correlational, not
# marginal". If accuracy falls with depth, the construction has leaked legibility into mu and
# mu is doing kappa_p's job.
E30_ACCURACY_FLOOR = 0.90
E28_ACCURACY_AT_BOTTOM_OF_RANGE = 0.257      # measured, RESULTS_V4_5.md
E28_ACCURACY_AT_MIDPOINT = 0.904             # measured, RESULTS_V4_5.md


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def n21_verdict(mu_effect: float, beta_false_depth: float,
                mu_effect_at_high_beta: float, mu_effect_at_low_beta: float) -> dict:
    """The gate on E30. See ``n21_depth_not_effort`` for the deviation behind clause (b)."""
    ordering = bool(mu_effect_at_high_beta > 0 and mu_effect_at_low_beta > 0)
    ratio = (float(mu_effect / beta_false_depth) if abs(beta_false_depth) > 1e-12
             else float("inf"))
    passed = bool(ordering and ratio >= N21_DOMINANCE)
    return {
        "passed": passed,
        "verdict": "DEPTH_RECOVERED_NOT_EFFORT" if passed else "BETA_CAN_MANUFACTURE_DEPTH",
        "dominance_ratio": ratio,
        "dominance_required": N21_DOMINANCE,
        "ordering_holds_at_both_beta_levels": ordering,
    }


def e30_verdict(mu_grid, recovered_mu, accuracy, update_magnitude) -> dict:
    """E30's outcome, with every branch written before the run.

    FOUR BRANCHES, and the third is the one a careless read would report as the prediction.

    * DEPTH_IS_SEPARABLE — update scales with depth AND accuracy holds everywhere. The
      predicted result, and the one that says mu does work neither existing gate can express.
    * DEPTH_MOVES_NOTHING — accuracy holds but update does not scale. mu is recoverable and
      inert, which would mean depth is not what gates uptake and C1's central claim is wrong.
    * DEPTH_IS_LEGIBILITY — update scales but accuracy falls with depth. mu is then doing
      kappa_p's job under a new name: shallow content would have to be harder to read, and the
      construction says it is exactly as readable. This branch existing is what stops the
      welcome answer from being the only reachable one.
    * DEPTH_DOES_NEITHER — neither holds.
    """
    rho_recovery = spearman(mu_grid, recovered_mu)
    rho_update = spearman(mu_grid, update_magnitude)
    acc = np.asarray(accuracy, dtype=float)
    accuracy_holds = bool(np.all(acc >= E30_ACCURACY_FLOOR))
    update_scales = bool(rho_update >= E30_UPDATE_RHO)

    if update_scales and accuracy_holds:
        verdict = "DEPTH_IS_SEPARABLE"
    elif accuracy_holds:
        verdict = "DEPTH_MOVES_NOTHING"
    elif update_scales:
        verdict = "DEPTH_IS_LEGIBILITY"
    else:
        verdict = "DEPTH_DOES_NEITHER"

    return {
        "verdict": verdict,
        "recovery_rho": rho_recovery,
        "recovery_rho_required": E30_RECOVERY_RHO,
        "depth_is_recoverable": bool(rho_recovery >= E30_RECOVERY_RHO),
        "update_rho": rho_update,
        "update_rho_required": E30_UPDATE_RHO,
        "update_scales_with_depth": update_scales,
        "accuracy_holds_at_every_depth": accuracy_holds,
        "accuracy_floor": E30_ACCURACY_FLOOR,
        "min_accuracy": float(acc.min()) if acc.size else float("nan"),
        "comparison_to_e28": {
            "e28_accuracy_at_bottom_of_range": E28_ACCURACY_AT_BOTTOM_OF_RANGE,
            "e28_accuracy_at_midpoint": E28_ACCURACY_AT_MIDPOINT,
            "claim": ("V5 E30 requires accuracy to hold across a WIDER range than beta "
                      "achieved. beta collapsed to 0.257 at the bottom of its grid because "
                      "low-beta content carries no goal information; depth is constructed so "
                      "that shallow content carries exactly as much as deep content, so "
                      "accuracy must hold at EVERY depth or the construction has leaked."),
        },
    }


# --------------------------------------------------------------------------- #
# The locked payload.
# --------------------------------------------------------------------------- #
def build_preregistration_v5(cfg: Config) -> dict:
    payload = {
        "version": "V5",
        "scope": ("C1 only: mu as estimated model depth, its dissociation null N21, and E30. "
                  "C2/C3/C4 and E31-E34 are not covered and must pre-register separately."),
        "construction": {
            "mu_levels": list((1, 2, 3)),
            "n_subgoals": int(cfg.get("v5.subgoals.n", 4)),
            "dwell": float(cfg.get("v5.subgoals.dwell", 9.2)),
            "delta": float(cfg.get("v5.depth.delta", 1.0)),
            "mode_js_floor": float(cfg.get("v5.depth.mode_js_floor", 0.10)),
            "asserted_at_construction": [
                "mu = 1 reproduces V4's A[0] elementwise, tolerance 1e-12",
                "beta at mu = 1 reduces to V4.5's sig_EXPLORE, tolerance 1e-12",
                "execution modes average EXACTLY to sig[g] (depth is correlational)",
                "sub-goal chains doubly stochastic with uniform stationary distribution",
                "execution modes pairwise separated above the JS floor",
                "execution modes carry < 0.10 mass in the FOREIGN block",
                "three to four mode blocks fit a 24-step artifact",
            ],
        },
        "N21": {
            "role": "gate on E30. E30 does not count unless this passes.",
            "cells": [list(c) for c in N21_CELLS],
            "clause_a": "recovered mu higher at mu = 3 than mu = 1, at BOTH beta levels",
            "clause_b": (f"mu effect exceeds beta's FALSE-DEPTH effect (the mu = 1 row) by "
                         f"{N21_DOMINANCE}x"),
            "clause_b_deviation": (
                "clause (b) originally averaged beta's simple effects over both mu levels; it "
                "failed at 1.67 and was restated to score the mu = 1 row. V5 deviation 1. The "
                "original is retained, computed, reported, and decides nothing. The threshold "
                "value was NOT changed."),
        },
        "E30": {
            "design": ("a NEW design, not a re-run of E28 (V5 decision 8): under route A the "
                       "creator is hierarchical, so there is no version of E28's stimulus to "
                       "re-run. E28 and E30 are reported side by side as two constructs on "
                       "comparable designs, and E28 is not deleted."),
            "mu_grid": list(E30_MU_GRID),
            "delta_grid": list(E30_DELTA_GRID),
            "predicted": ("update magnitude scales with recovered depth while goal accuracy "
                          "holds at EVERY depth — a wider range than beta achieved, which "
                          "collapsed to 0.257 at the bottom of its grid"),
            "recovery_rho": E30_RECOVERY_RHO,
            "update_rho": E30_UPDATE_RHO,
            "accuracy_floor": E30_ACCURACY_FLOOR,
            "outcomes": ["DEPTH_IS_SEPARABLE", "DEPTH_MOVES_NOTHING", "DEPTH_IS_LEGIBILITY",
                         "DEPTH_DOES_NEITHER"],
        },
        "unchanged_from_v4_5": {
            "E8": "stays withheld with its xfail(strict) marker",
            "E27": "the V3 residual stays open and is not touched",
            "E28/E29": "stand as measurements of mis-specified constructs, labelled as such",
            "null_suite": "unchanged and must continue to pass",
        },
    }
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return payload


def _canonical(payload: dict) -> str:
    return json.dumps({k: v for k, v in payload.items() if k != "content_hash"},
                      sort_keys=True, separators=(",", ":"))


def write_preregistration_v5(cfg: Config, path: Path, force: bool = False) -> dict:
    payload = build_preregistration_v5(cfg)
    path = Path(path)
    if path.exists() and not force:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if existing is not None and existing.get("content_hash") != payload["content_hash"]:
            raise RuntimeError(
                f"{path.name} already exists with a DIFFERENT content hash.\n"
                f"  on disk: {existing.get('content_hash')}\n"
                f"  now:     {payload['content_hash']}\n"
                "V5's criteria were pre-registered and must not change after the fact. If any "
                "V5 experiment has already run, changing this file is the failure V4 spec §7 "
                "names. Investigate, or pass force=True and record it as a deviation.")
        if existing is not None:
            return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def assert_prereg_locked_v5(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. No V5 experiment may run before its criteria are "
            "pre-registered.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_hash")
    recomputed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if stated != recomputed:
        raise RuntimeError(
            f"{path.name} has been modified since it was written (hash {stated} != recomputed "
            f"{recomputed}). The pre-registered criteria are not trustworthy; the V5 "
            f"programme will not run.")
    return payload
