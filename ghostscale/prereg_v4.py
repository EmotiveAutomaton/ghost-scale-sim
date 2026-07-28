"""V4 acceptance criteria, as executable, hash-locked code.

Same discipline as ``prereg_v3``: the criteria live here as functions the experiments and the
nulls both call, and the JSON file records their parameters plus a content hash. The written
criterion and the applied criterion are the same object, so they cannot drift.

V4 adds one thing V3 did not need. ``sig_EXPLORE`` is an ARRAY, not a threshold, and V4 spec
§6 and §7 both name it as the most tunable object in the build and the one most likely to be
quietly adjusted if E19 disappoints. So the pre-registration stores the signature itself,
elementwise, and N17 asserts the constructed array still matches. Changing the feature
partition, the peak mass, the floor, or the averaging rule all move that array and all fail
the lock.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import foreign as FN
from .config import Config

# --------------------------------------------------------------------------- #
# N17 — sig_EXPLORE is not a free parameter.
# --------------------------------------------------------------------------- #
EXPLORE_ATOL = 1.0e-12           # elementwise tolerance against the pre-registered array

# --------------------------------------------------------------------------- #
# N18 — foreign content is goal-directed. The reframe's core claim.
# Defaults mirror config v4.foreign.*; the config is authoritative and is recorded.
# --------------------------------------------------------------------------- #
N18_MI_FLOOR_DEFAULT = 0.8

# --------------------------------------------------------------------------- #
# E19 — the decisive experiment. Thresholds fixed BEFORE the run.
#
# The observer's goal space has 5 states when EXPLORE is on, so a flat posterior puts 0.20 on
# EXPLORE and has entropy ln(5) = 1.609 nats. Both thresholds below are set well clear of
# those baselines so that "absorbed" cannot be satisfied by an observer that simply failed to
# rule EXPLORE out.
# --------------------------------------------------------------------------- #
E19_ABSORB_MASS = 0.50           # mean posterior mass on EXPLORE, against a 0.20 flat baseline
E19_CONVERGED_ENTROPY = 0.50     # mean final goal-posterior entropy, against ln(5) = 1.609

# CORRECTION MADE BEFORE THE RUN, AND BEFORE THIS FILE WAS FIRST WRITTEN TO results/.
# Logged as V4 deviation 2 because it is a change to a stated criterion and the fact that it
# happened at smoke scale does not exempt it from being declared.
#
# The first draft made absorption conjunctive on THREE clauses: mass, convergence, and
# sustained engagement (>= 0.50 of free timesteps DEEP). A smoke run showed the engagement
# clause is invalid as a component of absorption, for a reason visible in the control cells:
#
#   human_directed / explore_off    accuracy 1.00   entropy 0.00007   engaged 0.000
#   human_directed / explore_on     accuracy 1.00   entropy 0.011     engaged 0.000
#
# Disengagement is what SUCCESS looks like. An observer that resolves the goal has nothing
# left to learn and correctly stops paying. The canonical success case in the entire model
# scores zero on engagement, so requiring high engagement would have failed E19's positive
# control (human_exploratory/explore_on measured mass 0.871, entropy 0.265, engaged 0.009)
# for behaving exactly as the theory says it should.
#
# The generative crash was never "disengages". It is "disengages WITHOUT having resolved
# anything", which is the joint of low engagement and HIGH entropy. That joint is now measured
# and reported by ``crash_signature`` as its own quantity, rather than smuggled into
# absorption where it inverts the meaning of one of its own clauses.
#
# THIS CORRECTION CANNOT MANUFACTURE THE WELCOME ANSWER, and that was checked before making
# it. The foreign cell fails absorption on MASS (measured 0.198 against a 0.20 flat baseline
# and a 0.50 threshold), which is the primary clause and is untouched here. Dropping the
# engagement clause rescues the positive control and does not move the decisive cell.
E19_ENGAGED_FRACTION = 0.50      # retained for the crash signature, NOT for absorption


def _canonical(payload: dict) -> str:
    scrubbed = {k: v for k, v in payload.items() if k != "content_hash"}
    return json.dumps(scrubbed, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# The criteria as functions.
# --------------------------------------------------------------------------- #
def explore_absorbed(mean_explore_mass: float, mean_final_entropy: float,
                     mean_engaged_fraction: float = float("nan")) -> dict:
    """Did EXPLORE absorb this content?

    CONJUNCTIVE on two clauses. Absorption means the observer put real mass on EXPLORE AND
    converged rather than staying spread. Mass alone is not enough: high EXPLORE mass with
    high entropy is an observer that cannot decide and has parked probability on the vaguest
    hypothesis available, which is not the same as explaining anything.

    ``mean_engaged_fraction`` is accepted and recorded but is NOT part of the test. See the
    correction note above ``E19_ENGAGED_FRACTION`` for why: in this model disengagement is
    what success looks like, so an engagement clause would fail the positive control.
    """
    m = float(mean_explore_mass)
    h = float(mean_final_entropy)
    e = float(mean_engaged_fraction)
    mass_ok = bool(np.isfinite(m) and m >= E19_ABSORB_MASS)
    conv_ok = bool(np.isfinite(h) and h <= E19_CONVERGED_ENTROPY)
    return {
        "explore_mass": m, "final_entropy": h, "engaged_fraction": e,
        "mass_ok": mass_ok, "converged_ok": conv_ok,
        "absorbed": bool(mass_ok and conv_ok),
        "criterion": (f"EXPLORE mass >= {E19_ABSORB_MASS} AND final entropy <= "
                      f"{E19_CONVERGED_ENTROPY}"),
    }


def crash_signature(mean_final_entropy: float, mean_engaged_fraction: float) -> dict:
    """The generative crash, stated as its own quantity rather than folded into absorption.

    The crash was never "the observer disengages". A successful reader disengages too, as soon
    as it has resolved the goal and has nothing left to buy. The crash is disengagement WITHOUT
    resolution: the observer stops paying while still not knowing what the artifact was for.
    """
    h = float(mean_final_entropy)
    e = float(mean_engaged_fraction)
    unresolved = bool(np.isfinite(h) and h > E19_CONVERGED_ENTROPY)
    disengaged = bool(np.isfinite(e) and e < E19_ENGAGED_FRACTION)
    return {
        "final_entropy": h, "engaged_fraction": e,
        "unresolved": unresolved, "disengaged": disengaged,
        "crashed": bool(unresolved and disengaged),
        "criterion": (f"final entropy > {E19_CONVERGED_ENTROPY} (unresolved) AND engaged "
                      f"fraction < {E19_ENGAGED_FRACTION} (gave up)"),
    }


def e19_verdict(cells: dict) -> dict:
    """The verdict on hypothesis-space poverty.

    ``cells`` maps "<content>/<arm>" to the dict returned by ``explore_absorbed``.

    THE CRASH SURVIVES if foreign content is NOT absorbed by an EXPLORE-enabled observer.
    THE CRASH WAS AN ARTIFACT if it is. That second outcome is the unwelcome one, it is the
    reason E19 runs first, and V4 spec §2 requires it be reported plainly and prominently
    rather than explained away.

    The exploratory-human cell is the positive control. If EXPLORE fails to absorb
    exploratory HUMAN content then EXPLORE is not doing its job at all, and the foreign
    result carries no weight either way: a hypothesis that explains nothing cannot be said to
    have failed to explain foreign content in particular. That case is reported as
    INCONCLUSIVE, not as a win.
    """
    foreign = cells.get("foreign/explore_on")
    human_ex = cells.get("human_exploratory/explore_on")
    if foreign is None or human_ex is None:
        return {"verdict": "INCOMPLETE",
                "statement": "E19 did not produce both of the cells the verdict depends on"}

    control_ok = bool(human_ex["absorbed"])
    foreign_absorbed = bool(foreign["absorbed"])

    if not control_ok:
        verdict = "INCONCLUSIVE"
        statement = (
            "EXPLORE failed to absorb exploratory HUMAN content, which is its positive "
            "control. A fallback hypothesis that explains nothing cannot be credited with "
            "having failed to explain foreign content in particular, so E19 cannot rule on "
            "hypothesis-space poverty. Fix the control before reading the foreign cell.")
    elif foreign_absorbed:
        verdict = "CRASH_IS_AN_ARTIFACT"
        statement = (
            "EXPLORE absorbed foreign content: the observer converged on a human-shaped "
            "exploratory goal, stayed engaged, and was confident. The V1-V3 crash was then "
            "at least partly an artifact of an artificially impoverished hypothesis space. "
            "This does not invalidate the trust-exploit or expertise results, but it "
            "substantially weakens the central claim and the README must say so.")
    else:
        verdict = "CRASH_SURVIVES"
        statement = (
            "EXPLORE absorbed exploratory human content but NOT foreign content. The most "
            "generous fallback the theory permits does not rescue out-of-family structure, "
            "so the crash is not an artifact of hypothesis-space poverty. This is a stronger "
            "result than anything in V1-V3, because the obvious deflationary explanation was "
            "given its best shot and did not take.")
    return {
        "verdict": verdict,
        "statement": statement,
        "positive_control_passed": control_ok,
        "foreign_absorbed": foreign_absorbed,
        "cells": cells,
    }


# --------------------------------------------------------------------------- #
# The pre-registration file.
# --------------------------------------------------------------------------- #
def build_preregistration_v4(cfg: Config) -> dict:
    """Construct the payload. Includes sig_EXPLORE elementwise (N17)."""
    sig_true = FN.build_human_signatures(cfg)
    sig_explore = FN.build_explore_signature(sig_true)
    basis = FN.build_foreign_basis(cfg, int(cfg.get("v4.foreign.draw_seed", 41)))

    payload = {
        "version": "V4",
        "written_before": "any V4 experiment is run",
        "C1_feature_partition": {
            "num_features": FN.NUM_FEATURES_V4,
            "human_features": FN.HUMAN_FEATURES,
            "foreign_features": FN.FOREIGN_FEATURES,
            "human_pairs": FN.HUMAN_PAIRS,
            "foreign_pairs": FN.FOREIGN_PAIRS,
            "why_16_not_8": (
                "V4 spec C1 requires foreign_basis to occupy a partition disjoint from the "
                "dominant support of sig_true. At V1-V3 cardinality F=8 and the four goal "
                "pairs tile all eight features, so no disjoint partition exists. The only "
                "alternative was to overlap the observer's support, which is the spec's own "
                "pre-mortem failure #1. Logged as V4 deviation 1: V4 does not share a seed "
                "stream with V1-V3 and N16 is a behavioural regression, not a bit-exact one."),
        },
        "C2_explore": {
            "construction": "normalize(mean_g(sig_true[g])) over the four REAL goals",
            "index_in_observer_goal_space": FN.EXPLORE,
            "signature": [float(x) for x in sig_explore],
            "human_block_mass": float(sig_explore[FN.HUMAN_FEATURES].sum()),
            "foreign_block_mass": float(sig_explore[FN.FOREIGN_FEATURES].sum()),
            "linf_from_global_uniform": float(
                np.max(np.abs(sig_explore - 1.0 / FN.NUM_FEATURES_V4))),
            "atol": EXPLORE_ATOL,
            "measured_before_the_build": {
                "at_V1_V3_cardinality_sig_explore_was_exactly_uniform": True,
                "linf_from_uniform_at_F8": 1.4e-17,
                "kl_synth_to_explore_at_F8": 0.565,
                "kl_synth_to_sig0_at_F8": 2.113,
                "consequence": (
                    "the prescribed construction and the construction the spec explicitly "
                    "forbids were the same object at F=8, and EXPLORE sat four times closer "
                    "to synthetic content than any real goal. E19 could only have returned "
                    "CRASH_IS_AN_ARTIFACT, for a reason about the feature partition rather "
                    "than about the theory."),
            },
        },
        "N18_foreign_is_goal_directed": {
            "mi_floor": float(cfg.get("v4.foreign.mi_floor", N18_MI_FLOOR_DEFAULT)),
            "observer_mi_ceiling": float(cfg.get("v4.foreign.observer_mi_ceiling", 0.05)),
            "entropy_ceiling": float(cfg.get("v4.foreign.entropy_ceiling", 2.0)),
            "foreign_draw_seed": int(cfg.get("v4.foreign.draw_seed", 41)),
            "foreign_concentration": float(cfg.get("v4.foreign.concentration", 0.25)),
            "measured_mi_features_foreign_goal": float(FN.mi_features_goal(basis)),
            "why": (
                "foreign content that is not itself goal-informative has silently reverted to "
                "V3's goal-empty model, and every V4 result would be a V3 result in new "
                "vocabulary (V4 spec §6, §7)"),
        },
        "E19_criteria": {
            "absorb_mass": E19_ABSORB_MASS,
            "converged_entropy": E19_CONVERGED_ENTROPY,
            "engaged_fraction_for_crash_signature_only": E19_ENGAGED_FRACTION,
            "flat_posterior_baselines": {
                "explore_mass_if_flat": 1.0 / (FN.NUM_REAL_GOALS + 1),
                "entropy_if_flat_nats": float(np.log(FN.NUM_REAL_GOALS + 1)),
            },
            "rule": ("conjunctive: absorption requires mass AND convergence, so that parking "
                     "probability on the vaguest available hypothesis does not count as "
                     "explaining the content"),
            "deviation_2_engagement_clause_removed": {
                "when": "before this file was first written to results/, at smoke scale",
                "why": (
                    "the first draft also required sustained engagement. That is invalid as a "
                    "component of absorption: in this model disengagement is what SUCCESS "
                    "looks like, and the canonical success case (human_directed, accuracy "
                    "1.00, entropy 0.00007) scores 0.000 on engagement. The clause would have "
                    "failed E19's own positive control for behaving as the theory predicts."),
                "checked_before_making_it": (
                    "the correction cannot manufacture the welcome answer: the foreign cell "
                    "fails absorption on MASS (0.198 at smoke scale, against a 0.20 flat "
                    "baseline and a 0.50 threshold), which is the primary clause and is "
                    "untouched. Dropping engagement rescues the control and does not move the "
                    "decisive cell."),
                "what_replaced_it": (
                    "the crash is now measured as its own quantity by crash_signature(): "
                    "disengagement WITHOUT resolution, which is the joint of low engagement "
                    "and high entropy"),
            },
            "positive_control": "human_exploratory/explore_on must be absorbed",
            "outcomes": ["CRASH_SURVIVES", "CRASH_IS_AN_ARTIFACT", "INCONCLUSIVE"],
            "both_outcomes_written_before_the_run": True,
        },
        "D4_posterior_arity_rule": (
            "metrics that consume a goal posterior take the four REAL goals, renormalised; "
            "EXPLORE mass is reported separately rather than folded in. Fixed before E19 so "
            "the choice cannot be made after seeing which is kinder."),
    }
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return payload


def write_preregistration_v4(cfg: Config, path: Path, force: bool = False) -> dict:
    """Write the V4 criteria, refusing to silently overwrite a differing set."""
    payload = build_preregistration_v4(cfg)
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
                "V4's criteria, and sig_EXPLORE in particular, were pre-registered and must "
                "not change after the fact (V4 spec §6). If E19 has already run, changing "
                "this file is the exact failure §7 names. Investigate the config change, or "
                "pass force=True and record it as a deviation in RESULTS_V4.md.")
        if existing is not None:
            return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def assert_prereg_locked_v4(path: Path) -> dict:
    """Load the V4 criteria and verify they have not been edited since being written."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. No V4 experiment may run before its criteria are "
            "pre-registered (V4 spec §6). Run: python run_v4_stage1.py --prereg-only")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_hash")
    recomputed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if stated != recomputed:
        raise RuntimeError(
            f"{path.name} has been modified since it was written "
            f"(hash {stated} != recomputed {recomputed}). The pre-registered criteria are not "
            "trustworthy; the V4 programme will not run.")
    return payload


def assert_explore_matches_prereg(cfg: Config, path: Path) -> dict:
    """N17 — the constructed sig_EXPLORE matches the pre-registered array, elementwise."""
    payload = assert_prereg_locked_v4(path)
    registered = np.asarray(payload["C2_explore"]["signature"], dtype=float)
    built = FN.build_explore_signature(FN.build_human_signatures(cfg))
    assert built.shape == registered.shape, (
        f"N17: sig_EXPLORE has shape {built.shape}, pre-registered {registered.shape}. The "
        f"feature partition changed after pre-registration.")
    delta = float(np.max(np.abs(built - registered)))
    assert delta <= EXPLORE_ATOL, (
        f"N17: constructed sig_EXPLORE differs from the pre-registered array by {delta:.3e} "
        f"(tolerance {EXPLORE_ATOL}). EXPLORE is the single most tunable object in V4 and "
        f"the one V4 spec §7 predicts will be quietly adjusted if E19 disappoints. It has "
        f"moved. Either revert the change or record it as a deviation and re-register.")
    return {"max_abs_delta": delta, "n_features": int(built.size)}
