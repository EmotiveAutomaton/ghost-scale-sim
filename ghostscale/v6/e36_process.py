"""E36 — process recovery: the measure the project never had, and the reason E30 came back null.

THE ARGUMENT, BECAUSE IT REWRITES AN EXISTING RESULT RATHER THAN ADDING A NEW ONE.

Every uptake measure in V1-V5 scores GOAL recovery. Depth is constructed so that the goal is
EXACTLY as recoverable at every depth -- that is the design commitment that stops depth being
legibility wearing a new name -- so a goal-uptake measure could not have moved with depth
whatever was true of the reader. E30's own write-up conceded the point ("the measure written
down in advance could not have moved") and the repair pass then bounded the effect under a tenth
of a nat.

Under the theory the experiment was measuring the wrong quantity. The claim is that the reader
recovers the maker's PROCESS, and that depth is how much of that process reaches the surface. The
reader has carried a posterior over the execution chain in every V5 run ever made, and nobody
ever scored it.

THREE THINGS ARE MEASURED HERE.

  H6.4  Does depth move PROCESS uptake, where it provably cannot move goal uptake?
  H6.3  Does goal recovery GATE process recovery? The author's ordering claim, which appears in
        neither the essay nor the preprint and came out of a walkthrough: intent is the key that
        unlocks the method, because once you know what someone was for you can read their
        actions as being in service of it.
  H6.12 Does the extraction run on a SLICE of the artifact the way it runs on the whole? The
        scaled version of the fractal claim, which cannot be built in full without recursion.

NULL N28 IS THE GATE. At mu = 1 there is no process to recover -- every execution mode emits the
goal signature exactly -- so process recovery must sit at chance whatever the goal recovery is.
If it does not, the measure is reading goal information and is goal recovery renamed, which is
precisely the trap depth itself was built to avoid.

HOW GOAL RECOVERY IS MADE TO VARY, since H6.3 needs cells where the reader sometimes fails and
depth is built so it never does. The rationality knob attenuates WHICH GOAL the execution modes
are in service of while leaving the mode structure standing -- "you can see the craft and not
what it was for", which is the aesthetic category V5 named when it built the parameter. That is
exactly the dissociation H6.3 needs, and it is already in the model.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import metrics
from .. import v6_model as V6
from ..config import Config
from ..prereg_v6 import (BOOTSTRAP_DRAWS, h63_verdict, h64_verdict, h612_verdict,
                         percentile_interval)
from ..v5_model import MU_LEVELS, make_v5_observer
from . import harness as H
from . import SEED_OFFSET, v6_dir

# Goal belief counts as SETTLED below this entropy, which is where the temporal ordering test
# splits each rollout. A quarter of the four-goal ceiling.
RESOLVED_ENTROPY = 0.35

MU_GRID = (1, 2, 3)
BETA_GRID = (1.0, 0.25, 0.10)      # 1.0 = goal fully legible; 0.10 = craft visible, goal not


def _cell(world, cfg_r, n_mu, n_sub, ng, mu: int, beta: float, n_obs: int,
          n_timesteps: int, forced_k: int, base_seed: int) -> list:
    kappa = float(world.cfg.signal_model.kappa)
    recs = []
    for i in range(int(n_obs)):
        art_rng = np.random.default_rng(base_seed * 31 + i)
        creator, artifact, env = H.make_artifact_and_env(
            world, cfg_r, int(art_rng.integers(ng)), int(mu), float(beta),
            n_timesteps, art_rng)
        agent = make_v5_observer(world, np.random.default_rng(base_seed * 7907 + i))
        enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                              np.random.default_rng(base_seed * 7907 + i),
                              n_timesteps, forced_k, n_sub, n_mu, ng, kappa)
        sw = V6.subwindow_recovery(enc.subgoal_posteriors, enc.true_modes, n_sub)

        # THE TEMPORAL FORM OF THE ORDERING CLAIM, ADDED AFTER THE PRE-REGISTERED FORM RETURNED
        # A NULL. Declared as an addition; the original decides nothing here and is reported
        # beside it, exactly as this project has done with every other post-hoc measure.
        #
        # The pre-registered test compares readers who ended up right about the goal against
        # readers who ended up wrong. That is a BETWEEN-READER contrast, and the claim it is
        # meant to test is a WITHIN-READER, TEMPORAL one: once you know what someone was for,
        # you can read their actions as being in service of it. Those are different statements
        # and only the second is what "intent is the key that unlocks the process" says.
        #
        # So: find the step at which this reader's goal belief settles, and compare its process
        # recovery before that point with its recovery after.
        ents = [float(metrics.within_observer_entropy(p)) for p in enc.goal_posteriors_by_step]
        settled = next((t for t, h in enumerate(ents) if h <= RESOLVED_ENTROPY), None)
        before = after = float("nan")
        if settled is not None and 2 <= settled <= len(ents) - 3:
            before = V6.process_recovery(enc.subgoal_posteriors[:settled],
                                         enc.true_modes[:settled], n_sub)["process_error_reduction"]
            after = V6.process_recovery(enc.subgoal_posteriors[settled:],
                                        enc.true_modes[settled:], n_sub)["process_error_reduction"]

        recs.append({
            "settled_at": -1 if settled is None else int(settled),
            "process_before_settling": before,
            "process_after_settling": after,
            "mu": int(mu), "beta": float(beta), "observer": i,
            "goal_correct": int(enc.correct),
            "goal_error_reduction": float(enc.error_reduction),
            "goal_movement": float(enc.movement),
            "process_accuracy": float(enc.process["process_accuracy"]),
            "process_error_reduction": float(enc.process["process_error_reduction"]),
            "window_accuracy": float(sw["window_accuracy_mean"]),
            "whole_accuracy": float(sw["whole_accuracy"]),
            "recovered_mu": float(enc.recovered_mu),
            "final_entropy": float(enc.final_entropy),
        })
    return recs


def run(cfg: Config, n_obs: int = 40, n_timesteps: int = 24, forced_k: int = 24,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)

    recs = []
    for mu in MU_GRID:
        for beta in BETA_GRID:
            recs.extend(_cell(world, cfg_r, n_mu, n_sub, ng, mu, beta, n_obs,
                              n_timesteps, forced_k, 30_000 + mu * 100 + int(beta * 100)))
    df = pd.DataFrame(recs)
    out = v6_dir("e36_process")
    df.to_csv(out / "e36_process.csv", index=False)

    cells = df.groupby(["mu", "beta"]).agg(
        goal_accuracy=("goal_correct", "mean"),
        goal_error_reduction=("goal_error_reduction", "mean"),
        process_accuracy=("process_accuracy", "mean"),
        process_error_reduction=("process_error_reduction", "mean"),
        window_accuracy=("window_accuracy", "mean"),
        whole_accuracy=("whole_accuracy", "mean")).reset_index()
    cells.to_csv(out / "e36_cells.csv", index=False)

    # ---- N28: at mu = 1 there is no process, so recovery must be at chance ----
    #
    # WHICH STATISTIC DECIDES THIS NULL, AND WHY IT IS NOT THE OBVIOUS ONE. The pre-registration
    # locks the null ("at mu = 1 process recovery is at chance") without naming a statistic, and
    # the first implementation used ACCURACY. That was wrong, and it is worth recording because
    # it is the fourth instance in this project of an instrument answering a nearby question --
    # caught this time before it produced a result rather than afterwards.
    #
    # At mu = 1 every execution mode emits the goal signature exactly, so the sub-goal posterior
    # never leaves its uniform prior. The argmax of a flat posterior is a TIE, broken to index
    # zero, while the true mode is autocorrelated: the chain dwells, so one artifact sits in one
    # mode for most of its length. So accuracy measures "how often did this artifact happen to
    # be in mode zero", which is not a chance-level statistic and came out at 0.15 against a
    # nominal 0.25 -- BELOW chance, which no amount of information could produce.
    #
    # The information measure has none of that: mean log-probability of the truth against a
    # uniform baseline is exactly zero when the posterior has not moved, whatever the truth was.
    # It decides. Accuracy is reported beside it, per the standing rule that nothing is dropped.
    chance = 1.0 / float(n_sub)
    mu1 = df[df.mu == 1]
    n28_acc = float(mu1.process_accuracy.mean())
    n28_info = float(mu1.process_error_reduction.mean())
    n28_passed = bool(abs(n28_info) <= 0.05)

    # ---- H6.4: depth moves process uptake, at full goal legibility ----
    full = df[df.beta == 1.0]
    deep = full[full.mu == max(MU_GRID)].process_error_reduction.to_numpy()
    shal = full[full.mu == min(MU_GRID)].process_error_reduction.to_numpy()
    contrast = float(np.mean(deep) - np.mean(shal))
    rng = np.random.default_rng(SEED_OFFSET + 20260801)
    draws = [float(np.mean(rng.choice(deep, deep.size, replace=True))
                   - np.mean(rng.choice(shal, shal.size, replace=True)))
             for _ in range(BOOTSTRAP_DRAWS)]
    h64 = h64_verdict(contrast, percentile_interval(draws))

    # The goal-uptake comparison on the SAME cells, which is what E30 measured. Reported beside
    # it rather than instead of it: the point is that one measure can move where the other
    # provably cannot, and that is only visible with both.
    g_deep = full[full.mu == max(MU_GRID)].goal_error_reduction.to_numpy()
    g_shal = full[full.mu == min(MU_GRID)].goal_error_reduction.to_numpy()
    g_contrast = float(np.mean(g_deep) - np.mean(g_shal))
    g_draws = [float(np.mean(rng.choice(g_deep, g_deep.size, replace=True))
                     - np.mean(rng.choice(g_shal, g_shal.size, replace=True)))
               for _ in range(BOOTSTRAP_DRAWS)]
    g_interval = percentile_interval(g_draws)

    # ---- H6.3: goal recovery gates process recovery ----
    deepish = df[(df.mu > 1)]
    right = deepish[deepish.goal_correct == 1].process_accuracy.to_numpy()
    wrong = deepish[deepish.goal_correct == 0].process_accuracy.to_numpy()
    h63 = h63_verdict(right, wrong)
    h63["n_goal_right"] = int(right.size)
    h63["n_goal_wrong"] = int(wrong.size)

    # ---- H6.3b: the TEMPORAL form of the ordering claim (added, declares nothing locked) ----
    tmp = deepish.dropna(subset=["process_before_settling", "process_after_settling"])
    if len(tmp):
        b = float(tmp.process_before_settling.mean())
        a = float(tmp.process_after_settling.mean())
        rng2 = np.random.default_rng(SEED_OFFSET + 20260802)
        d = tmp.process_after_settling.to_numpy() - tmp.process_before_settling.to_numpy()
        draws = [float(np.mean(rng2.choice(d, d.size, replace=True)))
                 for _ in range(BOOTSTRAP_DRAWS)]
        lo, hi = percentile_interval(draws)
        h63b = {"process_before_settling": b, "process_after_settling": a,
                "gain_after_settling": a - b, "interval": [lo, hi],
                "n_rollouts": int(len(tmp)),
                "excludes_zero_positive": bool(np.isfinite(lo) and lo > 0.0),
                "outcome": ("RESOLVING_THE_GOAL_UNLOCKS_THE_PROCESS"
                            if np.isfinite(lo) and lo > 0.0
                            else "NO_TEMPORAL_UNLOCK_EITHER")}
    else:
        h63b = {"outcome": "NOT_MEASURABLE_NO_ROLLOUT_SETTLED_IN_THE_MIDDLE",
                "n_rollouts": 0}

    # ---- H6.12: scale invariance ----
    h612 = h612_verdict(float(full[full.mu > 1].whole_accuracy.mean()),
                        float(full[full.mu > 1].window_accuracy.mean()))

    verdict = {
        "experiment": "E36",
        "hypotheses": ["H6.3", "H6.4", "H6.12"],
        "question": ("Does the reader recover the maker's PROCESS, does the goal gate that "
                     "recovery, and does depth move it where it cannot move goal uptake?"),
        "plain_language": (
            "Every measure of what a reader takes on, in every version of this project, has "
            "scored how much of the maker's PURPOSE it got. Depth is built so the purpose is "
            "equally readable however deep the work is, so that measure could not move with "
            "depth whatever was true. This scores what the reader got of the maker's METHOD, "
            "which the reader has been quietly tracking all along and nobody ever read out."),
        "null_n28": {
            "statement": "at the shallowest depth there is no process, so recovery is at chance",
            "decided_on": "process error reduction (mean log-probability of the truth vs uniform)",
            "measured_information": n28_info, "tolerance": 0.05, "passed": n28_passed,
            "accuracy_reported_beside_it": n28_acc, "accuracy_chance": chance,
            "why_not_accuracy": (
                "at mu = 1 the sub-goal posterior never leaves its uniform prior, so its argmax "
                "is a tie broken to index zero, while the true mode is autocorrelated because "
                "the chain dwells. Accuracy then measures how often the artifact happened to sit "
                "in mode zero and came out BELOW nominal chance, which no amount of information "
                "could produce. The information measure is exactly zero on an unmoved posterior "
                "whatever the truth was. Declared rather than quietly substituted."),
            "why_it_is_the_gate": ("if process recovery moved at mu = 1 it would be reading "
                                   "goal information, which is the exact trap depth was built "
                                   "to avoid"),
        },
        "H6.4": h64,
        "H6.4_goal_comparison": {
            "deepest_minus_shallowest": g_contrast, "interval": list(g_interval),
            "note": ("the measure E30 used, on these same cells. Reported beside the process "
                     "measure rather than instead of it: the claim is that one can move where "
                     "the other provably cannot."),
        },
        "H6.3": h63,
        "H6.3b_temporal": h63b,
        "H6.12": h612,
        "cells": cells.to_dict(orient="records"),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "design_note": ("goal recovery is made to vary with the rationality knob, which "
                        "attenuates which goal the execution modes serve while leaving the "
                        "mode structure standing. That dissociation is already in the model "
                        "and is what H6.3 needs."),
    }
    if not n28_passed:
        verdict["INTERPRETABILITY"] = (
            "NULL N28 FAILED. Process recovery is not at chance where there is no process, so "
            "the measure is reading goal information and every number above is uninterpretable.")
    (v6_dir() / "e36_process.json").write_text(json.dumps(verdict, indent=2, default=str),
                                               encoding="utf-8")
    return verdict
