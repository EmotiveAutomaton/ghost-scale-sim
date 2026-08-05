"""T-5 — auxiliary. If process is the source, is process the better detector?

NOT ONE OF THE FOUR REQUESTED TESTS. It exists because T-1 came back with a structure nobody
asked about and that structure has a direct engineering consequence.

T-1 found the triangle is not symmetric and not the chain the framework assumed. Supplying the
PROCESS moves the reader's belief about the goal and about depth; supplying the GOAL moves
nothing. Process is a source and goal is a sink. Every detection instrument this project and
Sounding Line have built scores the GOAL -- purpose recovery, purpose breadth, goal error
reduction. If T-1 is right, they are all reading the downstream variable.

SO THE ENGINEERING QUESTION: does a statistic computed on the reader's SUB-GOAL posterior
separate content-with-a-maker from content-without-one better than a statistic computed on its
GOAL posterior?

THREE CONTENT CLASSES, all already validated in this repository:

  hierarchical  a HierarchicalCreator at depth mu. There is a real execution chain to recover.
  foreign       a real policy over a goal the reader has no hypothesis for. E19 and E20 both
                establish this is the regime that sustains attention indefinitely -- the reader
                keeps looking and never resolves. The hard negative.
  ghost         pure synthetic emission. No maker, no chain. The easy negative.

SCORING. Every candidate statistic is scored as the area under the ROC curve for separating
hierarchical from each negative, with a bootstrap interval and a LABEL-PERMUTATION NULL. AUC is
used rather than accuracy because it needs no threshold, and a threshold fitted on the test set
is exactly the flaw T-4 found in S-3.

WHAT THIS IS NOT. It is not evidence that any of this works on text. It is a statement about
which internal quantity of this model's reader carries the discriminating information, which is
worth knowing before an instrument is pointed at the other one.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ... import constants as K
from ... import metrics
from ...config import Config
from ...v5_model import make_v5_observer, marginal_goal, marginal_mu, marginal_subgoal
from ...v6 import SEED_OFFSET, harness as H
from ...methods import gates as G
from ...methods import provenance as PROV
from . import sl_dir
from .common import build
from . import t_common as T

N_TIMESTEPS = 24
FORCED_K = 24


def _stats(res, n_mu, n_sub, ng) -> dict:
    """Every summary a detector could compute from one reading, with no access to the truth.

    THE NO-TRUTH RULE IS THE POINT. A detector in the field has the artifact and its own
    posteriors and nothing else. Anything scored against the true goal or the true mode is an
    oracle statistic and would make this test the thing E45 was withdrawn for. So: entropies,
    movement, volatility, concentration -- all computed from the reader's own beliefs.
    """
    rows = np.asarray(res.goal_posterior, dtype=float)
    g = [marginal_goal(r, n_mu, ng, n_sub) for r in rows]
    s = [marginal_subgoal(r, n_mu, ng, n_sub) for r in rows]
    m = [marginal_mu(r, n_mu, ng, n_sub) for r in rows]
    gh = np.array([metrics.within_observer_entropy(x) for x in g])
    sh = np.array([metrics.within_observer_entropy(x) for x in s])
    mh = np.array([metrics.within_observer_entropy(x) for x in m])
    S = np.asarray(s, dtype=float)
    # Step-to-step movement of the sub-goal belief: a real execution chain makes this belief
    # travel in a structured way, and synthetic content gives it nothing to travel on.
    sub_step = np.array([float(np.abs(S[t] - S[t - 1]).sum()) for t in range(1, len(S))])
    G = np.asarray(g, dtype=float)
    goal_step = np.array([float(np.abs(G[t] - G[t - 1]).sum()) for t in range(1, len(G))])
    prior = np.asarray(res.goal_prior, dtype=float)
    return {
        # ---- goal-side statistics: what every existing instrument scores -------------------
        "goal_final_entropy": float(gh[-1]),
        "goal_mean_entropy": float(gh.mean()),
        "goal_entropy_drop": float(gh[0] - gh[-1]),
        "goal_movement": float(metrics.kl_divergence(
            marginal_goal(np.asarray(res.final_goal_posterior, dtype=float), n_mu, ng, n_sub),
            marginal_goal(prior, n_mu, ng, n_sub))),
        "goal_max_posterior": float(np.max(G[-1])),
        "goal_step_movement_mean": float(goal_step.mean()),
        # ---- process-side statistics: what T-1 says is upstream ----------------------------
        "subgoal_min_entropy": float(sh.min()),
        "subgoal_mean_entropy": float(sh.mean()),
        "subgoal_effective_modes": float(np.exp(sh).mean()),
        "subgoal_entropy_range": float(sh.max() - sh.min()),
        "subgoal_entropy_std": float(sh.std()),
        "subgoal_step_movement_mean": float(sub_step.mean()),
        "subgoal_step_movement_std": float(sub_step.std()),
        "subgoal_max_posterior_mean": float(np.max(S, axis=1).mean()),
        # ---- depth-side ---------------------------------------------------------------------
        "depth_final_entropy": float(mh[-1]),
        "depth_entropy_drop": float(mh[0] - mh[-1]),
        # ---- attention, which is what E19/E20 already use -----------------------------------
        "engaged_fraction": float(np.mean(np.asarray(res.attention) == K.DEEP)),
    }


def _read(world, cfg_r, artifact, env, n_mu, n_sub, ng, rng, forced_k=FORCED_K):
    from ...observer import rollout_observer
    agent = make_v5_observer(world, rng)
    res = rollout_observer(agent, artifact, env, cfg_r, rng, n_timesteps=N_TIMESTEPS,
                           force_deep_k=int(forced_k), initial_glance=True, early_stop=False)
    return _stats(res, n_mu, n_sub, ng)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC, ties counted as half. Threshold-free by construction."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks over ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def run(cfg: Config, n_obs: int = 400) -> dict:
    world, _b, cfg_r, n_mu, n_sub, ng = build(cfg)
    rng = np.random.default_rng(SEED_OFFSET + 91_500)
    rows = []

    # THE LOOK LENGTH IS A DIFFICULTY DIAL AND IT HAS TO BE SWEPT. At a full 24-step forced look
    # both sides of this comparison sit at AUC 1.00 against both negatives -- a ceiling, so
    # "process beats goal" cannot be asked there at all. Short looks are where the two separate,
    # and a short look is also the realistic case for any instrument that has to triage.
    for forced_k in (24, 6, 3, 1):
      for mu in (2, 3):
        for beta in (1.0, 0.25):
            base = 67_000 + mu * 137 + int(beta * 100) + forced_k * 1013
            for i in range(int(n_obs)):
                a_rng = np.random.default_rng(base * 31 + i)
                o_rng = np.random.default_rng(base * 7907 + i)
                g = int(i % ng)
                # -- hierarchical: a maker with a real chain
                _c, art, env = H.make_artifact_and_env(world, cfg_r, g, mu, beta, N_TIMESTEPS,
                                                       a_rng, provenance=K.CREATOR)
                r = _read(world, cfg_r, art, env, n_mu, n_sub, ng, o_rng, forced_k)
                r.update({"cls": "hierarchical", "mu": mu, "beta": beta, "i": i,
                          "forced_k": forced_k})
                rows.append(r)
                # -- foreign: a real policy the reader has no hypothesis for (the hard negative)
                a_rng = np.random.default_rng(base * 31 + i)
                o_rng = np.random.default_rng(base * 7907 + i)
                fart, fenv = H.make_foreign_artifact_and_env(world, cfg_r, g % 2, N_TIMESTEPS,
                                                             a_rng, omega=0.10)
                r = _read(world, cfg_r, fart, fenv, n_mu, n_sub, ng, o_rng, forced_k)
                r.update({"cls": "foreign", "mu": mu, "beta": beta, "i": i,
                          "forced_k": forced_k})
                rows.append(r)
                # -- ghost: pure synthetic (the easy negative)
                a_rng = np.random.default_rng(base * 31 + i)
                o_rng = np.random.default_rng(base * 7907 + i)
                _c2, gart, genv = H.make_artifact_and_env(world, cfg_r, g, mu, beta, N_TIMESTEPS,
                                                          a_rng, provenance=K.GHOST)
                r = _read(world, cfg_r, gart, genv, n_mu, n_sub, ng, o_rng, forced_k)
                r.update({"cls": "ghost", "mu": mu, "beta": beta, "i": i,
                          "forced_k": forced_k})
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(sl_dir() / "t5_detection_points.csv", index=False)
    (df.groupby(["cls", "mu", "beta", "forced_k"]).mean(numeric_only=True)
       .to_csv(sl_dir() / "t5_detection_summary.csv"))

    feats = [c for c in df.columns if c not in ("cls", "mu", "beta", "i", "forced_k")]
    goal_side = [f for f in feats if f.startswith("goal_")]
    proc_side = [f for f in feats if f.startswith("subgoal_")]

    out = {}
    for neg_cls in ("foreign", "ghost"):
        per_cell = {}
        for forced_k in (24, 6, 3, 1):
          for mu in (2, 3):
            for beta in (1.0, 0.25):
                d = df[(df.mu == mu) & (df.beta == beta) & (df.forced_k == forced_k)]
                pos = d[d.cls == "hierarchical"]
                neg = d[d.cls == neg_cls]
                res = {}
                for f in feats:
                    a = auc(pos[f].to_numpy(), neg[f].to_numpy())
                    # bootstrap over artifacts
                    draws = [auc(rng.choice(pos[f].to_numpy(), len(pos), replace=True),
                                 rng.choice(neg[f].to_numpy(), len(neg), replace=True))
                             for _ in range(200)]
                    # permutation null: shuffle the labels
                    pool = np.concatenate([pos[f].to_numpy(), neg[f].to_numpy()])
                    perm = []
                    for _ in range(200):
                        p = rng.permutation(pool)
                        perm.append(auc(p[:len(pos)], p[len(pos):]))
                    # ORIENTED AUC. An AUC of 0.00 is a PERFECT detector with its sign the other
                    # way round -- goal_final_entropy is exactly that, and reported raw it reads
                    # as total failure. Direction is free to a detector, so every comparison here
                    # is made on max(auc, 1 - auc), with the raw value kept for the sign.
                    res[f] = {
                        "auc": float(a),
                        "auc_oriented": float(max(a, 1.0 - a)) if np.isfinite(a) else float("nan"),
                        "direction": ("higher_means_maker" if a >= 0.5
                                      else "lower_means_maker"),
                        "interval": [float(np.percentile(draws, 2.5)),
                                     float(np.percentile(draws, 97.5))],
                        "permutation_null_mean": float(np.mean(perm)),
                        "permutation_null_p95_abs_dev": float(
                            np.percentile(np.abs(np.array(perm) - 0.5), 95)),
                        "beats_permutation_null": bool(
                            abs(a - 0.5) > np.percentile(np.abs(np.array(perm) - 0.5), 95)),
                    }
                best_goal = max(goal_side, key=lambda f: res[f]["auc_oriented"])
                best_proc = max(proc_side, key=lambda f: res[f]["auc_oriented"])
                per_cell[f"k{forced_k}_mu{mu}_beta{beta}"] = {
                    "per_feature": res,
                    "best_goal_side": {"feature": best_goal, **res[best_goal]},
                    "best_process_side": {"feature": best_proc, **res[best_proc]},
                    "process_minus_goal": float(res[best_proc]["auc_oriented"]
                                                - res[best_goal]["auc_oriented"]),
                    "process_beats_goal": bool(res[best_proc]["auc_oriented"]
                                               > res[best_goal]["auc_oriented"]),
                    "both_at_ceiling": bool(min(res[best_proc]["auc_oriented"],
                                                res[best_goal]["auc_oriented"]) >= 0.99),
                }
        out[neg_cls] = per_cell

    summary = {}
    for neg_cls, cells in out.items():
        contested = {k: c for k, c in cells.items() if not c["both_at_ceiling"]}
        wins = sum(1 for c in contested.values() if c["process_beats_goal"])
        summary[neg_cls] = {
            "cells": len(cells),
            "cells_at_ceiling_where_the_question_cannot_be_asked": len(cells) - len(contested),
            "contested_cells": len(contested),
            "process_side_wins_among_contested": wins,
            "median_process_minus_goal_among_contested": (
                float(np.median([c["process_minus_goal"] for c in contested.values()]))
                if contested else None),
            "best_process_auc_by_cell": {k: c["best_process_side"]["auc_oriented"]
                                         for k, c in cells.items()},
            "best_goal_auc_by_cell": {k: c["best_goal_side"]["auc_oriented"]
                                      for k, c in cells.items()},
        }

    # ---- standing gates ---------------------------------------------------------------------
    gr = G.GateReport()
    _full = df[df.forced_k == 24]
    _pos = _full[_full.cls == "hierarchical"]
    _neg = _full[_full.cls == "ghost"]
    gr.positive("synthetic_content_is_separable_at_a_full_look",
                auc(_pos.goal_max_posterior.to_numpy(), _neg.goal_max_posterior.to_numpy()),
                1.0, 0.12,
                detail="a maker with a real chain versus pure synthetic emission, read for the "
                       "full 24 steps, is the easiest discrimination this model can pose. The "
                       "tolerance is 0.12 rather than 0.05 deliberately: a positive control has "
                       "to separate WORKING from BROKEN, not perfect from near-perfect. At full "
                       "scale this returns 0.997-1.000 and at smoke scale 0.909; a broken reader "
                       "would return ~0.5, which is nine tolerances away.")
    _perm_worst = 0.0
    for cells in out.values():
        for c in cells.values():
            for r in c["per_feature"].values():
                _perm_worst = max(_perm_worst, abs(r["permutation_null_mean"] - 0.5))
    gr.no_oracle("label_permutation_null_sits_at_chance", _perm_worst, 0.05,
                 detail="shuffling the class labels must leave every AUC at 0.5. If it does not, "
                        "the scoring is reading something other than the class.")

    verdict = {
        "test": "T-5 (auxiliary) — is the process posterior a better maker-detector than the "
                "goal posterior?",
        "for": "Sounding Line, which internal quantity a detection instrument should read",
        "WHY_THIS_EXISTS": (
            "T-1 found process is a source and goal is a sink. Every instrument this project and "
            "Sounding Line have built scores the goal. This asks whether that is the wrong "
            "variable to point at."),
        "no_oracle_statistics": (
            "every feature is computed from the reader's own posteriors only. Nothing is scored "
            "against the true goal or the true mode, because a detector in the field has neither "
            "and a statistic that used them would be the thing E45 was withdrawn for."),
        "scoring": ("Mann-Whitney AUC, threshold-free, with a bootstrap interval and a "
                    "label-permutation null. No threshold is fitted anywhere -- that is the flaw "
                    "T-4 found in S-3."),
        "summary": summary,
        "by_negative_class": out,
        "what_would_have_falsified_it": (
            "goal-side statistics matching or beating process-side ones. Then T-1's asymmetry, "
            "whatever it says about the reader's internals, would have no consequence for what "
            "an instrument should measure."),
        "what_this_cannot_show": (
            "anything about text. 'Foreign' here is a real policy over an unmodelled goal, which "
            "is this repository's stand-in for content whose maker cannot be reconstructed. It is "
            "not machine-generated text and this is not a detector."),
    }
    PROV.stamp(verdict, __file__, gr)
    (sl_dir() / "t5_detection.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
