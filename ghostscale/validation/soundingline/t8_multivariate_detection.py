"""T-8 — the detector T-5 could not build: many features, combined, on held-out data.

T-5 ASKED WHETHER PROCESS-SIDE STATISTICS BEAT GOAL-SIDE ONES and answered no, with a caveat it
stated plainly: single features only, no held-out split, and the twelve features were written down
by hand. All three of those are fixable and all three matter for an instrument.

WHAT CHANGES HERE.

  MANY FEATURES.  catch22 is a canonical set distilled from ~7000 candidates in hctsa by removing
                  redundancy and ranking on classification performance. It is applied to the
                  reader's per-step sub-goal entropy and goal entropy trajectories, which is
                  exactly the kind of series it was built for, and which T-5's finding pointed at:
                  the best single signal there was how far a posterior TRAVELS, not where it ends.
  COMBINED.       A logistic model over standardised features, rather than the best single one.
  HELD OUT.       Fitted on even-indexed artifacts, scored on odd-indexed ones. A feature picked
                  out of fifty on the same data it is scored on is a multiple-comparisons problem
                  wearing a lab coat.
  CONFIRMED ON FRESH SEEDS.  The model is then scored a third time on artifacts generated from a
                  seed block that did not exist when it was fitted. In a simulator this is free
                  and it is strictly stronger than a reusable holdout: T-2's difficulty control
                  flipped sign between n = 40 and n = 200, and a model selected over fifty
                  features is at least that fragile.

THE HONEST RISK, STATED UP FRONT. More features and a fitted model can only raise apparent
performance. The number that means anything is the CONFIRMATION AUC on fresh seeds, and the
number that means most is the confirmation AUC MINUS the best single hand-picked feature's --
because if a fifty-feature fitted model cannot beat one number somebody thought of, the extra
machinery is not earning its place and should not be transported to real text.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ... import constants as K
from ... import metrics
from ...config import Config
from ...methods import gates as G
from ...methods import provenance as PROVENANCE
from ...methods import trajectory as TRAJ
from ...v5_model import make_v5_observer, marginal_goal, marginal_subgoal
from ...v6 import SEED_OFFSET, harness as H
from . import sl_dir
from .common import build
from .t5_detection import auc

N_TIMESTEPS = 24
#: Two look lengths: a full forced read, and a triage-length glance where T-5 found the two sides
#: actually separate. Cells where every statistic saturates cannot answer the question.
LOOKS = (24, 3)
CELLS = [(3, 1.0), (3, 0.25)]
NEGATIVES = ("foreign", "ghost")


def _series(res, n_mu, n_sub, ng) -> dict:
    """The trajectories a detector could compute. No truth is consulted anywhere."""
    rows = np.asarray(res.goal_posterior, dtype=float)
    g = np.asarray([marginal_goal(r, n_mu, ng, n_sub) for r in rows], dtype=float)
    s = np.asarray([marginal_subgoal(r, n_mu, ng, n_sub) for r in rows], dtype=float)
    return {
        "subgoal_entropy": np.array([metrics.within_observer_entropy(x) for x in s]),
        "goal_entropy": np.array([metrics.within_observer_entropy(x) for x in g]),
        "subgoal_travel": np.concatenate([[0.0], np.abs(np.diff(s, axis=0)).sum(axis=1)]),
        "goal_travel": np.concatenate([[0.0], np.abs(np.diff(g, axis=0)).sum(axis=1)]),
    }


def _features(res, n_mu, n_sub, ng) -> dict:
    """T-5's hand-picked features, plus catch22 over each trajectory."""
    ser = _series(res, n_mu, n_sub, ng)
    sh, gh = ser["subgoal_entropy"], ser["goal_entropy"]
    st, gt = ser["subgoal_travel"][1:], ser["goal_travel"][1:]
    out = {
        # -- the hand-picked baseline, exactly T-5's set --------------------------------------
        "hand_goal_final_entropy": float(gh[-1]),
        "hand_goal_mean_entropy": float(gh.mean()),
        "hand_goal_entropy_drop": float(gh[0] - gh[-1]),
        "hand_subgoal_min_entropy": float(sh.min()),
        "hand_subgoal_mean_entropy": float(sh.mean()),
        "hand_subgoal_entropy_std": float(sh.std()),
        "hand_subgoal_travel_mean": float(st.mean()) if st.size else 0.0,
        "hand_subgoal_travel_std": float(st.std()) if st.size else 0.0,
        "hand_goal_travel_mean": float(gt.mean()) if gt.size else 0.0,
        "hand_engaged_fraction": float(np.mean(np.asarray(res.attention) == K.DEEP)),
    }
    for name, series in (("sub", sh), ("goal", gh), ("subtrav", ser["subgoal_travel"])):
        f = TRAJ.features(series)
        if "skipped" not in f:
            for k, v in f.items():
                out[f"c22_{name}_{k}"] = float(v) if np.isfinite(v) else 0.0
    return out


def _one(world, cfg_r, n_mu, n_sub, ng, cls, mu, beta, g, forced_k, a_rng, o_rng) -> dict:
    from ...observer import rollout_observer
    if cls == "foreign":
        art, env = H.make_foreign_artifact_and_env(world, cfg_r, g % 2, N_TIMESTEPS, a_rng,
                                                   omega=0.10)
    else:
        prov = K.GHOST if cls == "ghost" else K.CREATOR
        _c, art, env = H.make_artifact_and_env(world, cfg_r, g, mu, beta, N_TIMESTEPS, a_rng,
                                               provenance=prov)
    agent = make_v5_observer(world, o_rng)
    res = rollout_observer(agent, art, env, cfg_r, o_rng, n_timesteps=N_TIMESTEPS,
                           force_deep_k=int(forced_k), initial_glance=True, early_stop=False)
    return _features(res, n_mu, n_sub, ng)


def _collect(world, cfg_r, n_mu, n_sub, ng, n_obs, seed_base, tag) -> pd.DataFrame:
    rows = []
    for forced_k in LOOKS:
        for (mu, beta) in CELLS:
            base = seed_base + mu * 137 + int(beta * 100) + forced_k * 1013
            for cls in ("hierarchical",) + NEGATIVES:
                for i in range(int(n_obs)):
                    r = _one(world, cfg_r, n_mu, n_sub, ng, cls, mu, beta, int(i % ng), forced_k,
                             np.random.default_rng(base * 31 + i),
                             np.random.default_rng(base * 7907 + i))
                    r.update({"cls": cls, "mu": mu, "beta": beta, "forced_k": forced_k,
                              "i": i, "block": tag})
                    rows.append(r)
    return pd.DataFrame(rows)


def _fit_and_score(train: pd.DataFrame, test: pd.DataFrame, confirm: pd.DataFrame,
                   feats: list, neg: str) -> dict:
    """Logistic model on standardised features. Returns test and fresh-seed AUCs."""
    try:
        from statsmodels.discrete.discrete_model import Logit
    except Exception as exc:                                  # noqa: BLE001
        return {"skipped": f"statsmodels not installed ({type(exc).__name__})"}

    def xy(df):
        d = df[df.cls.isin(["hierarchical", neg])]
        X = d[feats].to_numpy(dtype=float)
        y = (d.cls.to_numpy() == "hierarchical").astype(float)
        return X, y

    Xtr, ytr = xy(train)
    # IMPUTE, DO NOT DROP. catch22 refuses a constant series -- several of its features are
    # undefined on zero variance -- and a saturated reader produces exactly that: at beta = 1.0
    # with a full look the goal posterior collapses and stops moving. Dropping any feature with a
    # single missing row silently deleted the whole catch22 block in the easy cells, which looked
    # like "catch22 does not help" when it means "catch22 was not present". Missing values are
    # filled with the TRAINING median, which is computed before the test block is touched.
    n_missing = int(np.isnan(Xtr).any(axis=0).sum())
    med = np.nanmedian(Xtr, axis=0)
    med[~np.isfinite(med)] = 0.0
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    mu_, sd_ = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd_[sd_ < 1e-9] = 1.0
    keep = np.isfinite(mu_) & np.isfinite(sd_) & (Xtr.std(axis=0) > 1e-9)
    if keep.sum() < 2 or len(np.unique(ytr)) < 2:
        return {"skipped": "degenerate training block"}

    def z(X):
        X = np.where(np.isnan(X), med, X)
        return np.column_stack([np.ones(len(X)), (X[:, keep] - mu_[keep]) / sd_[keep]])

    try:
        # L2 regularisation: fifty features on a few hundred rows separates perfectly otherwise,
        # and a perfectly separating unregularised logit has no finite maximum likelihood.
        fit = Logit(ytr, z(Xtr)).fit_regularized(alpha=1.0, L1_wt=0.0, disp=0, maxiter=200)
    except Exception as exc:                                  # noqa: BLE001
        return {"skipped": f"fit failed ({type(exc).__name__})"}
    out = {}
    for name, df in (("test", test), ("confirm", confirm)):
        X, y = xy(df)
        if len(np.unique(y)) < 2:
            out[f"{name}_auc"] = float("nan")
            continue
        p = np.asarray(fit.predict(z(X)), dtype=float)
        out[f"{name}_auc"] = float(auc(p[y == 1], p[y == 0]))
    out["n_features_offered"] = int(Xtr.shape[1])
    out["n_features_used"] = int(keep.sum())
    out["n_features_with_missing_values"] = n_missing
    out["n_train"] = int(len(ytr))
    return out


def _best_single(train: pd.DataFrame, test: pd.DataFrame, confirm: pd.DataFrame,
                 feats: list, neg: str) -> dict:
    """Pick the best single feature ON TRAIN ONLY, then score it on test and on fresh seeds."""
    def split(df):
        d = df[df.cls.isin(["hierarchical", neg])]
        return d[d.cls == "hierarchical"], d[d.cls == neg]
    ptr, ntr = split(train)
    scored = {}
    for f in feats:
        a = auc(ptr[f].to_numpy(), ntr[f].to_numpy())
        if np.isfinite(a):
            scored[f] = max(a, 1.0 - a)
    if not scored:
        return {"skipped": "no usable features"}
    best = max(scored, key=scored.get)
    direction = 1.0 if auc(ptr[best].to_numpy(), ntr[best].to_numpy()) >= 0.5 else -1.0
    out = {"feature": best, "train_auc": float(scored[best])}
    for name, df in (("test", test), ("confirm", confirm)):
        p, n = split(df)
        a = auc(direction * p[best].to_numpy(), direction * n[best].to_numpy())
        out[f"{name}_auc"] = float(a)
    return out


def run(cfg: Config, n_obs: int = 150) -> dict:
    world, _b, cfg_r, n_mu, n_sub, ng = build(cfg)
    gr = G.GateReport()
    ok, why = TRAJ.available()

    disc = _collect(world, cfg_r, n_mu, n_sub, ng, n_obs, 71_000, "discovery")
    conf = _collect(world, cfg_r, n_mu, n_sub, ng, max(n_obs // 2, 40), 88_311, "confirmation")
    df = pd.concat([disc, conf], ignore_index=True)
    df.to_csv(sl_dir() / "t8_multivariate_detection_points.csv", index=False)

    feats_all = [c for c in disc.columns
                 if c not in ("cls", "mu", "beta", "forced_k", "i", "block")]
    feats_hand = [c for c in feats_all if c.startswith("hand_")]
    feats_c22 = [c for c in feats_all if c.startswith("c22_")]

    results = {}
    for neg in NEGATIVES:
        for forced_k in LOOKS:
            for (mu, beta) in CELLS:
                key = f"k{forced_k}_mu{mu}_beta{beta}_vs_{neg}"
                d = disc[(disc.forced_k == forced_k) & (disc.mu == mu) & (disc.beta == beta)]
                c = conf[(conf.forced_k == forced_k) & (conf.mu == mu) & (conf.beta == beta)]
                tr, te = d[d.i % 2 == 0], d[d.i % 2 == 1]
                if not len(tr) or not len(te) or not len(c):
                    continue
                single = _best_single(tr, te, c, feats_hand, neg)
                multi_hand = _fit_and_score(tr, te, c, feats_hand, neg)
                multi_all = (_fit_and_score(tr, te, c, feats_all, neg) if feats_c22
                             else {"skipped": why})
                row = {"best_single_hand_picked": single,
                       "multivariate_hand_picked": multi_hand,
                       "multivariate_with_catch22": multi_all}
                if "confirm_auc" in single and "confirm_auc" in multi_all:
                    row["catch22_gain_on_fresh_seeds"] = float(
                        multi_all["confirm_auc"] - single["confirm_auc"])
                if "confirm_auc" in multi_all and "test_auc" in multi_all:
                    row["shrinkage_test_to_fresh"] = float(
                        multi_all["test_auc"] - multi_all["confirm_auc"])
                results[key] = row

    gains = [r["catch22_gain_on_fresh_seeds"] for r in results.values()
             if "catch22_gain_on_fresh_seeds" in r]
    shrink = [r["shrinkage_test_to_fresh"] for r in results.values()
              if "shrinkage_test_to_fresh" in r]

    # ---- gates ---------------------------------------------------------------------------------
    if not ok:
        gr.skip("catch22_features_available", "live", why)
    else:
        used = [r["multivariate_with_catch22"].get("n_features_used", 0)
                for r in results.values() if "skipped" not in r["multivariate_with_catch22"]]
        gr.live("catch22_features_reach_the_model", float(max(used) if used else 0),
                float(len(feats_hand)) + 5.0,
                detail=("the fitted model must actually receive more features than the "
                        "hand-picked baseline in at least one cell. A first version dropped every "
                        "catch22 column wherever one row had a constant series, so the "
                        "'multivariate with catch22' arm was silently identical to the "
                        "hand-picked one and reported no gain."))
        gr.live("catch22_produced_features", float(len(feats_c22)), 20.0,
                detail="catch22 returns 22 features per series over three series; a near-empty "
                       "feature block would make the comparison meaningless while still running.")
    full = disc[(disc.forced_k == 24) & (disc.mu == 3) & (disc.beta == 1.0)]
    if len(full):
        a = auc(full[full.cls == "hierarchical"].hand_goal_final_entropy.to_numpy(),
                full[full.cls == "ghost"].hand_goal_final_entropy.to_numpy())
        gr.positive("easiest_discrimination_is_easy", max(a, 1.0 - a), 1.0, 0.12,
                    detail="a real chain versus pure synthetic, read for a full 24 steps, is the "
                           "easiest separation this model can pose. A broken reader returns 0.5.")
    if shrink:
        gr.no_oracle("held_out_scoring_is_not_optimistic", float(max(shrink)), 0.25,
                     detail="test AUC minus fresh-seed AUC. Large positive shrinkage means the "
                            "held-out split inside one seed block is still leaking -- artifacts "
                            "from one block share a world draw even when the indices differ.")

    verdict = {
        "test": "T-8 — a multivariate maker-detector on held-out and fresh-seed data",
        "for": "Sounding Line; the instrument question T-5 could only pose",
        "method": {
            "features": {"hand_picked": len(feats_hand), "catch22": len(feats_c22),
                         "series": ["subgoal_entropy", "goal_entropy", "subgoal_travel"]},
            "model": "L2-regularised logistic regression on standardised features",
            "split": "fit on even-indexed artifacts, score on odd-indexed, then on a fresh "
                     "seed block that did not exist when the model was fitted",
            "no_oracle_statistics": "every feature is computed from the reader's own posteriors. "
                                    "Nothing is scored against the true goal or the true mode.",
            "n_per_class_per_cell": int(n_obs),
        },
        "by_cell": results,
        "summary": {
            "median_catch22_gain_on_fresh_seeds": (float(np.median(gains)) if gains else None),
            "worst_catch22_gain_on_fresh_seeds": (float(min(gains)) if gains else None),
            "best_catch22_gain_on_fresh_seeds": (float(max(gains)) if gains else None),
            "cells_where_catch22_helps": int(sum(g > 0.01 for g in gains)),
            "cells_total": len(gains),
            "median_shrinkage_test_to_fresh": (float(np.median(shrink)) if shrink else None),
        },
        "how_to_read": (
            "the only number that means anything is catch22_gain_on_fresh_seeds: a fifty-feature "
            "fitted model's AUC on artifacts it has never seen, minus the AUC of the single best "
            "hand-picked feature chosen on the training block alone. Positive means the extra "
            "machinery earns its place and is worth transporting. Near zero or negative means one "
            "number somebody thought of is as good, and the machinery is fitting noise."),
        "what_would_have_falsified_it": (
            "a large catch22 gain on the held-out split that vanishes on fresh seeds. That is the "
            "signature of feature selection on noise and is exactly what the third block exists "
            "to catch."),
        "what_this_cannot_show": (
            "anything about text. 'Foreign' is a real policy over an unmodelled goal, this "
            "repository's stand-in for content whose maker cannot be reconstructed. It is not "
            "machine-generated writing and this is not a deployable detector."),
    }
    PROVENANCE.stamp(verdict, __file__, gr)
    (sl_dir() / "t8_multivariate_detection.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
