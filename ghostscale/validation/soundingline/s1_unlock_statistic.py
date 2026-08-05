"""S-1 — is Sounding Line's unlock ratio measuring what E36's measure measures?

THE QUESTION. Sounding Line's primary is a count ratio with no ground truth in it:

    unlock = decisions_recovered_after_purpose_settles / decisions_recovered_before

E36's is ``process_error_reduction``: the mean log-probability the reader assigns to the maker's
TRUE execution mode, against a uniform baseline. Those are different quantities, and E36's own
file records that a count-style statistic was tried there first and was wrong -- it came out below
nominal chance, which no amount of information can produce.

TWO THINGS THIS SETTLES.

  1  Do the two statistics agree, cell by cell? If they do not, Sounding Line's primary does not
     inherit E36's support and has to earn its own.

  2  N28. At mu = 1 the construction guarantees there is no process to recover: every execution
     mode emits the goal signature exactly, so the sub-goal posterior never leaves its prior.
     ``process_error_reduction`` is built to read 0 there and does. **A count ratio has no such
     guarantee**, and if it moves at mu = 1 then it is reading something that is not process --
     in an environment where we can prove nothing is there.

WHAT WOULD FALSIFY THE WORRY. A count ratio that sits at 1.0 at mu = 1 and correlates with
process_error_reduction across cells. Then the two are interchangeable here and Sounding Line's
primary is fine.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ...config import Config
from ...prereg_v6 import BOOTSTRAP_DRAWS, percentile_interval
from ...v6 import SEED_OFFSET
from ...methods import provenance as PROVENANCE
from . import sl_dir
from .common import concentration, process_gain, ratio, resolved_steps, rollouts, usable

# A sub-goal posterior counts as RESOLVED below this share of its own maximum entropy. Swept
# rather than chosen, because a threshold picked once is a result about the threshold.
# Chosen AFTER looking at where this reader's sub-goal entropy actually lives, and that is
# reported rather than hidden: the median sub-goal posterior sits at 96.5% of maximum entropy, so
# thresholds of 0.25 and 0.50 never fire once in 288 steps and every ratio is undefined. The
# diffuseness is itself part of the answer.
THRESHOLDS = (0.75, 0.90, 0.95)


def run(cfg: Config, n_obs: int = 120, n_timesteps: int = 24, forced_k: int = 24) -> dict:
    rows = []
    for rec in rollouts(cfg, n_obs=n_obs, n_timesteps=n_timesteps, forced_k=forced_k):
        if not usable(rec):
            continue
        enc, split, n_sub = rec["enc"], rec["settled"], rec["n_sub"]
        row = {"mu": rec["mu"], "beta": rec["beta"], "settled_at": split,
               "process_gain": process_gain(enc, split, n_sub)}
        for th in THRESHOLDS:
            b, a = resolved_steps(enc, split, n_sub, th)
            row[f"count_before@{th}"] = b
            row[f"count_after@{th}"] = a
            row[f"count_ratio@{th}"] = ratio(b, a)
        cb, ca = concentration(enc, split, n_sub)
        row["conc_before"], row["conc_after"] = cb, ca
        row["conc_ratio"] = ratio(cb, ca)
        rows.append(row)

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED_OFFSET + 90_100)

    # ---- N28: at mu = 1 there is no process, so the ratio must not move ---- #
    n28 = {}
    for th in THRESHOLDS:
        sub = df[(df.mu == 1)][f"count_ratio@{th}"].replace([np.inf, -np.inf], np.nan).dropna()
        if not len(sub):
            n28[str(th)] = {"n": 0, "verdict": "NOT_MEASURABLE"}
            continue
        v = sub.to_numpy()
        draws = [float(np.mean(rng.choice(v, v.size, replace=True))) for _ in range(BOOTSTRAP_DRAWS)]
        lo, hi = percentile_interval(draws)
        holds = bool(lo <= 1.0 <= hi)
        n28[str(th)] = {"mean_ratio": float(v.mean()), "interval": [lo, hi], "n": int(v.size),
                        "contains_one": holds,
                        "verdict": "PASSES_N28" if holds else "FAILS_N28"}

    # ---- does the count ratio track the measure it is standing in for? ---- #
    corr = {}
    cells = df.groupby(["mu", "beta"])
    for th in THRESHOLDS:
        cell_means = cells.agg(pg=("process_gain", "mean"),
                               cr=(f"count_ratio@{th}", lambda s: s.replace(
                                   [np.inf, -np.inf], np.nan).dropna().mean())).dropna()
        if len(cell_means) >= 3:
            r = float(np.corrcoef(cell_means.pg, cell_means.cr)[0, 1])
        else:
            r = float("nan")
        # Per-rollout, which is the harder and fairer version of the same question.
        pr = df[["process_gain", f"count_ratio@{th}"]].replace([np.inf, -np.inf], np.nan).dropna()
        r_row = (float(np.corrcoef(pr.process_gain, pr[f"count_ratio@{th}"])[0, 1])
                 if len(pr) >= 3 else float("nan"))
        corr[str(th)] = {"across_cells": r, "n_cells": int(len(cell_means)),
                         "across_rollouts": r_row, "n_rollouts": int(len(pr))}

    # ---- the threshold-free variant, on the same rollouts ------------------ #
    cr = df.conc_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    c_mu1 = df[df.mu == 1].conc_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    conc = {}
    if len(c_mu1):
        v = c_mu1.to_numpy()
        draws = [float(np.mean(rng.choice(v, v.size, replace=True))) for _ in range(BOOTSTRAP_DRAWS)]
        lo, hi = percentile_interval(draws)
        conc["n28_at_mu_1"] = {"mean_ratio": float(v.mean()), "interval": [lo, hi],
                               "n": int(v.size), "contains_one": bool(lo <= 1.0 <= hi),
                               "verdict": "PASSES_N28" if lo <= 1.0 <= hi else "FAILS_N28"}
    pr = df[["process_gain", "conc_ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
    conc["agreement_across_rollouts"] = (float(np.corrcoef(pr.process_gain, pr.conc_ratio)[0, 1])
                                         if len(pr) >= 3 else float("nan"))
    conc["n_defined"] = int(len(cr))
    conc["of"] = int(len(df))

    # ---- how often is it simply undefined? --------------------------------- #
    undefined = {str(th): {
        "nan_or_inf": int(df[f"count_ratio@{th}"].replace([np.inf, -np.inf], np.nan).isna().sum()),
        "of": int(len(df))} for th in THRESHOLDS}

    df.to_csv(sl_dir() / "s1_unlock_statistic_points.csv", index=False)
    verdict = {
        "test": "S-1 — is the unlock ratio measuring what process_error_reduction measures?",
        "for": "Sounding Line, Gate 3 primary",
        "n_rollouts": int(len(df)),
        "resolved_thresholds_swept": list(THRESHOLDS),
        "n28_at_mu_1": n28,
        "agreement_with_process_error_reduction": corr,
        "undefined_ratios": undefined,
        "threshold_free_concentration_ratio": conc,
        "what_would_have_falsified_the_worry": (
            "a count ratio whose interval covers 1.0 at mu = 1, and which correlates with "
            "process_error_reduction across cells. Then the two are interchangeable in an "
            "environment with ground truth and the primary inherits E36's support."),
        "what_this_cannot_show": (
            "nothing about real text. 'A decision was recovered' is mapped here onto 'the "
            "sub-goal posterior resolved below an entropy threshold', which is the nearest "
            "honest analogue of a statistic that never consults the truth. A different mapping "
            "could behave differently and the threshold is swept for that reason."),
    }
    PROVENANCE.stamp(verdict, __file__)
    (sl_dir() / "s1_unlock_statistic.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
