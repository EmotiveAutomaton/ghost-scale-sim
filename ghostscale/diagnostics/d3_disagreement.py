"""D-3 — the disagreement estimator. Is "no two readers agree" a measurement or an artefact?

`metrics.between_observer_entropy` is the instrument behind the half of the headline the validation
pass concluded the theory is entitled to. It takes each reader's ARGMAX goal, forms the empirical
distribution of those votes, and returns its Shannon entropy. That has two problems, and they pull in
opposite directions.

-----------------------------------------------------------------------------------------
PROBLEM ONE: IT DEPENDS ON THE NUMBER OF READERS.

The plug-in entropy of a count vector is downward biased by about (K - 1) / 2N nats, the Miller-Madow
bias. At the 4000 readers version 1's E2 ran, that is 0.0004 nats and irrelevant. At the 60 readers
the validation pass ran at, it is 0.025. At 16 it is 0.099.

This is why the validation pass's robustness matrix needed its boolean gate made conditional on the
baseline: the disagreement clause was failing in the DEFAULT cell at reduced scale. That was worked
around rather than corrected, and correcting it is one line. Until it is, reduced-scale disagreement
numbers are not comparable to headline-scale ones and cross-experiment comparisons are confounded
wherever the reader counts differ.

-----------------------------------------------------------------------------------------
PROBLEM TWO, WHICH MATTERS MORE: IT CANNOT TELL DISAGREEMENT FROM ARGMAX NOISE.

A reader whose posterior is (0.26, 0.25, 0.25, 0.24) casts a vote as decisive as one at (0.99, ...).
So a population of readers who are all HONESTLY UNCERTAIN, each with a nearly flat posterior, will
scatter their argmaxes at random and produce near-ceiling "disagreement" while agreeing perfectly
about how little they know. That is not the claim. The claim is confident readers landing in
different places.

Whether this bites is an empirical question with a clean answer, and the answer differs by cell:

* the label-effect cell has within-reader entropy near 0.09 against a ceiling of 1.386, so its
  readers are sharply peaked and its disagreement is real;
* the foreign-content cells have within-reader entropy near 1.26 against the same ceiling, so their
  readers are nearly flat and their disagreement may be nothing at all.

THE NULL, AND WHY THE OBVIOUS ONE IS NOT ENOUGH. Resample each reader's modal goal FROM ITS OWN
POSTERIOR and recompute the statistic. Where posteriors are flat that resampling is a coin toss and
the null reproduces the observed value, which is the diagnosis. Where posteriors are PEAKED the
resampling is nearly deterministic, so the null returns each reader's own vote and reproduces the
observed value again — for the opposite reason.

**That is not a null failing, it is the finding.** A statistic computed from vote counts alone cannot
tell N confident readers spread across K goals from N unsure readers coin-tossing across K goals. The
two produce the same count vector and therefore the same entropy. So the between-observer number is
not identified on its own, in either regime, and its value is close to a deterministic function of
the mean within-observer entropy and the reader count.

That does not invalidate the project's claim, because the claim is always the CONJUNCTION: readers
are individually confident AND they differ. The fabrication index multiplies exactly those two
things. What it does invalidate is quoting the disagreement number by itself as a second,
independent piece of evidence, which is how it reads in several places.

So this module reports three things rather than one: the vote-resampling null, a direct measurement
of how much of the between number is predictable from the within number, and **the statistic the
project should have been using** — mean pairwise Jensen-Shannon divergence between readers' full
posteriors, which is near zero when everyone is equally unsure and large only when readers' beliefs
genuinely differ, and which therefore separates the two cases the entropy conflates.

This is a reanalysis wherever the posterior was persisted, and a small fresh probe where it was not.
Nothing existing is re-run and nothing outside `results/diagnostics/` is written.
"""
from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from ..config import Config
from ..experiments._common import RESULTS_DIR
from . import criteria as CR
from . import diagnostics_dir


# --------------------------------------------------------------------------- #
# The corrected estimator. Offered, never substituted.
# --------------------------------------------------------------------------- #
def between_observer_entropy_corrected(posteriors: list) -> float:
    """`metrics.between_observer_entropy` with the Miller-Madow bias correction added.

        H_MM = H_plugin + (m - 1) / 2N

    where m is the number of goals actually OBSERVED among the votes, not the number available.
    Using the observed support is what makes the correction a correction rather than a constant
    offset: a genuinely unanimous population has m = 1 and gets no adjustment.

    THIS DOES NOT REPLACE THE SHIPPED METRIC and must not. Every committed number was produced by
    the plug-in form, and silently swapping the estimator would change what those numbers mean
    without changing the files. It is reported beside them.
    """
    if len(posteriors) == 0:
        return 0.0
    num_goals = np.asarray(posteriors[0]).size
    modal = np.array([int(np.argmax(p)) for p in posteriors])
    counts = np.bincount(modal, minlength=num_goals).astype(float)
    n = float(counts.sum())
    plug_in = metrics.shannon_entropy(counts / n)
    m = float(np.count_nonzero(counts))
    return float(plug_in + (m - 1.0) / (2.0 * n))


def miller_madow_bias(num_goals: int, n_readers: int) -> float:
    """The leading-order bias of the plug-in estimator, as a negative number."""
    if n_readers <= 0:
        return float("nan")
    return -float(num_goals - 1) / (2.0 * float(n_readers))


def mean_pairwise_js(posteriors: list, max_pairs: int = 20000,
                     rng: np.random.Generator | None = None) -> float:
    """Mean Jensen-Shannon divergence between readers' FULL posteriors, in nats.

    THE STATISTIC THE PROJECT SHOULD HAVE BEEN USING for "readers disagree", and the reason is that
    it cannot be fooled by shared uncertainty. If every reader is equally unsure their posteriors are
    nearly identical, so the divergence between them is near zero however scattered their argmaxes
    are. If readers are confident about different things their posteriors are far apart and the
    divergence is large. It therefore separates the two populations that produce the same vote counts
    and the same modal-goal entropy.

    Sub-sampled above ``max_pairs`` because the pair count is quadratic and 200 readers is 19,900
    pairs, which is affordable, while 4000 is eight million, which is not.
    """
    P = np.asarray([np.asarray(p, dtype=float) for p in posteriors], dtype=float)
    P = P / P.sum(axis=1, keepdims=True)
    n = P.shape[0]
    if n < 2:
        return 0.0
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        a = rng.integers(0, n, max_pairs)
        b = rng.integers(0, n, max_pairs)
        idx = [(int(i), int(j)) for i, j in zip(a, b) if i != j]
    return float(np.mean([metrics.js_divergence(P[i], P[j]) for i, j in idx]))


def argmax_noise_null(posteriors: list, n_draws: int = 400,
                      rng: np.random.Generator | None = None) -> dict:
    """The null: each reader votes by sampling from its OWN posterior.

    Returns the null distribution's mean and spread, plus the observed value, so a caller can say
    how far outside the null the observation sits in the null's own units.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    P = np.asarray([np.asarray(p, dtype=float) for p in posteriors], dtype=float)
    P = P / P.sum(axis=1, keepdims=True)
    n, k = P.shape
    observed = metrics.between_observer_entropy(list(P))
    cum = np.cumsum(P, axis=1)
    null = np.empty(n_draws)
    for d in range(n_draws):
        u = rng.random((n, 1))
        votes = (u > cum).sum(axis=1)              # inverse-CDF sample, one per reader
        counts = np.bincount(votes, minlength=k).astype(float)
        null[d] = metrics.shannon_entropy(counts / counts.sum())
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    # How often the resample returns the reader's own argmax. Near 1 means the null is DEGENERATE
    # for this cell: peaked posteriors resample to themselves, so the null cannot destroy the
    # structure it is supposed to destroy, and its agreement with the observation says nothing.
    determinism = float(np.mean(P.max(axis=1)))
    return {
        "n_readers": int(n),
        "num_goals": int(k),
        "mean_within_reader_entropy": float(np.mean([metrics.shannon_entropy(p) for p in P])),
        "entropy_ceiling": float(np.log(k)),
        "mean_pairwise_js": mean_pairwise_js(list(P), rng=rng),
        "js_ceiling": float(np.log(2.0)),
        "null_determinism": determinism,
        "null_is_degenerate": bool(determinism > 0.90),
        "observed_between": float(observed),
        "null_mean": mu,
        "null_sd": sd,
        "standard_deviations_above_null": (float((observed - mu) / sd) if sd > 0
                                           else float("nan")),
        "excess_over_null": float(observed - mu),
        "indistinguishable_from_argmax_noise": bool(abs(observed - mu)
                                                    <= CR.D3_ARGMAX_NULL_TOL),
    }


# --------------------------------------------------------------------------- #
# Reading the committed posteriors.
# --------------------------------------------------------------------------- #
def _parse_posteriors(series) -> list:
    out = []
    for s in series:
        if isinstance(s, str):
            out.append(np.asarray(ast.literal_eval(s), dtype=float))
        else:
            out.append(np.asarray(s, dtype=float))
    return out


SOURCES = (
    # (label, file, group columns, cell filter, why this cell)
    ("E2 machine work passed off as human", "e2_points.csv",
     {"true_provenance": "GHOST", "declared_signal": "SIG_CREATOR"},
     "the headline. Within-reader entropy near 0.09, so the readers are sharply peaked and the "
     "disagreement should be real."),
    ("E2 machine work labelled honestly", "e2_points.csv",
     {"true_provenance": "GHOST", "declared_signal": "SIG_GHOST"},
     "the honest-label comparison cell, where readers are nearly flat. If the null explains this "
     "one and not the cell above, the contrast between them is exactly what it is claimed to be."),
    ("E2 human work read correctly", "e2_points.csv",
     {"true_provenance": "CREATOR", "declared_signal": "SIG_CREATOR"},
     "the positive control. Readers agree and are right, so both observed and null must be near "
     "zero."),
    ("E19 foreign content, fallback available", "e19_explore.csv",
     {"arm": "explore_on", "content": "foreign"},
     "the foreign-content cell. Within-reader entropy near 1.25 against a 1.386 ceiling, so this "
     "is where the null is most likely to explain the whole number."),
    ("E19 exploratory human content", "e19_explore.csv",
     {"arm": "explore_on", "content": "human_exploratory"},
     "the cell whose absorption the fallback is supposed to explain."),
)


def _from_committed(cfg: Config, rng: np.random.Generator) -> list:
    rows = []
    for label, fname, filt, why in SOURCES:
        path = RESULTS_DIR / fname
        if not path.exists():
            rows.append({"cell": label, "source": fname, "available": False,
                         "why_this_cell": why,
                         "note": "not present on disk; regenerate with run_all.py"})
            continue
        df = pd.read_csv(path)
        post_col = next((c for c in ("final_posterior", "posterior", "real_goal_posterior")
                         if c in df.columns), None)
        if post_col is None:
            rows.append({"cell": label, "source": fname, "available": False,
                         "why_this_cell": why,
                         "note": f"no posterior column in {list(df.columns)}"})
            continue
        sub = df
        for col, val in filt.items():
            if col not in sub.columns:
                sub = sub.iloc[0:0]
                break
            sub = sub[sub[col].astype(str) == str(val)]
        if not len(sub):
            rows.append({"cell": label, "source": fname, "available": False,
                         "why_this_cell": why, "note": f"cell {filt} not present"})
            continue
        # Per seed_rep, because that is the unit the statistic is computed over: every reader in a
        # seed sees the SAME artifact, which is what makes disagreement between them meaningful.
        per_seed = []
        for _, g in sub.groupby("seed_rep") if "seed_rep" in sub.columns else [(0, sub)]:
            P = _parse_posteriors(g[post_col])
            per_seed.append(argmax_noise_null(P, n_draws=200, rng=rng))
        agg = {k: float(np.mean([p[k] for p in per_seed]))
               for k in ("observed_between", "null_mean", "null_sd", "excess_over_null",
                         "mean_within_reader_entropy", "entropy_ceiling", "mean_pairwise_js",
                         "null_determinism")}
        agg["n_readers"] = int(np.mean([p["n_readers"] for p in per_seed]))
        agg["n_seeds"] = len(per_seed)
        agg["standard_deviations_above_null"] = (
            float(agg["excess_over_null"] / agg["null_sd"]) if agg["null_sd"] > 0 else float("nan"))
        agg["indistinguishable_from_argmax_noise"] = bool(
            abs(agg["excess_over_null"]) <= CR.D3_ARGMAX_NULL_TOL)
        agg["flatness"] = float(agg["mean_within_reader_entropy"] / agg["entropy_ceiling"])
        agg["null_is_degenerate"] = bool(agg["null_determinism"] > 0.90)
        agg["js_ceiling"] = float(np.log(2.0))
        rows.append({"cell": label, "source": fname, "available": True,
                     "why_this_cell": why, **agg})
    return rows


# --------------------------------------------------------------------------- #
# A fresh probe for the cells whose posteriors were never persisted.
# --------------------------------------------------------------------------- #
def _probe_unpersisted(cfg: Config, rng: np.random.Generator, n_readers: int = 200,
                       n_seeds: int = 6) -> list:
    """E20, E31 and E32 drop the posterior before writing, so their cells need generating.

    This is a NEW run written only into results/diagnostics/, not a re-run of a committed
    experiment: no existing verdict is recomputed and no headline number moves. The design mirrors
    the source cell closely enough for the null to be meaningful, and the differences are recorded.
    """
    from .. import foreign as FN
    from ..creators import HumanCreator
    from ..environment import Artifact, Environment
    from ..exact import make_exact_v4_observer
    from ..experiments import _common as C
    from ..observer import rollout_observer
    from ..v4_model import build_v4_world, load_v4_config

    out = []
    designs = (
        ("E20/E32 fully foreign content (omega = 0)", 0.0, 0.0,
         "where the readability sweep and the two-dimensions result both take their most extreme "
         "cell, and where within-reader entropy is highest"),
        ("E20 partial overlap (omega = 0.10)", 0.10, 0.0,
         "the interior peak. The fabrication index multiplies confidence by DISAGREEMENT, so if "
         "the disagreement term is argmax noise here the index is measuring one thing and not two"),
        ("E32 unskilled reader on human work (d = 0.945)", None, 0.945,
         "the matched arm of the two-dimensions result. Confidently wrong readers should produce "
         "disagreement the null cannot explain, and that is the claim"),
    )
    for label, omega, d_i, why in designs:
        c = load_v4_config(include_explore=False)
        c.set("inference.exact", True)
        c.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
        c.set("cardinalities.num_features",
              int(c.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))
        world = build_v4_world(c, omega=(0.0 if omega is None else omega), include_explore=False)
        creators = {g: HumanCreator(c, world.sigs.sig_true, g)
                    for g in range(FN.NUM_REAL_GOALS)}
        per_seed = []
        for s in range(n_seeds):
            env = Environment(c, world.gm, np.random.default_rng(9_000 + s), honesty=1.0,
                              signing_rate=0.0, creator_bank=creators,
                              foreign_sig=world.sigs.sig_foreign)
            art_rng = np.random.default_rng(50_000 + s)
            k = int(art_rng.integers(FN.NUM_REAL_GOALS))
            if omega is None:                       # human content, unskilled reader
                art = Artifact(provenance=K.CREATOR, goal=k, declared_signal=K.UNSIGNED)
            else:                                   # foreign content, expert reader
                art = Artifact(provenance=K.GHOST, goal=k, declared_signal=K.UNSIGNED,
                               foreign_goal=k)
            P = []
            for i in range(n_readers):
                r = C.observer_rng(777, 0, s, i)
                agent = make_exact_v4_observer(world, r, d_i=float(d_i))
                res = rollout_observer(agent, art, env, c, r, n_timesteps=24, force_deep_k=10,
                                       initial_glance=True)
                P.append(np.asarray(res.final_goal_posterior, dtype=float))
            per_seed.append(argmax_noise_null(P, n_draws=200, rng=rng))
        agg = {kk: float(np.mean([p[kk] for p in per_seed]))
               for kk in ("observed_between", "null_mean", "null_sd", "excess_over_null",
                          "mean_within_reader_entropy", "entropy_ceiling", "mean_pairwise_js",
                          "null_determinism")}
        agg["n_readers"] = n_readers
        agg["n_seeds"] = n_seeds
        agg["standard_deviations_above_null"] = (
            float(agg["excess_over_null"] / agg["null_sd"]) if agg["null_sd"] > 0 else float("nan"))
        agg["indistinguishable_from_argmax_noise"] = bool(
            abs(agg["excess_over_null"]) <= CR.D3_ARGMAX_NULL_TOL)
        agg["flatness"] = float(agg["mean_within_reader_entropy"] / agg["entropy_ceiling"])
        agg["null_is_degenerate"] = bool(agg["null_determinism"] > 0.90)
        agg["js_ceiling"] = float(np.log(2.0))
        out.append({"cell": label, "source": "fresh probe in results/diagnostics/",
                    "available": True, "why_this_cell": why, **agg})
    return out


# --------------------------------------------------------------------------- #
def run(cfg: Config, workers: int = 1) -> dict:
    rng = np.random.default_rng(20260730)
    out = diagnostics_dir("d3_disagreement")

    # Part one: the scale bias, against the reader counts the project actually reports at.
    num_goals = 4
    reported_at = {
        "E2 (version 1 headline)": 4000, "E17": 4000, "E19": 200, "E20": 200, "E21": 200,
        "E32": 200, "E31": 60,
        "validation pass reduced scale": 60,
        "validation V-2b per random draw": 15,
    }
    bias_rows = []
    for where, n in reported_at.items():
        b = miller_madow_bias(num_goals, n)
        bias_rows.append({
            "reported_at": where, "n_readers": n, "bias_nats": b,
            "bias_as_fraction_of_ceiling": abs(b) / float(np.log(num_goals)),
            "scale_sensitive": bool(abs(b) / float(np.log(num_goals)) > CR.D3_BIAS_FRACTION),
        })

    # An empirical confirmation of the analytic bias, so the number is not taken on trust.
    empirical = []
    for n in (16, 30, 60, 200, 1000, 4000):
        vals = []
        for _ in range(300):
            modal = rng.integers(0, num_goals, n)
            P = np.zeros((n, num_goals))
            P[np.arange(n), modal] = 1.0
            vals.append(metrics.between_observer_entropy(list(P)))
        empirical.append({"n_readers": n, "measured": float(np.mean(vals)),
                          "analytic_prediction": float(np.log(num_goals)
                                                       + miller_madow_bias(num_goals, n)),
                          "ceiling": float(np.log(num_goals))})

    # Part two: the argmax-noise null.
    cells = _from_committed(cfg, rng) + _probe_unpersisted(cfg, rng)
    pd.DataFrame(cells).to_csv(out / "argmax_noise_null.csv", index=False)
    pd.DataFrame(bias_rows).to_csv(out / "scale_bias.csv", index=False)

    live = [c for c in cells if c.get("available")]
    scale_sensitive = [r["reported_at"] for r in bias_rows if r["scale_sensitive"]]

    # HOW MUCH OF THE BETWEEN NUMBER IS JUST THE WITHIN NUMBER. If the between-reader entropy is
    # predictable from the mean within-reader entropy, then reporting both as separate evidence
    # double-counts one measurement. Scored as the rank correlation and as the fraction of variance
    # a straight line on the within number explains.
    redundancy = {"n_cells": len(live)}
    if len(live) >= 3:
        w = np.array([c["mean_within_reader_entropy"] for c in live], dtype=float)
        b = np.array([c["observed_between"] for c in live], dtype=float)
        j = np.array([c["mean_pairwise_js"] for c in live], dtype=float)
        redundancy["spearman_between_vs_within"] = CR.spearman(w, b)
        redundancy["spearman_js_vs_within"] = CR.spearman(w, j)
        for name, y in (("between_from_within", b), ("js_from_within", j)):
            if np.ptp(w) > 0:
                fit = np.polyfit(w, y, 1)
                resid = y - np.polyval(fit, w)
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                redundancy["r2_" + name] = (float(1.0 - np.sum(resid ** 2) / ss_tot)
                                            if ss_tot > 0 else float("nan"))

    degenerate = [c["cell"] for c in live if c.get("null_is_degenerate")]
    flat_noise = [c["cell"] for c in live
                  if not c.get("null_is_degenerate")
                  and c.get("indistinguishable_from_argmax_noise")]

    # Does the alternative statistic separate the cells the entropy measure conflates? The test:
    # pairs of cells whose between-entropy is nearly identical and whose mean pairwise JS is not.
    # Each such pair is a case the shipped instrument cannot distinguish and this one can.
    separated = []
    for i in range(len(live)):
        for k2 in range(i + 1, len(live)):
            a, b2 = live[i], live[k2]
            d_between = abs(a["observed_between"] - b2["observed_between"])
            d_js = abs(a["mean_pairwise_js"] - b2["mean_pairwise_js"])
            if d_between < 0.05 and d_js > 0.05:
                separated.append({
                    "cell_a": a["cell"], "cell_b": b2["cell"],
                    "between_entropy_a": a["observed_between"],
                    "between_entropy_b": b2["observed_between"],
                    "between_gap": d_between,
                    "pairwise_js_a": a["mean_pairwise_js"],
                    "pairwise_js_b": b2["mean_pairwise_js"],
                    "js_gap": d_js,
                })

    r2 = redundancy.get("r2_between_from_within", float("nan"))
    pairs_txt = "; ".join(
        'the cells "{}" and "{}" both score about {:.2f} on modal-goal entropy while scoring '
        "{:.3f} against {:.3f} on pairwise divergence".format(
            s["cell_a"], s["cell_b"], s["between_entropy_a"],
            s["pairwise_js_a"], s["pairwise_js_b"]) for s in separated[:3])

    if not live:
        verdict = "NOT_ASSESSABLE"
        statement = "No cell had a posterior available, so nothing is claimed either way."
    elif separated:
        verdict = "DISAGREEMENT_NUMBER_IS_NOT_IDENTIFIED_ON_ITS_OWN"
        statement = (
            "The disagreement statistic cannot be read on its own, and this is a fact about the "
            "statistic rather than about any result.\n\n"
            "It counts each reader's single best guess and measures how spread out the guesses are. "
            "Two completely different populations produce the same spread: readers who are each "
            "certain of a DIFFERENT answer, and readers who are all equally unsure and therefore "
            "guessing. Both give the same count vector and so the same entropy. The vote-resampling "
            "null confirms it from both directions at once, reproducing the observed value in "
            "%d cells because peaked posteriors resample to themselves and in %d because flat ones "
            "resample to noise.\n\n"
            "Measured directly: %.0f%% of the variation in the between-reader number across the "
            "checked cells is explained by a straight line on the mean WITHIN-reader number. It is "
            "close to a restatement of how unsure the readers are.\n\n"
            "**This does not invalidate the headline**, because the headline is always the "
            "conjunction, and the fabrication index multiplies confidence by disagreement rather "
            "than reporting either alone. What it does invalidate is quoting the disagreement "
            "figure by itself as a second independent piece of evidence, which is how it reads in "
            "the results files and in the README.\n\n"
            "**The fix is a better statistic and it is one line.** Mean pairwise Jensen-Shannon "
            "divergence between readers' full posteriors is near zero when everyone is equally "
            "unsure, because their beliefs are then nearly identical, and large only when readers "
            "are confident about different things. It separates %d pair(s) of cells the shipped "
            "statistic reports as indistinguishable: %s."
            % (len(degenerate), len(flat_noise), 100 * r2, len(separated), pairs_txt))
    elif flat_noise:
        verdict = "SOME_DISAGREEMENT_IS_ARGMAX_NOISE"
        statement = (
            "In %d cell(s) the disagreement number is inside what resampling each reader's vote "
            "from its own posterior produces, and those are the cells where readers are nearly "
            "flat. There the number measures the readers' shared uncertainty rather than any "
            "difference between them: %s." % (len(flat_noise), "; ".join(flat_noise)))
    else:
        verdict = "DISAGREEMENT_SURVIVES_THE_NULL"
        statement = ("Every checked cell's disagreement exceeds what argmax noise on the readers' "
                     "own posteriors produces, and the alternative statistic agrees with it "
                     "everywhere, so the shipped instrument is carrying real information.")

    if scale_sensitive:
        statement += (
            "\n\nSeparately, the estimator is downward biased by about (K-1)/2N nats, so numbers "
            "reported at %s are not on the same scale as those reported at 4000 readers. The bias "
            "is analytic, confirmed empirically here, and correctable in one line; the correction "
            "is offered as a separate function rather than substituted, because swapping the "
            "estimator would change what every committed number means without changing the files."
            % ", ".join(scale_sensitive))

    payload = {
        "check": "D-3",
        "question": ("Is 'no two readers agree' measuring disagreement, or the readers' own "
                     "uncertainty scattering their argmaxes?"),
        "plain_language": (
            "The project's disagreement number counts each reader's single best guess and measures "
            "how spread out those guesses are. That works when readers are sure of themselves and "
            "differ. It breaks when readers are all equally unsure, because then their best guess "
            "is close to a coin toss and the coin tosses scatter on their own. This check builds "
            "the coin-toss version of each population, using each reader's actual uncertainty, and "
            "asks whether the real number is any bigger."),
        "criteria": {"argmax_null_tolerance": CR.D3_ARGMAX_NULL_TOL,
                     "bias_fraction": CR.D3_BIAS_FRACTION},
        "scale_bias": bias_rows,
        "scale_bias_empirical_check": empirical,
        "argmax_noise_null": cells,
        "null_degenerate_because_peaked": degenerate,
        "null_reproduces_because_flat": flat_noise,
        "redundancy_with_within_observer": redundancy,
        "pairs_the_shipped_statistic_cannot_separate": separated,
        "recommended_replacement": (
            "mean pairwise Jensen-Shannon divergence over the full posteriors, reported beside the "
            "modal-goal entropy rather than instead of it, so the existing numbers stay readable."),
        "verdict": verdict,
        "statement": statement,
    }
    (diagnostics_dir() / "d3_disagreement.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload
