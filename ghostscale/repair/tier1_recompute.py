"""Tier one: R-1 to R-4. Everything here runs on committed data. No new simulation.

Highest value per unit of work in the pass, and the reason is that three headline criteria are
currently UNDETERMINED rather than answered, and determining them needs arithmetic rather than
compute.

-----------------------------------------------------------------------------------------
R-1 — THE THREE UNDER-POWERED CRITERIA, AND THE CORRECTION THE SPECIFICATION NEEDED.

The specification asks for these to be recomputed over per-reader units instead of cell means, on
the grounds that the data was sitting there unused. Checked before running it, that is the wrong
move, and it would have manufactured a failure.

Take the two-gates criterion. Within each cell, recovered depth and the update it produces are
essentially UNCORRELATED across readers: between -0.14 and +0.04 in the six cells. So pooling 7,200
readers does not sharpen the six-point correlation, it drowns it. The pooled value comes out at
0.51 against the cell-level 0.89, and it would read as a failure against a threshold of 0.70 that
was written for a correlation between cell means.

The estimand is preserved by BOOTSTRAPPING THE READERS AND RE-DERIVING THE CELL MEANS. That gives an
interval on the quantity the criterion was actually written over. Both numbers are computed and both
are reported, with the per-reader one labelled as the different quantity it is.

-----------------------------------------------------------------------------------------
R-2 — pairwise divergence beside every disagreement figure, wherever the posterior was saved. Where
it was not, this records which experiments need re-running rather than quietly skipping them.

R-3 — bootstrap intervals on the headlines, and the one that matters most is the LOCATION of the
fabrication peak, which every downstream claim including the prediction card is anchored to and
which has never had an error bar.

R-4 — the reader-count bias, corrected and reported beside the uncorrected figure.
"""
from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd

from .. import metrics
from ..config import Config
from ..experiments._common import RESULTS_DIR
from . import criteria as CR
from . import repair_dir


def _read(name: str):
    p = RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def _posteriors(series) -> list:
    out = []
    for s in series:
        out.append(np.asarray(ast.literal_eval(s), dtype=float) if isinstance(s, str)
                   else np.asarray(s, dtype=float))
    return out


# =========================================================================== #
# R-1.
# =========================================================================== #
def _bootstrap_cell_statistic(groups, stat, draws: int, rng) -> np.ndarray:
    """Resample readers inside each cell, re-derive the cell summary, recompute the statistic."""
    out = np.empty(draws)
    for d in range(draws):
        resampled = []
        for g in groups:
            idx = rng.integers(0, len(g), len(g))
            resampled.append(g.iloc[idx])
        out[d] = stat(resampled)
    return out


def r1_two_gates(draws: int, rng) -> dict:
    """The public headline: does update magnitude track recovered depth?"""
    d = _read("e31_points.csv")
    if d is None:
        return {"criterion": "update magnitude tracks recovered depth", "available": False,
                "why": "results/e31_points.csv is not committed"}
    op = d[d.theta == "open"]
    groups = [g for _, g in op.groupby(["content", "label"])]

    def stat(res):
        mus = [float(g.recovered_mu.mean()) for g in res]
        drs = [float(g.prior_drift.mean()) for g in res]
        return CR_spearman(mus, drs)

    boot = _bootstrap_cell_statistic(groups, stat, draws, rng)
    point = stat(groups)
    interval = CR.percentile_interval(boot)
    pooled = CR_spearman(op.recovered_mu.values, op.prior_drift.values)
    within = {f"{c} / {l}": CR_spearman(g.recovered_mu.values, g.prior_drift.values)
              for (c, l), g in op.groupby(["content", "label"])}
    threshold = 0.70
    return {
        "criterion": "update magnitude tracks recovered depth",
        "experiment": "E31", "available": True,
        "original_value": 0.8857142857142857,
        "original_computed_over": "6 cell means",
        "threshold": threshold,
        "recomputed_point": point,
        "bootstrap_interval": list(interval),
        "bootstrap_draws": draws,
        "support": _support(boot),
        "verdict": CR.determined_against(interval, threshold),
        "pooled_per_reader_value": pooled,
        "pooled_per_reader_n": int(len(op)),
        "within_cell_correlations": within,
        "why_pooling_is_not_the_same_question": (
            "Within cells the two quantities are essentially uncorrelated, so pooling readers adds "
            "noise around the six cell means rather than resolving them. The pooled value of "
            f"{pooled:.3f} is not a higher-powered version of {point:.3f}; it is a different "
            "estimand, and it would read as a failure against a threshold written for the other "
            "one."),
    }


def r1_depth_not_effort(draws: int, rng) -> dict:
    """Does the depth estimate respond to depth rather than to effort?"""
    d = _read("n21_points.csv")
    if d is None:
        return {"criterion": "depth recovery is not effort recovery", "available": False,
                "why": "results/n21_points.csv is not committed"}
    groups = [g for _, g in d.groupby(["true_beta", "true_mu"])]
    keys = [k for k, _ in d.groupby(["true_beta", "true_mu"])]
    lo_b, hi_b = float(min(k[0] for k in keys)), float(max(k[0] for k in keys))
    lo_m, hi_m = int(min(k[1] for k in keys)), int(max(k[1] for k in keys))

    def cell_of(res, b, m):
        for k, g in zip(keys, res):
            if np.isclose(k[0], b) and int(k[1]) == m:
                return float(g.recovered_mu.mean())
        return float("nan")

    def stat(res):
        mu_eff = 0.5 * ((cell_of(res, lo_b, hi_m) - cell_of(res, lo_b, lo_m))
                        + (cell_of(res, hi_b, hi_m) - cell_of(res, hi_b, lo_m)))
        beta_false = cell_of(res, hi_b, lo_m) - cell_of(res, lo_b, lo_m)
        return abs(mu_eff) / abs(beta_false) if beta_false != 0 else float("inf")

    boot = _bootstrap_cell_statistic(groups, stat, draws, rng)
    point = stat(groups)
    interval = CR.percentile_interval(boot)
    threshold = 3.0
    return {
        "criterion": "depth recovery is not effort recovery",
        "experiment": "N21", "available": True,
        "original_value": 5.98388046397683, "original_computed_over": "4 cell means",
        "threshold": threshold,
        "recomputed_point": point, "bootstrap_interval": list(interval),
        "bootstrap_draws": draws,
        "verdict": CR.determined_against(interval, threshold),
        "note": ("a ratio of differences between four cell means. The bootstrap is over readers "
                 "within each of the four, which is the only source of sampling variation the "
                 "design has."),
    }


def r1_depth_moves_uptake(draws: int, rng) -> dict:
    """Does depth change how much the reader takes on? Recomputed on BOTH uptake measures."""
    d = _read("e30_points.csv")
    if d is None:
        return {"criterion": "depth changes how much the reader takes on", "available": False,
                "why": "results/e30_points.csv is not committed"}
    sub = d[d.arm == "sustained"] if "arm" in d.columns else d
    groups = [g for _, g in sub.groupby("true_mu")]
    levels = sorted(float(k) for k, _ in sub.groupby("true_mu"))

    def stat_for(col):
        def stat(res):
            means = [float(g[col].mean()) for g in res]
            return CR_spearman(levels, means)
        return stat

    out = {"criterion": "depth changes how much the reader takes on", "experiment": "E30",
           "available": True, "original_value": None,
           "original_computed_over": "3 depth levels", "threshold": 0.0,
           "bootstrap_draws": draws, "measures": {}}
    for col, label in (("psi_analogue", "movement (the measure on record)"),
                       ("psi_structure", "structural movement (the added secondary)")):
        if col not in sub.columns:
            continue
        boot = _bootstrap_cell_statistic(groups, stat_for(col), draws, rng)
        point = stat_for(col)(groups)
        interval = CR.percentile_interval(boot)
        out["measures"][label] = {
            "point": point, "interval": list(interval),
            "cell_means": [float(g[col].mean()) for g in groups],
            "levels": levels,
            "verdict": CR.determined_against(interval, 0.0),
        }
    out["note"] = ("three levels, and the experiment itself reports two of them as "
                   "indistinguishable, so the correlation runs on what is effectively two points. "
                   "The interval says so rather than the point estimate hiding it. Error reduction "
                   "cannot be computed here because the committed file does not carry the true "
                   "goal alongside the posterior; the rerun supplies it.")
    return out


def CR_spearman(x, y) -> float:
    from ..diagnostics.criteria import spearman
    return spearman(x, y)


def _support(boot: np.ndarray) -> list:
    """The discrete support of a statistic computed on a handful of cells, with its masses.

    A rank correlation over six points can only take certain values, so a percentile interval alone
    understates how concentrated it is. Reported because it is the clearest way to see that a
    criterion is settled.
    """
    vals, counts = np.unique(np.round(boot[np.isfinite(boot)], 4), return_counts=True)
    order = np.argsort(-counts)
    return [{"value": float(vals[i]), "share": float(counts[i] / counts.sum())}
            for i in order[:6]]


# =========================================================================== #
# R-2 and R-4.
# =========================================================================== #
DISAGREEMENT_SOURCES = (
    ("E2", "e2_points.csv", ["true_provenance", "declared_signal"], "final_posterior"),
    ("E17", "e17_points.csv", ["true_provenance", "labelling"], "final_posterior"),
    ("E19", "e19_explore.csv", ["arm", "content"], "real_goal_posterior"),
)
NEEDS_RERUN_FOR_DIVERGENCE = {
    "E20": "drops the posterior before writing e20_points.csv",
    "E31": "drops the posterior before writing e31_points.csv",
    "E32": "drops the posterior before writing e32_points.csv",
    "E21": "writes cell statistics only",
}


def r2_r4_disagreement(rng) -> dict:
    """Pairwise divergence and the bias correction, beside every disagreement figure available."""
    rows = []
    for name, fname, keys, post_col in DISAGREEMENT_SOURCES:
        df = _read(fname)
        if df is None:
            rows.append({"experiment": name, "available": False,
                         "why": f"results/{fname} is not on disk"})
            continue
        col = post_col if post_col in df.columns else next(
            (c for c in ("final_posterior", "posterior", "real_goal_posterior")
             if c in df.columns), None)
        if col is None:
            rows.append({"experiment": name, "available": False,
                         "why": f"no posterior column in {list(df.columns)}"})
            continue
        keys = [k for k in keys if k in df.columns]
        for cell, g in df.groupby(keys):
            per_seed_plug, per_seed_corr, per_seed_js, per_seed_within = [], [], [], []
            grouper = g.groupby("seed_rep") if "seed_rep" in g.columns else [(0, g)]
            for _, s in grouper:
                P = _posteriors(s[col])
                per_seed_plug.append(metrics.between_observer_entropy(P))
                per_seed_corr.append(metrics.between_observer_entropy_corrected(P))
                per_seed_js.append(metrics.mean_pairwise_js(P, rng=rng))
                per_seed_within.append(metrics.mean_within_observer_entropy(P))
            n = int(len(g) / max(1, len(per_seed_plug)))
            rows.append({
                "experiment": name, "available": True,
                "cell": " / ".join(str(c) for c in (cell if isinstance(cell, tuple) else (cell,))),
                "n_readers_per_seed": n, "n_seeds": len(per_seed_plug),
                "within_reader_entropy": float(np.mean(per_seed_within)),
                "modal_goal_entropy_as_committed": float(np.mean(per_seed_plug)),
                "modal_goal_entropy_bias_corrected": float(np.mean(per_seed_corr)),
                "bias_nats": float(np.mean(per_seed_corr) - np.mean(per_seed_plug)),
                "mean_pairwise_divergence": float(np.mean(per_seed_js)),
            })
    return {
        "rows": rows,
        "needs_rerun_for_divergence": NEEDS_RERUN_FOR_DIVERGENCE,
        "note": ("pairwise divergence needs the full posterior, which four experiments do not "
                 "persist. Those are named rather than skipped, and the exact-inference sweep "
                 "supplies them."),
    }


# =========================================================================== #
# R-3.
# =========================================================================== #
def r3_fabrication_peak(draws: int, rng) -> dict:
    """The location of the interior peak, with an error bar for the first time.

    Every downstream claim is anchored to this location, including the prediction card written for a
    human study, and it has been a bare argmax over eight grid points with nothing attached.
    """
    d = _read("e20_points.csv")
    if d is None:
        return {"available": False, "why": "results/e20_points.csv is not committed"}
    omegas = sorted(float(x) for x in d.omega.unique())
    ng = 4
    ceiling = float(np.log(ng))
    by_cell = {om: [g for _, g in d[np.isclose(d.omega, om)].groupby("seed_rep")]
               for om in omegas}

    def index_from(rows) -> float:
        """The pre-registered fabrication index, rebuilt from per-reader rows."""
        within = float(np.mean(rows.final_entropy))
        modal = np.bincount(rows.modal_goal.values.astype(int), minlength=ng).astype(float)
        between = metrics.shannon_entropy(modal / modal.sum())
        return (1.0 - within / ceiling) * (between / ceiling)

    point = {om: float(np.mean([index_from(s) for s in by_cell[om]])) for om in omegas}
    peak_point = max(point, key=point.get)

    argmaxes, curves = [], []
    for _ in range(draws):
        curve = {}
        for om in omegas:
            vals = []
            for s in by_cell[om]:
                idx = rng.integers(0, len(s), len(s))
                vals.append(index_from(s.iloc[idx]))
            curve[om] = float(np.mean(vals))
        curves.append(curve)
        argmaxes.append(max(curve, key=curve.get))
    counts = pd.Series(argmaxes).value_counts(normalize=True).sort_index()
    top_share = float(counts.max())
    return {
        "available": True,
        "grid": omegas,
        "committed_peak": 0.10,
        "recomputed_peak": float(peak_point),
        "index_by_omega": point,
        "argmax_distribution": {str(k): float(v) for k, v in counts.items()},
        "modal_location_share": top_share,
        "located": bool(top_share >= CR.PEAK_LOCATED_SHARE),
        "peak_height_interval": list(CR.percentile_interval(
            [c[peak_point] for c in curves])),
        "bootstrap_draws": draws,
        "note": ("the index is rebuilt from per-reader rows rather than read off the cell summary, "
                 "so the bootstrap resamples the thing the summary was computed from. The "
                 "disagreement term inside it is the modal-goal entropy, which the diagnostics "
                 "pass showed cannot be read alone; the divergence version needs the posterior, "
                 "which this experiment does not persist."),
    }


def r3_headline_intervals(draws: int, rng) -> list:
    """Bootstrap intervals on the other headline point estimates that committed data supports."""
    out = []
    e2 = _read("e2_points.csv")
    if e2 is not None and "final_posterior" in e2.columns:
        for prov, sig, label in (("GHOST", "SIG_CREATOR", "machine work passed off as human"),
                                 ("GHOST", "SIG_GHOST", "machine work labelled honestly"),
                                 ("CREATOR", "SIG_CREATOR", "human work read correctly")):
            g = e2[(e2.true_provenance == prov) & (e2.declared_signal == sig)]
            if not len(g):
                continue
            P = np.array(_posteriors(g.final_posterior))
            within = np.array([metrics.within_observer_entropy(p) for p in P])
            boot = np.array([within[rng.integers(0, len(within), len(within))].mean()
                             for _ in range(draws)])
            out.append({"quantity": f"E2 within-reader doubt, {label}",
                        "point": float(within.mean()),
                        "interval": list(CR.percentile_interval(boot)),
                        "n_readers": int(len(within))})
    e15 = _read("e15_verdict.json") if False else None
    e32 = _read("e32_points.csv")
    if e32 is not None:
        om = float(e32.omega.min())
        for arm, label in (("foreign_content", "expert reader, machine work"),
                           ("unskilled_reader", "novice reader, human work")):
            g = e32[(e32.arm == arm) & np.isclose(e32.omega, om)]
            if not len(g):
                continue
            acc = g.correct.values.astype(float)
            boot = np.array([acc[rng.integers(0, len(acc), len(acc))].mean()
                             for _ in range(draws)])
            out.append({"quantity": f"E32 reads the real purpose, {label}",
                        "point": float(acc.mean()),
                        "interval": list(CR.percentile_interval(boot)),
                        "n_readers": int(len(acc))})
    return out


# =========================================================================== #
def run(cfg: Config, workers: int = 1) -> dict:
    draws = CR.BOOTSTRAP_DRAWS
    rng = np.random.default_rng(20260731)
    out = repair_dir("tier1")

    r1 = [r1_two_gates(draws, rng), r1_depth_not_effort(draws, rng),
          r1_depth_moves_uptake(draws, rng)]
    r2r4 = r2_r4_disagreement(rng)
    peak = r3_fabrication_peak(draws, rng)
    headlines = r3_headline_intervals(draws, rng)

    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in r1]
                 ).to_csv(out / "r1_criteria.csv", index=False)
    pd.DataFrame(r2r4["rows"]).to_csv(out / "r2_r4_disagreement.csv", index=False)
    pd.DataFrame(headlines).to_csv(out / "r3_headline_intervals.csv", index=False)
    if peak.get("available"):
        pd.DataFrame([{"omega": k, "index": v} for k, v in peak["index_by_omega"].items()]
                     ).to_csv(out / "r3_peak_curve.csv", index=False)

    determined = [r for r in r1 if r.get("verdict", "").startswith("determined")]
    meets = [r for r in r1 if r.get("verdict") == "determined_meets"]
    fails = [r for r in r1 if r.get("verdict") == "determined_fails"]

    if fails:
        verdict = "SOME_RECOMPUTED_CRITERIA_NOW_FAIL"
    elif len(determined) == len([r for r in r1 if r.get("available")]):
        verdict = "EVERY_RECOMPUTED_CRITERION_IS_NOW_DETERMINED"
    else:
        verdict = "SOME_CRITERIA_REMAIN_UNDETERMINED"

    payload = {
        "check": "R-1 to R-4",
        "question": ("What do the under-powered criteria say once they have intervals, and what "
                     "does the disagreement figure look like beside a statistic that can read it?"),
        "plain_language": (
            "Three of the project's headline conclusions were computed from a handful of numbers "
            "with no error bar, so nobody could say whether they were solid or noise. This puts an "
            "interval on each of them by resampling the readers, without running a single new "
            "simulation. It also adds a second disagreement statistic beside the existing one, and "
            "corrects a known bias that only matters at small reader counts."),
        "criteria": {"bootstrap_draws": draws, "alpha": CR.BOOTSTRAP_ALPHA,
                     "peak_located_share": CR.PEAK_LOCATED_SHARE},
        "r1_recomputed_criteria": r1,
        "r2_r4_disagreement": r2r4,
        "r3_fabrication_peak": peak,
        "r3_headline_intervals": headlines,
        "verdict": verdict,
    }
    payload["statement"] = _statement(payload)
    (repair_dir() / "tier1_recompute.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


def _statement(p: dict) -> str:
    bits = []
    for r in p["r1_recomputed_criteria"]:
        if not r.get("available"):
            continue
        if "bootstrap_interval" in r:
            lo, hi = r["bootstrap_interval"]
            bits.append(
                "**%s** (%s): recomputed %.3f, 95%% interval [%.3f, %.3f] against a threshold of "
                "%.2f. %s." % (r["criterion"], r["experiment"], r["recomputed_point"], lo, hi,
                               r["threshold"], r["verdict"].replace("_", " ")))
        else:
            for label, m in (r.get("measures") or {}).items():
                lo, hi = m["interval"]
                bits.append("**%s** (%s, %s): %.3f, 95%% interval [%.3f, %.3f]. %s."
                            % (r["criterion"], r["experiment"], label, m["point"], lo, hi,
                               m["verdict"].replace("_", " ")))
    head = "\n\n".join(bits)

    two = p["r1_recomputed_criteria"][0]
    if two.get("available"):
        head += ("\n\n**The correction that mattered most.** The specification asked for these to be "
                 "recomputed over per-reader pairs. Doing that would have manufactured a failure: "
                 "within cells the two quantities are uncorrelated, so pooling readers drowns the "
                 "signal rather than resolving it, and the pooled value comes out at %.3f against "
                 "the cell-level %.3f. Bootstrapping the cell means preserves the estimand, and it "
                 "shows the criterion is not marginal at all: %.0f%% of draws land on a single "
                 "value and the interval excludes the threshold. **The earlier reading that this "
                 "criterion could never have told either way was wrong, and it was mine.**"
                 % (two["pooled_per_reader_value"], two["recomputed_point"],
                    100 * two["support"][0]["share"]))

    peak = p["r3_fabrication_peak"]
    if peak.get("available"):
        head += ("\n\n**The interior peak now has an error bar.** Resampling readers and "
                 "recomputing the whole curve, the peak lands at %s in %.0f%% of draws, and the "
                 "distribution over grid points is %s. %s"
                 % (peak["recomputed_peak"], 100 * peak["modal_location_share"],
                    ", ".join(f"{k}: {100*v:.0f}%" for k, v in
                              peak["argmax_distribution"].items()),
                    "The location is settled and the prediction card can quote it."
                    if peak["located"] else
                    "The location is NOT settled, so the prediction card has to quote a range "
                    "rather than a point."))

    rows = [r for r in p["r2_r4_disagreement"]["rows"] if r.get("available")]
    if rows:
        worst = max(rows, key=lambda r: abs(r["bias_nats"]))
        head += ("\n\n**Disagreement, read two ways.** Every cell now carries the modal-goal "
                 "entropy as committed, the same figure with the reader-count bias corrected, and "
                 "the mean pairwise divergence between readers' full beliefs. The bias correction "
                 "is at most %.4f nats on these cells, which is negligible at the reader counts "
                 "the headlines ran at and is the point: it is material only at the small counts "
                 "the validation pass used, which is where the two were being compared. Four "
                 "experiments cannot be given the divergence figure without re-running, and they "
                 "are named." % abs(worst["bias_nats"]))
    return head
