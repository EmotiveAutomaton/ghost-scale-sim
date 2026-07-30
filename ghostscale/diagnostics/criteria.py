"""Pre-registered criteria for the diagnostics pass, hash-locked before any sweep runs.

A NEW FILE, NOT AN EDIT TO THE VALIDATION LOCK, per the spec's section 3. That lock has reported and
is sealed; a lock that acquires new content after its experiments run is not a lock.

WHAT IS COMMITTED AND WHAT IS NOT. Everything the spec commits in its sections 1.3, 1.5 and 2.5 is
here verbatim, plus the classification thresholds for the five added diagnostics. Grid resolutions,
seed counts and scales are NOT committed: they are engineering decisions and fixing them now would
be pre-registering guesses as though they were design.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import Config

# --------------------------------------------------------------------------- #
# P-1 — parameter recovery. Section 1.3, transcribed.
# --------------------------------------------------------------------------- #
P1_MIN_GRID = 9                # at least 9 true values per parameter
P1_MIN_SEEDS = 16              # at least 16 seeds per grid point

P1_RECOVERED_RHO = 0.90        # rank correlation floor for RECOVERED
P1_RECOVERED_SLOPE = (0.70, 1.30)
P1_RECOVERED_USABLE = 0.70     # usable range as a fraction of the swept range
P1_COMPRESSED_SLOPE = 0.70     # below this, with rho still >= 0.90, is COMPRESSED
P1_PARTIAL_USABLE = (0.40, 0.70)
P1_UNIDENTIFIABLE_RHO = 0.70   # below this, or usable < 0.40, is UNIDENTIFIABLE

# "Usable range" needs an operational definition or the classification cannot be applied
# mechanically. Two adjacent grid points count as DISTINGUISHABLE when the gap between their mean
# recovered values exceeds this many pooled standard errors of those means. Two, because that is the
# conventional reading of "distinguishable" and because it is the same constant V-1 used for
# "within its own spread" — reusing it keeps the two passes commensurable.
P1_DISTINGUISHABLE_SE = 2.0

# Section 1.4 — the joint check. A pair is TRADING OFF when the correlation between the two
# recovery errors exceeds this in absolute value.
P1_TRADEOFF_CORRELATION = 0.50

# Section 1.2 — the cheap pre-check. The goal-marginal likelihood counts as INVARIANT in the
# parameter when the largest elementwise change across the swept range is below this. At that point
# the parameter carries no information in the marginal and all evidence about it comes from joint
# coupling, which predicts compressed and unreliable recovery for structural reasons.
P1_MARGINAL_INVARIANT_TOL = 1e-9

# --------------------------------------------------------------------------- #
# P-2 — the goal-difficulty probe. Section 2.5, transcribed.
# --------------------------------------------------------------------------- #
P2_TARGET_BAND = (0.55, 0.85)      # goal recovery accuracy band that counts as genuinely uncertain
P2_CEILING = 0.90                  # above this across the whole sweep is NO REGIME
P2_UPTAKE_VARIANCE_FACTOR = 1.25   # "materially above its value at the current default"

# --------------------------------------------------------------------------- #
# D-1 — channel accounting.
# --------------------------------------------------------------------------- #
# The label channel BEATS the content channel when its per-observation log-likelihood ratio for the
# declared tier exceeds the content's for the true tier. There is no threshold to commit: the
# comparison is an inequality between two computed numbers. What is committed is the reporting rule.
D1_REPORT_CROSSOVER = True
# A claim is ROBUST TO CHANNEL COMPETITION when the parameters it was measured at sit at least this
# far, in nats per observation, from the crossover. Below that margin the claim's mechanism depends
# on the run length, which is an engineering parameter.
D1_SAFE_MARGIN_NATS = 0.50

# --------------------------------------------------------------------------- #
# D-2 — the uptake response curve.
# --------------------------------------------------------------------------- #
# Uptake counts as NON-MONOTONE in recovery quality when the curve's interior minimum sits this far
# below both endpoints, as a fraction of the endpoint range. A non-monotone uptake measure cannot be
# regressed on a difficulty manipulation without knowing where the arms sit.
D2_NONMONOTONE_DIP = 0.10

# --------------------------------------------------------------------------- #
# D-3 — the disagreement estimator.
# --------------------------------------------------------------------------- #
# The plug-in entropy of modal-goal counts is downward biased by roughly (K - 1) / 2N nats
# (Miller-Madow). A reported disagreement number is SCALE-SENSITIVE when that bias exceeds this
# fraction of the number itself.
D3_BIAS_FRACTION = 0.02
# Disagreement is INDISTINGUISHABLE FROM ARGMAX NOISE when the observed between-reader entropy is
# within this many nats of the value produced by resampling each reader's modal goal from its own
# posterior — a null that preserves within-reader uncertainty and destroys systematic between-reader
# difference.
D3_ARGMAX_NULL_TOL = 0.05

# --------------------------------------------------------------------------- #
# D-5 — criterion power.
# --------------------------------------------------------------------------- #
# A pre-registered criterion is UNDER-POWERED when it is computed over fewer independent units than
# this. Twelve, because a rank correlation over fewer than about a dozen points cannot distinguish
# a strong effect from a moderate one at any conventional level, and rank correlation is the form
# several of this project's primary criteria take.
D5_MIN_UNITS = 12

# --------------------------------------------------------------------------- #
# D-6 — seed independence.
# --------------------------------------------------------------------------- #
# Any collision at all inside a single (cell, seed) group is a defect, because the between-reader
# statistic assumes independent readers within exactly that group. Cross-cell collisions are
# reported separately, with their direction, because they are not automatically a defect.
D6_WITHIN_GROUP_COLLISIONS_ALLOWED = 0


def _canonical(payload: dict) -> str:
    trimmed = {k: v for k, v in payload.items() if k != "content_hash"}
    return json.dumps(trimmed, sort_keys=True, separators=(",", ":"), default=str)


def criteria_payload(cfg: Config) -> dict:
    return {
        "document": "Ghost Scale Simulation — diagnostics pass criteria",
        "spec": "DIAGNOSTICS_P1_P2_SPEC.md",
        "written_before": "any diagnostic sweep ran",
        "separate_from": ("results/validation/criteria.json, which has reported and is sealed"),
        "p1": {
            "min_grid_points": P1_MIN_GRID,
            "min_seeds_per_point": P1_MIN_SEEDS,
            "classification": {
                "RECOVERED": {"rho_at_least": P1_RECOVERED_RHO,
                              "slope_in": list(P1_RECOVERED_SLOPE),
                              "usable_range_at_least": P1_RECOVERED_USABLE},
                "COMPRESSED": {"rho_at_least": P1_RECOVERED_RHO,
                               "slope_below": P1_COMPRESSED_SLOPE},
                "PARTIALLY_RECOVERED": {"usable_range_between": list(P1_PARTIAL_USABLE)},
                "UNIDENTIFIABLE": {"rho_below": P1_UNIDENTIFIABLE_RHO,
                                   "or_usable_range_below": P1_PARTIAL_USABLE[0]},
            },
            "distinguishable_standard_errors": P1_DISTINGUISHABLE_SE,
            "tradeoff_correlation": P1_TRADEOFF_CORRELATION,
            "marginal_invariance_tolerance": P1_MARGINAL_INVARIANT_TOL,
            "consequence": ("A parameter classified UNIDENTIFIABLE cannot support a claim, and "
                            "every claim resting on it has to be restated or withdrawn. "
                            "COMPRESSED is survivable and changes what may be said: directions "
                            "transfer, magnitudes do not."),
            "estimator_note": (
                "THE SPEC'S SECTION 1.1 SAYS 'run the model's own inference on those observations'. "
                "That is well defined for mu, which is a hidden state the observer has a posterior "
                "over. It is NOT defined for kappa, omega or theta, none of which is a hidden "
                "state and none of which the observer can represent. Those three are estimated by "
                "MAXIMISING THE EXACT SEQUENCE LOG-EVIDENCE over a grid, using a model that does "
                "contain the parameter, on a fixed observation tape. That is the standard "
                "simulate-then-fit form of parameter recovery, and it answers a different question "
                "from the mu case: whether an ideal analyst holding the correct model could "
                "identify the value, not whether the observer inside the model can. For omega the "
                "distinction is the entire point of the version 4 reframe, and it is reported "
                "beside every omega number rather than left for a reader to notice."),
        },
        "p2": {
            "target_band": list(P2_TARGET_BAND),
            "ceiling": P2_CEILING,
            "uptake_variance_factor": P2_UPTAKE_VARIANCE_FACTOR,
            "outcomes": ["REGIME_FOUND", "ACCURACY_MOVES_UPTAKE_DOES_NOT", "NO_REGIME"],
            "knob_note": (
                "The spec names three knobs: signature separation, observations before the "
                "decision, and number of goals. Measured before this lock was written, the first "
                "and third do not move goal accuracy at all: peak mass from 0.90 down to 0.22 and "
                "goal count from 4 up to 8 both leave accuracy at 1.000, because every deep "
                "observation is an independent draw from a goal-dependent distribution and any "
                "non-zero per-observation evidence accumulates to certainty over 24 looks. Those "
                "two are therefore run as single confirmatory cells at their extremes and "
                "reported as measured-dead rather than swept. A FOURTH KNOB IS ADDED AND MADE "
                "PRIMARY: reader inexpertise, which introduces systematic rather than sampling "
                "error and so does not average out over the run. The addition is declared here, "
                "before the sweep, with the measurement that motivated it."),
        },
        "d1": {"report_crossover": D1_REPORT_CROSSOVER, "safe_margin_nats": D1_SAFE_MARGIN_NATS},
        "d2": {"nonmonotone_dip": D2_NONMONOTONE_DIP},
        "d3": {"bias_fraction": D3_BIAS_FRACTION, "argmax_null_tolerance": D3_ARGMAX_NULL_TOL},
        "d5": {"min_independent_units": D5_MIN_UNITS},
        "d6": {"within_group_collisions_allowed": D6_WITHIN_GROUP_COLLISIONS_ALLOWED},
        "constraints": [
            "Nothing in results/ outside results/diagnostics/ is written.",
            "No existing experiment is re-run and no existing verdict is recomputed.",
            "Exact inference throughout. If a sweep is unaffordable at exact, the grid is reduced "
            "rather than falling back to the approximation, and the dropped cells are named.",
            "Both P-1 and P-2 must be able to return the unwelcome answer. UNIDENTIFIABLE for a "
            "load-bearing parameter and NO_REGIME are both live possibilities.",
            "No repairs. Nothing in the model changes. The one code change made alongside this "
            "pass is D-4's: ExactAgent gains exact expected-free-energy accessors and a sequence "
            "log-evidence, neither of which any committed number depends on.",
        ],
    }


def write_criteria(cfg: Config, path: Path) -> dict:
    payload = criteria_payload(cfg)
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def ensure_criteria(cfg: Config, path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        write_criteria(cfg, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_hash")
    recomputed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if stated != recomputed:
        raise RuntimeError(
            f"{path.name} has been modified since it was written ({stated} != {recomputed}). "
            "The pre-registered diagnostics criteria are not trustworthy; the pass will not run.")
    return payload


# --------------------------------------------------------------------------- #
# Scoring, applied mechanically.
# --------------------------------------------------------------------------- #
def _tied_ranks(v: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, which is what Spearman's rho is defined on.

    THE REASON THIS IS NOT `argsort(argsort(v))`. That form assigns distinct ranks to tied values in
    whatever order they happen to arrive, so a CONSTANT vector gets ranks 0, 1, 2, ... and correlates
    perfectly with anything ascending. P-1 caught it doing exactly that: an estimator that returned
    the same value at every grid point scored a rank correlation of 1.00, which reads as perfect
    recovery and is the precise opposite. Ties have to be averaged, and a variable with no variance
    has to return NaN rather than a number.
    """
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman's rho with tie-averaged ranks. NaN when either variable is constant.

    NaN rather than 0 for a constant variable, because "no monotone relationship" and "the estimator
    returned the same number every time" are different facts and the classification treats them
    differently: the first is a weak result and the second is an unidentifiable parameter.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return float("nan")
    rx, ry = _tied_ranks(x), _tied_ranks(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def usable_range_fraction(true_vals, means, sems,
                          k: float = P1_DISTINGUISHABLE_SE) -> tuple[float, list]:
    """The fraction of the swept range over which adjacent true values are distinguishable.

    Walks the grid and marks each adjacent pair as distinguishable when the gap between their mean
    recovered values exceeds ``k`` pooled standard errors. The usable fraction is the summed width
    of the distinguishable intervals over the total swept width, so a flat region in the middle
    costs exactly its own width and no more.
    """
    t = np.asarray(true_vals, dtype=float)
    m = np.asarray(means, dtype=float)
    s = np.asarray(sems, dtype=float)
    order = np.argsort(t)
    t, m, s = t[order], m[order], s[order]
    total = float(t[-1] - t[0])
    if total <= 0:
        return float("nan"), []
    usable = 0.0
    flags = []
    for i in range(len(t) - 1):
        pooled = float(np.sqrt(np.nan_to_num(s[i]) ** 2 + np.nan_to_num(s[i + 1]) ** 2))
        gap = abs(float(m[i + 1] - m[i]))
        # A POOLED STANDARD ERROR OF ZERO IS THE BEST CASE, NOT AN EXCLUSION. Requiring
        # `pooled > 0` scored a NOISELESSLY recovered step as indistinguishable, which is exactly
        # backwards: two means that differ with no uncertainty at all are as distinguishable as two
        # numbers can be. P-1 caught this on the value gate, whose estimator is exact over half its
        # range and was being classified unidentifiable for being too precise. Both branches are
        # written out rather than folded into one expression, because the one-expression version is
        # what hid it.
        if pooled > 0:
            ok = bool(gap > k * pooled)
        else:
            ok = bool(gap > 0)
        flags.append({"from": float(t[i]), "to": float(t[i + 1]),
                      "gap": gap, "pooled_se": pooled, "distinguishable": ok})
        if ok:
            usable += float(t[i + 1] - t[i])
    return usable / total, flags


def classify_recovery(rho: float, slope: float, usable: float) -> str:
    """P-1's four-way classification, applied mechanically and in the spec's own order."""
    if not np.isfinite(rho) or rho < P1_UNIDENTIFIABLE_RHO or \
            (np.isfinite(usable) and usable < P1_PARTIAL_USABLE[0]):
        return "UNIDENTIFIABLE"
    if rho >= P1_RECOVERED_RHO and np.isfinite(slope) and slope < P1_COMPRESSED_SLOPE:
        return "COMPRESSED"
    if np.isfinite(usable) and P1_PARTIAL_USABLE[0] <= usable < P1_PARTIAL_USABLE[1]:
        return "PARTIALLY_RECOVERED"
    if rho >= P1_RECOVERED_RHO and np.isfinite(slope) \
            and P1_RECOVERED_SLOPE[0] <= slope <= P1_RECOVERED_SLOPE[1] \
            and np.isfinite(usable) and usable >= P1_RECOVERED_USABLE:
        return "RECOVERED"
    # Everything left over: the ordering survives and something else does not. Named rather than
    # forced into one of the four, because a silent reclassification is how a bad parameter passes.
    return "PARTIALLY_RECOVERED"
