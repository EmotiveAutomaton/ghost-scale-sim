"""Pre-registered criteria for the repair pass, hash-locked before any recomputation runs.

A THIRD LOCK, separate from the validation and diagnostics locks, both of which have reported and
are sealed.

The constraint that shapes most of what follows, from the specification's section 8: **every original
number is retained and reported beside its replacement.** No committed value is silently superseded.
That applies without exception to the recomputed criteria, the disagreement figures, the bias
correction, the uptake decomposition and the seed change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import Config

# --------------------------------------------------------------------------- #
# R-1 / R-3 — recomputation and intervals.
# --------------------------------------------------------------------------- #
BOOTSTRAP_DRAWS = 4000          # the specification asks for at least 2000
BOOTSTRAP_ALPHA = 0.05          # 95% percentile intervals throughout

# A criterion is DETERMINED when its bootstrap interval excludes its own pre-registered threshold,
# in either direction. That is the whole point of the exercise: a criterion that was previously
# undetermined becomes determined by having an interval at all, and it can become determined
# AGAINST the claim, which is a live outcome.
#
# The interval is computed on the ESTIMAND THE CRITERION WAS WRITTEN OVER, which for the rank
# correlations means resampling readers and re-deriving the cell means, not pooling readers into one
# big correlation. Pooling changes the question; see the package docstring.
DETERMINED_EXCLUDES_THRESHOLD = True

# R-3's headline item. The fabrication peak's LOCATION is what every downstream claim is anchored
# to, including the prediction card, and it has never had an error bar. The bootstrap distribution
# of the argmax is reported as a distribution over grid points, and the peak counts as LOCATED when
# one grid point holds at least this share of the draws.
PEAK_LOCATED_SHARE = 0.80

# --------------------------------------------------------------------------- #
# R-5 — the uptake decomposition.
# --------------------------------------------------------------------------- #
# Movement and error reduction disagree about an experiment when they rank its conditions
# differently. That disagreement is itself a finding and is reported rather than resolved.
UPTAKE_DISAGREEMENT_IS_A_FINDING = True

# --------------------------------------------------------------------------- #
# R-6 / R-8 — estimators and the identifiability map.
# --------------------------------------------------------------------------- #
# Both observability standards are run and reported as a bracket. Neither is primary in the code;
# the narrow one is primary in the PROSE, because it is the conservative standard and it is what an
# experiment on a person could collect.
OBSERVABILITY = ("narrow", "wide")
OBSERVABILITY_PRIMARY = "narrow"

# A parameter is IDENTIFIABLE OVER A RANGE when adjacent true values inside that range produce
# estimates separated by more than this many pooled standard errors. Same constant the diagnostics
# pass used, so the two are commensurable.
IDENTIFIABLE_SE = 2.0
# And the map reports the FRACTION of the swept range that is identifiable, rather than a verdict.
# These thresholds only bucket the fraction for reading; the fraction itself is the result.
MAP_FULL = 0.90
MAP_PARTIAL = 0.40

# A SLOPE FLOOR, ADDED DURING THE PASS AND DECLARED AS AN ADDITION RATHER THAN APPLIED QUIETLY.
#
# The fraction above is a purely STATISTICAL criterion: adjacent values count as distinguishable
# when their gap beats the noise. That is necessary and it is not sufficient, and the learned-trust
# check found the gap. A reader whose belief moved by 0.005 across a true range of 0.8 was scored as
# identifiable across 100% of the range, because its readers were near-deterministic so the noise
# was smaller still. Statistically separable, and it had recovered 0.7% of the signal.
#
# So identifiability now needs a signal term as well: the estimate must track at least this fraction
# of the true variation. The value is the diagnostics pass's own COMPRESSED boundary halved, on the
# reasoning that below a fifth the estimate is not carrying the quantity in any useful sense.
#
# THIS IS THE THIRD TIME A DISTINGUISHABILITY RULE IN THIS PROJECT HAS BEEN WRONG, and the three
# failures are the same shape from different sides: once it excluded a noiseless estimator for
# having no noise, once it scored a constant estimator as perfect, and now it has admitted a flat
# one for being quiet. The fraction-only rule is retained and still reported beside the new one.
MAP_SLOPE_FLOOR = 0.20

# --------------------------------------------------------------------------- #
# R-7 — the seed replacement.
# --------------------------------------------------------------------------- #
# The old function stays in the codebase and stays selectable, so the historical record remains
# independently regenerable. The new one becomes the default. A headline quantity counts as
# UNAFFECTED by the change when the two runs agree within this many across-seed standard errors.
SEED_CHANGE_TOLERANCE_SE = 2.0

# --------------------------------------------------------------------------- #
# R-12 — the exact-inference sweep.
# --------------------------------------------------------------------------- #
# A verdict MOVED when its outcome string changes. A quantity moved when it changes by more than
# this many of its own across-seed standard errors. Both are reported per experiment.
EXACT_SWEEP_TOLERANCE_SE = 2.0

# --------------------------------------------------------------------------- #
# R-13 — the reruns.
# --------------------------------------------------------------------------- #
# The reruns use ERROR REDUCTION as the primary outcome, which is a change of primary from the
# original pre-registration and is declared here before they run. Movement is retained and still
# computed. The original criterion is also still computed, and if the two disagree that is reported
# as the finding rather than resolved in favour of either.
RERUN_PRIMARY = "error_reduction"
RERUN_RETAINS_ORIGINAL = True
# The regime the difficulty probe located, and the constraint the uptake curve imposes on it.
RERUN_INEXPERTISE = 0.85
RERUN_OBSERVATIONS = 6
RERUN_TROUGH = 0.90


def _canonical(payload: dict) -> str:
    trimmed = {k: v for k, v in payload.items() if k != "content_hash"}
    return json.dumps(trimmed, sort_keys=True, separators=(",", ":"), default=str)


def criteria_payload(cfg: Config) -> dict:
    return {
        "document": "Ghost Scale Simulation — repair pass criteria",
        "spec": "docs/specs/SPEC_REPAIR.md",
        "written_before": "any repair recomputation ran",
        "separate_from": ["results/validation/criteria.json", "results/diagnostics/criteria.json"],
        "governing_rule": ("every change either makes something measurable that was not, or removes "
                           "something. An addition qualifies only if it converts a free choice into "
                           "a measured quantity."),
        "corrections_to_the_specification": [
            "The error-reduction formula is undefined as written: compared against a point mass on "
            "the truth every term diverges, and floored at an epsilon it returns a number that "
            "tracks the epsilon. The arguments are reversed, giving the reduction in the surprisal "
            "of the truth.",
            "Recomputing the rank correlations over per-reader pairs answers a different question "
            "and answers it worse, because within cells the two quantities are uncorrelated so "
            "pooling attenuates. The estimand is preserved by bootstrapping the cell means. Both "
            "are computed and both are reported, labelled.",
            "Trust is not structurally unidentifiable. It was fitted to the observation tape, which "
            "the world generates and which contains no trust parameter. Fitted to the reader's own "
            "behaviour it recovers over part of its range, and saturates above the point where the "
            "label already beats the content outright.",
        ],
        "r1_r3": {
            "bootstrap_draws": BOOTSTRAP_DRAWS, "alpha": BOOTSTRAP_ALPHA,
            "determined_excludes_threshold": DETERMINED_EXCLUDES_THRESHOLD,
            "peak_located_share": PEAK_LOCATED_SHARE,
        },
        "r5": {"disagreement_is_a_finding": UPTAKE_DISAGREEMENT_IS_A_FINDING,
               "quantities": ["movement", "error_reduction", "trust_factor"]},
        "r6_r8": {"observability": list(OBSERVABILITY), "primary": OBSERVABILITY_PRIMARY,
                  "identifiable_se": IDENTIFIABLE_SE,
                  "map_full": MAP_FULL, "map_partial": MAP_PARTIAL,
                  "map_slope_floor": MAP_SLOPE_FLOOR,
                  "slope_floor_added_during_the_pass": (
                      "the statistical criterion alone admitted an estimate that tracked 0.7% of "
                      "the true variation, because its readers were near-deterministic so the noise "
                      "was smaller still. A signal term was added. The fraction-only rule is "
                      "retained and still reported beside it."),
                  "note": ("both standards are run and the pair is reported as a bracket. Narrow is "
                           "a floor on what is measurable and wide is a ceiling, and any real "
                           "reporting process lies between them.")},
        "r7": {"tolerance_se": SEED_CHANGE_TOLERANCE_SE,
               "old_function_retained": True,
               "note": ("the collision-free function becomes the default and the old one stays "
                        "selectable, so every committed number remains independently regenerable "
                        "by the code that produced it")},
        "r12": {"tolerance_se": EXACT_SWEEP_TOLERANCE_SE},
        "r13": {"primary": RERUN_PRIMARY, "retains_original": RERUN_RETAINS_ORIGINAL,
                "inexpertise": RERUN_INEXPERTISE, "observations": RERUN_OBSERVATIONS,
                "uptake_trough": RERUN_TROUGH,
                "note": ("changing the primary outcome from movement to error reduction is a change "
                         "of primary from the original pre-registration, declared here before the "
                         "rerun. The original is retained and still computed.")},
        "constraints": [
            "Every original number is retained and reported beside its replacement.",
            "The withheld experiment stays withheld, its failing test stays in the suite, and the "
            "open residual stays open.",
            "No claim is upgraded on the strength of a repair alone. A criterion recomputed at "
            "higher power reports what it reports.",
            "REPAIR.md is written from the verdict files afterward, never from these expectations.",
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
            "The pre-registered repair criteria are not trustworthy; the pass will not run.")
    return payload


# --------------------------------------------------------------------------- #
# Shared scoring.
# --------------------------------------------------------------------------- #
def percentile_interval(samples, alpha: float = BOOTSTRAP_ALPHA) -> tuple:
    a = np.asarray(samples, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(a, 100 * alpha / 2)),
            float(np.percentile(a, 100 * (1 - alpha / 2))))


def determined_against(interval: tuple, threshold: float) -> str:
    """Does the interval settle the criterion, and which way?"""
    lo, hi = interval
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undetermined"
    if lo > threshold:
        return "determined_meets"
    if hi < threshold:
        return "determined_fails"
    return "undetermined"


def identifiable_fraction(true_vals, means, sems, k: float = IDENTIFIABLE_SE) -> tuple:
    """The fraction of a swept range over which adjacent values are distinguishable.

    A pooled standard error of ZERO with a nonzero gap is the most distinguishable case there is,
    not an excluded one. The diagnostics pass shipped that bug and classified an exact estimator as
    unidentifiable for being too precise; both branches are written out here for that reason.
    """
    t = np.asarray(true_vals, dtype=float)
    m = np.asarray(means, dtype=float)
    s = np.asarray(sems, dtype=float)
    order = np.argsort(t)
    t, m, s = t[order], m[order], s[order]
    total = float(t[-1] - t[0])
    if total <= 0:
        return float("nan"), []
    usable, flags = 0.0, []
    for i in range(len(t) - 1):
        pooled = float(np.sqrt(np.nan_to_num(s[i]) ** 2 + np.nan_to_num(s[i + 1]) ** 2))
        gap = abs(float(m[i + 1] - m[i]))
        ok = bool(gap > k * pooled) if pooled > 0 else bool(gap > 0)
        flags.append({"from": float(t[i]), "to": float(t[i + 1]), "gap": gap,
                      "pooled_se": pooled, "distinguishable": ok})
        if ok:
            usable += float(t[i + 1] - t[i])
    return usable / total, flags


def map_label(fraction: float, slope: float | None = None) -> str:
    """How to read an identifiability fraction, with the slope floor applied when a slope is given.

    Both terms are needed. The fraction says the estimate is separable from noise; the slope says it
    is separable from nothing. An estimate can pass the first and fail the second by being quiet
    rather than by being right, which is what the learned-trust check caught.
    """
    if not np.isfinite(fraction):
        return "not assessable"
    if slope is not None and np.isfinite(slope) and abs(slope) < MAP_SLOPE_FLOOR:
        return "statistically separable but flat: it tracks almost none of the true variation"
    if fraction >= MAP_FULL:
        return "measurable across the range"
    if fraction >= MAP_PARTIAL:
        return "measurable over part of the range"
    return "a free choice, not a measured quantity"
