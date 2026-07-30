"""Pre-registered criteria for the validation pass, hash-locked before any check runs.

The constants and the outcome branches in this file are the criteria. They are written here
rather than inside the code that scores them for one reason: the spec requires that every check
be able to return the unwelcome answer, and a threshold that lives next to its own measurement
can be nudged after the fact without anybody, including the person doing it, noticing.

WHERE EACH NUMBER COMES FROM. Two of them are inherited from the committed record and two are
new. The inherited ones are quoted with their source. The new ones are stated with the reasoning
that fixed them, so a reader can disagree with the reasoning rather than guess at it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import Config

# --------------------------------------------------------------------------- #
# V-1 — does the inference approximation distort the headline results?
# --------------------------------------------------------------------------- #
# The spec's own pass condition, in as many words: "peak location within one grid step; primary
# outcomes within their existing run-to-run spread."
#
# ONE GRID STEP is taken literally against E20's committed omega grid, so the tolerance is a
# property of the design rather than of the result.
V1_PEAK_TOLERANCE_STEPS = 1
# "Within their existing run-to-run spread" is operationalised as: the exact-solver cell mean
# must lie within this many across-seed standard errors of the approximate-solver cell mean, at
# the SAME reduced scale, so the comparison is not confounded by the scale reduction. Two
# standard errors is the conventional reading of "within its own spread" and is fixed here
# rather than chosen from the measurement.
V1_SPREAD_SE = 2.0
# A verdict is a verdict: any check whose primary outcome is a boolean (crash signature fires,
# a gate dissociates, an ordering holds) must return the same boolean under both solvers. There
# is no tolerance on a boolean and there should not be one.

# --------------------------------------------------------------------------- #
# V-2 — would a model of this shape produce these results anyway?
# --------------------------------------------------------------------------- #
# (a) SCRAMBLED PROVENANCE. Permuting the provenance-to-content mapping must make every effect
# vanish. "Vanish" is defined against the effect's own size rather than as a bare number: the
# scrambled effect must be under this fraction of the intact one. A tenth is the same standard
# the existing null suite applies (N4, the alpha-permutation control).
V2A_RESIDUAL_FRACTION = 0.10
# (b) RANDOM-PARAMETER FALSE-POSITIVE RATE. Draw the likelihood structure at random subject only
# to the format constraints, run the headline comparison unchanged, and count how often a
# randomly parameterised model of this architecture returns something that would have been
# reported as a finding.
#
# THE THRESHOLD IS COMMITTED HERE, BEFORE THE RUN, as the spec requires. A headline effect must
# fall outside the central 95% of the random distribution — that is, at or beyond the 97.5th
# percentile of random draws in the direction the finding claims. 5% two-sided is the ordinary
# convention and it is being used because it is ordinary, not because it flatters anything.
V2B_PERCENTILE = 97.5
V2B_ALPHA = 0.05
# Below this many usable draws the rate is not reported as a rate. A false-positive rate quoted
# from a handful of draws is worse than no rate at all.
V2B_MIN_DRAWS = 100

# --------------------------------------------------------------------------- #
# V-3 — knife-edge or robust?
# --------------------------------------------------------------------------- #
# Not a pass/fail. Each cell is HOLDS, WEAKENS or FLIPS, and the three are defined on the
# verdict's own primary quantity:
V3_WEAKEN_FRACTION = 0.50   # keeps its sign and direction but drops below half its headline size
# Plus the scoping judgement the spec asks for, which is the part that changes wording in the
# README: a result that survives only in a window narrower than this fraction of the swept range
# is reported as TUNED, whatever its verdict says.
V3_TUNED_WINDOW_FRACTION = 0.40

# --------------------------------------------------------------------------- #
# V-7 — seed and scale independence.
# --------------------------------------------------------------------------- #
# Effect sizes, not just verdicts. An effect that moves by more than this fraction of itself
# when the scale doubles is reported as under-powered, whether or not its verdict changed.
V7_SCALE_DRIFT_FRACTION = 0.25

# --------------------------------------------------------------------------- #
# V-8 — independent reimplementation.
# --------------------------------------------------------------------------- #
# The two-gates result is "same artifact, one word in the label, N-fold difference in uptake".
# A reimplementation from the prose alone cannot be expected to reproduce a multiple to two
# decimal places — it shares no code, no parameters and no random stream. What it can be
# expected to reproduce is the DIRECTION and the ORDER OF MAGNITUDE, and those are what is
# pre-registered. Anything stricter would be a test of whether two people chose the same
# arbitrary constants.
V8_DIRECTION_REQUIRED = True
V8_ORDER_OF_MAGNITUDE_FACTOR = 10.0   # reimplemented multiple within 10x of the original


def _canonical(payload: dict) -> str:
    trimmed = {k: v for k, v in payload.items() if k != "content_hash"}
    return json.dumps(trimmed, sort_keys=True, separators=(",", ":"), default=float)


def criteria_payload(cfg: Config) -> dict:
    """Everything that could decide a validation verdict, in one hashable object."""
    return {
        "document": "Ghost Scale Simulation — validation pass criteria",
        "spec": "docs/specs/SPEC_VALIDATION.md",
        "written_before": "any validation check ran",
        "v1": {
            "peak_tolerance_steps": V1_PEAK_TOLERANCE_STEPS,
            "spread_standard_errors": V1_SPREAD_SE,
            "boolean_outcomes_must_match": True,
            "targets": [
                "E20 — the collapse-and-invention sweep across the readability axis",
                "E19 — sustained attention on unreadable content",
                "E31 — the two-gates result",
                "E32 — the silent-versus-loud failure comparison",
                "E2  — the label-induced disagreement result",
            ],
        },
        "v2": {
            "scrambled_residual_fraction": V2A_RESIDUAL_FRACTION,
            "random_model_percentile": V2B_PERCENTILE,
            "random_model_alpha": V2B_ALPHA,
            "random_model_min_draws": V2B_MIN_DRAWS,
        },
        "v3": {
            "weaken_fraction": V3_WEAKEN_FRACTION,
            "tuned_window_fraction": V3_TUNED_WINDOW_FRACTION,
        },
        "v7": {"scale_drift_fraction": V7_SCALE_DRIFT_FRACTION},
        "v8": {
            "direction_required": V8_DIRECTION_REQUIRED,
            "order_of_magnitude_factor": V8_ORDER_OF_MAGNITUDE_FACTOR,
        },
        "scale": {
            "n_observers": int(cfg.get("validation.n_observers", 60)),
            "n_seeds": int(cfg.get("validation.n_seeds", 12)),
            "random_model_draws": int(cfg.get("validation.random_model_draws", 300)),
            "seed_block_offset": int(cfg.get("validation.seed_block_offset", 700003)),
            "note": ("Reduced from the headline runs, as the spec permits, and recorded beside "
                     "every number. The question is whether a verdict SURVIVES a change of "
                     "solver or parameter, which needs enough observers to resolve the effect "
                     "rather than the precision the headline number was quoted at."),
        },
    }


def write_criteria(cfg: Config, path: Path) -> dict:
    payload = criteria_payload(cfg)
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


def ensure_criteria(cfg: Config, path: Path) -> dict:
    """Write the criteria if absent, then assert they have not moved since."""
    path = Path(path)
    if not path.exists():
        write_criteria(cfg, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_hash")
    recomputed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if stated != recomputed:
        raise RuntimeError(
            f"{path.name} has been modified since it was written ({stated} != {recomputed}). "
            "The pre-registered validation criteria are not trustworthy; the pass will not run.")
    return payload


# --------------------------------------------------------------------------- #
# Scoring helpers used by more than one check.
# --------------------------------------------------------------------------- #
def within_spread(a: float, b: float, sem: float, k: float = V1_SPREAD_SE) -> bool:
    """Is ``a`` within ``k`` standard errors of ``b``? NaN sem means undecidable, not pass."""
    if not np.isfinite(sem) or sem <= 0:
        return bool(np.isclose(a, b, rtol=1e-6, atol=1e-9))
    return bool(abs(float(a) - float(b)) <= k * float(sem))


def robustness_label(headline: float, swept: float,
                     weaken_fraction: float = V3_WEAKEN_FRACTION) -> str:
    """HOLDS / WEAKENS / FLIPS for one cell of the V-3 matrix."""
    h, s = float(headline), float(swept)
    if not np.isfinite(s):
        return "undefined"
    if h == 0.0:
        return "holds" if s == 0.0 else "flips"
    if np.sign(s) != np.sign(h):
        return "flips"
    return "holds" if abs(s) >= weaken_fraction * abs(h) else "weakens"
