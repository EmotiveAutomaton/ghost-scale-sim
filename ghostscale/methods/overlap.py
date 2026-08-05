"""How far a result sits from its own null, as a number rather than a judgement.

THE MOTIVATING CASE IS IN THIS REPOSITORY. T-1's ``goal->process`` edge at mu3/beta1.0 returns
+0.0017 with an interval of [+0.0011, +0.0023], so ``excludes_zero`` is ``true``. It is also
plainly inside the noise: the same edge is -0.0012 one cell over, the budget-matched version of it
flickers between excluding and covering zero across duty cycles, and the effect is 0.3% of the
process->depth edge measured on the same scale. A reader has to be told all of that in prose to
know that a ``true`` flag means nothing here.

``excludes_zero`` answers "is the interval on one side of zero". That is a question about
PRECISION, and with n = 200 paired rollouts almost anything separates from zero. The question
worth asking is "does the effect distribution separate from the distribution the harness produces
when there is nothing to find" -- which is a question about SEPARATION, and needs a null built by
destroying the signal rather than by assuming it is zero.

So, following the sanity-checks-for-agentic-data-science framing:

  overlap coefficient   the shared area of the effect and null densities, in [0, 1]. 1.0 is total
                        overlap and 0.0 is complete separation. Threshold-free and unitless, so a
                        process edge measured in nats and a detection edge measured in AUC are
                        directly comparable -- which no interval in this repository currently is.
  bootstrap p           the fraction of the null distribution at least as extreme as the observed
                        effect. Reported alongside, because overlap alone does not carry a sign.

WHERE THE NULL COMES FROM MATTERS MORE THAN THE STATISTIC. A null built by assuming zero is not a
null, it is an assumption. The three honest constructions in this repository, in ascending order
of strength, are: a placebo arm (a manipulation at zero strength), a permuted-label arm, and a
swap arm (another artifact's true values). ``null_from_arms`` takes whichever you have and says
which one it used, so a strong claim can never be quoted off a weak null by accident.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12

#: Above this, the effect and null distributions are not usefully distinguishable. Taken from the
#: sanity-checks paper's tau = 0.2; kept as a module constant so it can be argued with in one
#: place rather than hard-coded at each call site.
TAU = 0.20


def overlap_coefficient(a, b, bins: int = 64) -> float:
    """Shared area of two empirical densities, in [0, 1].

    Histogram-based on a shared support, which is the estimator the sanity-checks framing uses.
    A kernel estimate would be smoother and would also introduce a bandwidth to argue about; the
    bin count is reported so the number is reproducible.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi - lo < _EPS:
        return 1.0                       # both degenerate at the same point: total overlap
    edges = np.linspace(lo, hi, int(bins) + 1)
    pa, _ = np.histogram(a, bins=edges, density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    return float(np.minimum(pa, pb).sum())


def separation(effect, null, bins: int = 64, tau: float = TAU, draws: int = 2000,
               seed: int = 20260805) -> dict:
    """Score one effect sample against one null sample.

    THE OVERLAP IS COMPUTED ON THE BOOTSTRAP DISTRIBUTIONS OF THE MEAN, NOT ON THE PER-UNIT
    VALUES, and getting this wrong is the first thing this module did.

    The per-unit version was calibrated against nothing and it fails an obvious check: T-1's
    ``process->depth`` edge is the largest in the study at +0.5147 with an interval nowhere near
    zero, and per-unit it scored overlap 0.335 -- "not separated" at tau = 0.2. That is not a
    finding about the edge, it is a statement that individual artifacts vary by more than the mean
    effect, which is true of almost every result in this repository and says nothing about whether
    the effect is real.

    The question this is meant to answer is the sanity-checks paper's: did the STATISTIC affirm
    the research question distinguishably from noise. The statistic is the mean, so the two
    distributions to compare are its sampling distribution under the data and under the null. On
    those, tau = 0.2 is the intended scale and a real effect separates cleanly.

    The per-unit standardised effect is still reported, as ``standardised_effect``, because it is
    the right number for "how big" as opposed to "how sure" -- and confusing those two is what the
    first version did.
    """
    effect = np.asarray(effect, dtype=float)
    null = np.asarray(null, dtype=float)
    e = effect[np.isfinite(effect)]
    n = null[np.isfinite(null)]
    if e.size < 2 or n.size < 2:
        return {"overlap": float("nan"), "separated": False,
                "reason": "need at least 2 finite values in each of effect and null"}
    rng = np.random.default_rng(int(seed))
    be = np.array([rng.choice(e, e.size, replace=True).mean() for _ in range(int(draws))])
    bn = np.array([rng.choice(n, n.size, replace=True).mean() for _ in range(int(draws))])
    ov = overlap_coefficient(be, bn, bins=bins)
    obs, centre = float(e.mean()), float(n.mean())
    null_sd = float(n.std(ddof=1))
    # How often a bootstrap draw of the null's mean is at least as extreme as the observed mean,
    # measured from the null's own centre. Uses the null's spread, which cannot flatter the result.
    p = float((np.abs(bn - centre) >= abs(obs - centre)).mean())
    return {
        "overlap": ov,
        "tau": float(tau),
        # NAMED FOR WHAT IT ACTUALLY TESTS. The scrambled nulls -- random values, or another
        # artifact's trajectory -- do not sit at zero: a channel that lies confidently HURTS, so
        # the null mean is well below zero. Separating from it therefore means "this effect needs
        # the channel to correspond to the truth", which is not the same claim as "this effect is
        # nonzero", and calling the field `separated` invited exactly that conflation. T-1's
        # goal->process edge at -0.0012 separates cleanly from a swap null at -0.0514 while being
        # a negative effect.
        "separated_from_null": bool(ov <= tau),
        "effect_exceeds_null": bool(obs > centre),
        "supports_a_positive_edge": bool(ov <= tau and obs > centre and obs > 0.0),
        "effect_mean": obs,
        "null_mean": centre,
        "null_sd": null_sd,
        "standardised_effect": (float((obs - centre) / null_sd) if null_sd > _EPS
                                else float("nan")),
        "per_unit_overlap": overlap_coefficient(e, n, bins=bins),
        "bootstrap_p_two_sided": p,
        "n_effect": int(e.size), "n_null": int(n.size), "bins": int(bins),
        "bootstrap_draws": int(draws),
        "how_to_read": (
            "THREE FIELDS, THREE DIFFERENT CLAIMS, and the point of separating them is that no "
            "single flag can carry all three. separated_from_null: the effect's bootstrapped mean "
            "is distinguishable from the scrambled-channel null, i.e. it needs the channel to "
            "correspond to the truth. effect_exceeds_null: it is on the helping side of that "
            "null. supports_a_positive_edge: both of those AND the effect is above zero -- the "
            "only one of the three that licenses 'this edge is alive'. "
            "None of them is a magnitude. standardised_effect is the magnitude, in null standard "
            "deviations, and an edge can be cleanly separated and still negligible: T-1's "
            "+0.0017 goal->process edge is real and is 0.3% of the process->depth edge on the "
            "same axis. per_unit_overlap is effect size against individual variation rather than "
            "confidence, and is high for almost everything here."),
    }


def relative_magnitude(entries: dict, key: str = "effect_mean") -> dict:
    """Each effect as a fraction of the largest in its family. The number that says "negligible".

    THIS IS THE PIECE NEITHER `excludes_zero` NOR THE OVERLAP CAN SUPPLY, and building the overlap
    made that obvious rather than fixing it.

    T-1's `goal->process` edge at mu3/beta1.0 is +0.0017. It excludes zero. Its bootstrapped mean
    also separates cleanly from every null, correctly, because it is a real effect. And its
    STANDARDISED effect is 1.15 -- larger than the +0.5147 `process->depth` edge's 0.81 -- because
    at beta = 1.0 the reader is saturated and the null's spread collapses, so dividing by it
    inflates everything in that cell. Every confidence-flavoured statistic in this module says the
    same thing about that edge: it is real.

    It is also 0.3% of the largest edge measured on the same axis, and that is the fact a reader
    needs. Confidence and magnitude are different questions and no single flag answers both; this
    answers the second, by comparing like with like inside one family rather than across cells
    whose variances are not comparable.
    """
    vals = {k: abs(float(v[key])) for k, v in entries.items()
            if isinstance(v, dict) and key in v and np.isfinite(v[key])}
    if not vals:
        return {"skipped": f"no finite {key} values"}
    biggest = max(vals.values())
    if biggest <= _EPS:
        return {"skipped": "every effect in the family is zero"}
    return {
        "largest_abs_effect": float(biggest),
        "largest_is": max(vals, key=vals.get),
        "fraction_of_largest": {k: float(v / biggest) for k, v in vals.items()},
        "negligible_below_5pc": sorted(k for k, v in vals.items() if v / biggest < 0.05),
        "how_to_read": (
            "magnitude relative to the biggest effect measured the same way in the same family. "
            "An edge can separate from every null, exclude zero, and still sit at 0.3% of the "
            "result next to it -- which is a fact about importance that no confidence statistic "
            "reports, and the reason `excludes_zero` was misleading in the first place."),
    }


def null_from_arms(placebo=None, permuted=None, swapped=None) -> tuple:
    """Pick the strongest available null and say which one it is.

    Strength order, weakest first:

      placebo   the manipulation at zero strength. Controls the harness, not the signal: it
                cannot tell a real effect from one the content would have produced anyway.
      permuted  the channel carrying random values through a likelihood that still claims
                fidelity. Controls the CONTENT of the signal.
      swapped   another artifact's true values, preserving marginal statistics and temporal
                structure and destroying only the correspondence to this artifact. Strongest,
                because everything except the thing being claimed is held fixed.

    Returned with its name so a claim can never be quoted against a weaker null than the caller
    believes it used.
    """
    for name, arm in (("swapped", swapped), ("permuted", permuted), ("placebo", placebo)):
        if arm is not None and len(np.asarray(arm, dtype=float)) > 0:
            return np.asarray(arm, dtype=float), name
    return np.asarray([], dtype=float), "none"


def score_against_best_null(effect, placebo=None, permuted=None, swapped=None,
                            bins: int = 64, tau: float = TAU) -> dict:
    """:func:`separation` against the strongest null supplied, with the null named in the output."""
    null, which = null_from_arms(placebo=placebo, permuted=permuted, swapped=swapped)
    if which == "none":
        return {"overlap": float("nan"), "separated": False, "null_used": "none",
                "reason": "no null arm supplied; separation cannot be scored"}
    out = separation(effect, null, bins=bins, tau=tau)
    out["null_used"] = which
    return out
