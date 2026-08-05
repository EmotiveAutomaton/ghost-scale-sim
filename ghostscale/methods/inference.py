"""Two things this repository's verdicts need and do not have: FDR control, and bounded nulls.

MULTIPLICITY. Batch two reports several hundred bootstrap intervals -- six edges times seven cells
times several fidelities, three negative controls each, sixteen detection cells times eighteen
features. Nothing is corrected. At the conventional level that is dozens of expected false
positives by construction, and the two smallest live edges in T-1 are exactly where that matters.

Benjamini-Hochberg is the right correction for this and Bonferroni is not. Bonferroni controls the
probability of ANY false claim, which is the correct target when a single claim carries a decision.
BH controls the expected PROPORTION of false claims among those made, which is the correct target
when you are deliberately scouring a space and expect most of what you look at to be null. The
second is what this project is doing now.

EQUIVALENCE. Half of what batch two found is nulls -- three dead edges in T-1, the depth axis in
T-2, the whole of T-3. "The interval covers zero" is a statement about failing to detect something.
"The effect is bounded below 0.02 with 95% confidence" is a statement about the effect. The second
is a claim; the first is the absence of one, and this repository's best results deserve the first.

TOST is the frequentist form and needs a bound to test against. Choosing that bound is the hard
part and it should not be automated: the honest default here is a fraction of a live effect
measured on the same axis in the same run, so ``smallest_effect_of_interest`` takes it explicitly
and records where it came from.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def available() -> tuple:
    try:
        import statsmodels                               # noqa: F401
        return True, ""
    except Exception as exc:                             # noqa: BLE001
        return False, f"statsmodels not installed ({type(exc).__name__})"


# --------------------------------------------------------------------------------------------- #
# Multiplicity
# --------------------------------------------------------------------------------------------- #
def bootstrap_p_from_interval(difference: float, interval, alpha: float = 0.05) -> float:
    """A two-sided p-value implied by a bootstrap interval, for correction purposes only.

    THIS IS AN APPROXIMATION AND IS LABELLED AS ONE. The verdicts in this repository store an
    interval, not a p-value, and BH needs p-values. Inverting a percentile interval to a normal
    p is exact only if the bootstrap distribution is symmetric, which it is not everywhere. It is
    good enough to RANK results for FDR, which is all BH uses it for, and it must not be reported
    as a p-value in its own right.
    """
    lo, hi = float(interval[0]), float(interval[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float("nan")
    from scipy.stats import norm
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    se = (hi - lo) / (2.0 * z_crit)
    if se <= _EPS:
        return 0.0 if abs(difference) > _EPS else 1.0
    return float(2.0 * (1.0 - norm.cdf(abs(float(difference)) / se)))


def control_fdr(entries: dict, alpha: float = 0.05, method: str = "fdr_bh") -> dict:
    """Apply Benjamini-Hochberg across a family of ``{name: {difference, interval}}`` results.

    ``entries`` is exactly the shape ``boot_paired`` and ``boot_diff`` already produce, so this
    can be pointed at an existing verdict block without reshaping anything.
    """
    ok, reason = available()
    names, ps = [], []
    for name, e in entries.items():
        if not isinstance(e, dict) or "difference" not in e or "interval" not in e:
            continue
        p = bootstrap_p_from_interval(e["difference"], e["interval"], alpha=alpha)
        if np.isfinite(p):
            names.append(name)
            ps.append(p)
    if not names:
        return {"skipped": "no correctable entries found"}
    if not ok:
        return {"skipped": reason, "n_entries": len(names)}
    from statsmodels.stats.multitest import multipletests
    reject, p_adj, _, _ = multipletests(ps, alpha=alpha, method=method)
    n_raw = int(sum(p <= alpha for p in ps))
    return {
        "method": method, "alpha": float(alpha), "n_tests": len(names),
        "n_significant_uncorrected": n_raw,
        "n_significant_corrected": int(reject.sum()),
        "n_lost_to_correction": int(n_raw - reject.sum()),
        "per_entry": {n: {"p_approx": float(p), "p_adjusted": float(pa),
                          "survives_correction": bool(r)}
                      for n, p, pa, r in zip(names, ps, p_adj, reject)},
        "how_to_read": (
            "p values are inverted from the stored bootstrap intervals under a normal "
            "approximation, which is exact only for a symmetric bootstrap distribution. That is "
            "adequate for RANKING under Benjamini-Hochberg, which is all it is used for here, and "
            "these numbers must not be quoted as p-values in their own right. "
            "n_lost_to_correction is the number to look at: it is how many claims the family size "
            "was buying."),
    }


# --------------------------------------------------------------------------------------------- #
# Equivalence
# --------------------------------------------------------------------------------------------- #
def equivalence(sample, bound: float, bound_source: str, alpha: float = 0.05) -> dict:
    """TOST: is this effect bounded inside [-bound, +bound]?

    A pass means "the effect is smaller than ``bound``", which is a positive claim about a null.
    A fail means the data cannot rule out an effect that large -- NOT that an effect exists.

    ``bound_source`` is required and is recorded verbatim. An equivalence bound pulled out of the
    air is worse than no equivalence test, because it converts an arbitrary choice into an
    authoritative-looking verdict, and a reader cannot audit a number whose provenance is missing.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return {"skipped": f"need at least 3 finite observations, got {x.size}"}
    ok, reason = available()
    if not ok:
        return {"skipped": reason}
    from statsmodels.stats.weightstats import ttost_1samp
    p, lower, upper = ttost_1samp(x, -abs(float(bound)), abs(float(bound)))
    return {
        "mean": float(x.mean()), "n": int(x.size),
        "bound": float(abs(bound)), "bound_source": str(bound_source),
        "p_tost": float(p),
        "p_lower": float(lower[1]), "p_upper": float(upper[1]),
        "equivalent_at_alpha": bool(p < alpha), "alpha": float(alpha),
        "how_to_read": (
            "equivalent = the effect is bounded inside +/- bound, which is a claim about the "
            "effect rather than a failure to detect one. NOT equivalent does not mean an effect "
            "exists; it means the data cannot rule out one that large. The bound is a judgement "
            "and bound_source records whose."),
    }


def equivalence_from_interval(difference: float, interval, bound: float, bound_source: str,
                              interval_level: float = 0.95) -> dict:
    """TOST by interval inclusion, so a null can be bounded from a COMMITTED VERDICT alone.

    A confidence interval lying entirely inside ``[-bound, +bound]`` is equivalence at the same
    alpha the interval was built at -- that is the standard CI form of the two one-sided tests, and
    it needs the interval rather than the raw sample. Every verdict in this repository stores an
    interval; almost none of them commits the per-rollout data the sample-based test would need,
    because the per-rollout files are gitignored on purpose. So this is the form that actually
    works on what is on disk, on a fresh clone, forever.

    It is CONSERVATIVE at a 95% interval: the exact correspondence is with a 90% interval, so a
    95% one demands more before it will call something equivalent. Being harder to pass in the
    direction of "we cannot bound this" is the right way round for a repository whose nulls are
    load-bearing.

    A pass means: *the effect is smaller than* ``bound``. A fail means the data cannot rule out an
    effect that large -- NOT that an effect exists.
    """
    lo, hi = float(interval[0]), float(interval[1])
    b = abs(float(bound))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return {"skipped": "interval is not finite"}
    inside = bool(lo >= -b and hi <= b)
    return {
        "difference": float(difference), "interval": [lo, hi],
        "bound": b, "bound_source": str(bound_source),
        "interval_level": float(interval_level),
        "equivalent": inside,
        "widest_excursion": float(max(abs(lo), abs(hi))),
        "headroom": float(b - max(abs(lo), abs(hi))),
        "how_to_read": (
            "equivalent = the whole interval lies inside +/- bound, so the effect is bounded "
            "below it. This is TOST in confidence-interval form and it is conservative at a 95% "
            "interval (the exact correspondence is with 90%). NOT equivalent does not mean an "
            "effect exists; it means the data cannot rule out one of that size. headroom is how "
            "much room was left -- a small positive headroom is a bound that only just held."),
    }


def smallest_effect_of_interest(reference_effect: float, fraction: float = 0.10,
                                label: str = "") -> tuple:
    """A defensible equivalence bound: a fraction of a live effect measured on the same axis.

    Returns ``(bound, source_string)`` so the provenance travels with the number. Using a live
    effect from the SAME run and the SAME units is the only construction here that does not
    import an outside convention -- there is no established smallest-effect-of-interest for
    process error reduction in nats, and inventing one would be exactly the arbitrary choice the
    docstring above warns about.
    """
    b = abs(float(reference_effect)) * float(fraction)
    src = (f"{fraction:.0%} of {label or 'a reference effect'} "
           f"({reference_effect:+.4f}) measured on the same axis in the same run")
    return b, src
