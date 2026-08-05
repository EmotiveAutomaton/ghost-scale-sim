"""Belief trajectories as time series, and the features nobody thought to compute.

WHY. T-5 asked which internal quantity of the reader best separates content-with-a-maker from
content-without-one, and the answer was neither the goal posterior nor the sub-goal posterior as
such -- it was how far the sub-goal posterior TRAVELS step to step, plus whether the reader is
allowed to disengage. Both are properties of the trajectory rather than of its endpoint. Every
instrument in this repository and in Sounding Line reads an endpoint.

That result came from about a dozen features I wrote down by hand, which is a poor way to search a
space. ``catch22`` is a canonical set distilled from roughly 7000 candidates in ``hctsa`` by
removing redundancy and ranking on classification performance, and a reader's per-step posterior
entropy is exactly the kind of series it was built for. This is the cheapest available route to an
exploratory measure nobody has proposed yet.

WHAT THIS DOES NOT DO. It does not select features against an outcome, and it must not be used to.
Extracting 22 features per series and reporting the best one is a multiple-comparisons problem
wearing a lab coat -- see ``methods.inference.control_fdr``, and note that a feature found this way
needs confirming on a fresh seed block before it is quoted. ``confirm_on_fresh_seeds`` in this
module is the protocol.

DEGRADES GRACEFULLY: without ``pycatch22`` this returns a recorded skip.
"""
from __future__ import annotations

import numpy as np


def available() -> tuple:
    try:
        import pycatch22                                 # noqa: F401
        return True, ""
    except Exception as exc:                             # noqa: BLE001
        return False, f"pycatch22 not installed ({type(exc).__name__})"


def features(series) -> dict:
    """The 22 canonical features of one series, as ``{name: value}``.

    A constant series is returned as a skip rather than as NaNs: several catch22 features are
    undefined on zero variance, and a block of silent NaNs downstream is worse than an explicit
    refusal. This matters here because a disengaged reader's posterior IS constant.
    """
    ok, reason = available()
    if not ok:
        return {"skipped": reason}
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return {"skipped": f"series too short ({x.size} points)"}
    if float(np.std(x)) < 1e-12:
        return {"skipped": "series is constant; most catch22 features are undefined"}
    import pycatch22
    r = pycatch22.catch22_all(x.tolist())
    return {n: float(v) for n, v in zip(r["names"], r["values"])}


def feature_table(series_by_id: dict) -> dict:
    """Features for many series. Returns ``{id: {feature: value}}`` plus a skip tally.

    The tally is the point of returning a dict rather than a frame: if a third of the trajectories
    were constant because the reader disengaged, that is a finding about engagement and it must
    not disappear into missing rows.
    """
    out, skipped = {}, {}
    for k, s in series_by_id.items():
        f = features(s)
        if "skipped" in f:
            skipped[k] = f["skipped"]
        else:
            out[k] = f
    return {"features": out, "n_ok": len(out), "n_skipped": len(skipped),
            "skip_reasons": skipped}


def confirm_on_fresh_seeds(discovery_fn, confirm_fn, feature_name: str) -> dict:
    """The protocol that makes exploratory feature search honest in a simulator.

    The standard answer to adaptive overfitting is a reusable holdout accessed through a
    differentially private mechanism. This project does not need one: it has a generator, so it
    has unlimited fresh data, and the strictly stronger move is to re-run the winner on a seed
    block that did not exist when the winner was chosen.

    This matters concretely rather than theoretically. T-2's difficulty control returned
    BREADTH_SEPARATES_DIVERSITY_FROM_DIFFICULTY at n = 40 and the opposite verdict at n = 200. The
    sign flipped. A feature picked out of 22 on one seed block is at least as fragile as that.

    ``discovery_fn`` and ``confirm_fn`` each return a scalar for ``feature_name``; the second must
    draw from seeds the first never saw.
    """
    d = float(discovery_fn(feature_name))
    c = float(confirm_fn(feature_name))
    same_sign = bool(np.sign(d) == np.sign(c)) if abs(d) > 0 and abs(c) > 0 else False
    shrink = (float(abs(c) / abs(d)) if abs(d) > 1e-12 else float("nan"))
    return {
        "feature": feature_name,
        "discovery_value": d, "confirmation_value": c,
        "same_sign": same_sign,
        "retained_fraction": shrink,
        "confirmed": bool(same_sign and np.isfinite(shrink) and shrink >= 0.5),
        "how_to_read": (
            "confirmed requires the sign to hold AND at least half the magnitude to survive on "
            "seeds that did not exist when the feature was chosen. Shrinkage toward zero is the "
            "expected signature of a feature selected on noise, and a simulator can always buy "
            "this check because it can always generate more data."),
    }
