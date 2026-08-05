"""Which parameters carry a finding, rather than how often it survives randomisation.

WHAT THIS REPLACES. Plate 5 of the walkthrough -- "I threw my own settings away" -- keeps the shape
of the model, randomises everything the theory specifies, and counts how often each finding still
appears. That is a severity rate: one scalar per finding, and its honest reading is "83% of
randomly parameterised models of this shape do it too, so most of this is architecture".

It is the most-caveated plate in the set, and the caveat is a consequence of the statistic rather
than of the result. A survival count cannot say WHICH parameters matter, so an architectural
finding and a finding driven by one specific setting look identical.

Sobol indices split the variance instead of counting survivals:

    S1   first-order. The share of the output's variance explained by moving this parameter alone.
    ST   total-order. Its share including every interaction it takes part in.

A parameter with ST near zero is provably not carrying the result. A large gap between ST and S1
means the parameter only matters in combination -- which is the same distinction PID draws between
unique and synergistic information, arrived at from the other direction. Together they convert
"83% of models of this shape do it" into "and these three settings are why", which is a positive
claim where the severity rate is only a caveat.

COST. Sobol needs N*(2D+2) model evaluations. At the repository's rollout cost of ~3 ms and a
dozen parameters, N = 256 is about 20000 runs, roughly a minute. Morris screening is available for
when it is not: far cheaper, ranks parameters rather than quantifying them, and is the right first
pass when the parameter list is long.

DEGRADES GRACEFULLY: without ``SALib`` this returns a recorded skip.
"""
from __future__ import annotations

import numpy as np


def available() -> tuple:
    try:
        import SALib                                     # noqa: F401
        return True, ""
    except Exception as exc:                             # noqa: BLE001
        return False, f"SALib not installed ({type(exc).__name__})"


def problem(bounds: dict) -> dict:
    """SALib's problem spec from ``{name: (low, high)}``, order preserved."""
    names = list(bounds)
    return {"num_vars": len(names), "names": names,
            "bounds": [list(map(float, bounds[n])) for n in names]}


def sobol(bounds: dict, evaluate, n: int = 256, seed: int = 20260805,
          calc_second_order: bool = False) -> dict:
    """First- and total-order Sobol indices for one scalar output.

    ``evaluate`` takes a dict ``{param: value}`` and returns a float -- typically the effect size
    a finding is stated in, so the indices decompose the variance OF THE FINDING rather than of
    some proxy for it.

    ``calc_second_order`` is off by default because it roughly doubles the sample requirement and
    ST already reports total interaction involvement, which is the question this is usually asked.
    """
    ok, reason = available()
    if not ok:
        return {"skipped": reason}
    from SALib.analyze import sobol as analyze
    from SALib.sample import sobol as sample

    prob = problem(bounds)
    X = sample.sample(prob, int(n), calc_second_order=calc_second_order, seed=int(seed))
    Y = np.array([float(evaluate(dict(zip(prob["names"], row)))) for row in X], dtype=float)
    if not np.all(np.isfinite(Y)):
        bad = int((~np.isfinite(Y)).sum())
        return {"skipped": f"{bad} of {Y.size} evaluations were not finite"}
    if float(np.var(Y)) < 1e-15:
        return {"skipped": "output has no variance; nothing to decompose",
                "constant_output_value": float(Y.mean())}
    S = analyze.analyze(prob, Y, calc_second_order=calc_second_order, print_to_console=False)
    per = {nm: {"S1": float(S["S1"][i]), "S1_conf": float(S["S1_conf"][i]),
                "ST": float(S["ST"][i]), "ST_conf": float(S["ST_conf"][i]),
                "interaction_share": float(S["ST"][i] - S["S1"][i])}
           for i, nm in enumerate(prob["names"])}
    ranked = sorted(per, key=lambda k: -per[k]["ST"])
    inert = [k for k in per if per[k]["ST"] + per[k]["ST_conf"] < 0.05]
    return {
        "n_base_samples": int(n), "n_evaluations": int(X.shape[0]),
        "output_mean": float(Y.mean()), "output_variance": float(np.var(Y)),
        "per_parameter": per,
        "ranked_by_total_order": ranked,
        "carries_most": ranked[0] if ranked else None,
        "provably_inert": inert,
        "sum_first_order": float(sum(v["S1"] for v in per.values())),
        "how_to_read": (
            "S1 is the variance share a parameter explains alone; ST includes every interaction "
            "it takes part in. ST near zero means the parameter is provably not carrying the "
            "finding. A large ST minus S1 means it only matters in combination. sum_first_order "
            "well below 1 means the result is mostly interactions -- which is itself the answer "
            "to whether a finding is architectural."),
    }


def morris(bounds: dict, evaluate, n: int = 64, seed: int = 20260805) -> dict:
    """Morris elementary-effects screening: cheap, ordinal, for long parameter lists.

    Use to cut a dozen parameters down to the three worth spending a Sobol budget on. ``mu_star``
    ranks overall influence; a large ``sigma`` relative to it means the effect is non-linear or
    interacting, so a parameter with high sigma is a signal to look closer rather than a nuisance.
    """
    ok, reason = available()
    if not ok:
        return {"skipped": reason}
    from SALib.analyze import morris as analyze
    from SALib.sample import morris as sample

    prob = problem(bounds)
    X = sample.sample(prob, int(n), num_levels=4, seed=int(seed))
    Y = np.array([float(evaluate(dict(zip(prob["names"], row)))) for row in X], dtype=float)
    if not np.all(np.isfinite(Y)):
        return {"skipped": "non-finite evaluations"}
    S = analyze.analyze(prob, X, Y, num_levels=4, print_to_console=False)
    per = {nm: {"mu_star": float(S["mu_star"][i]), "sigma": float(S["sigma"][i])}
           for i, nm in enumerate(prob["names"])}
    return {"n_evaluations": int(X.shape[0]), "per_parameter": per,
            "ranked_by_mu_star": sorted(per, key=lambda k: -per[k]["mu_star"]),
            "how_to_read": ("mu_star ranks overall influence. sigma large relative to mu_star "
                            "means non-linear or interacting, which is a reason to spend Sobol "
                            "budget on that parameter rather than a reason to drop it.")}
