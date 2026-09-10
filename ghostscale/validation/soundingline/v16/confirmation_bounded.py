"""Finite-sample paired-mean test and explicit variance-based power planning.

Inference: Maurer and Pontil (2009), Theorem 4 applied to -X and rescaled
from [0,1]. https://www.cs.mcgill.ca/~colt2009/papers/012.pdf
The sample variance uses n-1. Every observation is a fresh constructor with one
paired maker packet. Nested condition records cannot increase independent n.

Power is for an explicit three-point planning law with the discovery variance
and a declared mean beyond the null boundary. It is not a uniform power claim
over all bounded laws or an assurance at the null boundary. Monte Carlo error
is bounded separately using an exact binomial confidence bound.
"""
import math
import numpy as np
from .confirmation_math import upper_probability


def lower_bound(mean, variance, n, *, low=-1., high=1., alpha=.05/3):
    if type(n) is not int or n < 2 or not 0 < alpha < 1 or not low < high:
        raise ValueError("invalid independent sample or confidence allocation")
    if not all(math.isfinite(x) for x in [mean, variance, low, high]) or not low <= mean <= high or variance < 0:
        raise ValueError("invalid bounded paired moments")
    log = math.log(2/alpha)
    radius = math.sqrt(2*variance*log/n) + 7*(high-low)*log/(3*(n-1))
    return max(low, mean-radius)


def bounded_mean(packets, *, null_mean, low=-1., high=1., alpha=.05/3):
    if len(packets) < 2 or len({row["unit_id"] for row in packets}) != len(packets):
        raise ValueError("empty or duplicate independent maker packet")
    if len({row["constructor_id"] for row in packets}) != len(packets):
        raise ValueError("independent constructor requirement violated")
    values = [row["difference"] for row in packets]
    if not low < null_mean < high or any(not math.isfinite(x) or not low <= x <= high for x in values):
        raise ValueError("external paired outcome range or null boundary violated")
    n = len(values)
    mean = math.fsum(values)/n
    variance = math.fsum((x-mean)**2 for x in values)/(n-1)
    left, right = 1e-15, 1-1e-15
    if lower_bound(mean, variance, n, low=low, high=high, alpha=right) <= null_mean:
        p = 1.
    else:
        for _ in range(64):
            middle = (left+right)/2
            if lower_bound(mean, variance, n, low=low, high=high, alpha=middle) > null_mean:
                right = middle
            else:
                left = middle
        p = right
    lower = lower_bound(mean, variance, n, low=low, high=high, alpha=alpha)
    return {"n_independent_makers": n, "n_independent_constructors": n,
        "paired_mean": mean, "paired_sample_variance": variance, "null_mean": null_mean,
        "external_range": [low, high], "one_sided_lower_bound": lower,
        "alpha": alpha, "valid_one_sided_p": p, "criterion_state": "held" if lower > null_mean else "not established",
        "method": "Maurer-Pontil empirical Bernstein one-sided bound; each constructor contributes one maker packet"}


def planning_law(variance, mean):
    """Moment-matched [-1,0,1] law; unattainable moments are rejected explicitly."""
    if not math.isfinite(variance) or variance < 0 or not -1 < mean < 1:
        raise ValueError("invalid planning moments")
    second = variance+mean*mean
    probabilities = [(second-mean)/2, 1-second, (second+mean)/2]
    if any(p < -1e-12 or p > 1+1e-12 for p in probabilities):
        raise ValueError("discovery variance and planning mean do not define the registered three-point law")
    return [max(0., min(1., p)) for p in probabilities]


def rejection_count(n, probabilities, *, null_mean, alpha, trials, seed):
    counts = np.random.default_rng(seed).multinomial(n, probabilities, size=trials)
    means = (counts[:, 2]-counts[:, 0])/n
    variances = np.maximum(0, (counts[:, 0]+counts[:, 2]-n*means**2)/(n-1))
    log = math.log(2/alpha)
    lower = means-np.sqrt(2*variances*log/n)-14*log/(3*(n-1))
    return int(np.count_nonzero(lower > null_mean))


def plan_power(*, discovery_variance, null_mean, practical_increment, alpha=.05/3,
               target_power=.90, maximum_n=4096, trials=20000, seed=160903):
    if practical_increment <= 0 or trials < 1000 or not 2 <= maximum_n <= 4096:
        raise ValueError("invalid finite power allocation")
    alternative = null_mean+practical_increment
    probabilities = planning_law(discovery_variance, alternative)
    result = None
    for n in range(64, maximum_n+1, 64):
        successes = rejection_count(n, probabilities, null_mean=null_mean, alpha=alpha, trials=trials, seed=seed+n)
        # A separate 99% lower confidence bound on simulation power. This is
        # planning uncertainty, not an extra scientific test of the claim.
        lower = 1-upper_probability(trials-successes, trials, .01)
        result = {"n_independent_makers": n, "constructors": n, "planning_state": "adequate" if lower >= target_power else "inadequate",
            "null_mean": null_mean, "planning_mean": alternative, "practical_increment": practical_increment,
            "discovery_marginal_variance": discovery_variance, "support": [-1, 0, 1], "planning_probabilities": probabilities,
            "estimated_power": successes/trials, "power_99_percent_lower": lower, "simulation_trials": trials,
            "simulation_seed": seed+n, "alpha": alpha, "target_power": target_power,
            "power_scope": "Moment-matched planning law, not a uniform guarantee or 90 percent power at the null boundary",
            "inference_scope": "Finite-sample bound does not require the planning law to be correct"}
        if lower >= target_power:
            break
    return result


def holm(p_values, alpha=.05):
    if not p_values or len(p_values) > 3 or len({row["claim_id"] for row in p_values}) != len(p_values):
        raise ValueError("multiplicity ledger requires one to three distinct frozen claims")
    ordered = sorted(p_values, key=lambda row: (row["p"], row["claim_id"]))
    previous = 0.
    result = []
    for index, row in enumerate(ordered):
        if not math.isfinite(row["p"]) or not 0 <= row["p"] <= 1:
            raise ValueError("invalid primary p value")
        adjusted = min(1., max(previous, (len(ordered)-index)*row["p"]))
        previous = adjusted
        result.append({**row, "holm_adjusted_p": adjusted, "familywise_rejected": adjusted <= alpha})
    return result
