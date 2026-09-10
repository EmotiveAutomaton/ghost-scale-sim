"""Exact finite-binomial power and a bounded paired-equivalence envelope.

This is an optional confirmation instrument, not a selected claim. It never
turns zero empirical variance into a zero-sample power calculation. Paired
contexts/rivals can be grouped into one independently drawn maker packet.
"""
import math


def binomial_cdf(k, n, probability):
    if type(n) is not int or n < 1 or type(k) is not int or not 0 <= probability <= 1:
        raise ValueError("invalid finite binomial parameters")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability == 0:
        return 1.0
    if probability == 1:
        return 0.0
    log_p, log_q = math.log(probability), math.log1p(-probability)
    terms = [math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)+j*log_p+(n-j)*log_q for j in range(k+1)]
    anchor = max(terms)
    return min(1.0, math.exp(anchor)*math.fsum(math.exp(term-anchor) for term in terms))


def rare_event_cutoff(n, null_probability, alpha):
    if not 0 < alpha < 1 or not 0 < null_probability < 1:
        raise ValueError("invalid exact-test threshold")
    low, high = -1, n
    while high-low > 1:
        middle = (low+high)//2
        if binomial_cdf(middle, n, null_probability) <= alpha:
            low = middle
        else:
            high = middle
    return low


def upper_probability(k, n, alpha):
    if not 0 < alpha < 1 or not 0 <= k <= n:
        raise ValueError("invalid binomial confidence bound")
    if k == n:
        return 1.0
    left, right = 0.0, 1.0
    for _ in range(64):
        middle = (left+right)/2
        if binomial_cdf(k, n, middle) > alpha:
            left = middle
        else:
            right = middle
    return right


def equivalence_power(*, outcome_difference_bound, margin, equality_tolerance=1e-10,
                      alpha=.05/3, target_power=.90, interior_ratio=.1,
                      maximum_n=4096, minimum_n=64, step=32):
    bound = outcome_difference_bound
    if not 0 <= equality_tolerance < margin < bound or not 0 < interior_ratio < 1:
        raise ValueError("equivalence requires an external range bound and an interior alternative")
    if not 0 < target_power < 1 or minimum_n < 1 or maximum_n < minimum_n or step < 1:
        raise ValueError("invalid finite power allocation")
    null = (margin-equality_tolerance)/bound
    alternative = null*interior_ratio
    last = None
    for n in range(minimum_n, maximum_n+1, step):
        cutoff = rare_event_cutoff(n, null, alpha)
        power = binomial_cdf(cutoff, n, alternative)
        last = {"n_independent_maker_packets": n, "maximum_disagreements_for_rejection": cutoff,
            "exact_power_at_planning_alternative": power,
            "exact_size_at_boundary": binomial_cdf(cutoff, n, null),
            "null_event_probability": null, "planning_event_probability": alternative,
            "margin": margin, "outcome_difference_bound": bound, "equality_tolerance": equality_tolerance,
            "alpha": alpha, "target_power": target_power,
            "planning_state": "adequate" if power >= target_power else "inadequate",
            "scope": "one independent constructor/history per packet; all paired contexts and rivals collapse to one any-disagreement event",
            "alternative_scope": "explicit interior equivalence alternative; not a claim of power at the equivalence boundary"}
        if power >= target_power:
            return last
    return last


def bounded_equivalence(packets, *, outcome_difference_bound, margin, equality_tolerance=1e-10, alpha=.05/3):
    """Every packet is a same-layout vector of paired differences, including failures.

    For every component, |E[d]| <= tolerance + B*P(any |d|>tolerance).
    One exact event-rate bound therefore covers all declared components together.
    """
    if not packets or len({row["unit_id"] for row in packets}) != len(packets):
        raise ValueError("empty or duplicate independent maker packets")
    dimensions = len(packets[0]["differences"])
    if dimensions < 1 or not 0 <= equality_tolerance < margin < outcome_difference_bound:
        raise ValueError("invalid equivalence estimand or bound")
    if len({row["constructor_id"] for row in packets}) != len(packets):
        raise ValueError("exact binomial analysis requires independent constructor draws, not nested makers")
    events = 0
    for row in packets:
        differences = row["differences"]
        if len(differences) != dimensions or any(not math.isfinite(value) or abs(value) > outcome_difference_bound for value in differences):
            raise ValueError("paired layout or external outcome range violated")
        events += int(any(abs(value) > equality_tolerance for value in differences))
    n = len(packets)
    boundary = (margin-equality_tolerance)/outcome_difference_bound
    upper = upper_probability(events, n, alpha)
    return {"n_independent_maker_packets": n, "n_independent_constructors": n, "paired_components": dimensions,
        "disagreement_events": events, "exact_one_sided_p": binomial_cdf(events, n, boundary),
        "event_probability_upper": upper, "simultaneous_absolute_mean_bound": equality_tolerance+outcome_difference_bound*upper,
        "paired_means": [math.fsum(row["differences"][j] for row in packets)/n for j in range(dimensions)],
        "margin": margin, "outcome_difference_bound": outcome_difference_bound, "alpha": alpha,
        "equivalence_established_at_unadjusted_allocation": equality_tolerance+outcome_difference_bound*upper < margin,
        "multiplicity": "apply the separately frozen familywise ledger; this local alpha alone does not account for other claims"}
