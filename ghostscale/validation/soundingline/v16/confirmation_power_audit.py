"""Independent reproduction of the frozen confirmation power calculations."""
import math
import numpy as np
from scipy.stats import beta, binom
from .records import read, file_digest


def near(left, right):
    if not math.isfinite(left) or not math.isclose(left, right, rel_tol=0, abs_tol=1e-9):
        raise ValueError("independent confirmation power arithmetic differs")


def mean_power(record):
    variance, mean = record["discovery_marginal_variance"], record["planning_mean"]
    second = variance+mean*mean
    probabilities = [(second-mean)/2, 1-second, (second+mean)/2]
    if any(value < -1e-12 or value > 1+1e-12 for value in probabilities):
        raise ValueError("independent power law has impossible moments")
    probabilities = [max(0., min(1., value)) for value in probabilities]
    for actual, expected in zip(record["planning_probabilities"], probabilities):
        near(actual, expected)
    n = record["n_independent_makers"]
    alpha, trials = record["alpha"], record["simulation_trials"]
    draws = np.random.default_rng(record["simulation_seed"]).multinomial(n, probabilities, size=trials)
    means = (draws[:, 2]-draws[:, 0])/n
    variances = np.maximum(0., (draws[:, 0]+draws[:, 2]-n*means*means)/(n-1))
    radius = np.sqrt(2*variances*math.log(2/alpha)/n)+14*math.log(2/alpha)/(3*(n-1))
    successes = int(np.count_nonzero(means-radius > record["null_mean"]))
    lower = 0. if successes == 0 else float(beta.ppf(.01, successes, trials-successes+1))
    near(record["estimated_power"], successes/trials)
    near(record["power_99_percent_lower"], lower)
    if (record["planning_state"] == "adequate") != (lower >= record["target_power"]):
        raise ValueError("independent power adequacy disposition differs")
    return {"n": n, "simulation_successes": successes, "simulation_trials": trials,
        "power": successes/trials, "power_99_percent_lower": lower,
        "adequate": lower >= record["target_power"], "scope": record["power_scope"]}


def equivalence_power(record):
    n, alpha = record["n_independent_maker_packets"], record["alpha"]
    null = (record["margin"]-record["equality_tolerance"])/record["outcome_difference_bound"]
    alternative = record["planning_event_probability"]
    near(null, record["null_event_probability"])
    k = int(binom.ppf(alpha, n, null))
    while k >= 0 and binom.cdf(k, n, null) > alpha:
        k -= 1
    while k < n and binom.cdf(k+1, n, null) <= alpha:
        k += 1
    power, size = float(binom.cdf(k, n, alternative)), float(binom.cdf(k, n, null))
    if k != record["maximum_disagreements_for_rejection"]:
        raise ValueError("independent joint-event cutoff differs")
    near(record["exact_power_at_planning_alternative"], power)
    near(record["exact_size_at_boundary"], size)
    if (record["planning_state"] == "adequate") != (power >= record["target_power"]):
        raise ValueError("independent exact power adequacy differs")
    return {"n": n, "maximum_disagreements": k, "power": power, "boundary_size": size,
        "adequate": power >= record["target_power"], "scope": record["alternative_scope"]}


def run(root):
    selection = read(root/"confirmation-selection/CLAIMS.json")
    packet = read(root/"packets/confirmation-selection-1.json")
    if selection["packet_hash"] != packet["packet_hash"] or selection["selected"] != packet["identity"]["design"]["claims"]:
        raise ValueError("independent power audit requires the unchanged pre-data claim packet")
    results = []
    for claim in selection["selected"]:
        checked = equivalence_power(claim["power"]) if claim["kind"] == "equivalence" else mean_power(claim["power"])
        if not checked["adequate"] or checked["n"] != claim["n_constructor_packets"] or claim["total_fresh_acquisition_histories"] > 4096:
            raise ValueError("selected confirmation is underpowered or exceeds its frozen cap")
        results.append({"claim_id": claim["claim_id"], **checked})
    return {"instrument_state": "valid", "claims": results, "claim_count": len(results),
        "claims_sha256": file_digest(root/"confirmation-selection/CLAIMS.json"),
        "scope": "Reproduces power for each frozen selected n and declared planning alternative; not uniform power across bounded laws or power at the null boundary"}
