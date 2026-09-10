import math
import pytest
from ghostscale.validation.soundingline.v16.confirmation_bounded import (
    lower_bound, bounded_mean, planning_law, plan_power, rejection_count, holm)


def packets(values):
    return [{"unit_id": str(i), "constructor_id": str(i), "difference": x} for i, x in enumerate(values)]


def test_bound_keeps_zero_variance_penalty_and_constructor_independence():
    assert lower_bound(.1, 0, 64) < .1
    assert bounded_mean(packets([0]*64), null_mean=.05)["criterion_state"] == "not established"
    result = bounded_mean(packets([.5]*320), null_mean=.05)
    assert result["criterion_state"] == "held"
    assert result["valid_one_sided_p"] < .05/3
    rows = packets([.5]*320)
    rows[1]["constructor_id"] = rows[0]["constructor_id"]
    with pytest.raises(ValueError, match="constructor"):
        bounded_mean(rows, null_mean=.05)
    with pytest.raises(ValueError, match="range"):
        bounded_mean(packets([2, 0]), null_mean=.05)


def test_formula_matches_independent_rescaling_and_holm_keeps_failed_claim():
    n, values, alpha = 64, [0, .5, 1, 1]*16, .02
    mean = sum(values)/n
    variance = sum((v-mean)**2 for v in values)/(n-1)
    normalized = [(v+1)/2 for v in values]
    m = sum(normalized)/n
    s = sum((v-m)**2 for v in normalized)/(n-1)
    independently_scaled = 2*(m-math.sqrt(2*s*math.log(2/alpha)/n)-7*math.log(2/alpha)/(3*(n-1)))-1
    assert lower_bound(mean, variance, n, alpha=alpha) == pytest.approx(independently_scaled)
    result = holm([{"claim_id": "A", "p": .01}, {"claim_id": "B", "p": .04}, {"claim_id": "C", "p": .9}])
    assert [row["familywise_rejected"] for row in result] == [True, False, False]


def test_actual_paired_mean_power_and_null_controls():
    plan = plan_power(discovery_variance=.25, null_mean=.05, practical_increment=.05)
    assert plan["planning_state"] == "adequate"
    assert plan["power_99_percent_lower"] >= .90
    assert 64 < plan["n_independent_makers"] <= 4096
    probabilities = planning_law(.25, .05)
    null_rejections = rejection_count(plan["n_independent_makers"], probabilities, null_mean=.05,
        alpha=.05/3, trials=20000, seed=491161)
    assert null_rejections/20000 < .025
    fresh = rejection_count(plan["n_independent_makers"], plan["planning_probabilities"], null_mean=.05,
        alpha=.05/3, trials=20000, seed=919031)
    assert fresh/20000 > .90
