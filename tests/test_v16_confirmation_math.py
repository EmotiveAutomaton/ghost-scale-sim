import math
import pytest
from ghostscale.validation.soundingline.v16.confirmation_math import (
    binomial_cdf, upper_probability, rare_event_cutoff, equivalence_power, bounded_equivalence)


def test_binomial_scalar_matches_independent_finite_coin_enumeration():
    for n in [1, 2, 5, 12]:
        for p in [0, .1, .5, .9, 1]:
            for k in range(n+1):
                expected = sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k+1))
                assert abs(binomial_cdf(k,n,p)-expected) < 1e-12


def test_zero_events_still_need_positive_finite_sample_and_exact_confidence():
    assert abs(upper_probability(0,200,.05) - (1-.05**(1/200))) < 1e-12
    plan = equivalence_power(outcome_difference_bound=2, margin=.05)
    assert 64 <= plan["n_independent_maker_packets"] <= 4096
    assert plan["planning_state"] == "adequate"
    assert plan["exact_power_at_planning_alternative"] >= .90
    assert plan["exact_size_at_boundary"] <= .05/3


def test_known_boundary_fails_and_exact_calibrated_null_passes():
    plan = equivalence_power(outcome_difference_bound=2, margin=.05)
    n = plan["n_independent_maker_packets"]
    rows = [{"unit_id": str(i), "constructor_id": i, "differences": [0., 0.]} for i in range(n)]
    null = bounded_equivalence(rows, outcome_difference_bound=2, margin=.05)
    assert null["equivalence_established_at_unadjusted_allocation"]
    for row in rows:
        row["differences"][1] = .1
    broken = bounded_equivalence(rows, outcome_difference_bound=2, margin=.05)
    assert not broken["equivalence_established_at_unadjusted_allocation"]
    assert broken["exact_one_sided_p"] == 1


def test_shared_constructors_duplicate_units_and_false_range_are_rejected():
    rows = [{"unit_id": "a", "constructor_id": 1, "differences": [0.]},
            {"unit_id": "b", "constructor_id": 1, "differences": [0.]}]
    with pytest.raises(ValueError, match="independent constructor"):
        bounded_equivalence(rows, outcome_difference_bound=2, margin=.05)
    rows[1]["constructor_id"] = 2
    rows[1]["differences"] = [3.]
    with pytest.raises(ValueError, match="outcome range"):
        bounded_equivalence(rows, outcome_difference_bound=2, margin=.05)
    rows[1] = rows[0]
    with pytest.raises(ValueError, match="duplicate"):
        bounded_equivalence(rows, outcome_difference_bound=2, margin=.05)


def test_inadequate_capped_power_stays_inadequate():
    plan = equivalence_power(outcome_difference_bound=2, margin=.00001, maximum_n=64)
    assert plan["planning_state"] == "inadequate"
    assert plan["maximum_disagreements_for_rejection"] == -1
