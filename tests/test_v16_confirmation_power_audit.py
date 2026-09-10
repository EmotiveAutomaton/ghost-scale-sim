from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.confirmation_power_audit import mean_power, equivalence_power
from ghostscale.validation.soundingline.v16.confirmation_bounded import plan_power
from ghostscale.validation.soundingline.v16.confirmation_math import equivalence_power as original_equivalence_power


def test_independent_mc_tail_reproduces_power_and_rejects_a_fabricated_estimate():
    record = plan_power(discovery_variance=.25, null_mean=.05, practical_increment=.15, maximum_n=1024)
    assert mean_power(record)["adequate"]
    corrupt = deepcopy(record)
    corrupt["estimated_power"] -= .1
    with pytest.raises(ValueError):
        mean_power(corrupt)


def test_independent_exact_cutoff_reproduces_joint_event_power():
    record = original_equivalence_power(outcome_difference_bound=2., margin=.05, maximum_n=2048)
    result = equivalence_power(record)
    assert result["n"] == 320 and result["adequate"]
    record["maximum_disagreements_for_rejection"] += 1
    with pytest.raises(ValueError):
        equivalence_power(record)
