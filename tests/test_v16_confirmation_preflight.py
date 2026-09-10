from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.confirmation_runner import fixture_claims, validate_claims


def test_confirmation_cap_and_lineage_are_checked_before_generation():
    claims = fixture_claims()
    validate_claims(claims, fixture=True)
    for change in [
        {"n_constructor_packets": 4096, "total_fresh_acquisition_histories": 8192},
        {"histories_per_constructor_packet": 1, "total_fresh_acquisition_histories": 64},
        {"namespace": "an-old-discovery-lineage"},
        {"alpha_planning": .05},
    ]:
        altered = deepcopy(claims)
        altered[1].update(change)
        with pytest.raises(ValueError):
            validate_claims(altered, fixture=True)
    with pytest.raises(ValueError):
        validate_claims(claims+claims, fixture=True)
    validate_claims([], fixture=False)
