from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import partial_prior_review as R
from ghostscale.validation.soundingline.v19 import partial_prior_disclosure as P
from ghostscale.validation.soundingline.v19.randomized_source_storage import fixture


def library():
    return fixture([1,2,3], [1,2,1], 2, {'a':[1,3], 'b':[2], 'c':[3]},
                   [[F(1,2),F(1,3),F(1,6)], [F(1,6),F(2,3),F(1,6)], [F(0),F(0),F(1)]])


def test_known_answers():
    assert all(R.controls().values())


@pytest.mark.parametrize('mixture', R.MIXTURES)
def test_exhaustive_policy_against_conditional_optimizer(mixture):
    s = library()
    assert R.value(s, mixture) == P.value(s, mixture)


def test_infeasible_byte_charge_rejected():
    s = deepcopy(library()); s['candidates'][0]['used_bytes'] = 3
    with pytest.raises(AssertionError): R.value(s, (1,1,2))
