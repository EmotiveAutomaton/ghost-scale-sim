from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import noisy_prior_review as R
from ghostscale.validation.soundingline.v19 import noisy_prior_disclosure as P


def library():
    return P.fixture([1,2,3], [1,2,1], 2, {'a':[1,3], 'b':[2], 'c':[3]},
                     [[F(1,2),F(1,3),F(1,6)], [F(1,6),F(2,3),F(1,6)], [F(0),F(0),F(1)]])


def test_known_answers():
    assert all(R.controls().values())


@pytest.mark.parametrize('counts', R.MIXTURES)
@pytest.mark.parametrize('reliability', R.RELIABILITIES)
def test_whole_policy_matches_conditional_scores(counts,reliability):
    assert R.value(library(),counts,reliability) == P.value(library(),counts,P.channel(reliability))


def test_infeasible_charge_rejected():
    s=deepcopy(library());s['candidates'][0]['used_bytes']=3
    with pytest.raises(AssertionError): R.value(s,(1,1,2),(2,3))


def test_uninformative_label_can_hurt_certainty_but_not_optimum():
    for c in R.MIXTURES:
        r=R.value(library(),c,(1,3))
        assert r['information_value']==[0,1]
        assert F(*r['certainty_over_fixed']) <= 0
