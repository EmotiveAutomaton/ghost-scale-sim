from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import adaptive_second_cue as P
from ghostscale.validation.soundingline.v19 import adaptive_second_cue_review as R


def fixture():
    return P.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)],[F(1,2)]*2])


def test_known_answers():
    assert all(R.controls().values())


@pytest.mark.parametrize('counts', R.MIXTURES)
@pytest.mark.parametrize('rate', R.RELIABILITIES)
@pytest.mark.parametrize('dep', R.DEPENDENCES)
def test_independent_complete_policy_enumeration(counts,rate,dep):
    s=fixture()
    assert R.solve(s,counts,rate,dep)==P.solve(s,counts,rate,dep)


def test_reject_corrupted_capacity():
    s=deepcopy(fixture());s['candidates'][0]['used_bytes']=2
    with pytest.raises(AssertionError):R.solve(s,(2,2,0),(2,3),(0,1))


def test_impossible_labels_have_no_conditional_price():
    for a in R.solve(fixture(),(4,0,0),(1,1),(1,1))['fees']:
        for item in a['conditional'][1:]:
            assert item['conditional_break_even_fee'] is None
            assert item['optimal_decisions']==['decline','buy']
            assert item['selected_decision']=='decline'
            assert item['fee_contribution']==[0,1]
