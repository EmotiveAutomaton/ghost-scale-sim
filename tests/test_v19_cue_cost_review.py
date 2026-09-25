from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import cue_cost_review as R, cue_acquisition_cost as M
from test_v19_randomized_channel_regret import fixture


def test_controls():
    assert all(R.controls().values())


@pytest.mark.parametrize('counts', R.MIXTURES)
@pytest.mark.parametrize('reliability', R.RELIABILITIES)
def test_complete_independent_reconstruction(counts, reliability):
    s=fixture()
    assert R.solve(s,counts,reliability)==M.solve(s,counts,reliability)


def test_catches_corrupted_cost():
    s=fixture();s['candidates'][0]['used_bytes']=s['capacity_bytes']+1
    with pytest.raises(AssertionError):R.solve(s,(2,2,0),(2,3))


def test_break_even_exact_and_free_tie():
    s=fixture();initial=R.solve(s,(2,2,0),(2,3));threshold=F(*initial['break_even_fee'])
    levels=R.solve(s,(2,2,0),(2,3),[initial['break_even_fee']])['fee_levels']
    assert {x['acquire'] for x in levels[0]['optimal_choices']}=={False,True}
    assert not levels[0]['selected']['acquire']
    assert levels[0]['net_gain_over_no_access']==[0,1]


@pytest.mark.parametrize('field', ['break_even_fee','all_paid_policies','paid_optimal_policies','fee_levels','free_access'])
def test_corrupted_retained_table_rejected(field):
    s=fixture();answer=R.solve(s,(2,2,0),(2,3));bad=deepcopy(answer);bad[field]=[]
    assert bad!=R.solve(s,(2,2,0),(2,3))
