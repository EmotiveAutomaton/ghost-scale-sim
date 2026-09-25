from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v19 import robust_mass_review as R
from ghostscale.validation.soundingline.v19.source_mass_frontier import allocate


def records(times=(1,2,3),costs=(10,20,30),capacity=40):
    return [dict(times=list(times),costs=list(costs),capacity_bytes=capacity,prior=p,
        result=deepcopy(allocate(tuple(times),tuple(costs),capacity,p))) for p in R.PRIORS]


def test_controls():assert all(R.controls().values())


@pytest.mark.parametrize('capacity',[0,10,20,30,40,60,100])
def test_independent_library_matches_producer(capacity):
    from ghostscale.validation.soundingline.v19.robust_source_mass import from_parent
    rr=records(capacity=capacity)
    assert R.reconstruct(rr)==from_parent(rr)


def test_mask_tie_and_recent_choice():
    a=R.reconstruct(records((1,2),(1,1),1))
    assert a['optimal_masks']==[1,2] and a['selected_mask']==2
    assert len(a['candidates'])==2
    assert sum(len(r['identities']) for r in a['candidates'])==12


@pytest.mark.parametrize('field,value',[('used_bytes',1000),('mass_numerator',999),('selected_times',[3,2,1])])
def test_corrupted_parent_rejected(field,value):
    rr=records();rr[0]['result']['optimal'][field]=value
    with pytest.raises(AssertionError):R.reconstruct(rr)


def test_prior_order_invariant():
    rr=records();assert R.reconstruct(rr)==R.reconstruct(rr[::-1])


def test_mismatched_costs_rejected():
    rr=deepcopy(records());rr[1]['costs'][0]+=1
    with pytest.raises(AssertionError):R.reconstruct(rr)
