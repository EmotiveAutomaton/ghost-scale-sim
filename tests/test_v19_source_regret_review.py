from copy import deepcopy
from fractions import Fraction
import pytest
from ghostscale.validation.soundingline.v19 import source_regret_review as R
from ghostscale.validation.soundingline.v19.source_mass_frontier import allocate


def records(capacity):
    return [dict(times=[1,2,3],costs=[10,20,30],capacity_bytes=capacity,prior=p,
        result=deepcopy(allocate((1,2,3),(10,20,30),capacity,p))) for p in R.PRIORS]


def test_controls():
    assert all(R.controls().values())


@pytest.mark.parametrize('capacity',[0,10,20,30,40,60,100])
def test_independent_decision_and_exhaustive_prior_mixtures(capacity):
    from ghostscale.validation.soundingline.v19.source_prior_regret import minimize
    from ghostscale.validation.soundingline.v19.robust_source_mass import from_parent
    rr = records(capacity); lib = from_parent(rr); answer = R.reconstruct(rr)
    assert answer == dict(lib, **minimize(lib))
    for row in answer['candidates']:
        regrets = list(map(lambda x:Fraction(*x),row['prior_regrets']))
        losses = [Fraction(a,8)*regrets[0]+Fraction(b,8)*regrets[1]+Fraction(8-a-b,8)*regrets[2]
            for a in range(9) for b in range(9-a)]
        assert max(losses) == Fraction(*row['maximum_regret'])


@pytest.mark.parametrize('field,value',[('used_bytes',1000),('mass_numerator',999),('selected_times',[3,2,1])])
def test_parent_corruption_rejected(field,value):
    rr=records(40);rr[0]['result']['optimal'][field]=value
    with pytest.raises(AssertionError): R.reconstruct(rr)


def test_prior_order_invariant():
    rr=records(40);assert R.reconstruct(rr)==R.reconstruct(rr[::-1])


def test_regret_corruption_rejected():
    lib=R.reconstruct(records(40));lib['candidates'][0]['prior_regrets'][0]=[999,1]
    with pytest.raises(AssertionError): R.regret_decision(lib)
