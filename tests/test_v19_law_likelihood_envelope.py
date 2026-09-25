from fractions import Fraction
from itertools import product
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.law_likelihood_envelope import evaluate,posterior_bound,fixture,controls


def test_live_and_placebo():assert all(controls().values())


@pytest.mark.parametrize('ratio',[Fraction(1),Fraction(4,3),Fraction(9),Fraction(100)])
def test_rational_prior_enumeration_and_attaining_prior(ratio):
    bound=posterior_bound(math.log(ratio),1)
    for i in range(101):
        p=Fraction(i,100);q=ratio*p/(ratio*p+1-p)
        assert float(abs(q-p))<=bound+1e-15
    p=1/(math.sqrt(ratio)+1);q=float(ratio)*p/(float(ratio)*p+1-p)
    assert abs((q-p)-bound)<1e-15


@pytest.mark.parametrize('length',[1,8,32,128])
def test_explicit_likelihood_products(length):
    # Exact decimal fractions, independent of log-space multiplication.
    old=(Fraction(1,5),Fraction(4,5));new=(Fraction(21,100),Fraction(79,100))
    ratios=[new[i]/old[i] for i in range(2)]
    span=math.log(max(ratios))-math.log(min(ratios))
    for a in range(1,20):
        p=Fraction(a,20);x=p*old[0]**length;y=(1-p)*old[1]**length
        xx=p*new[0]**length;yy=(1-p)*new[1]**length
        assert float(abs(x/(x+y)-xx/(xx+yy)))<=posterior_bound(span,length)+1e-15


def test_support_loss_and_impossible_observation():
    law=np.zeros((16,4,8));law[:,:,0]=1-1e-9;law[:,:,1]=1e-9
    r=evaluate(law,'float16')
    assert r['lost_support'].sum()==64 and np.isinf(r['maximum_log_range'])
    assert np.all(r['posterior_tv_bounds']==1)
    assert not r['supported_observations'][:,2:].any()
    assert np.isnan(r['ratios'][:,:,2:]).all()


def test_exact_cast_and_constant_ratio():
    r=evaluate(fixture(),'float64');assert (r['posterior_tv_bounds']==0).all()
    assert posterior_bound(0,128)==0


def test_permutations():
    law=fixture();a=evaluate(law,'float16');b=evaluate(law[::-1,::-1,::-1],'float16')
    assert np.max(abs(a['posterior_tv_bounds']-b['posterior_tv_bounds']))<1e-12


def test_undersized_bound_fails_attainer():
    p=Fraction(1,4);q=9*p/(9*p+1-p)
    assert float(q-p)>posterior_bound(math.log(9),1)-1e-5


@pytest.mark.parametrize('case',['empty','negative','nan'])
def test_invalid_rows(case):
    law=fixture()
    if case=='empty':law[0,0]=0
    elif case=='negative':law[0,0,0]=-1
    else:law[0,0,0]=np.nan
    with pytest.raises(ValueError):evaluate(law,'float16')
