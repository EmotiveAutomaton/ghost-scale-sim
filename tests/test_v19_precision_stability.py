from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import precision_stability as M


def value(ranges,assignment,vertices):
    return max(sum(v[c]*ranges[c][assignment[c]] for c in range(len(assignment))) for v in vertices)


def test_controls():assert all(M.controls().values())


@pytest.mark.parametrize('first,second',list(product(product(range(2),repeat=2),repeat=2)))
def test_bounds_cover_every_reduced_interval_corner(first,second):
    lower=((1,3),(2,4));upper=((2,5),(4,7));vertices=((3,1),(1,3))
    bound=M.pair_bounds(lower,upper,first,second,vertices)
    values=[]
    for bits in product(range(2),repeat=4):
        x=[[upper[c][k] if bits[2*c+k] else lower[c][k] for k in range(2)] for c in range(2)]
        values.append(value(x,first,vertices)-value(x,second,vertices))
    assert bound['lower_numerator']<=min(values)<=max(values)<=bound['upper_numerator']
    # Interior points matter too; this bound does not assert corners suffice.
    mid=[[lower[c][k]+upper[c][k] for k in range(2)] for c in range(2)]
    delta=value(mid,first,vertices)-value(mid,second,vertices)
    assert 2*bound['lower_numerator']<=delta<=2*bound['upper_numerator']


def test_zero_width_exact_objective_difference():
    ranges=((1,7),(5,2));vertices=((3,1),(1,3))
    for a,b in product(product(range(2),repeat=2),repeat=2):
        r=M.pair_bounds(ranges,ranges,a,b,vertices)
        exact=value(ranges,a,vertices)-value(ranges,b,vertices)
        assert r['lower_numerator']==r['upper_numerator']==exact


def test_widening_and_context_permutation():
    lo=((1,3),(2,4));hi=((2,5),(4,7));a=(0,1);b=(1,0);v=((3,1),(1,3))
    r=M.pair_bounds(lo,hi,a,b,v)
    wide=M.pair_bounds(((0,2),(1,3)),((3,6),(5,8)),a,b,v)
    assert wide['lower_numerator']<=r['lower_numerator'] and wide['upper_numerator']>=r['upper_numerator']
    perm=M.pair_bounds(lo[::-1],hi[::-1],a[::-1],b[::-1],tuple(x[::-1] for x in v))
    assert all(perm[k]==r[k] for k in ('lower_numerator','upper_numerator'))


def test_binary_interval_conversion_and_costs():
    a=np.arange(12).reshape(4,3)/10;b=np.nextafter(a,np.inf)
    lower,upper,d=M.intervals(a,b)
    assert all(lower[c][k]/d==a[c,k] and upper[c][k]/d==b[c,k] for c in range(4) for k in range(3))
    with pytest.raises(ValueError):M.evaluate(a,b,[[0]]*3,charges=[0]*81)
    with pytest.raises(ValueError):M.intervals(a,np.full((4,3),np.inf))


def test_wider_box_cannot_remove_candidates():
    a=np.array([[3.,2.,0.]]*4);zero=M.evaluate(a,a,[[0]]*3)
    wide=M.evaluate(a,a+1,[[0]]*3)
    assert all(set(x['not_ruled_out'])<=set(y['not_ruled_out']) for x,y in zip(zero['summaries'],wide['summaries']))
