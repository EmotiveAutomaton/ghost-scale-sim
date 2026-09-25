from itertools import product
from fractions import Fraction
import pytest
from ghostscale.validation.soundingline.v19.source_mass_review import controls,optimum,rebuild_selection


def test_known_controls():
    assert all(controls().values())


@pytest.mark.parametrize('capacity',[0,2,5,9,30])
def test_exhaustive_small_all_subsets(capacity):
    times=[3,1,4,2];costs=[3,0,6,2];values=[3,1,4,2]
    candidates=[]
    for keep in product((0,1),repeat=4):
        if sum(c*k for c,k in zip(costs,keep))<=capacity:
            candidates.append((sum(v*k for v,k in zip(values,keep)),sum((1<<(t-1))*k for t,k in zip(times,keep))))
    assert optimum(times,costs,capacity,values)==max(candidates)


def test_ratio_greedy_can_lose():
    s=dict(times=[1,2,3],costs=[1,2,3],capacity_bytes=5,prior='time-proportional')
    # A distinct counterexample with equal-ratio ties prioritizing a recent item.
    s.update(times=[6,10,12],costs=[10,20,30],capacity_bytes=50)
    r=rebuild_selection(s)
    assert r['probability-per-byte']['retained_mass']<r['optimal']['retained_mass']


def test_permutation_and_exact_reciprocal_mass():
    s=dict(times=[1,3,7],costs=[2,4,5],capacity_bytes=6,prior='reciprocal-time')
    a=rebuild_selection(s);b=rebuild_selection(dict(s,times=[7,1,3],costs=[5,2,4]))
    assert a==b
    assert Fraction(a['optimal']['mass_numerator'],a['optimal']['mass_denominator'])==Fraction(4,3)/Fraction(31,21)


def test_corrupted_cost_fails():
    with pytest.raises(AssertionError):rebuild_selection(dict(times=[1],costs=[-1],capacity_bytes=2,prior='uniform'))
