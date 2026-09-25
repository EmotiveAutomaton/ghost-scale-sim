import itertools
import math
import struct
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.robust_context_precision import allocate, controls, VERTICES, ASSIGNMENTS
from ghostscale.validation.soundingline.v19.context_precision_allocation import validate_costs


def test_controls(): assert all(controls().values())


def scalar(ranges, budget):
    data = []
    for a in itertools.product(range(3), repeat=4):
        cost = sum(128*struct.calcsize(('e','f','d')[d]) for d in a)
        v = [math.fsum(float(ranges[c, a[c]])*(13 if c == t else 1) for c in range(4)) for t in range(4)]
        data.append((cost, v))
    feasible = [i for i, (cost, v) in enumerate(data) if cost <= budget]
    best = min((max(data[i][1]) for i in feasible), default=None)
    return data, [i for i in feasible if max(data[i][1]) == best]


@pytest.mark.parametrize('ranges', [np.zeros((4,3)), np.tile([.3,.1,0],(4,1)), np.random.default_rng(771).uniform(0,.02,(4,3)), np.full((4,3),math.inf)])
def test_exhaustive_scalar(ranges):
    a = allocate(ranges)
    for k,budget in enumerate((1280,1536,2048)):
        data, selected = scalar(ranges,budget)
        assert selected == np.flatnonzero(a['selected'][k]).tolist()
        assert [d[0] for d in data] == a['stored_law_bytes'].tolist()
        assert np.allclose([d[1] for d in data], a['vertex_log_ranges'][:,0])
        for v in range(4):
            feasible = [i for i,d in enumerate(data) if d[0] <= budget]
            best = min(data[i][1][v] for i in feasible)
            assert [i for i in feasible if data[i][1][v] == best] == np.flatnonzero(a['vertex_selected'][k,v]).tolist()


def test_rational_convex_mixtures():
    r=np.random.default_rng(619).uniform(0,.03,(4,3));a=allocate(r)
    for weights in itertools.product(range(5),repeat=4):
        if sum(weights)!=4:continue
        counts=[sum(weights[v]*VERTICES[v][c] for v in range(4))/4 for c in range(4)]
        for i,assignment in enumerate(ASSIGNMENTS):
            score=math.fsum(counts[c]*r[c,assignment[c]] for c in range(4))
            assert score<=a['worst_log_ranges'][i,0]+1e-15
    assert np.allclose(a['vertex_log_ranges'].mean(-1),a['balanced_log_ranges'])


def test_matched_context_permutation():
    r=np.random.default_rng(601).uniform(0,.03,(4,3));p=[2,0,3,1];a=allocate(r);b=allocate(r[p])
    for i,assignment in enumerate(ASSIGNMENTS):
        j=ASSIGNMENTS.index(tuple(assignment[c] for c in p))
        assert np.array_equal(a['selected'][:,i],b['selected'][:,j])
        assert np.allclose(a['vertex_log_ranges'][i][:,p],b['vertex_log_ranges'][j])


def test_balanced_can_be_worse():
    a=allocate(np.array([[6,0,0],[10,5,0],[0,0,0],[0,0,0.]]))
    best=np.flatnonzero(a['selected'][0]);b=np.flatnonzero(a['balanced_selected'][0])
    assert a['worst_log_ranges'][best[0],0] < min(a['worst_log_ranges'][i,0] for i in b)


def test_support_loss_and_undefined_regret():
    a=allocate(np.full((4,3),math.inf));assert np.all(a['worst_posterior_tv_bounds']==1)
    assert np.isnan(a['vertex_regrets']).all()


def test_exact_cast_all_ties_and_infeasible():
    a=allocate(np.zeros((4,3)),(1023,1280))
    assert not a['selected'][0].any()
    assert np.array_equal(a['selected'][1],a['stored_law_bytes']<=1280)
    assert not a['worst_posterior_tv_bounds'].any()


def test_undercharged_bytes():
    a=allocate(np.zeros((4,3)));cost=a['stored_law_bytes'].copy();cost[8]-=1
    with pytest.raises(ValueError,match='byte charge'):validate_costs(ASSIGNMENTS,cost)


@pytest.mark.parametrize('r,b',[(np.zeros((3,3)),(1280,)),(np.full((4,3),np.nan),(1280,)),(np.full((4,3),-1),(1280,)),(np.zeros((4,3)),(-1,))])
def test_bad_inputs(r,b):
    with pytest.raises(ValueError):allocate(r,b)
