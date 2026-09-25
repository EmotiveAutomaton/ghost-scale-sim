import itertools
import math
import struct
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.context_precision_regret import allocate, controls, ASSIGNMENTS, VERTICES


def test_controls(): assert all(controls().values())


@pytest.mark.parametrize('ranges', [np.zeros((4,3)),np.tile([.03,.02,0],(4,1)),
    np.random.default_rng(785).uniform(0,.04,(4,3)),np.full((4,3),math.inf)])
def test_exhaustive_scalar(ranges):
    a=allocate(ranges)
    costs=[sum(128*struct.calcsize(('e','f','d')[d]) for d in x) for x in itertools.product(range(3),repeat=4)]
    totals=[[math.fsum(float(ranges[c,x[c]])*(13 if c==v else 1) for c in range(4)) for v in range(4)] for x in ASSIGNMENTS]
    for k,budget in enumerate((1280,1536,2048)):
        feasible=[i for i,cost in enumerate(costs) if cost<=budget]
        best=[min(totals[i][v] for i in feasible) for v in range(4)]
        regret=[[math.nan if math.isinf(totals[i][v]) and math.isinf(best[v]) else totals[i][v]-best[v] for v in range(4)] for i in range(81)]
        eligible=[i for i in feasible if not any(math.isnan(x) for x in regret[i])]
        low=min((max(regret[i]) for i in eligible),default=None)
        assert [i for i in eligible if max(regret[i])==low]==np.flatnonzero(a['regret_selected'][k]).tolist()
        assert np.allclose(regret,a['vertex_regrets'][k,:,0],equal_nan=True)
        assert np.array_equal(a['stored_law_bytes'],costs)


def test_convex_mixtures_bounded_by_vertices():
    r=np.random.default_rng(286).uniform(0,.04,(4,3));a=allocate(r)
    for weights in itertools.product(range(5),repeat=4):
        if sum(weights)!=4:continue
        counts=[sum(weights[v]*VERTICES[v][c] for v in range(4))/4 for c in range(4)]
        scores=[math.fsum(float(r[c,x[c]])*counts[c] for c in range(4)) for x in ASSIGNMENTS]
        for k,budget in enumerate((1280,1536,2048)):
            best=min(scores[i] for i in range(81) if a['stored_law_bytes'][i]<=budget)
            assert np.all(np.array(scores)-best<=a['worst_vertex_regret'][k,:,0]+1e-14)


def test_permutation():
    r=np.random.default_rng(654).integers(0,100,(4,3));p=[2,0,3,1];a=allocate(r);b=allocate(r[p])
    for i,x in enumerate(ASSIGNMENTS):
        j=ASSIGNMENTS.index(tuple(x[c] for c in p))
        assert np.array_equal(a['regret_selected'][:,i],b['regret_selected'][:,j])
        assert np.array_equal(a['worst_vertex_regret'][:,i],b['worst_vertex_regret'][:,j])


def test_zero_ties_and_infeasible():
    a=allocate(np.zeros((4,3)),(1023,1280))
    assert not a['regret_selected'][0].any()
    assert np.array_equal(a['regret_selected'][1],a['stored_law_bytes']<=1280)


def test_counterexample_and_bound_tradeoff():
    a=allocate(np.array([[6,0,0],[10,5,0],[0,0,0],[0,0,0.]]))
    regret=np.flatnonzero(a['regret_selected'][0])[0];absolute=np.flatnonzero(a['selected'][0])[0]
    assert a['worst_vertex_regret'][0,regret,0]<a['worst_vertex_regret'][0,absolute,0]
    assert a['worst_log_ranges'][regret,0]>a['worst_log_ranges'][absolute,0]


def test_undefined_is_not_zero():
    a=allocate(np.full((4,3),math.inf))
    assert not a['regret_defined'].any() and not a['regret_selected'].any()
    assert np.isnan(a['worst_vertex_regret']).all()
