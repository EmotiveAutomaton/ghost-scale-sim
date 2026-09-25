import itertools
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.context_precision_allocation import allocate, controls, ASSIGNMENTS, validate_costs


def test_controls(): assert all(controls().values())


def test_exhaustive_independent_enumeration():
    ranges = np.random.default_rng(92).uniform(0, .02, (4, 3)); a = allocate(ranges)
    scalar = []
    for assignment in itertools.product((2, 4, 8), repeat=4):
        cost = sum(16*8*item for item in assignment)
        score = sum(float(ranges[c, (2, 4, 8).index(item)]) for c, item in enumerate(assignment))
        scalar.append((cost, score))
    assert [x[0] for x in scalar] == a['stored_law_bytes'].tolist()
    for k, budget in enumerate((1280, 1536, 2048)):
        best = min(score for cost, score in scalar if cost <= budget)
        assert np.flatnonzero(a['selected'][k]).tolist() == [i for i, (cost, score) in enumerate(scalar) if cost <= budget and score == best]
    for i, (cost, score) in enumerate(scalar):
        assert a['dominated'][i] == any(c <= cost and s <= score and (c < cost or s < score) for c, s in scalar)


def test_equal_contexts_and_all_ties():
    a = allocate(np.zeros((4, 3)))
    for k, b in enumerate((1280, 1536, 2048)):
        assert np.array_equal(a['selected'][k], a['stored_law_bytes'] <= b)
    a = allocate(np.tile([.03, .02, 0], (4, 1)))
    assert a['selected'][0].sum() == 4


def test_greedy_failure():
    ranges = np.array([[10, 9, 0], [6, 0, 0], [6, 0, 0], [0, 0, 0]], dtype=float)
    a = allocate(ranges)
    # Benefit per added byte chooses context1,context2 single, then context0 single.
    greedy = (1, 1, 1, 0); exact = (2, 0, 1, 0)
    assert sum(ranges[c, exact[c]] for c in range(4)) == 6
    assert sum(ranges[c, greedy[c]] for c in range(4)) == 9
    assert a['selected'][2, ASSIGNMENTS.index(exact)]
    assert not a['selected'][2, ASSIGNMENTS.index(greedy)]


def test_permutation():
    r = np.random.default_rng(111).uniform(0, 1, (4, 3)); a = allocate(r); perm = [2, 0, 3, 1]; b = allocate(r[perm])
    for i, assignment in enumerate(ASSIGNMENTS):
        j = ASSIGNMENTS.index(tuple(assignment[c] for c in perm))
        assert np.allclose(a['accumulated_log_ranges'][i], b['accumulated_log_ranges'][j], atol=1e-13)
        assert np.array_equal(a['selected'][:, i], b['selected'][:, j])


def test_support_loss_and_infeasible():
    a = allocate(np.full((4, 3), math.inf))
    assert np.all(a['posterior_tv_bounds'] == 1)
    assert not allocate(np.zeros((4, 3)), (1023,))['selected'].any()


def test_bytes_independently_count_actual_buffers():
    a = allocate(np.zeros((4, 3)))
    for i, assignment in enumerate(ASSIGNMENTS):
        actual = sum(np.empty((16, 8), dtype=('float16', 'float32', 'float64')[d]).nbytes for d in assignment)
        assert a['stored_law_bytes'][i] == actual
    corrupt = a['stored_law_bytes'].copy(); corrupt[5] -= 1
    with pytest.raises(ValueError, match='byte charge'): validate_costs(ASSIGNMENTS, corrupt)


@pytest.mark.parametrize('ranges,budgets', [(np.zeros((4, 2)), (1280,)), (np.full((4, 3), np.nan), (1280,)), (np.full((4, 3), -1), (1280,)), (np.zeros((4, 3)), (-1,))])
def test_invalid(ranges, budgets):
    with pytest.raises(ValueError): allocate(ranges, budgets)
