import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.context_precision_regret import allocate
from ghostscale.validation.soundingline.v19.precision_regret_review import reconstruct
from ghostscale.validation.soundingline.v19.likelihood_envelope_review import check_arrays


@pytest.mark.parametrize('ranges', [np.zeros((4, 3)), np.tile([.03, .02, 0], (4, 1)),
    np.random.default_rng(719).uniform(0, .04, (4, 3)), np.full((4, 3), math.inf),
    np.array([[6, 0, 0], [10, 5, 0], [0, 0, 0], [0, 0, 0.]])])
def test_all_arrays(ranges):
    check_arrays(allocate(ranges), reconstruct(ranges))


@pytest.mark.parametrize('field', ['stored_law_bytes', 'selected', 'balanced_selected', 'vertex_selected',
    'vertex_regrets', 'vertex_posterior_tv_bounds', 'count_vertices', 'assignments',
    'regret_defined', 'worst_vertex_regret', 'regret_selected'])
def test_corruption(field):
    r = np.random.default_rng(897).uniform(0, .04, (4, 3)); actual = allocate(r)
    a = actual[field]; a.flat[0] = not a.flat[0] if a.dtype == bool else a.flat[0]+1
    with pytest.raises(AssertionError): check_arrays(actual, reconstruct(r))


def test_undefined_and_infeasible():
    r = np.full((4, 3), math.inf)
    a = reconstruct(r, (1023, 1280))
    check_arrays(allocate(r, (1023, 1280)), a)
    assert not a['regret_selected'].any() and not a['regret_defined'].any()


def test_known_absolute_regret_disagreement():
    a = reconstruct(np.array([[6, 0, 0], [10, 5, 0], [0, 0, 0], [0, 0, 0.]]))
    assert set(np.flatnonzero(a['selected'][0])).isdisjoint(np.flatnonzero(a['regret_selected'][0]))
