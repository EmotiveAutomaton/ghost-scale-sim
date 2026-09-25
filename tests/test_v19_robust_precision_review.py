import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.robust_context_precision import allocate
from ghostscale.validation.soundingline.v19.robust_precision_review import reconstruct
from ghostscale.validation.soundingline.v19.likelihood_envelope_review import check_arrays


@pytest.mark.parametrize('ranges', [np.zeros((4, 3)), np.tile([.03, .02, 0], (4, 1)),
    np.random.default_rng(719).uniform(0, .04, (4, 3)), np.full((4, 3), math.inf),
    np.array([[6, 0, 0], [10, 5, 0], [0, 0, 0], [0, 0, 0.]])])
def test_all_arrays(ranges):
    check_arrays(allocate(ranges), reconstruct(ranges))


@pytest.mark.parametrize('field', ['stored_law_bytes', 'selected', 'balanced_selected', 'vertex_selected',
    'vertex_regrets', 'vertex_posterior_tv_bounds', 'count_vertices', 'assignments'])
def test_corruption(field):
    r = np.random.default_rng(897).uniform(0, .04, (4, 3)); actual = allocate(r)
    a = actual[field]; a.flat[0] = not a.flat[0] if a.dtype == bool else a.flat[0]+1
    with pytest.raises(AssertionError): check_arrays(actual, reconstruct(r))


def test_infeasible_and_infinite_regrets():
    r = np.full((4, 3), math.inf)
    check_arrays(allocate(r, (1023, 1280)), reconstruct(r, (1023, 1280)))
    assert np.isnan(reconstruct(r)['vertex_regrets']).all()


def test_saturated_bounds_do_not_define_ties():
    r = np.array([[10000., 9000, 8000]]*4)
    a = reconstruct(r)
    assert a['worst_posterior_tv_bounds'].min() == 1
    assert a['selected'][0].sum() < 81
