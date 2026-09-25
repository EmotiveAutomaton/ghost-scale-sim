import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.context_precision_allocation import allocate
from ghostscale.validation.soundingline.v19.context_precision_review import reconstruct
from ghostscale.validation.soundingline.v19.likelihood_envelope_review import check_arrays


@pytest.mark.parametrize('ranges', [np.zeros((4, 3)), np.tile([.03, .02, 0], (4, 1)),
    np.random.default_rng(716).uniform(0, .04, (4, 3)), np.full((4, 3), math.inf),
    np.array([[10, 9, 0], [6, 0, 0], [6, 0, 0], [0, 0, 0.]])])
def test_independent_full_enumeration(ranges):
    check_arrays(allocate(ranges), reconstruct(ranges))


@pytest.mark.parametrize('field', ['stored_law_bytes', 'selected', 'dominated', 'posterior_tv_bounds', 'assignments'])
def test_corruption_rejected(field):
    r = np.random.default_rng(807).uniform(0, .04, (4, 3))
    actual = allocate(r); value = actual[field]
    value.flat[0] = not value.flat[0] if value.dtype == bool else value.flat[0]+1
    with pytest.raises(AssertionError): check_arrays(actual, reconstruct(r))


def test_infeasible_and_saturated_ties():
    r = np.array([[10000., 9000, 8000]]*4)
    a = allocate(r, (1023, 1280)); b = reconstruct(r, (1023, 1280))
    check_arrays(a, b)
    assert not b['selected'][0].any()
    assert b['posterior_tv_bounds'].min() == 1
    assert b['selected'][1].sum() == 4


def test_float_tie_identity_is_not_tolerance_identity():
    r = np.tile([.03, .02, 0], (4, 1))
    original = reconstruct(r)
    changed = r.copy(); changed[0, 1] = np.nextafter(changed[0, 1], 0.)
    # One ULP may disappear when summed; a bounded accumulated perturbation does not.
    changed[0, 1] -= 1e-14
    scalar = reconstruct(changed)
    assert original['selected'][0].sum() == 4
    assert scalar['selected'][0].sum() == 1
    assert np.allclose(original['accumulated_log_ranges'], scalar['accumulated_log_ranges'], atol=2e-12)
