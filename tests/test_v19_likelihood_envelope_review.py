import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.likelihood_envelope_review import reconstruct, check_arrays
from ghostscale.validation.soundingline.v19.law_likelihood_envelope import evaluate, fixture


@pytest.mark.parametrize('dtype', ['float64', 'float32', 'float16'])
def test_independent_scalar_reconstruction(dtype):
    law = np.random.default_rng(471).uniform(.00001, 1, (16, 4, 8)); law /= law.sum(-1, keepdims=True)
    check_arrays(evaluate(law, dtype), reconstruct(law, dtype))


def test_exact_null():
    assert not reconstruct(np.full((16, 4, 8), .125), 'float16')['posterior_tv_bounds'].any()


def test_support_loss_and_impossible_observation():
    law = np.zeros((16, 4, 8)); law[:, :, 0] = 1-1e-9; law[:, :, 1] = 1e-9
    r = reconstruct(law, 'float16'); check_arrays(evaluate(law, 'float16'), r)
    assert np.all(r['posterior_tv_bounds'] == 1) and r['lost_support'].sum() == 64
    assert np.isnan(r['log_ranges'][:, 2:]).all()


@pytest.mark.parametrize('field', ['ratios', 'log_ratios', 'lower', 'upper', 'log_ranges', 'posterior_tv_bounds', 'lost_support'])
def test_corruption_fails(field):
    law = fixture(); r = evaluate(law, 'float16'); r[field].flat[0] += 1
    with pytest.raises(AssertionError): check_arrays(r, reconstruct(law, 'float16'))


def test_permutation():
    law = fixture(); a = reconstruct(law, 'float16'); b = reconstruct(law[::-1, ::-1, ::-1], 'float16')
    assert np.allclose(a['posterior_tv_bounds'], b['posterior_tv_bounds'], atol=1e-13, rtol=0)
