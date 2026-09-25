import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.schedule_envelope_review import reconstruct
from ghostscale.validation.soundingline.v19.likelihood_envelope_review import check_arrays
from ghostscale.validation.soundingline.v19.schedule_likelihood_envelope import evaluate, fixture


@pytest.mark.parametrize('dtype', ['float64', 'float32', 'float16'])
def test_all_fields(dtype):
    law = np.random.default_rng(951).uniform(.00001, 1, (16, 4, 8)); law /= law.sum(-1, keepdims=True)
    check_arrays(evaluate(law, dtype), reconstruct(law, dtype))


@pytest.mark.parametrize('field', ['context_log_ranges', 'attaining_endpoints', 'schedule_log_ranges', 'schedule_posterior_tv_bounds', 'schedule_0_8', 'schedule_1_128'])
def test_corruptions(field):
    law = fixture(); r = evaluate(law, 'float16')
    r[field].flat[0] = (not r[field].flat[0]) if r[field].dtype == bool else r[field].flat[0]+1
    with pytest.raises(AssertionError): check_arrays(r, reconstruct(law, 'float16'))


def test_exact_and_support_loss():
    check_arrays(evaluate(np.full((16, 4, 8), .125), 'float16'), reconstruct(np.full((16, 4, 8), .125), 'float16'))
    law = np.zeros((16, 4, 8)); law[:, :, 0] = 1-1e-9; law[:, :, 1] = 1e-9
    r = reconstruct(law, 'float16'); check_arrays(evaluate(law, 'float16'), r)
    assert np.all(r['schedule_posterior_tv_bounds'] == 1)
