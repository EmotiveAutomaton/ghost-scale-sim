import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.state_error import noisy_bank


def test_zero_state_error_has_no_noise():
    bank=np.tile(np.full(16,1/16),5)
    noisy,receipt=noisy_bank(bank,bank,np.random.default_rng(1))
    assert np.array_equal(noisy,bank)
    assert receipt['pre_projection_squared_error']==receipt['realized_squared_error']==0


def test_noise_matches_declared_error_before_probability_repair():
    exact=np.tile(np.full(16,1/16),5);learned=exact.copy();learned[0]+=.01;learned[1]-=.01
    noisy,r=noisy_bank(exact,learned,np.random.default_rng(2))
    assert r['pre_projection_squared_error']==pytest.approx(r['requested_squared_error'])
    assert np.min(noisy)>=0
    assert np.max(abs(noisy.reshape(5,16).sum(1)-1))<1e-12
