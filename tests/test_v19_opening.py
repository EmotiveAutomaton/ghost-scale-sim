import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.algebra import audit, controls, bank_error
from ghostscale.validation.soundingline.v19.retained import paired_interval


def test_identification_controls_distinguish_truncation_from_null():
    assert all(controls().values())


def test_exact_bank_known_answer_and_noisy_amplification():
    atoms=np.eye(3);target=np.array([[1.,0.],[.5,.5],[0.,1.]])
    receipt,mapping=audit(atoms,target);p=np.array([.2,.3,.5])
    exact=bank_error(atoms,target,p,p@atoms,mapping)
    noisy=bank_error(atoms,target,p,p@atoms+[.1,0,-.1],mapping)
    assert exact['exact_bank_max_readout_error']<1e-12
    assert exact['target_weighted_squared_error']==0
    assert noisy['target_weighted_squared_error']==pytest.approx(.02)


def test_constant_paired_difference_has_zero_interval_width():
    result=paired_interval(np.full(48,-.125))
    assert result['mean']==result['low']==result['high']==-.125
    assert result['lineages']==48


def test_zero_bank_cannot_predict_distinguishing_target():
    receipt,_=audit(np.full((3,2),.5),np.eye(3))
    assert receipt['full_numerical_rank']==1
    assert receipt['tolerance_sweep'][0]['target_span_residual']>.5


def test_invalid_law_rejected():
    with pytest.raises(ValueError):audit(np.eye(3),np.eye(2))
