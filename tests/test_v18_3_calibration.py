import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_3 import calibration as C,world as W


def test_credible_coverage_known_and_deliberately_reversed_model():
    likelihood=np.array([[.99,.01],[.01,.99]])
    joint=(likelihood[:,:,None]*likelihood[:,None,:]/2).reshape(2,-1)
    assert C.credible_coverage(joint,joint)['coverage']>=.95
    assert C.credible_coverage(joint,joint[::-1])['coverage']<.03
    assert C.native_coverage(W.make_world(0,90871),'matched')['coverage']>=.95-1e-12
    with pytest.raises(ValueError):C.credible_coverage(joint*2,joint)


def test_reliability_exact_and_overconfident_answers():
    truth=np.tile([.6,.4],(20,1))
    assert C.reliability(truth,truth)['expected_absolute_calibration_gap']==0
    result=C.reliability(np.tile([.9,.1],(20,1)),truth)
    assert abs(result['expected_absolute_calibration_gap']-.3)<1e-12
    assert result['overconfidence']>0
