import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import n2_robust as m

def test_floor_and_extreme_score_known_answers():assert m.controls()['passed']

def test_pool_refuses_unidentified_or_invalid_uncertainty():
    for errors in ([0,0,0],[float('nan'),1,2],[-1,1,2]):
        with pytest.raises(ValueError):m.robust_pool([0,1,2],errors)

def test_tempering_neutral_and_identity():
    for power in m.POWERS:assert m.loss(power*0,1)==m.loss(0,1)
    assert m.loss(512,0)==512
    assert 0<m.loss(512,1)<1e-200
    assert m.robust_pool([2,2,2],[1e-158,1,2])['mean']==2
