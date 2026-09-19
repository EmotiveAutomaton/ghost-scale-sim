import numpy as np
import pytest
from runners.analyze_v18_3 import contrasts
from ghostscale.validation.soundingline.v18_3 import world as W


def test_factorial_contrasts_on_known_additive_and_interaction_world():
    f=2*np.array(W.FACTORS)-1
    response=3+2*f[:,0]-f[:,1]+.5*f[:,0]*f[:,1]
    effects=contrasts(response)
    assert effects['decision rule']==4
    assert effects['acquisition interference']==-2
    assert effects['decision rule x acquisition interference']==2
    assert all(abs(value)<1e-12 for name,value in effects.items() if name not in ('decision rule','acquisition interference','decision rule x acquisition interference'))
    with pytest.raises(ValueError):contrasts(response[:-1])
