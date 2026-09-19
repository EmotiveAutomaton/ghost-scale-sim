import numpy as np
from ghostscale.validation.soundingline.v18_3 import purpose_readout as R,purpose_data as D,world as W


def test_new_purpose_readout_balanced_null_and_known_positive():
    target=np.array([D.roles(np.eye(len(W.STATES))[i]) for i in range(len(W.STATES))])
    null=np.ones((len(target),4));model,_=R.fit(null,target,null,target)
    p=R.predict(model,null)
    assert np.allclose(p,target.mean(0))
    assert abs(R.loss(target,p)-(np.log(3)+3*np.log(2))/4)<1e-12
    model,_=R.fit(target*4,target,target*4,target)
    p=R.predict(model,target*4)
    assert R.loss(target,p)<.1
    assert all(np.array_equal(p[:,s].argmax(1),target[:,s].argmax(1)) for s in R.SLICES)
