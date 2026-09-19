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


def test_role_accuracy_respects_exact_and_numerical_ties():
    truth=np.array([D.roles(np.eye(len(W.STATES))[i]) for i in range(len(W.STATES))])
    p=np.tile(np.array([1/12]*3+[1/8]*6),(len(truth),1))
    role,whole=R.accuracies(truth,p)
    assert np.allclose(role,(1/3+3/2)/4) and np.allclose(whole,1/24)
    jitter=p.copy();jitter[:,0]+=1e-13;jitter[:,1]-=1e-13;jitter[:,3]+=1e-13;jitter[:,4]-=1e-13
    a,b=R.accuracies(truth,jitter)
    assert np.array_equal(role,a) and np.array_equal(whole,b)
    a,b=R.accuracies(truth,truth)
    assert np.all(a==1) and np.all(b==1)
