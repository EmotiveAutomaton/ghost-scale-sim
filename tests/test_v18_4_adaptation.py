import itertools
import numpy as np
from ghostscale.validation.soundingline.v18_4 import adaptation as A


def test_filter_matches_enumerated_joint_paths():
    kernels=np.array([[[.8,.2],[.1,.9]],[[1.,0.],[0.,1.]]])
    p=np.array([[.3,.7],[.6,.4]]);likelihood=np.array([.2,.8])
    posterior,evidence=A.filter_step(p,kernels,likelihood)
    for m in range(2):
        joint=np.zeros(2)
        for i,j in itertools.product(range(2),repeat=2):joint[j]+=p[m,i]*kernels[m,i,j]*likelihood[j]
        assert np.allclose(posterior[m],joint/joint.sum(),atol=1e-14)
        assert abs(evidence[m]-joint.sum())<1e-14


def test_zero_information_and_no_change_known_answers():
    names,kernels=A.model_bank();p=np.ones((len(names),24))/24
    result,evidence=A.filter_step(p,kernels,np.ones(24)*.25)
    assert np.allclose(result,p,atol=1e-14)
    assert np.allclose(evidence,.25,atol=1e-14)
    assert np.array_equal(kernels[0],np.eye(24))


def test_trace_scalar_native_replay_and_copy_identity():
    unit=A.unit(184,cell=3,length=12,copy_span=3)
    assert unit['history'][0]==unit['history'][1]==unit['history'][2]
    assert A.verify(unit)
