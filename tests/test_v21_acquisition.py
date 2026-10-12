import numpy as np
from ghostscale.validation.soundingline.v21.acquisition import controls,describe,choose
from ghostscale.validation.soundingline.v21.acquisition_check import reference


def test_known_answer_information_and_cost():
    assert all(controls().values())
    joint=np.array([[.5,0],[0,.5]])
    info,uncertainty=describe(joint)
    assert np.isclose(info,np.log(2)) and np.isclose(uncertainty,np.log(2))
    assert np.allclose(reference(joint)[2],np.eye(2))


def test_random_action_trap_and_independent_reference():
    independent=np.outer([.2,.8],np.ones(8)/8)
    assert abs(describe(independent)[0])<1e-12
    information=np.array([.3,.1,.05,0]);entropies=np.array([.5,1,.7,np.log(8)])
    choices=choose(information,entropies,0,1)
    assert choices==[0,3,2,1,-1]
    rng=np.random.default_rng(53);joint=rng.random((9,7));joint/=joint.sum()
    assert np.allclose(describe(joint),reference(joint)[:2],atol=1e-12)
