import numpy as np
from ghostscale.validation.soundingline.v21.answer_policy import frontier,decide,assess,controls
from ghostscale.validation.soundingline.v21.answer_policy_check import reference_frontier,check


def test_known_answers():
    assert all(controls().values())


def test_every_subset_against_scalar_enumeration():
    p=np.random.default_rng(71000850).dirichlet(np.ones(128),size=4)
    m,v,q=frontier(p)
    for i,row in enumerate(p):
        a,b,c=reference_frontier(row)
        assert np.array_equal(m[i],a) and np.array_equal(v[i],b)
        assert np.max(abs(q[i]-c))<1e-12
    for ratio in (1.,2.,5.,10.):check(p,np.array([0,31,99,127]),assess(decide(m,v,q,ratio),np.array([0,31,99,127]),ratio),ratio)


def test_joint_partial_answer_and_no_truth_in_decision():
    p=np.zeros((1,128));p[0,[0,1]]=.5
    m,v,q=frontier(p);a=decide(m,v,q,10.)
    assert a['asserted'][0]==6 and a['mask'][0]==126 and a['confidence'][0]==1
    assert assess(a,np.array([0]),10.)['loss'][0]==assess(a,np.array([1]),10.)['loss'][0]
