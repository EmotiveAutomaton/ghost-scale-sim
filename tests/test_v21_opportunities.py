import numpy as np
from ghostscale.validation.soundingline.v21.opportunities import controls,law,channel,CONTEXTS,posterior,select,POLICIES
from ghostscale.validation.soundingline.v21.opportunities_check import reference

def test_known_answer_controls():
    assert all(controls().values())

def test_forced_context_cannot_identify_preference_and_free_context_can():
    known=law(71000831);contexts=np.array([[1,0.],[1,4.],[1,-4.]])
    q=posterior(channel(contexts,known),[0,1,2],[1,1,1])
    assert abs(q[:2].sum()-.5)<1e-14 and q[1]+q[3]>.5
    q=posterior(channel(CONTEXTS,known),[1],[1])
    assert q[2:].sum()>.5

def test_policy_prefixes_use_no_unselected_outcomes_and_saturate_same_evidence():
    known=law(71000832);lik=channel(CONTEXTS,known);responses=np.array([1,0,1,1,0,1,1,0]);full=[]
    for policy in POLICIES:
        chosen=[];ys=[];p=np.full(4,.25)
        for _ in range(8):
            i=select(policy,p,lik,CONTEXTS,chosen,np.arange(8)[::-1]);assert i not in chosen
            chosen.append(i);ys.append(responses[i]);p=posterior(lik,chosen,ys)
            assert np.allclose(p,reference(CONTEXTS[chosen],ys,known),atol=1e-14,rtol=0)
        full.append(p)
    assert np.allclose(full,full[0],atol=1e-14,rtol=0)

def test_numerically_tied_information_uses_smallest_index():
    known=law(71000835);contexts=np.array([[1,0.],[1,4.],[1,-4.]])
    likelihood=channel(contexts,known)
    for p in (np.full(4,.25),np.array([.1,.2,.3,.4])):
        assert select('target-information',p,likelihood,contexts,[],np.arange(3))==0
