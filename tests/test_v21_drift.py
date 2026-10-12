import numpy as np
from ghostscale.validation.soundingline.v21.drift import channels,infer,controls,STATES
from ghostscale.validation.soundingline.v21.drift_check import reference

def test_known_answer_controls():
    assert all(controls().values())

def test_scalar_reference_and_permutation_witness():
    c=channels(71000762)
    history=[0]*16+[3]*16+[1]*64
    for name in ('static','all-fast','all-slow','role-mixture','direct-chronological'):
        q,_=infer(history,c['emission'],name)
        r=reference(history,c['preference'],c['skill'],c['context'],name)
        assert np.allclose(q,r,atol=1e-12)
    a,_=infer(history,c['emission'],'static');b,_=infer(history[::-1],c['emission'],'static')
    assert np.allclose(a,b,atol=1e-12)
    a,_=infer(history,c['emission'],'all-fast');b,_=infer(history[::-1],c['emission'],'all-fast')
    assert np.max(np.abs(a-b))>.1

def test_role_transition_distinct_from_direct():
    # Known almost-exact observations: one-bit and three-bit changes differ.
    channel=np.full((8,8),.001);np.fill_diagonal(channel,.993)
    q,roles=infer([0]*24+[7]*24,channel,'role-mixture')
    direct,_=infer([0]*24+[7]*24,channel,'direct-chronological')
    assert direct[7]>q[7] and direct[7]>.99
    assert abs(roles.sum()-1)<1e-12
