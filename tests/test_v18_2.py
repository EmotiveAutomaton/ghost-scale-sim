"""Known-answer, boundary and independent replay checks, not scientific criteria."""
import json
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_2 import model as m, verify as v


def test_independent_enumeration_and_acquired_expertise():
    w=m.world('test',0)
    for state in m.STATES:
        for signal in (None,0,1):
            c=m.context(None,signal,1,.4)
            programs,p=v.reference(w,state,c)
            assert programs==list(m.PROGRAMS)
            assert np.allclose(p,m.policy(w,state,c),atol=1e-12)
            assert abs(sum(p)-1)<1e-12
    assert m.library(tuple(map(tuple,w['groups'])),0)==()
    assert len(m.library(tuple(map(tuple,w['groups'])),1))==1
    p0=m.policy(w,(0,1,0,0),m.context(0,0))
    p1=m.policy(w,(1,1,0,0),m.context(0,0))
    assert sum(p0[i] for i,p in enumerate(m.PROGRAMS) if len(p)==3)==0
    assert sum(p1[i] for i,p in enumerate(m.PROGRAMS) if len(p)==3)>0


def test_reference_posterior_boundary_and_temporal_separation():
    case=m.make_case('admission',0)
    assert v.check_case(case)['passed']
    for tier in m.TIERS:
        payload=m.public_packet(case,tier);public=json.loads(payload)
        assert np.allclose(v.posterior(public),m.infer(payload)['posterior'])
        public['truth']=case['truth']
        with pytest.raises(ValueError):m.infer(m.canonical(public))
    before=m.public_packet(case,'process-history')
    case['truth']['future_state']=[0,0,0,0];case['probes'][0]['observed']['artifact']=15
    assert before==m.public_packet(case,'process-history')


def test_perspective_and_ambiguity_known_answers():
    w=m.world('test',0);s=(1,2,0,0)
    assert np.array_equal(m.policy(w,s,m.context(0,0,0)),m.policy(w,s,m.context(0,0,1)))
    assert not np.allclose(m.policy(w,s,m.context(0,0,1)),m.policy(w,s,m.context(0,1,1)))
    # Known goal and received signal hide default goal and prior belief completely.
    assert np.array_equal(m.policy(w,(1,2,0,0),m.context(0,1)),m.policy(w,(1,2,1,1),m.context(0,1)))
    assert not np.allclose(m.policy(w,(1,2,0,0),m.context(None,None)),m.policy(w,(1,2,1,0),m.context(None,None)))
