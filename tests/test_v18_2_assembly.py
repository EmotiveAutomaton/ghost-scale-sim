import json
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_2 import assembly_maker as a,model as m


def test_assembly_topology_execution_and_expertise():
    counts=[];live=False
    for split in ('train','dev','test'):
        w=a.world(split,0);programs,outputs,libraries=a.catalog(m.canonical(w).decode())
        counts.append(len(programs))
        assert len(libraries[0])==0 and len(libraries[1])>0
        for program,output in zip(programs,outputs):assert a.replay(w,program)==output
        for state in m.STATES:
            p=a.distribution(w,state,m.context(None,None))
            assert np.isfinite(p).all() and abs(p.sum()-1)<1e-12
        live|=not np.allclose(a.distribution(w,(0,1,0,0),m.context(0,0)),a.distribution(w,(1,1,0,0),m.context(0,0)))
    assert counts==[17,49,9]
    assert live
    with pytest.raises(ValueError):a.replay(a.world('test',0),[2,9])


def test_assembly_public_boundary_and_reference_contraction():
    case=a.make_case('assembly-admission',0,'test');payload=a.packet(case,0);public=json.loads(payload)
    programs,_,_=a.catalog(m.canonical(case['world']).decode())
    joint=[]
    for state in m.STATES:
        value=1/36
        for obs in case['history']:value*=a.distribution(case['world'],state,obs['context'])[programs.index(tuple(obs['program']))]
        joint.append(value)
    weights=np.array(joint)/sum(joint)
    expected=sum(weight*a.marginal(case['world'],a.distribution(case['world'],state,public['current'])) for weight,state in zip(weights,m.STATES))
    assert np.allclose(expected,a.infer(payload),atol=1e-12)
    public['truth']=case['truth']
    with pytest.raises(ValueError):a.infer(m.canonical(public))
    before=a.features(payload);case['truth']=[2,2,1,1]
    assert np.array_equal(before,a.features(a.packet(case,0)))


def test_selective_physical_dependency_and_shared_update():
    from ghostscale.validation.soundingline.v18_2.selective_assembly import evaluate
    unit=evaluate('selective-admission',0)
    assert len(unit['rows'])==48
    for row in unit['rows']:
        if '-weight-0.0-' in row['method']:
            assert row['library']==[] and row['beta_parameters']==[1.,9.]
        for output in row['outputs']:
            assert output['charged_total']<=256
            if output['execution']['successfully_stopped']:
                a.replay(row['world'],output['plan']['program'])
