import copy
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_3 import world as W,active,dynamics,provenance,revision,verify,runtime
from ghostscale.validation.soundingline.v16.records import read,write,file_digest
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v18_3.native import ProcessClock


def test_native_clock_and_factor_pairing():
    import os
    if os.name=='nt':
        clock=ProcessClock(os.getpid())
        try:assert clock.seconds()>=0
        finally:clock.close()
    a=W.make_world(0,17);b=W.make_world(15,17)
    assert a['groups']==b['groups'] and a['price']==b['price'] and a['temperature']==b['temperature']


def test_native_acquisition_and_independent_execution():
    for coupled in (False,True):
        w=dict(W.make_world(),coupled=coupled)
        for skill in (1,2):
            acq=W.acquisition(tuple(map(tuple,w['groups'])),skill,coupled)
            assert acq.library==(tuple(w['groups'][2-skill if coupled else skill-1]),)
            assert all(verify.execute(p)==target for p,target in zip(acq.attempts,acq.targets))
    target=7
    assert W.enact((0,1),target)['success']
    assert not W.enact((),target)['success']


def test_independent_finite_policy_all_architectures():
    for cell in range(16):
        w=W.make_world(cell,0)
        for s in (0,7,16,23):
            for c in (W.context(),W.context(goal=1,budget=1),W.context(signal=0,offered=[0,2,3])):
                p=W.matrix(w,c)[s]
                verify.distribution(p)
                assert np.allclose(p,verify.reference_policy(w,s,c),atol=1e-12,rtol=0)


def test_ambiguity_and_discriminating_intervention():
    w=dict(W.make_world(),endogenous=False)
    a=W.STATES.index((1,0,0,0));b=W.STATES.index((1,1,1,0))
    assert np.allclose(W.matrix(w,W.context())[a],W.matrix(w,W.context())[b])
    assert not np.allclose(W.matrix(w,W.context(signal=0))[a],W.matrix(w,W.context(signal=0))[b])


def test_public_boundary_and_future_taint():
    w=W.make_world();h=W.initial_history(w,1,W.rng('boundary'))
    payload=W.packet(w,h);p=W.posterior(payload)
    changed=json.loads(payload);changed['truth']=1
    with pytest.raises(ValueError):W.posterior(W.canonical(changed))
    changed=json.loads(payload);changed['history'][0]['future']=2
    with pytest.raises(ValueError):W.posterior(W.canonical(changed))
    assert np.array_equal(p,W.posterior(payload))
    assert W.cross_entropy([1.,0.],[0.,1.])==float('inf')
    assert W.loss_record(float('inf'))=={'value':None,'infinite':True}


def test_no_information_access_normalization_and_query_permutation():
    w=W.make_world();weights=np.ones(len(W.STATES)*3)/(len(W.STATES)*3)
    tables=active.bank(w,uninformative=True)
    for t in tables:assert np.allclose(t.sum(axis=1),1.)
    for purpose in ('prediction','historical-state'):
        gains=active.expected_gains(weights,tables,w,purpose,7)
        assert np.max(abs(gains))<1e-12
    tables=active.bank(w)
    gains=active.expected_gains(weights,tables,w,'prediction',7)
    assert np.allclose(gains[::-1],active.expected_gains(weights,tables[::-1],w,'prediction',7))


def test_complete_active_unit_equal_evidence_and_failed_access():
    unit=active.unit(0,mode='query-dependent',split='pilot')
    assert len(unit['rows'])==len(active.METHODS)*len(active.PURPOSES)
    assert verify.check_unit(unit,True)['executions']>0
    for row in unit['rows']:
        assert row['costs']['attempted_queries']>=row['costs']['responses']
        assert row['same_evidence_direct_error']<1e-12


def test_transition_preserves_slow_roles_and_dynamics():
    t=W.transition('fast',1.)
    assert np.allclose(t.sum(axis=1),1.)
    for i,a in enumerate(W.STATES):
        for j,b in enumerate(W.STATES):
            if a[0]!=b[0] or a[3]!=b[3]:assert t[i,j]==0
    unit=dynamics.unit(0,condition='return',split='pilot')
    assert verify.check_unit(unit,True)['distributions']==len(dynamics.METHODS)*32
    assert unit['evaluator']['states'][0]==unit['evaluator']['states'][-1]
    assert unit['evaluator']['states'][0]!=unit['evaluator']['states'][unit['evaluator']['change']]


def test_copy_control_and_independent_correction():
    p=provenance.posterior([1]*24,[(0,)*24])
    naive=provenance.posterior([1]*24,[tuple(range(24))])
    assert abs(p[1]-.8)<1e-10
    assert naive[1]>.999
    for mode in provenance.MODES:
        unit=provenance.unit(1,roots=2,copies=3,mode=mode,split='pilot')
        assert verify.check_unit(unit)['distributions']>0
    one=provenance.unit(1,roots=2,copies=1,mode='copied',split='pilot')
    many=provenance.unit(1,roots=2,copies=6,mode='copied',split='pilot')
    assert one['evaluator']['root_values']==many['evaluator']['root_values']
    assert one['public']['independent_correction']==many['public']['independent_correction']


def test_revision_orders_share_evidence_and_future():
    a=revision.unit(0,order='early',split='pilot');b=revision.unit(0,order='late',split='pilot')
    assert a['public']==b['public'] and a['evaluator']==b['evaluator']
    assert a['rows'][0]==b['rows'][0]
    assert a['revision']['prefix_length']==0 and b['revision']['prefix_length']==8
    assert verify.check_unit(a,True)['distributions']>0


def prepared(tmp_path,expired=False):
    campaign=tmp_path/'campaign';root=campaign/'fixture';source=tmp_path/'source'
    write(campaign/'ACCEPTANCE.json',dict(prior_cpu_seconds=0,cumulative_cpu_ceiling_seconds=1000,
          report_start=(datetime.now(timezone.utc)+timedelta(hours=-1 if expired else 1)).isoformat()))
    files={p:file_digest(runtime.REPO/p) for p in runtime.source_files()}
    design=dict(block_size=1,units=[dict(family='C',index=i,roots=1,copies=2,mode='copied',split='pilot') for i in range(2)])
    runtime.freeze(source,root,design,dict(passed=True,sources=files))
    return campaign,root,source


def test_resume_immutable_corruption_and_cutoff(tmp_path):
    campaign,root,source=prepared(tmp_path)
    assert runtime.run(root,campaign,max_blocks=1)=='checkpointed'
    first=(root/'raw/block-000000_points.json.gz').read_bytes()
    assert runtime.run(root,campaign)=='complete'
    assert first==(root/'raw/block-000000_points.json.gz').read_bytes()
    assert runtime.run(root,campaign)=='complete'
    (root/'raw/block-000000_points.json.gz').write_bytes(b'broken')
    with pytest.raises(ValueError,match='corruption'):runtime.run(root,campaign)
    campaign2,root2,_=prepared(tmp_path/'expired',True)
    assert runtime.run(root2,campaign2)=='resource_cutoff'
    assert not (root2/'COMPLETE.json').exists()


def test_source_change_and_owner_fail_closed(tmp_path):
    campaign,root,_=prepared(tmp_path)
    plan=read(root/'PLAN.json');plan['sources']['runners/run_v18_3.py']='bad'
    write(root/'PLAN.json',plan,immutable=False)
    with pytest.raises(ValueError,match='source identity'):runtime.run(root,campaign)
    with local_owner(campaign/'scientific-worker-owner'):
        with pytest.raises(RuntimeError):runtime.run(root,campaign)
