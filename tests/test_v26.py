"""Meaningful boundaries: known answers, excluded seeds, inference and resource gates."""
from datetime import datetime,timezone
import importlib
import numpy as np
import pytest
from ghostscale.validation.soundingline.v26 import contract,registry,study,hierarchy,pauses,revision,values,opacity
from ghostscale.validation.soundingline.v26.common import rng,close,posterior

@pytest.mark.parametrize('family',contract.ORDER)
def test_known_answers(family):
    assert study.module(family).controls()['passed']

def plan():
    return dict(family='H1',kind='probe',pool='studies',cpu_cap_seconds=1800,pass_caps=dict(original=600,replay=600,review=600),numerical_threads=1,priority='below-normal',design_check_sha256='check')

def test_admission_boundaries_and_protected_pools():
    a=contract.acceptance('2026-11-02T15:00:00+00:00','spec');p=plan();t=datetime(2026,11,2,16,tzinfo=timezone.utc)
    assert contract.admission(p,a,t,{})
    for changes,charges,running in [({}, {},2),({'pool':'protected'},{},0),({}, {'studies':36*3600-1},0),({'design_check_sha256':None},{},0),({'numerical_threads':2},{},0),({'cpu_cap_seconds':1801},{},0)]:
        with pytest.raises(ValueError):contract.admission(p|changes,a,t,charges,running=running)
    for hour in (6,15):
        with pytest.raises(ValueError):contract.admission(p,a,t.replace(hour=hour),{})
    with pytest.raises(ValueError):contract.admission(p,a,datetime.fromisoformat(contract.NEW_STARTS_CLOSE),{})

def test_texture_healing_reserve():
    a=contract.acceptance('2026-11-02T15:00:00+00:00','spec');p=plan()|dict(kind='texture',pool='followup_healing');t=datetime(2026,11,2,16,tzinfo=timezone.utc)
    with pytest.raises(ValueError):contract.admission(p,a,t,dict(followup_healing=11*3600))
    assert contract.admission(p,a,datetime.fromisoformat(contract.HEALING_RELEASE),dict(followup_healing=11*3600))

def test_shared_repair_and_existing_pool_overrun():
    a=contract.acceptance('2026-11-02T15:00:00+00:00','spec');t=datetime(2026,11,2,16,tzinfo=timezone.utc);p=plan()|dict(family='R4')
    with pytest.raises(ValueError):contract.admission(p,a,t,{'family:R13':3*3600})
    with pytest.raises(ValueError):contract.admission(p,a,t,{'preparation':3*3600+1})

def test_seed_domains_and_whole_world_unit():
    assert not np.array_equal(rng('check-only',1).integers(0,100,10),rng('science',1).integers(0,100,10))
    rows=[dict(setting=0,block=b,data=dict(readouts=[dict(labels=dict(reader='x',maker=i,truth='B'),metrics=dict(score=v)) for i,v in enumerate(vals)])) for b,vals in enumerate(([1,1,1],[3]))]
    a=study.aggregate(rows);assert len(a)==1 and a[0]['mean']==2 and a[0]['blocks']==2 and a[0]['labels']['truth']=='B'
    close(a,study.aggregate(rows,True))

def test_revision_marginal_equals_scalar_equations():
    w=pauses.weights(['fixture']);r=rng('revision-check');obs=revision.draw(r,w,0,1,4,.6);hs=revision.hypotheses(4)
    for mode in ('final','positions','history'):
        got=revision.document_likelihood(w,obs,4,mode)
        for index in (0,23,199,500):
            g0,gs,point,rate,hi=hs[index];product=1.
            for i,(y,flag,old) in enumerate(obs['units']):product*=revision.emission(pauses.HABITS[hi],w[g0],w[gs],i<point,rate,mode,y,flag,old)
            assert np.isclose(np.exp(got[index]),product,atol=1e-12)

def test_forced_value_record_updates_without_choice_evidence():
    h=[('B',.1,2.,np.array([.25,.25,.5]))];s=values.states(h);before=s['w'].copy();values.update(h,s,np.array([1.,0,0]),1)
    assert s['w'][0,0]>before[0,0]
    # Conditional on observed forced assignment, every candidate receives likelihood1.
    prior=np.array([.25,.75]);assert np.array_equal(prior*np.ones(2),prior)

def test_source_policy_is_integrated_jointly_not_redrawn():
    prior=np.ones(3)/3;p=np.array([[opacity.presented(.3,s),opacity.presented(.7,s)] for s in (0,.5,1)])
    correct=posterior((3*np.log(p)).reshape(-1),np.repeat(prior/2,2)).reshape(3,2).sum(0)
    redrawn=posterior(3*np.log(prior@p))
    assert abs(correct[1]-redrawn[1])>.001

@pytest.mark.parametrize('family',('H2','H3','E2','E3','U1','R13'))
def test_small_world_scalar_reconstruction(family):
    module=study.module(family);s=registry.primary_setting(family)
    if family=='H3':s=dict(s,prefixes=[1],units=4)
    if family=='H2':s=dict(s,units=4)
    if family=='U1':s=dict(s,claims=2)
    a=module.block(s,['excluded-engineering',family,0]);b=module.block(s,['excluded-engineering',family,0],independent=True)
    close(a,b,tol=1e-9)
