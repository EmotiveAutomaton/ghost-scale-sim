"""Adversarial known-answer controls for all V20 study consumers."""
import copy
from datetime import datetime,timezone
import numpy as np
import pytest
from ghostscale.validation.soundingline.v20 import world as W,checker as C,transport as T
from ghostscale.validation.soundingline.v20.readers import Readers,score
from ghostscale.validation.soundingline.v20.studies import reference,inputs,posterior,run_study
from ghostscale.validation.soundingline.v20.runtime import validate_design
from runners.watch_v20 import eligible

@pytest.mark.parametrize('lineage',[0,24,56])
@pytest.mark.parametrize('shift',['native','changed-tool','presentation-shift'])
def test_exhaustive_independent_transition_policy(lineage,shift):
    a=W.enumerate_world(lineage,shift);assert a.shape==(24576,15)
    assert C.verify_world(a,W.law(lineage,shift),W.MAKERS,W.CONTEXTS)['exact_mass_before_normalization']=='1'

@pytest.mark.parametrize('tier',W.TIERS)
@pytest.mark.parametrize('observed',[False,True])
def test_independent_history_posterior_and_all_arms_support(tier,observed):
    a=W.enumerate_world(1);s=W.sampled(a,64,'control');x,hh=inputs(a,s,tier,4,'control',observed)
    q,support=posterior(reference(a,tier,observed),x[:,:13],[W.project(a[h],tier,observed) for h in hh])
    assert C.verify_posterior(a,x,tier,W.CONTEXTS,q,observed)
    m=Readers(x,s[:,10].astype(int),epochs=2)
    for method in ('prior','template','direct-table','direct-mlp','factorized-mlp','structured-joint'):
        p=m.predict(x,method,support);assert np.max(p[~support],initial=0)==0
        C.probabilities(p);C.reaggregate(p,s[:,10].astype(int),score(p,s[:,10].astype(int)))

def test_corrupt_world_and_forecast_and_score_rejected():
    a=W.enumerate_world(2).copy();a[0,11]=1-a[0,11]
    with pytest.raises(ValueError):C.verify_world(a,W.law(2),W.MAKERS,W.CONTEXTS)
    for p in [[[.3,.3]],[[1.2,-.2]],[[float('nan'),1]]]:
        with pytest.raises(ValueError):C.probabilities(p)
    p=np.eye(128)[:2];y=np.array([0,1]);scores=score(p,y);scores['log_loss'][0]=3
    with pytest.raises(ValueError):C.reaggregate(p,y,scores)

def test_corrupt_posterior_rejected():
    a=W.enumerate_world(3);s=W.sampled(a,4,'corrupt');x=W.project(s,'context');q,_=posterior(reference(a,'context'),x,[])
    q[0]=1/128
    with pytest.raises(ValueError):C.verify_posterior(a,x,'context',W.CONTEXTS,q)

def test_exact_alias_and_cooccurring_goal_sets():
    w=C.exact_witnesses();assert w['factorized_impossible_mass']=='1/2';assert 'A+B' in w['cooccurring_goal_sets']
    a=W.enumerate_world(4);by={}
    for row in a:
        key=tuple(row[11:14]);by.setdefault(key,set()).add(int(row[2]))
    assert any(routes=={0,1} for routes in by.values())
    # Skill does not gate tool availability: all four skill/provenance cells exist.
    assert {(W.MAKERS[int(r[0])][1],int(r[2])) for r in a}=={(0,0),(0,1),(1,0),(1,1)}

@pytest.mark.parametrize('tier',W.TIERS)
def test_reader_boundary_and_anonymous_identity(tier):
    row=W.enumerate_world(5)[1234];p=T.packet(row,tier,(5,1234));assert T.validate(p)
    changed=copy.deepcopy(p);changed['case_id']='arbitrary-anonymous-id';assert T.validate(changed)
    assert changed['evidence']==p['evidence']
    for where in ('top','evidence','event'):
        bad=copy.deepcopy(p)
        if where=='top':bad['true_goal']=1
        elif where=='evidence':bad['evidence']['fields']['private_skill']=1
        else:
            bad=T.packet(row,'complete',(5,1234));bad['observed_events'][0]['payload']['true_goal']=1
        with pytest.raises(ValueError):T.validate(bad)

@pytest.mark.parametrize('branch',['G0','G1','G2','G3','G4','G5','G6','G7'])
def test_complete_consumers(tmp_path,branch):
    d=dict(branch=branch,tier='context',lineages=[24],training_lineages=[0,1],train_cases=128,test_cases=32,fit_seed=0,epochs=2,history=1,copied=True)
    result=run_study(tmp_path,d,lambda **kw:None);assert all(result['controls'].values())
    if branch in ('G1','G2','G7'):
        assert len(result['population'])==7
        assert (tmp_path/'reader/24_points.npz').exists()
    if branch=='G6':
        rows=result['population'];stay=[r for r in rows if r['update']=='unchanged'];assert len({r['metrics']['log_loss'] for r in stay})==1
    if branch=='G4':
        import json
        policy=json.loads((tmp_path/'policies/24.json').read_bytes());assert set(policy['learned'])<={0,1}
    if branch=='G5':
        rows={r['method']:r for r in result['population']};assert rows['unknown']['metrics']['unknown_mass']>0
        assert rows['generated-expansion']['metrics']['candidate_count']<=80
        assert rows['supplied-hypothesis-control']['metrics']['represented_truth_fraction']==1

def test_deadline_dependencies_and_lineage_leakage():
    now=datetime(2026,9,28,tzinfo=timezone.utc)
    assert not eligible(dict(requires=['a']),set(),now)
    assert not eligible(dict(not_before='2026-10-01T00:00:00+00:00'),set(),now)
    assert eligible(dict(requires=['a']),{'a'},now)
    d=dict(branch='G1',lineages=[24],training_lineages=[0],epochs=120,train_cases=128,test_cases=32,cpu_cap_seconds=600)
    assert validate_design(d)
    for override in [dict(lineages=[0]),dict(branch='unimplemented'),dict(branch='G7',split='discovery'),dict(epochs=999),dict(history=999)]:
        with pytest.raises(ValueError):validate_design(dict(d,**override))

def test_retained_v19_order_witness_is_regression_only():
    from ghostscale.validation.soundingline.v19.process_sufficiency import synthetic_controls
    assert all(synthetic_controls().values())

def test_secondary_score_corruption_and_zero_mass():
    p=np.eye(128)[:2];y=np.array([0,1]);m=score(p,y)
    for metric in ('tool_brier','goal_brier','order_brier','inspection_brier','confidence','unsupported_attribution','zero_truth_probability'):
        bad={k:v.copy() for k,v in m.items()};bad[metric][0]+=1
        with pytest.raises(ValueError):C.reaggregate(p,y,bad)

def test_long_history_masks_precede_logspace_normalization():
    x=np.tile(np.array([[0]*13,[1]*13],dtype=np.int8),(8,257));y=np.tile(np.array([0,127]),8)
    m=Readers(x,y,epochs=0);support=np.zeros((2,128),bool);support[:,1]=True
    p=m.predict(x[:2],'structured-joint',support);assert np.all(p[:,1]==1)

def test_finite_registered_forest_and_reserved_split():
    from runners.admit_v20 import designs
    opening,forest,confirmation=designs()
    assert {d['branch'] for d in opening}==set('G'+str(i) for i in range(7))
    assert len(opening+forest+confirmation)<1500
    assert len(confirmation)>0 and all(d['lineages']==list(range(96,128)) for d in confirmation)
    assert all(set(d['lineages']).isdisjoint(d['training_lineages']) for d in opening+forest+confirmation)
    assert all(d['train_cases']<=32768 for d in forest if d['history']==256)

def test_candidate_unknown_transport_normalization(tmp_path):
    d=dict(branch='G5',tier='context',lineages=[18],training_lineages=[0,1],train_cases=128,test_cases=32,fit_seed=0,epochs=2,history=0,candidate_budget=0)
    run_study(tmp_path,d,lambda **kw:None)
    for path in (tmp_path/'raw').glob('*transport_points.npz'):
        with np.load(path) as a:assert np.allclose(a['candidate_mass'].sum(1)+a['unknown_mass'],1)

def test_whole_lineage_bootstrap_not_rows():
    from ghostscale.validation.soundingline.v20.analysis import interval
    result=interval([.1,.2,-.1,.3]);assert result['lineages']==4
    assert result['lower']<=result['mean']<=result['upper']


def test_factorized_saturation_preserves_legal_low_probability():
    x=np.array([[0]*13,[1]*13],dtype=np.int8);m=Readers(x,np.array([0,127]),epochs=0)
    m.f2[:]=0;m.fb2[:]=1000
    support=np.zeros((2,128),bool);support[:,0]=True
    assert np.all(m.predict(x,'factorized-mlp',support)[:,0]==1)
    m.w2[:]=0;m.b2[:]=0;m.b2[127]=1000
    assert np.all(m.predict(x,'direct-mlp',support)[:,0]==1)
