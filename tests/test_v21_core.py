import numpy as np
import pytest
from ghostscale.validation.soundingline.v21 import world as W,oracle as O,metrics as M
from ghostscale.validation.soundingline.v20 import world as V,checker as C,studies as S

def test_vector_world_matches_independent_fraction_checker():
    for shift in ('native','changed-tool'):
        w=W.law(71000001,shift);rows=W.enumerate_law(w)
        assert C.verify_world(rows,w,W.MAKERS,W.CONTEXTS)['trajectories']==24576

def test_regression_v20_probability():
    w=V.law(97);assert np.allclose(W.enumerate_law(w),V.enumerate_world(97),rtol=1e-12,atol=2e-16)

def test_supplied_posterior_independently_checked():
    rows=W.enumerate_world(71000002);chosen=W.sampled(rows,7,'engineering')
    x,_=W.inputs(rows,chosen,'context',4,'eng')
    lp,_=O.posterior(x,'context',[W.law(71000002)])
    assert C.verify_posterior(rows,x,'context',W.CONTEXTS,np.exp(lp))

def test_mixture_weights_evidence_not_posteriors():
    rows=W.enumerate_world(71000003);x,_=W.inputs(rows,W.sampled(rows,5,'mix'),'context',4,'mix')
    laws=[W.law(71000003),W.law(71000004)];layout=O.Layout(x,'context')
    a,b=[layout.log_joint(w) for w in laws];lp,_=O.posterior(x,'context',laws)
    joint=np.exp(a)+np.exp(b);joint/=joint.sum(1,keepdims=True)
    assert np.allclose(np.exp(lp),joint)
    assert not np.allclose(np.exp(lp),(np.exp(O.normalize_log(a))+np.exp(O.normalize_log(b)))/2,atol=1e-6)

def test_asymmetric_history_and_long_log_domain():
    rows=W.enumerate_world(71000005);x,_=W.inputs(rows,W.sampled(rows,3,'long'),'artifact',64,'long',history_tier='complete',history_observed=True)
    assert (x[:,:13][:,3:]==-1).all() and (x[:,13:].reshape(3,64,13)[:,:,10]>=0).all()
    lp,_=O.posterior(x,'artifact',[W.law(71000005)],history_tier='complete',history_observed=True)
    assert np.allclose(np.exp(lp).sum(1),1)

def test_law_prior_cardinality():
    assert sum(1 for _ in W.all_laws())==12500
    assert sum(1 for _ in W.all_laws(regime='wide'))==40500

def test_score_corruption_and_uncapped_floor():
    lp=np.array([[-40.,np.log1p(-np.exp(-40.))]]);y=np.array([0])
    # Independent arithmetic accepts arbitrary finite outcome count.
    a={'log_loss':np.array([40.]),'capped_log_loss':np.array([-np.log(1e-12)]),'brier':np.array([2*(1-np.exp(-40.))**2])}
    assert M.check_scores(lp,y,a)
    a['log_loss'][0]=39
    with pytest.raises(ValueError):M.check_scores(lp,y,a)

@pytest.mark.parametrize('history',[0,4,64])
def test_independent_law_mixture_asymmetric(history):
    from ghostscale.validation.soundingline.v21.checker import scalar_mixture
    laws=[W.law(71000010),W.law(71000011)];worlds=[W.enumerate_law(w) for w in laws]
    x,_=W.inputs(worlds[0],W.sampled(worlds[0],3,'ind'),'artifact',history,'ind',history_tier='complete',history_observed=True)
    lp,_=O.posterior(x,'artifact',laws,history_tier='complete',history_observed=True)
    expected=scalar_mixture(worlds,x,'artifact',W.CONTEXTS,history_tier='complete',history_observed=True)
    assert np.allclose(np.exp(lp),expected,atol=2e-12,rtol=2e-10)

def test_splits_and_budget_fail_closed():
    from ghostscale.validation.soundingline.v21.checker import validate_design
    from runners.watch_v21 import budget_fits
    with pytest.raises(ValueError):validate_design(dict(handler='headroom',lineages=[4],training_lineages=[4]))
    with pytest.raises(ValueError):validate_design(dict(handler='ceiling',lineages=[71900000]))
    assert not budget_fits(30,40,100,{'cpu_cap_seconds':31})
    assert budget_fits(30,40,100,{'cpu_cap_seconds':30})

def test_learned_readers_known_answer_and_permutation():
    from ghostscale.validation.soundingline.v21.readers import Direct,SharedMaker,GroupedTable
    rng=np.random.default_rng(8);x=rng.integers(0,2,(256,26),dtype=np.int8);y=x[:,0].astype(int)
    direct=Direct(x,y,x,y,max_epochs=35,patience=6)
    assert (direct.predict(x).argmax(1)==y).mean()>.98
    latent=SharedMaker(x,y,states=4,iterations=10)
    assert np.allclose(np.exp(latent.predict(x)).sum(1),1)
    table=GroupedTable(x,y);assert (table.predict(x).argmax(1)==y).mean()==1

def test_missing_records_independent_scalar_and_complete():
    from ghostscale.validation.soundingline.v21.records import generate,record_posterior,FIELDS,missing_prob
    rows=W.enumerate_world(71000020);chosen=W.sampled(rows,2,'records-test');rng=np.random.default_rng(3)
    rec=generate(chosen,.5,'event-dependent',rng);lp=record_posterior(rows,chosen[:,11:14],rec,.5,'event-dependent','aware')
    expected=[]
    for current,record in zip(chosen[:,11:14],rec):
        masses=np.zeros(128)
        for row in rows:
            if tuple(row[11:14])!=tuple(current):continue
            p=float(row[-1])
            for j,field in enumerate(FIELDS):
                prob=(.75 if row[5] else .25) if j==3 else .5
                if record[j]<0:p*=prob
                else:p*=int(row[field]==record[j])*(1-prob)
            masses[int(row[10])]+=p
        expected.append(masses/masses.sum())
    assert np.allclose(np.exp(lp),expected)
    full=generate(chosen,0,'independent',rng)
    a=record_posterior(rows,chosen[:,11:14],full,0,'independent','aware')
    b=record_posterior(rows,chosen[:,11:14],full,0,'independent','observed-record')
    assert np.allclose(np.exp(a),np.exp(b))

def test_unambiguous_id_permutation_and_corrupt_hidden_field():
    from ghostscale.validation.soundingline.v20.transport import packet,validate
    row=W.enumerate_world(71000030)[55];p=packet(row,'artifact',('x',1));q=packet(row,'artifact',('permuted',8))
    assert p['evidence']==q['evidence'] and p['artifact']==q['artifact']
    p['truth']=int(row[10])
    with pytest.raises(ValueError):validate(p)

def test_six_hour_checkpoints_do_not_reset_on_other_events(tmp_path):
    from datetime import datetime,timezone
    from runners.watch_v18_4_events import checkpoint_events
    from ghostscale.validation.soundingline.v18_3.io import read,file_digest
    config={'checkpoints':[{'id':'h0','kind':'six_hour_native_health','at':'2026-10-10T23:30:00+00:00'},
                           {'id':'h1','kind':'six_hour_native_health','at':'2026-10-11T05:30:00+00:00'}]}
    checkpoint_events(tmp_path,{},config,datetime(2026,10,11,0,tzinfo=timezone.utc));old=file_digest(tmp_path/'events/h0.json')
    checkpoint_events(tmp_path,{},config,datetime(2026,10,11,6,tzinfo=timezone.utc))
    assert file_digest(tmp_path/'events/h0.json')==old and (tmp_path/'events/h1.json').exists()

def test_sparse_current_shift_independent_mixture():
    from ghostscale.validation.soundingline.v21.checker import scalar_mixture
    laws=[W.law(71000040,'changed-tool'),W.law(71000041,'changed-tool')]
    worlds=[W.enumerate_law(w) for w in laws];past=[W.enumerate_law(dict(w,shift='native')) for w in laws]
    x,_=W.inputs(past[0],W.sampled(worlds[0],2,'sparse-shift'),'context',4,'sparse-shift',history_tier='complete',history_observed=True)
    lp,_=O.posterior(x,'context',laws,history_tier='complete',history_observed=True,shift='changed-tool',history_shift='native')
    q=scalar_mixture(worlds,x,'context',W.CONTEXTS,history_tier='complete',history_observed=True,history_worlds=past)
    assert np.allclose(np.exp(lp),q,rtol=2e-10,atol=2e-12)

def test_execution_kernel_adaptation_and_zero_label_invariance():
    from ghostscale.validation.soundingline.v21.adaptation import execution_weight,finetune
    from ghostscale.validation.soundingline.v21.readers import Direct
    table=np.full((128,8,8),.01);table[:,:,0]=.93
    x=np.zeros((16,13),dtype=np.int8);x[:,1]=1;y=np.zeros(16,int)
    weight=execution_weight(table,x,y);assert weight[1]>.999999
    model=Direct(x,y,x,y,max_epochs=3)
    assert np.array_equal(finetune(model,x[:0],y[:0]).predict(x),model.predict(x))
