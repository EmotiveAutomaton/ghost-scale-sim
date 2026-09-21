import numpy as np
from ghostscale.validation.soundingline.v19 import recursive_update as U
from ghostscale.validation.soundingline.v19 import probability_convergence as C
from ghostscale.validation.soundingline.v19 import probability_head as P

def test_update_known_answers_and_duplicate_identity():
    assert all(U.controls().values())

def test_reservoir_skips_copies():
    codes=np.arange(32)[None,:]%32;novel=np.ones((1,32),dtype=bool);novel[:,1]=False
    h,wi,wr=U.reservoir(codes,novel,0)
    assert np.array_equal(h[:,0],h[:,1])
    assert np.linalg.norm(wr,2)<=.5000000001

def test_convergence_keeps_objective_and_known_labels():
    assert all(C.controls().values())
    x=np.array([[1.,-1.],[1.,1.]])
    y=np.array([[.1,.9,.2,.3,.5],[.8,.2,.4,.4,.2]])
    w,trace=C.fit(x,y,[2,3]);obj,g,lp=P.objective_gradient(x,y,w,[2,3])
    assert abs(trace[-1]['objective']-obj)<1e-12
    assert np.max(abs(g))<1e-6
    assert np.allclose(np.exp(lp[:,:2]).sum(1),1)
    assert np.allclose(np.exp(lp[:,2:]).sum(1),1)

def test_update_fixture_complete_roles_and_denominators(tmp_path):
    cfg=dict(train_lineages=[-9101],development_lineages=[-9102],training_draws=[-9103],fit_seeds=[9104],streams_per_lineage=2,lengths=[8,32])
    result=U.run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert all(result['controls'].values()) and result['rows']==28
    with np.load(tmp_path/'reader/development--9103-independent.npz') as a:assert set(a.files)=={'codes','novel'}
    with np.load(tmp_path/'reader/train--9103-independent.npz') as a:assert set(a.files)=={'codes','novel','target','before_bank_teacher'}
    for condition in ('independent','copied'):
        with np.load(tmp_path/f'evaluator/development--9103-{condition}.npz') as a:
            assert np.allclose(a['target'].reshape(2,32,4,8).sum(-1),1)
            if condition=='copied':assert np.array_equal(a['target'][:,1::2],a['target'][:,::2])
        for length in (8,32):
            f=next(r for r in result['cells'] if r['condition']==condition and r['length']==length and r['arm']=='full-history')
            c=next(r for r in result['cells'] if r['condition']==condition and r['length']==length and r['arm']=='cached-history')
            assert f['loss']==c['loss'] and f['brier']==c['brier']
