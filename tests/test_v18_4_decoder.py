import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_4 import decoder_readout as R,neural_data as D


def test_decoder_live_placebo_and_scalar_rulers():
    assert all(R.controls().values())
    y=np.eye(2);p=np.array([[.8,.2],[.3,.7]])
    assert R.loss(y,p)==pytest.approx((-math.log(.8)-math.log(.7))/2)
    with pytest.raises(ValueError):R.loss(y,np.ones((2,2)))


def test_train_only_transforms_determinism_and_nonlinear_interactions():
    x=np.tile(np.array([[-1,-1],[-1,1],[1,-1],[1,1.]]),(16,1))
    y=np.eye(2)[(x[:,0]*x[:,1]>0).astype(int)]
    linear,_=R.fit(x,y,x,y,'linear')
    nonlinear,_=R.fit(x,y,x,y,'nonlinear',width=64)
    again,_=R.fit(x,y,x,y,'nonlinear',width=64)
    assert R.loss(y,R.predict(linear,x))==pytest.approx(math.log(2))
    assert R.loss(y,R.predict(nonlinear,x))<.01
    assert all(np.array_equal(nonlinear[k],again[k]) for k in nonlinear)
    old_center=nonlinear['center'].copy();R.predict(nonlinear,x+999)
    assert np.array_equal(old_center,nonlinear['center'])


def test_permutation_preserves_question_strata_and_changes_alignment():
    y=np.arange(80).reshape(40,2);q=np.repeat([[0.,1.],[1.,0.]],20,axis=0)
    shuffled=R.permute_targets(y,q,2,44)
    assert not np.array_equal(shuffled,y)
    for start in (0,20):assert sorted(map(tuple,shuffled[start:start+20]))==sorted(map(tuple,y[start:start+20]))
    assert np.array_equal(shuffled,R.permute_targets(y,q,2,44))


def test_fresh_decoder_worlds_and_disjoint_splits():
    _,old=D.make_split('pilot',8,support='all')
    data,fresh=D.make_split('pilot',8,support='all',namespace='v18.4-decoder')
    _,dev=D.make_split('dev',8,support='all',namespace='v18.4-decoder')
    assert [r['world'] for r in old]!=[r['world'] for r in fresh]
    assert [r['world'] for r in fresh]!=[r['world'] for r in dev]
    assert np.allclose(data['target'].sum(1),1,atol=1e-6)
    assert all(c not in D.TRAIN_QUERIES+D.NEW_QUERIES for c in D.FAR_QUERIES)


def test_decoder_preparation_binds_both_verified_parents(tmp_path,monkeypatch):
    from ghostscale.validation.soundingline.v18_4 import decoder_data as P
    from ghostscale.validation.soundingline.v18_3.io import write,file_digest
    parents={}
    for label in ('even','odd'):
        parent=tmp_path/label;parent.mkdir();parents[label]=parent
        write(parent/'SUMMARY.json',{'fixture':label});write(parent/'COMPLETE.json',{'fixture':label})
        write(parent/'INDEPENDENT_REPLAY.json',dict(passed=True,summary_sha256=file_digest(parent/'SUMMARY.json')))
        best=parent/'neural/selected/BEST.pt';best.parent.mkdir(parents=True);best.write_bytes(label.encode())
        write(parent/'neural/COMPLETE.json',dict(predictions={'flat-seed701':{'test':{'selected':'selected'}}},
            fits={'selected':{'best_sha256':file_digest(best)}}))
    root=tmp_path/'decoder';(root/'reader').mkdir(parents=True)
    monkeypatch.setattr(D,'prepare',lambda *a,**k:{'fixture':True})
    result=P.prepare(root,parents,{})
    assert set(result['encoders'])=={'even-flat-seed701','odd-flat-seed701'}
    assert set(result['parent_complete_sha256'])=={'even','odd'}
    for label in parents:assert (root/'reader'/f'{label}-flat-seed701-ENCODER.pt').read_bytes()==label.encode()
    (parents['odd']/'neural/selected/BEST.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='weights changed'):P.prepare(root,parents,{})


def test_decoder_scores_use_declared_baseline_and_independent_targets(tmp_path):
    from ghostscale.validation.soundingline.v18_4.neural_runtime import score_E
    from ghostscale.validation.soundingline.v18_3.io import write,file_digest
    D.prepare(tmp_path/'data',train_per_cell=8,dev_per_cell=8,test_per_cell=8,namespace='v18.4-decoder-fixture')
    baseline='raw-history-linear-n128-aligned';reader=baseline+'-seed0'
    write(tmp_path/'PLAN.json',dict(design=dict(study='E-decoder',baseline_method=baseline)))
    predictions={};(tmp_path/'neural').mkdir()
    for condition in D.TEST_QUERIES:
        with np.load(tmp_path/'data'/f'{condition}-TRUTH.npz') as z:p=z['exact']
        target=tmp_path/'neural'/f'{condition}-PREDICTIONS.npz';np.savez_compressed(target,probabilities=p)
        predictions[condition]=dict(file=target.name,sha256=file_digest(target))
    write(tmp_path/'neural/COMPLETE.json',dict(predictions={reader:predictions},fits={},benchmarks={},environment={}))
    result=score_E(tmp_path)
    assert result['family']=='L3' and result['checks']['independent_target_and_exact_reconstructions']>0
    for condition in D.TEST_QUERIES:
        assert result['paired'][condition+'|'+baseline+' minus '+baseline]['expected_loss']['mean']==0
