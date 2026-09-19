import copy
import json
import numpy as np
from ghostscale.validation.soundingline.v18_2 import model as m, branches as b, learned as n, verify as v


def test_counterfactuals_and_off_model_branch():
    case=m.make_case('admission-branches',0)
    for branch in ('g1','g2','g3','g5'):
        rows=b.evaluate_branch(case,dict(branch=branch,dependency=True,paid_access=True))
        assert rows
        assert all(r['instrument'] in ('valid','model_mismatch') for r in rows)
        for r in rows:
            if 'probabilities' in r['result'] and r['result']['probabilities'] is not None:
                assert abs(sum(r['result']['probabilities'])-1)<1e-6
    rows=b.perspective(case,{})
    def predictions(condition):
        return [r['result']['probabilities'] for r in rows if r['condition']==condition and r['method']=='persistent']
    assert np.allclose(predictions('false-belief'),predictions('reader-only-correction'))
    assert not np.allclose(predictions('false-belief'),predictions('maker-correction'))
    # Force a known legal support violation, never a measured science outcome.
    w=case['world'];program=w['groups'][0]+[w['groups'][1][0]]
    case['history'][-1].update(program=program,artifact=v.replay(program))
    revised=b.family_revision(case,{})
    assert all(r['revision']['disconfirmed'] for r in revised)
    assert all(not r['result']['mismatch'] for r in revised if r['method']=='bounded-expansion')
    assert all(r['revision']['independent_context_sources']==1 for r in revised)


def test_neural_gradient_and_learned_positive_control():
    rng=np.random.default_rng(123)
    x=np.zeros((128,n.HISTORY+n.CURRENT),np.float32);x[:,0]=rng.normal(size=128)
    y=(x[:,0]>0).astype(int)
    for kind in ('split','flat'):
        net=n.Network(kind,3)
        grad=net.gradients(x[:8],y[:8])
        p=net.parameters[-1];old=p[0].copy();epsilon=1e-3
        def loss():
            q,_=net.forward(x[:8]);return float(-np.log(q[np.arange(8),y[:8]]).mean())
        p[0]=old+epsilon;a=loss();p[0]=old-epsilon;c=loss();p[0]=old
        assert abs((a-c)/(2*epsilon)-grad[-1][0])<2e-4
    models,receipt=n.train_pair(x,y,x,y,2,lambda:False,lambda **kw:None,epochs=35,cap=60)
    assert receipt['epochs_completed']==35
    for net in models.values():
        q,_=net.forward(x);assert (q.argmax(1)==y).mean()>.85
    assert max(receipt['parameters'].values())<100000
    assert max(receipt['parameters'].values())/min(receipt['parameters'].values())<1.1


def test_transform_and_feature_truth_boundary():
    case=m.make_case('feature-check',0);payload=m.public_packet(case,'process-history')
    f=n.features(payload)
    case['truth']['future_state']=[2,2,1,1]
    assert np.array_equal(f,n.features(m.public_packet(case,'process-history')))
    rng=np.random.default_rng(4);x=rng.normal(size=(80,70));y=rng.integers(0,16,80)
    predictions,record=n.geometry(x[:10],x,y,x[:10],y[:10],None,4)
    assert record['full_inverse_max_error']<1e-12
    assert all(np.allclose(q.sum(1),1) for q in predictions.values())
    w=m.world('test',0);ctx=m.context(2,0)
    assert np.allclose(m.policy(w,(1,1,0,0),ctx),v.reference(w,(1,1,0,0),ctx)[1])
