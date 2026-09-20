from copy import deepcopy
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_4 import family_expansion as F


def test_live_and_placebo_known_answers():
    assert all(F.controls().values())
    p=np.array([[.3,.7],[.8,.2]]);lik=np.array([[.4,.9],[.7,.1]])
    actual,scores=F.update(p,np.zeros(2),lik)
    for m in range(2):
        evidence=sum(p[m,s]*lik[m,s] for s in range(2))
        assert np.allclose(actual[m],[p[m,s]*lik[m,s]/evidence for s in range(2)])
        assert scores[m]==pytest.approx(math.log(evidence))


def test_copy_invariance_no_purchase_and_from_start_equivalence():
    a,_=F.make_case(981,0,'new-rule','old-first',1,8)
    b,_=F.make_case(981,0,'new-rule','old-first',3,8)
    for method in F.METHODS:assert F.read_stream(a,method)==F.read_stream(b,method)
    for paid,base,expanded in [('paid-select-1.5','select','expanded-select'),('paid-mixture-1.5','mixture','expanded-mixture')]:
        blocked=F.read_stream(a,paid,allow_expansion=False);original=F.read_stream(a,base)
        early=F.read_stream(a,paid,force_at=0);ceiling=F.read_stream(a,expanded)
        for key in ('trace','final','final_state_weights','final_log_evidence','likelihood_evaluations'):
            assert blocked[key]==original[key]
            assert early[key]==ceiling[key]


def test_prefix_causality_conflicting_copy_and_order_invariant_posterior():
    a,_=F.make_case(982,2,'outside-menu','old-first',1,16)
    b=deepcopy(a);b['history'][-1]['program']=[];b['history'][-1]['artifact']=0
    for method in F.METHODS:
        x=F.read_stream(a,method);y=F.read_stream(b,method)
        assert x['trace']==y['trace']
    c,_=F.make_case(982,2,'outside-menu','composed-first',1,16)
    for method in ('fixed','select','mixture','expanded-select','expanded-mixture'):
        x=F.read_stream(a,method);y=F.read_stream(c,method)
        assert np.allclose(x['final_state_weights'],y['final_state_weights'],atol=1e-12)
        assert np.allclose(x['final_log_evidence'],y['final_log_evidence'],atol=1e-12)
    b=deepcopy(a);b['history'].append(dict(b['history'][0],artifact=15))
    with pytest.raises(ValueError,match='conflicting'):F.read_stream(b,'mixture')


def test_delayed_purchase_replays_history_and_charges_retroactive_work():
    a,_=F.make_case(985,0,'new-rule','old-first',1,16)
    paid=F.read_stream(a,'paid-mixture-1.5',force_at=8)
    base=F.read_stream(a,'mixture');expanded=F.read_stream(a,'expanded-mixture')
    assert paid['purchase_step']==8 and paid['trace'][:8]==base['trace'][:8]
    assert paid['final']==expanded['final']
    assert paid['likelihood_evaluations']==expanded['likelihood_evaluations']==3*len(F.W.STATES)*16
    # Exercise the automatic sensor, not just the forced timing diagnostic.
    automatic=F.read_stream(a,'paid-mixture-1.5')
    candidates=[t for t in range(8,16) if math.fsum(automatic['sensor_surprises'][t-4:t])/4>1.5]
    assert automatic['purchase_step']==(min(candidates) if candidates else 17)


def test_scalar_native_cost_batch_reconstruction_and_corruption():
    u=F.unit(983,cell=2,kind='new-rule',order='interleaved',copy_span=3,length=8)
    assert F.verify(u)
    bad=deepcopy(u);bad['rows'][0]['trace'][0]['prediction'][0]+=.1
    with pytest.raises(ValueError):F.verify(bad)
    bad=deepcopy(u);bad['rows'][0]['likelihood_evaluations']+=1
    with pytest.raises(ValueError):F.verify(bad)


def test_frozen_dispatch_means_and_whole_unit_replay(tmp_path):
    from ghostscale.validation.soundingline.v18_4 import runtime as R
    from ghostscale.validation.soundingline.v18_3.io import write,file_digest
    from runners.replay_v18_4 import finite
    request=dict(family='R1',index=984,cell=0,kind='in-family',order='old-first',copy_span=1,length=8)
    write(tmp_path/'PLAN.json',dict(design=dict(units=[request],block_size=1)))
    u=R.dispatch(request);R.keep(tmp_path,'block-000000',[u],0.,0.);R.aggregate(tmp_path)
    write(tmp_path/'COMPLETE.json',dict(blocks=['block-000000'],summary_sha256=file_digest(tmp_path/'SUMMARY.json')))
    proof=finite(tmp_path);assert proof['independent_means']==len(F.METHODS)*len(F.METRICS)
    assert proof['whole_unit_replays']==[0]
