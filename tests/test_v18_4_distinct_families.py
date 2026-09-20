from copy import deepcopy
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_4 import distinct_families as F


def test_known_distinct_laws_uniform_null_and_duplicate_class_prior():
    assert all(F.controls().values())
    predictions=np.array([[.8,.2],[.1,.9]])
    scores=np.log([.3,.7]);priors=np.array([.5,.5])
    mixed=F.combine(predictions,scores,priors)
    assert np.allclose(mixed,[.31,.69])
    assert np.array_equal(F.combine(predictions,scores,priors,True),predictions[1])
    duplicated=F.combine(predictions[[0,0,1]],scores[[0,0,1]],np.array([.25,.25,.5]))
    assert np.allclose(duplicated,mixed)
    assert np.allclose(F.combine(np.ones((2,2))*.5,scores,priors),[.5,.5])
    # Deliberately keep equal label priors instead: a known non-invariant prior.
    assert not np.allclose(F.combine(predictions[[0,0,1]],scores[[0,0,1]],np.ones(3)),mixed)


def test_catalog_fixture_has_behavioral_contrast_and_fixed_missing_truth():
    public,truth=F.make_case(992,0,'new-rule','old-first',1,8)
    a,b,c=F.catalog(public['world'])
    assert (a['rule'],b['rule'],c['rule'])==('softmax','satisficing','lexicographic')
    assert truth['world']==c
    assert np.max(abs(F.W.matrix(a,F.W.QUERIES[0]).mean(0)-F.W.matrix(b,F.W.QUERIES[0]).mean(0)))>.01
    _,other=F.make_case(992,0,'outside-menu','old-first',1,8)
    assert other['world'] not in (a,b,c)


@pytest.mark.parametrize('prior',['equal-class','duplicate-supplied'])
def test_scalar_prefix_reconstruction_and_corruption(prior):
    u=F.unit(993,cell=2,kind='new-rule',order='interleaved',copy_span=1,length=16,prior_mode=prior)
    assert F.verify(u)
    bad=deepcopy(u);bad['rows'][1]['trace'][1]['prediction']=[1/16]*16
    with pytest.raises(ValueError):F.verify(bad)
    bad=deepcopy(u);bad['rows'][0]['model_entropy']+=.1
    with pytest.raises(ValueError):F.verify(bad)


def test_past_only_trigger_copy_invariance_and_delayed_full_refit():
    a,_=F.make_case(994,8,'initial-alternative','old-first',1,16);a['prior_mode']='duplicate-supplied'
    b=deepcopy(a);b['history']=[h for o in a['history'] for h in [o,o,o]]
    for m in F.METHODS:assert F.read_stream(a,m)==F.read_stream(b,m)
    base=F.read_stream(a,'mixture');full=F.read_stream(a,'expanded-mixture')
    paid=F.read_stream(a,'paid-mixture-1.5',force_at=8)
    assert paid['trace'][:8]==base['trace'][:8] and paid['final']==full['final']
    assert paid['likelihood_evaluations']==full['likelihood_evaluations']
    later=deepcopy(a);later['history'][-1]['program']=[]
    assert F.read_stream(a,'paid-mixture-1.5')['trace']==F.read_stream(later,'paid-mixture-1.5')['trace']
    auto=F.read_stream(a,'paid-mixture-1.5')
    eligible=[t for t in range(8,16) if math.fsum(auto['sensor_surprises'][t-4:t])/4>1.5]
    assert auto['purchase_step']==(min(eligible) if eligible else 17)


def test_frozen_dispatch_replay(tmp_path):
    from ghostscale.validation.soundingline.v18_4 import runtime as R
    from ghostscale.validation.soundingline.v18_3.io import write,file_digest
    from runners.replay_v18_4 import finite
    request=dict(family='R2',index=995,cell=0,kind='in-family',order='old-first',copy_span=1,length=8,prior_mode='equal-class')
    write(tmp_path/'PLAN.json',dict(design=dict(units=[request],block_size=1)))
    R.keep(tmp_path,'block-000000',[R.dispatch(request)],0.,0.);R.aggregate(tmp_path)
    write(tmp_path/'COMPLETE.json',dict(blocks=['block-000000'],summary_sha256=file_digest(tmp_path/'SUMMARY.json')))
    proof=finite(tmp_path)
    assert proof['independent_means']==len(F.METHODS)*len(F.METRICS)
    assert proof['whole_unit_replays']==[0]
