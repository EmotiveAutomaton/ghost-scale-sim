from copy import deepcopy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import purchase_evaluation as E
from ghostscale.validation.soundingline.v19 import purchase_calibration as C
from ghostscale.validation.soundingline.v18_4 import distinct_families as D

def fixture():
    p,t=D.make_case(-19731,2,'new-rule','interleaved',1);p['prior_mode']='equal-class'
    return p,t

def test_complete_forecasts_actions_trigger_and_cost_reconstruction():
    p,t=fixture();bars={'raw':3.,'centered':.5};rows,_=E.evaluate(p,t,bars)
    unit=dict(public=p,evaluator=t,rows=rows,length=32,copy_span=1,prior_mode=p['prior_mode'])
    assert E.verify(unit,bars)
    bad=deepcopy(unit);bad['rows'][-1]['trace'][11]['prediction'][0]+=.001
    with pytest.raises(ValueError):E.verify(bad,bars)
    bad=deepcopy(unit);bad['rows'][-1]['likelihood_evaluations']+=1
    with pytest.raises(ValueError):E.verify(bad,bars)
    bad=deepcopy(unit);bad['rows'][-1]['purchase_step']=31
    with pytest.raises(ValueError):E.verify(bad,bars)

def test_decision_ties_no_purchase_and_copy_identity():
    assert E.trigger([1.]*24,1.) is None and E.trigger([1.]*24,.99)==8
    p,_=fixture();bars={'raw':100.,'centered':100.}
    a=E.read_stream(p,'calibrated-raw',bars);assert not a['expanded'] and a['purchase_step']==33
    q=deepcopy(p);q['history']=[o for o in q['history'] for _ in range(3)]
    assert E.read_stream(q,'calibrated-raw',bars)==a
    q['history'][1]=dict(q['history'][1],artifact=999)
    with pytest.raises(ValueError):E.read_stream(q,'calibrated-raw',bars)

def test_behaviorally_inert_purchase_has_exact_extra_cost(monkeypatch):
    p,t=fixture();monkeypatch.setattr(D,'catalog',lambda w:[dict(w)]*3)
    a=D.score(E.read_stream(p,'calibrated-raw',{'raw':-1.}),t)
    b=D.score(E.read_stream(p,'calibrated-raw',{'raw':100.}),t)
    assert a['purchase_step']==8 and not b['expanded']
    assert np.allclose([x['prediction'] for x in a['trace']],[x['prediction'] for x in b['trace']],rtol=0,atol=1e-12)
    assert a['likelihood_evaluations']-b['likelihood_evaluations']==24*32
    assert abs((a['net_match_low']-b['net_match_low'])+(1+1e-5*24*32)/32)<1e-12
    assert abs((a['net_match_high']-b['net_match_high'])+(4+1e-4*24*32)/32)<1e-12

def test_future_outcomes_cannot_change_earlier_forecasts_or_purchase():
    p,_=fixture();q=deepcopy(p)
    for obs in q['history'][16:]:obs.update(program=[],artifact=0)
    for method in ('calibrated-raw','calibrated-centered'):
        a=E.read_stream(p,method,{'raw':2.,'centered':.3});b=E.read_stream(q,method,{'raw':2.,'centered':.3})
        assert a['trace'][:17]==b['trace'][:17]

def test_handler_input_binding_and_complete_small_fixture(tmp_path):
    from ghostscale.validation.soundingline.v18_3.io import write,file_digest,read
    bars=[dict(cell=2,order=o,prior=p,sensor=s,threshold=3. if s=='raw' else .5) for o in D.ORDERS for p in C.PRIORS for s in C.SENSORS]
    write(tmp_path/'inputs/THRESHOLDS.json',dict(calibration_indices=[-19699],evaluation_indices=[-19799],strata=bars))
    h=file_digest(tmp_path/'inputs/THRESHOLDS.json')
    write(tmp_path/'inputs/INDEPENDENT_REVIEW.json',dict(passed=True,thresholds_sha256=h))
    plan=dict(design=dict(thresholds_sha256=h,calibration_review_sha256=file_digest(tmp_path/'inputs/INDEPENDENT_REVIEW.json'),evaluation_indices=[-19799],cells=[2]))
    summary=E.run(tmp_path,plan,lambda **k:None)
    assert summary['streams']==24 and summary['strata']==216 and all(summary['controls'].values())
    assert len(read(tmp_path/'STRATA.json')['strata'])==216
    bad=deepcopy(plan);bad['design']['thresholds_sha256']='0'*64
    with pytest.raises(ValueError,match='threshold artifact'):E.run(tmp_path,bad,lambda **k:None)
