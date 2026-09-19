import numpy as np
from ghostscale.validation.soundingline.v18_3 import world as W,revision as D,source_uptake as U,provenance as C,verify


def test_duplicate_evidence_cannot_change_source_aware_fit_but_can_change_naive():
    w=W.make_world(0,98120);context=W.context();program=int(np.argmax(np.ptp(W.matrix(w,context),axis=0)))
    h=[dict(context=context,program=list(W.PROGRAMS[program]),artifact=int(W.ARTIFACTS[program]),source='root')]
    a,sa=D.fit(w,h,deduplicate=True);b,sb=D.fit(w,h*3,deduplicate=True);naive,sn=D.fit(w,h*3)
    assert np.array_equal(a,b) and sa==sb
    assert not np.allclose(a,naive) and sn!=sa


def test_source_revision_pairing_and_independence_controls():
    for cell in (0,1):
        a=D.unit(90990,cell=cell,source_reader='source-aware',split='pilot')
        b=D.unit(90990,cell=cell,source_reader='naive',split='pilot')
        assert a['public']==b['public'] and a['evaluator']==b['evaluator']
        assert a['independent_roots']==(4 if cell==1 else 12)
        if cell==0:assert a['rows']==b['rows']
        assert verify.check_unit(a,True)['distributions']>0


def test_native_recommendation_success_failure_and_stronger_public_checker():
    unit=C.unit(81821,mode='shared-error',split='pilot')
    for truth in (0,1):
        unit['evaluator']['truth']=truth
        unit['rows']=[dict(method='correct',instrument='valid',posterior=[1-truth,truth],after_correction=[[1-truth,truth]]),dict(method='wrong',instrument='valid',posterior=[truth,1-truth],after_correction=[[truth,1-truth]])]
        result=U.apply(unit);by={r['method']:r for r in result['rows']}
        assert by['correct']['initial']['success'] and not by['wrong']['initial']['success']
        assert by['public-law-task-checker']['initial']['success']
        assert by['public-law-task-checker']['counterfactual_check_work']>0
