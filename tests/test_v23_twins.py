from copy import deepcopy
import os
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import registry,ledger,twin_ledger
from ghostscale.validation.soundingline.v23 import twin_writing,twin_chess,twin_wiki,twin_panel
from ghostscale.validation.soundingline.v23.common import canonical


def fixture(family,dirty=False):
    v=deepcopy(registry.variants(family)[0]);v['settings']['dirty']=dirty
    if family=='T1':
        v['settings']['personal_effect']=.6
        shape=dict(records_per_maker=[8],sessions_per_maker=[2],options_per_decision=4,
            latency_quantiles_s=dict(probabilities=[0,1],values=[1,4]),
            keystroke_quantiles=dict(probabilities=[0,1],values=[0,12]),handling_marginals=[.5,.22,.23,.05])
        p,t=twin_writing.maker(v,'engineering-only',0,0,shape)
    elif family=='T2':
        p,t=twin_chess.player(v,'engineering-only',0,0,games=2,
            clock_quantiles=dict(probabilities=[0,1],seconds=[1,120]))
    elif family=='T3':
        rows=list(twin_wiki.world(v,'engineering-only',0,makers=3,records=3))
        p=dict(maker=0,history=[x for x,_ in rows]);t=dict(marks=[y for _,y in rows])
    else:p,t=twin_panel.rater(v,'engineering-only',0,0,semantics='nominal_yes_no_unsure',repeat=1,records=4)
    return v,p,t


@pytest.mark.parametrize('family',['T1','T2','T3','T4'])
@pytest.mark.parametrize('dirty',[False,True])
def test_constructed_twin_independent_inference_and_ledger_roundtrip(family,dirty):
    v,p,t=fixture(family,dirty)
    if family=='T1':
        prediction,detail=twin_writing.evaluate(v,p);assert twin_writing.verify(v,p,detail)
    elif family=='T2':
        prediction,detail=twin_chess.evaluate(v,p);assert twin_chess.verify_posterior(v,p,prediction,detail)
    elif family=='T3':
        prediction,detail=twin_wiki.evaluate(v,p['history'],p['maker']);assert twin_wiki.verify(v,p['history'],p['maker'],detail)
    else:
        prediction,detail=twin_panel.evaluate(v,p);assert twin_panel.verify(v,p,detail)
    assert all(np.allclose(q.sum(-1),1) for q in prediction.values())
    schema,_=ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    bundle=twin_ledger.emit(v,p,schema,['engineering-twin',family,dirty]);ledger.validate(bundle,schema)
    assert canonical(twin_ledger.decode(v,bundle,schema))==canonical(p)
    assert b'evaluator-only-canary-482117' not in canonical(bundle)


def test_zero_personal_writing_effect_is_exact_population_parity():
    v,p,_=fixture('T1');v['settings']['personal_effect']=0
    prediction,_=twin_writing.evaluate(v,p)
    np.testing.assert_allclose(prediction['individual'],prediction['population'],atol=1e-12,rtol=0)


def test_twin_checkers_reject_corrupt_posteriors():
    for name,module in [('T1',twin_writing),('T4',twin_panel)]:
        v,p,_=fixture(name);_,detail=module.evaluate(v,p);detail['type_posterior'][0]+=.1
        with pytest.raises(ValueError,match='posterior mismatch'):module.verify(v,p,detail)
    v,p,_=fixture('T3');_,detail=twin_wiki.evaluate(v,p['history'],0);detail['joint']['purpose_posterior'][0]+=.1
    with pytest.raises(ValueError,match='posterior mismatch'):twin_wiki.verify(v,p['history'],0,detail)


def test_minimax_and_person_balanced_flood_controls():
    levels=twin_chess.solve(3,3,'engineering-tree');assert twin_chess.verify_tree(levels,3)==0
    bad=[x.copy() for x in levels];bad[0][0]+=.1
    with pytest.raises(ValueError,match='minimax'):twin_chess.verify_tree(bad,3)
    before=twin_panel.population([[.9,.1],[.1,.9]],[1,1]);after=twin_panel.population([[.9,.1],[.1,.9]],[100,1])
    assert before['person_balanced']==after['person_balanced']
    assert after['contribution_weighted'][0]>before['contribution_weighted'][0]+.3


def test_wiki_global_article_revisions_preserve_multiple_maker_attribution():
    v,p,_=fixture('T3');latest={}
    for row in p['history']:
        if row['article'] in latest:np.testing.assert_allclose(row['state'],latest[row['article']],atol=1e-12)
        latest[row['article']]=np.array(row['state'])+row['diff']
    schema,_=ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    bundle=twin_ledger.emit(v,p,schema,['engineering','attribution'])
    assert len(bundle['rows']['Maker'])==3
    assert len(bundle['rows']['Artifact'][0]['maker_ids'])==3


def test_nominal_and_ordinal_panel_models_are_separate():
    x=np.array([[1,1,1,1],[0,0,0,0]])
    nominal=twin_panel.ratings(x,'nominal_yes_no_unsure');ordinal=twin_panel.ratings(x,'ordinal_3')
    assert np.max(np.abs(nominal-ordinal))>.01
    assert np.allclose(ordinal.sum(-1),1) and np.min(ordinal)>0
