from copy import deepcopy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import devaluation as d, ledger
from ghostscale.validation.soundingline.v23.common import canonical, read


def test_known_answers_and_predeclared_grid():
    assert d.controls()['passed']
    assert len(d.variants()) == 12
    assert all(len({v['settings']['mechanism'] for v in d.variants() if v['settings']['history']==n})==4 for n in (0,8,32))


def test_paired_public_inputs_and_no_forecast_target_leak():
    inputs=[]
    for v in d.variants()[::3]:
        p,t=d.sample(v,'fixture',13,2);inputs.append(p)
        forecast,detail,_=d.evaluate(v,p,t)
        altered=dict(t,y=[(x+1)%6 for x in t['y']])
        other,_,_=d.evaluate(v,p,altered)
        assert np.array_equal(forecast['exact'],other['exact'])
        assert np.allclose(detail['before'],.25)
    assert all(canonical(p)==canonical(inputs[0]) for p in inputs)


def test_probability_and_log_space_agree_under_low_reliability_noise():
    for world in (0,7,15):
        for v in d.variants():
            p,t=d.sample(v,'noisy-fixture',world,9)
            for subset in (None,{2},{3,4},{0,1}):
                for hidden in (False,True):
                    assert np.allclose(d.posterior(p,t['y'],subset,hidden),d.independent(p,t['y'],subset,hidden),atol=1e-12,rtol=0)


def test_observable_goal_is_necessary_to_identify_habitual_goal_selection():
    full=d.laws(d.CONTEXTS)
    assert not np.array_equal(full[0],full[3])
    action=full.reshape(4,16,6,2,3).sum(-2)
    assert np.array_equal(action[0],action[3])
    assert len({tuple(d.intended(m,c) for c in d.CONTEXTS) for m in d.MECHANISMS})==4


def test_canonical_roundtrip_and_query_leak_rejected():
    import os
    schema,_=ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    for v in d.variants()[:3]:
        p,_=d.sample(v,'ledger-fixture',0,0)
        bundle=d.emit(p,schema,['fixture',v['id']])
        assert canonical(d.decode(bundle,schema))==canonical(p)
        bad=deepcopy(bundle)
        row=next(r for r in bad['rows']['Decision'] if r['context']['channel']=='query')
        row['taken']=row['options'][0]['option_id']
        with pytest.raises(ValueError,match='query target leaked'): d.decode(bad,schema)
