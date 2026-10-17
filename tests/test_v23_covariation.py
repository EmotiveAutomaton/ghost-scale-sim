from copy import deepcopy
import os
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import covariation as m,ledger
from ghostscale.validation.soundingline.v23.common import canonical

def test_known_answers_and_grid():
    assert len(m.variants())==12
    assert m.controls()['passed']

def test_all_observation_maps_agree_with_probability_products():
    for v in m.variants():
        p,t=m.sample(v,'fixture',0,1)
        for mode in m.READOUTS:
            assert np.allclose(m.posterior(p,mode),m.posterior(p,mode,product=True),atol=2e-12,rtol=0)

def test_masked_inputs_cannot_change_restricted_readers():
    p,t=m.sample(m.variants()[-1],'fixture',15,7);q=deepcopy(p)
    q['traits']=[1-x for x in p['traits']];q['consensus']=[1-x for x in p['consensus']]
    q['history']=[[x//4*4+(3-x%4) for x in history[::-1]] for history in p['history']]
    assert np.array_equal(m.posterior(p,'pervasiveness'),m.posterior(q,'pervasiveness'))
    q=deepcopy(p);q['traits']=[1-x for x in p['traits']]
    q['history']=[[x//4*4+(3-x%4) for x in history] for history in p['history']]
    assert np.array_equal(m.posterior(p,'covariation'),m.posterior(q,'covariation'))

def test_future_and_evaluator_labels_do_not_enter_forecast():
    v=m.variants()[5];p,t=m.sample(v,'fixture',5,2)
    a,_,_=m.evaluate(v,p,t)
    b,_,_=m.evaluate(v,p,dict(t,y=[(y+1)%8 for y in t['y']],mechanism=0))
    assert np.array_equal(a['exact'],b['exact'])
    assert np.array_equal(m.features(v,p),m.features(m.variants()[0],p))

def test_ledger_roundtrip_and_query_leak_rejection():
    schema,_=ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    p,_=m.sample(m.variants()[-1],'ledger-fixture',0,0)
    b=m.emit(p,schema,['fixture'])
    assert canonical(m.decode(b,schema))==canonical(p)
    q=next(r for r in b['rows']['Decision'] if r['context']['channel']=='query');q['taken']=q['options'][0]['option_id']
    with pytest.raises(ValueError,match='query target'):m.decode(b,schema)
