from copy import deepcopy
import os
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import consideration as m,ledger
from ghostscale.validation.soundingline.v23.common import canonical

def test_grid_and_known_answers():
    assert len(m.variants())==12
    assert m.controls()['passed']

def test_all_posteriors_independently_reconstruct():
    for v in m.variants():
        p,t=m.sample(v,'noisy-fixture',0,8)
        for mode in ('all','offers','timing','actions'):
            assert np.allclose(m.posterior(p,t['y'],mode),m.posterior(p,t['y'],mode,product=True),atol=1e-12,rtol=0)

def test_hidden_timing_and_future_target_cannot_leak():
    v=m.variants()[4];p,t=m.sample(v,'fixture',11,3);assert not p['timing']
    changed=dict(t,y=[y//2*2+1-y%2 for y in t['y']])
    a,d,s=m.evaluate(v,p,t);b,e,z=m.evaluate(v,p,changed)
    assert np.array_equal(a['exact'],b['exact']) and canonical(d)==canonical(e)
    assert s['timing_gain']==pytest.approx(0,abs=1e-14)

def test_schema_roundtrip_and_no_query_taken():
    schema,_=ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    p,t=m.sample(m.variants()[-1],'ledger-fixture',15,7);bundle=m.emit(p,schema,['fixture'])
    assert canonical(m.decode(bundle,schema))==canonical(p)
    q=next(r for r in bundle['rows']['Decision'] if r['context']['channel']=='query');q['taken']=q['options'][0]['option_id']
    with pytest.raises(ValueError,match='query target'):m.decode(bundle,schema)
