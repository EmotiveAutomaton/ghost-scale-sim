from copy import deepcopy
import os
import numpy as np
from ghostscale.validation.soundingline.v23 import falsifier as m,ledger
from ghostscale.validation.soundingline.v23.common import canonical

def test_calibration_and_known_answers():
    assert len(m.variants())==9
    assert m.controls()['passed']
    for v in m.variants():
        c=m.calibration(v)
        assert c['unit']=='half-Brier'
        assert abs(c['fit_oracle_gain']-v['settings']['strength']*c['margin'])<.05*c['margin']
        assert np.isclose(np.mean(1-np.sum(m.LAW[:,0]**2,axis=-1))*.5,.2891156462585034)

def test_no_future_or_evaluator_inputs_and_ledger():
    schema,_=ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    for v in m.variants():
        p,t=m.sample(v,'fixture',7,3);a,_,_=m.evaluate(v,p,t)
        changed=dict(t,mechanism=0,y=[(x+1)%4 for x in t['y']],personal=np.full((8,4),.25).tolist())
        b,_,_=m.evaluate(v,p,changed)
        for key in ('records','exact','tempered'):assert np.array_equal(a[key],b[key])
        assert canonical(m.decode(m.emit(p,schema,['fixture',v['id']]),schema))==canonical(p)
        assert np.allclose(a['records'].sum(-1),1)

def test_primary_bars_and_falsification_preconditions():
    summaries=[]
    for v in m.variants():
        supported=v['settings']['strength']>0
        rows=[dict(powered=supported,supported=supported,falsifiable_local=True,low_gain=not supported) for _ in range(16)]
        summaries.append(dict(variant=v,block_readouts=rows))
    out=m.study_readout(summaries)
    assert out['primary']['P3_passed']
    broken=deepcopy(summaries)
    for item in broken:
        if item['variant']['id']=='N13-0-81':item['block_readouts'][0]['supported']=True
    result=m.study_readout(broken)
    assert not result['primary']['P3_passed']
    assert 'FALSIFIED' not in result['verdicts']['N13-0-32']['verdicts']

def test_zero_goal_alias_and_half_brier_null():
    for v in [x for x in m.variants() if x['settings']['strength']==0]:
        p,t=m.sample(v,'fixture',0,0);pred,_,d=m.evaluate(v,p,t)
        assert np.allclose(pred['records'],pred['exact'],atol=1e-13,rtol=0)
        assert abs(d['primary'])<1e-13
