from datetime import datetime,timezone
import pytest
from ghostscale.validation.soundingline.v25 import contract,pieces,setpoints

def test_piece_controls():assert pieces.controls()['passed']
def test_hidden_controller_controls():assert setpoints.controls()['passed']

def test_tr4_calibrated_followup_uses_w8_filter(monkeypatch):
    from ghostscale.validation.soundingline.v25 import followups
    observed={}
    def fake(setting,identity,independent=False):
        observed.update(setting=setting,identity=identity,independent=independent)
        return {'public':{},'truth_and_observations':[],'readouts':[]}
    monkeypatch.setattr(followups.volatile,'block',fake)
    setting=dict(followup='tr4-dose',process='walk',rate=.002)
    assert followups.block(setting,['fixture'],True)['readouts']==[]
    assert observed==dict(setting=setting,identity=['fixture'],independent=True)

def fixture():
    a=contract.acceptance('2026-10-25T19:00:00+00:00','x'*64)
    p=dict(family='W1',kind='probe',pool='studies',cpu_cap_seconds=1800,pass_caps=dict(original=600,replay=600,review=600),numerical_threads=1,priority='below-normal')
    return a,p

def test_opening_and_dst_boundary():
    a,p=fixture();assert contract.admission(p,a,datetime.fromisoformat('2026-10-25T20:00:00+00:00'),{})
    for t in ('2026-10-25T18:59:00+00:00','2026-11-02T04:00:00+00:00'):
        with pytest.raises(ValueError):contract.admission(p,a,datetime.fromisoformat(t),{})

def test_protected_pool_and_shared_slots():
    a,p=fixture();now=datetime.fromisoformat(a['opened'])
    with pytest.raises(ValueError):contract.admission(p,a,now,{},running=2)
    with pytest.raises(ValueError):contract.admission(p,a,now,{'studies':32*3600})
    with pytest.raises(ValueError):contract.admission(dict(p,pool='protected'),a,now,{})

def test_followup_generation_and_healing_reserve():
    a,p=fixture();p.update(kind='followup',pool='followup_healing',generation=1)
    with pytest.raises(ValueError):contract.admission(p,a,datetime.fromisoformat(a['opened']),{'followup':10000})
    with pytest.raises(ValueError):contract.admission(dict(p,generation=2),a,datetime.fromisoformat(a['opened']),{})

def test_exact_control_really_detects_mutation(monkeypatch):
    import numpy as np
    monkeypatch.setattr(pieces,'laws',lambda e,a,*args,**kwargs:np.full((2,len(a)),1/len(a)))
    with pytest.raises(AssertionError):pieces.controls()
