"""Long-history stability, paired readout grids and finite follow-up accounting."""
import math
from decimal import Decimal, localcontext
from datetime import datetime, timezone
import numpy as np
import pytest
from ghostscale.validation.soundingline.v26 import values, opacity, contract
from ghostscale.validation.soundingline.v26.common import close, write
from runners.run_v26_followups import charges, family_charged


def test_extended_type_odds_against_decimal_without_probability_floor():
    ll=np.array([-2000.,-2001.,-1000.,-1002.]);prior=np.ones(4)/4
    kinds=np.array(['A','A','B','B'])
    with localcontext() as ctx:
        ctx.prec=60
        sums=[sum((Decimal(str(x)).exp()/4 for x in ll[kinds==k]),Decimal(0)) for k in ('A','B')]
        expected=float((sums[1]/sums[0]).ln())
    for independent in (False,True):
        assert abs(values.type_log_evidence(ll,prior,kinds,'B','A',independent)-expected)<1e-12
    assert math.isfinite(expected) and expected>999


def test_e1_longer_horizon_preserves_prefix_and_reader_roster():
    setting=dict(environment='stationary',eta=.005,theta=.6,truth_types=['B'])
    a=values.block(setting|dict(max_history=256),['excluded-followup-prefix'])
    b=values.block(setting|dict(max_history=512,histories=[256,512]),['excluded-followup-prefix'])
    assert a['public']==b['public']
    close(a['truth_and_observations'][0]['records'],b['truth_and_observations'][0]['records'][:256])
    assert len(b['truth_and_observations'])==1
    for row in b['readouts']:
        assert row['labels']['truth']=='B'
    close(b,values.block(setting|dict(max_history=512,histories=[256,512]),['excluded-followup-prefix'],True),tol=1e-9)


def test_u1_finer_grid_has_identical_observations_and_shared_readouts():
    s=dict(s=.5,tau=1.)
    a=opacity.block(s,['excluded-followup-grid'])
    b=opacity.block(s|dict(training_lengths=list(range(1,65))),['excluded-followup-grid'])
    for x,y in zip(a['truth_and_observations'],b['truth_and_observations']):
        assert x['training']==y['training'] and x['claims']==y['claims']
        assert all(v==y['source_posteriors'][k] for k,v in x['source_posteriors'].items())
    assert a['readouts']==[r for r in b['readouts'] if r['labels']['history'] in [1,2,4,8,16,32,64]]
    close(b,opacity.block(s|dict(training_lengths=list(range(1,65))),['excluded-followup-grid'],True),tol=1e-9)


def test_stationary_and_shift_are_exact_aliases_for_specialists():
    for t in (1,127,128,256,1024,2048):
        from ghostscale.validation.soundingline.v26.common import rng
        a=values.menu_at(rng('excluded-shift-alias',t),t,dict(environment='stationary',environment_dimension=1))
        b=values.menu_at(rng('excluded-shift-alias',t),t,dict(environment='shift',environment_dimension=1))
        assert np.array_equal(a[0],b[0]) and a[1]==b[1]


def test_followup_carries_prior_charges_and_reserved_failures(tmp_path):
    write(tmp_path/'PARENT_CHARGES.json',dict(charges={'followup_healing':5000.,'texture':5000.,'preparation':500.}))
    (tmp_path/'accounting').mkdir()
    write(tmp_path/'accounting'/'failed.json',dict(pool='followup_healing',kind='followup',family='E1',cpu_seconds=10.,reserved_cpu_seconds=1800.))
    c=charges(tmp_path)
    assert c['followup_healing']==6800 and c['followup']==1800 and family_charged(tmp_path,'E1')==1800
    a=contract.acceptance('2026-11-02T15:00:00+00:00','spec')
    p=dict(family='E1',kind='followup',pool='followup_healing',generation=1,numerical_threads=1,priority='below-normal',design_check_sha256='bound',cpu_cap_seconds=9100,pass_caps=dict(original=3000,replay=3000,review=3000))
    with pytest.raises(ValueError,match='follow-up subtotal'):
        contract.admission(p,a,datetime(2026,11,3,14,tzinfo=timezone.utc),c)


@pytest.mark.parametrize('setting',[{'training_lengths':[]},{'training_lengths':[0,64]},{'training_lengths':[1,1]}])
def test_reject_invalid_training_grid(setting):
    with pytest.raises(ValueError):opacity.block(dict(s=.5,tau=1.)|setting,['excluded-invalid'])


def test_followup_monitor_skips_old_events_without_moving_acceptance(tmp_path):
    from runners.watch_v26 import scheduled
    accepted=contract.acceptance('2026-11-02T15:07:04+00:00','spec')
    write(tmp_path/'ACCEPTANCE.json',accepted)
    write(tmp_path/'MONITOR_START.json',dict(at='2026-11-03T13:16:00+00:00'))
    scheduled(tmp_path,datetime.fromisoformat('2026-11-03T13:20:00+00:00'))
    assert not list((tmp_path/'events').glob('*.json'))
    scheduled(tmp_path,datetime.fromisoformat('2026-11-03T15:08:00+00:00'))
    events=list((tmp_path/'events').glob('*.json'))
    assert len(events)==1 and events[0].stem=='v26-health12-02'
    from ghostscale.validation.soundingline.v26.common import read
    assert read(tmp_path/'ACCEPTANCE.json')==accepted
