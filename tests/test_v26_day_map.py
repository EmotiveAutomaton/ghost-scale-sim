"""Resource-only G3-T09 exception: no changes to the frozen scientific design."""
from datetime import datetime, timezone
import pytest
from ghostscale.validation.soundingline.v26 import day


def setup():
    a=day.supplement('2026-11-07T00:00:00+00:00','packet','prior',{})
    p=dict(card='G3-T09',family='G3',kind='probe',lane='texture',pool='followup_healing',
        cpu_cap_seconds=460,pass_caps=dict(original=150,replay=150,review=150),
        projected_wall_seconds=1800,numerical_threads=1,priority='below-normal',
        design_check_sha256='case',controls_sha256='controls',resource_supplement_sha256='bound')
    s=dict(schema='v26-day-g3-t09-allocation-20261107',card='G3-T09',
        previous_texture_seconds=1800,texture_seconds=3600,total_cpu_seconds=110*3600,
        pools_seconds=day.POOLS,new_starts_close=a['new_starts_close'],drain=a['drain'])
    return a,p,s,datetime(2026,11,7,12,tzinfo=timezone.utc)


def test_held_packet_admitted_only_with_scoped_supplement():
    a,p,s,t=setup();charges={'texture:G3':1388.421875,'followup_healing':14947.671875}
    with pytest.raises(ValueError,match='texture ceiling'):day.admission(p,a,t,charges)
    assert day.admission(p,a,t,charges,texture_supplement=s)
    with pytest.raises(ValueError,match='texture ceiling'):
        day.admission(p,a,t,{'texture:G3':3600-459},texture_supplement=s)


@pytest.mark.parametrize('change',[
    {'card':'G3-T08'}, {'family':'G2'}, {'lane':'primary','pool':'studies'},
    {'resource_supplement_sha256':None}, {'kind':'healing','healing_number':1}])
def test_supplement_cannot_admit_other_scope(change):
    a,p,s,t=setup()
    with pytest.raises(ValueError,match='invalid G3-T09'):day.admission(p|change,a,t,{},texture_supplement=s)


@pytest.mark.parametrize('change',[{'texture_seconds':3601},{'total_cpu_seconds':111*3600},
    {'drain':'2026-11-10T06:00:00+00:00'},{'previous_texture_seconds':0}])
def test_supplement_cannot_change_limits(change):
    a,p,s,t=setup()
    with pytest.raises(ValueError,match='invalid G3-T09'):day.admission(p,a,t,{},texture_supplement=s|change)


def test_supplement_preserves_pool_slots_and_deadline():
    a,p,s,t=setup()
    with pytest.raises(ValueError,match='pool ceiling'):
        day.admission(p,a,t,{'followup_healing':day.POOLS['followup_healing']-459},texture_supplement=s)
    with pytest.raises(ValueError,match='two shared'):
        day.admission(p,a,t,{},2,s)
    with pytest.raises(ValueError,match='outside new-start'):
        day.admission(p,a,datetime(2026,11,9,4,tzinfo=timezone.utc),{},texture_supplement=s)
