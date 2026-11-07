from datetime import datetime,timezone,timedelta
import pytest
from ghostscale.validation.soundingline.v27 import contract as c


def fixture():
    a=c.acceptance('2026-11-07T18:18:53+00:00','packet');t=datetime(2026,11,7,19,tzinfo=timezone.utc)
    p=dict(family='K2',kind='study',lane='study',pool='studies',cpu_cap_seconds=100,
        projected_wall_seconds=200,pass_caps=dict(original=30,replay=30,review=30),
        numerical_threads=1,priority='below-normal',design_check_sha256='d',controls_sha256='c',
        protocol_sha256='p',source_manifest_sha256='s',blocks=16)
    return a,p,t


def test_pool_isolation_measured_costs_and_four_slots():
    a,p,t=fixture()
    assert c.admission(p,a,t,{'followup_healing':c.POOLS['followup_healing']+1},running=3)
    with pytest.raises(ValueError,match='own pool'):c.admission(p,a,t,{'studies':c.POOLS['studies']})
    with pytest.raises(ValueError,match='shared worker'):c.admission(p,a,t,{},running=4)
    with pytest.raises(ValueError,match='shared worker'):c.admission(p,a,t,{},running=2,worker_limit=2)
    assert c.accounted(dict(pool='followup_healing',state='COMPLETE',cpu_seconds=10,reserved_cpu_seconds=100))==10
    assert c.accounted(dict(pool='followup_healing',state='FAILED',cpu_seconds=110,reserved_cpu_seconds=100))==110
    assert c.accounted(dict(pool='followup_healing',state='RUNNING',cpu_seconds=10,reserved_cpu_seconds=100))==100


def test_descendant_precision_and_reread_limits():
    a,p,t=fixture();p=p|dict(kind='followup',lane='followup',pool='followup_healing',generation=2)
    with pytest.raises(ValueError,match='primary condition'):c.admission(p,a,t,{})
    assert c.admission(p|dict(primary_positive_or_sign_reversal=True),a,t,{})
    with pytest.raises(ValueError,match='generation'):c.admission(p|dict(generation=3),a,t,{})
    q=p|dict(kind='precision',lane='precision',parent_complete_sha256='parent',parent_blocks=16,blocks=32)
    assert c.admission(q,a,t,{})
    with pytest.raises(ValueError,match='doubles'):c.admission(q|dict(blocks=64),a,t,{})
    q=p|dict(kind='reread',lane='reread',reread_level=2,retained_inputs={'raw':'hash'})
    assert c.admission(q,a,t,{})
    with pytest.raises(ValueError,match='reread'):c.admission(q|dict(reread_level=3),a,t,{})


def test_deadlines_shedding_and_protected_healing():
    a,p,t=fixture()
    with pytest.raises(ValueError,match='outside'):c.admission(p,a,datetime.fromisoformat(c.NEW_STARTS_CLOSE),{})
    with pytest.raises(ValueError,match='split'):c.admission(p|dict(projected_wall_seconds=43201),a,t,{})
    with pytest.raises(ValueError,match='shed'):c.admission(p,a,t,{'family:K2':40*3600-1})
    with pytest.raises(ValueError,match='trigger 1'):c.admission(p|dict(shedding_exhausted=True),a,t,{'family:K2':60*3600})
    q=p|dict(kind='texture',lane='texture',pool='followup_healing')
    with pytest.raises(ValueError,match='protected'):c.admission(q,a,t,{'followup_healing':470*3600})
    assert c.admission(q,a,datetime.fromisoformat(c.HEALING_RELEASE),{'followup_healing':470*3600})


def test_load_rule_requires_ten_minutes_and_hysteresis():
    assert c.load_limit([(i,.95) for i in range(0,600,30)])==4
    assert c.load_limit([(i,.95) for i in range(0,601,30)])==2
    assert c.load_limit([(i,.85) for i in range(0,601,30)],2)==2
    assert c.load_limit([(i,.7) for i in range(0,601,30)],2)==4
    assert c.load_limit([(0,.95),(600,.95)])==4
