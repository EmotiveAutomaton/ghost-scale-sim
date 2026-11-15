from datetime import datetime,timezone,timedelta
import pytest
from runners.run_v30 import admissible,equivalent,resource_contract
from ghostscale.validation.soundingline.v30.common import write,digest
import numpy as np

def test_reservations_and_deadline_are_real_boundaries():
    now=datetime.now(timezone.utc)
    a=dict(cpu_limits={"P":100,"reserve":20},total_cpu_seconds=110,
           new_starts_close=(now+timedelta(hours=2)).isoformat(),
           drain=(now+timedelta(hours=3)).isoformat())
    p=dict(pool="P",cpu_cap_seconds=40,wall_cap_seconds=60)
    assert admissible(a,{"P":60,"reserve":0},p,now)
    assert not admissible(a,{"P":61,"reserve":0},p,now)
    assert not admissible(a,{"P":60,"reserve":11},p,now)
    assert not admissible(a,{"P":0,"reserve":0},p,now+timedelta(hours=2))

def test_replay_rejects_missing_or_changed_raw(tmp_path):
    a=tmp_path/"a";b=tmp_path/"b";a.mkdir();b.mkdir()
    (a/"raw").write_bytes(b"123");(b/"raw").write_bytes(b"123")
    equivalent(a,b,True)
    (b/"raw").write_bytes(b"124")
    with pytest.raises(ArithmeticError):equivalent(a,b,True)

def test_reconstruction_checks_world_metrics_not_just_means(tmp_path):
    a=tmp_path/"a";b=tmp_path/"b";a.mkdir();b.mkdir()
    write(a/"SUMMARY.json",dict(mean=2));write(b/"SUMMARY.json",dict(mean=2))
    np.savez(a/"cell_measures.npz",scores=np.array([1.,3.]))
    np.savez(b/"cell_measures.npz",scores=np.array([2.,2.]))
    with pytest.raises(ArithmeticError):equivalent(a,b)

def test_pool_supplement_preserves_acceptance_and_other_boundaries(tmp_path):
    a=dict(cpu_limits={'P':100,'reserve':20},total_cpu_seconds=120,
           gear=2,drain='fixed-drain',new_starts_close='fixed-starts')
    write(tmp_path/'ACCEPTANCE.json',a)
    original=digest(tmp_path/'ACCEPTANCE.json')
    assert resource_contract(tmp_path)==a
    s=dict(approved=True,owner_authorization='explicit fixture approval',
           acceptance_sha256=original,total_cpu_seconds=120,cpu_limits={'P':60,'reserve':60},
           gear=99,drain='attempted-override')
    write(tmp_path/'RESOURCE_REALLOCATION.json',s)
    actual=resource_contract(tmp_path)
    assert actual==dict(a,cpu_limits={'P':60,'reserve':60})
    assert digest(tmp_path/'ACCEPTANCE.json')==original

@pytest.mark.parametrize('change',[
    {'approved':False},{'owner_authorization':''},{'acceptance_sha256':'wrong'},
    {'total_cpu_seconds':121},{'cpu_limits':{'P':100,'reserve':21}},
    {'cpu_limits':{'P':-1,'reserve':121}}, {'cpu_limits':{'wrong':60,'reserve':60}},
])
def test_invalid_resource_supplement_cannot_admit_work(tmp_path,change):
    write(tmp_path/'ACCEPTANCE.json',dict(cpu_limits={'P':100,'reserve':20},total_cpu_seconds=120))
    s=dict(approved=True,owner_authorization='explicit fixture approval',
           acceptance_sha256=digest(tmp_path/'ACCEPTANCE.json'),total_cpu_seconds=120,
           cpu_limits={'P':60,'reserve':60})
    s.update(change);write(tmp_path/'RESOURCE_REALLOCATION.json',s)
    with pytest.raises(ValueError):resource_contract(tmp_path)
