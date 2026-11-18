"""Storage admission preserves headroom without converting holds into failures."""
from types import SimpleNamespace
from runners.run_v31 import storage_admissible
import pytest
from ghostscale.validation.soundingline.v31.common import write,digest
from runners.run_v31 import verify_recovery_parent


def test_storage_accounts_for_active_output_and_headroom(tmp_path,monkeypatch):
    gib=1024**3
    monkeypatch.setattr('runners.run_v31.shutil.disk_usage',lambda p:SimpleNamespace(free=40*gib))
    plan=dict(storage_required_bytes=8*gib,storage_floor_bytes=30*gib)
    assert storage_admissible(tmp_path,tmp_path,plan,{})
    active={'one':dict(job=tmp_path,plan=dict(storage_required_bytes=4*gib))}
    assert not storage_admissible(tmp_path,tmp_path,plan,active)


def test_releasing_active_reservation_reopens_admission(tmp_path,monkeypatch):
    gib=1024**3
    monkeypatch.setattr('runners.run_v31.shutil.disk_usage',lambda p:SimpleNamespace(free=31*gib))
    plan=dict(storage_required_bytes=2*gib,storage_floor_bytes=30*gib)
    assert not storage_admissible(tmp_path,tmp_path,plan,{})
    monkeypatch.setattr('runners.run_v31.shutil.disk_usage',lambda p:SimpleNamespace(free=33*gib))
    assert storage_admissible(tmp_path,tmp_path,plan,{})


def test_campaign_metadata_volume_is_protected(tmp_path,monkeypatch):
    gib=1024**3;c=tmp_path/'campaign';job=tmp_path/'job'
    monkeypatch.setattr('runners.run_v31.shutil.disk_usage',lambda p:SimpleNamespace(free=(4 if p==c else 100)*gib))
    assert not storage_admissible(c,job,dict(storage_required_bytes=gib,storage_floor_bytes=30*gib),{})


def test_recovery_keeps_contract_and_reproduces_retained_bytes(tmp_path):
    parent=tmp_path/'jobs/old';successor=tmp_path/'jobs/new'
    original=dict(setting={'seed_identity':17},module='frozen',source_root='source',
                  source_manifest_sha256='a',acceptance_sha256='b',
                  cpu_cap_seconds=12,wall_cap_seconds=30,interpreter='python')
    write(parent/'PLAN.json',original);write(parent/'COMPLETE.json',dict(state='FAILED'))
    write(parent/'output/pilot-original/RAW.json',{'retained':1})
    write(successor/'output/pilot-original/RAW.json',{'retained':1})
    plan=dict(original,recovery_parent='old',recovery_parent_plan_sha256=digest(parent/'PLAN.json'),
              recovery_parent_completion_sha256=digest(parent/'COMPLETE.json'),
              retained_parent_files={'output/pilot-original/RAW.json':digest(parent/'output/pilot-original/RAW.json')})
    verify_recovery_parent(tmp_path,plan,successor)
    with pytest.raises(ValueError,match='scientific contract'):
        verify_recovery_parent(tmp_path,dict(plan,setting={'seed_identity':18}))
    write(successor/'output/pilot-original/RAW.json',{'retained':2},immutable=False)
    with pytest.raises(ArithmeticError,match='reproduce'):verify_recovery_parent(tmp_path,plan,successor)


def test_verified_parent_cannot_be_recovered(tmp_path):
    parent=tmp_path/'jobs/old'
    original=dict(setting={},module='frozen',source_root='source',source_manifest_sha256='a',
                  acceptance_sha256='b',cpu_cap_seconds=12,wall_cap_seconds=30,interpreter='python')
    write(parent/'PLAN.json',original);write(parent/'COMPLETE.json',dict(state='VERIFIED'))
    plan=dict(original,recovery_parent='old',recovery_parent_plan_sha256=digest(parent/'PLAN.json'),
              recovery_parent_completion_sha256=digest(parent/'COMPLETE.json'))
    with pytest.raises(ValueError,match='not the retained failed'):verify_recovery_parent(tmp_path,plan)
