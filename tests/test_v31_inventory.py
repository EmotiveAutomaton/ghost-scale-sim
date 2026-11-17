"""Queue optimization must still see admissions, failures and released budgets."""
from datetime import datetime, timezone
import pytest
from ghostscale.validation.soundingline.v31.common import write, digest
from runners.run_v31 import QueueInventory, AccountingInventory, resource_contract, admissible


def test_new_admission_and_failure_are_visible(tmp_path):
    first=tmp_path/'jobs/first'
    write(first/'PLAN.json',dict(order=2,family='A'))
    inv=QueueInventory()
    assert len(inv.snapshot(tmp_path,0)[0])==1
    write(first/'COMPLETE.json',dict(state='FAILED'))
    write(tmp_path/'jobs/second/PLAN.json',dict(order=1,family='B'))
    assert inv.snapshot(tmp_path,5)[1]=={'first':'FAILED'}
    jobs,done=inv.snapshot(tmp_path,60)
    assert [p.parent.name for p,_ in jobs]==['second','first']
    assert done=={'first':'FAILED'}


def test_unchanged_receipts_are_not_reread(tmp_path,monkeypatch):
    write(tmp_path/'jobs/a/PLAN.json',dict(order=0))
    write(tmp_path/'jobs/a/COMPLETE.json',dict(state='VERIFIED'))
    inv=QueueInventory();inv.snapshot(tmp_path,0)
    monkeypatch.setattr('runners.run_v31.read',lambda p:pytest.fail('reread immutable evidence'))
    assert inv.snapshot(tmp_path,60)[1]=={'a':'VERIFIED'}


def test_accounting_observes_reservation_release_and_external_charge(tmp_path):
    write(tmp_path/'ACCEPTANCE.json',dict(cpu_limits=dict(A=100,reserve=50)))
    write(tmp_path/'accounting/a.json',dict(pool='A',cpu_seconds=1,reservation=80))
    inv=AccountingInventory()
    assert inv.totals(tmp_path)==dict(A=80,reserve=0)
    write(tmp_path/'accounting/a.json',dict(pool='A',cpu_seconds=21,reservation=0),immutable=False)
    write(tmp_path/'accounting/audit.json',dict(pool='reserve',cpu_seconds=3,reservation=0))
    assert inv.totals(tmp_path)==dict(A=21,reserve=3)


def test_reallocation_retains_clocks_and_total(tmp_path):
    original=dict(cpu_limits=dict(A=100,B=100),total_cpu_seconds=200,
                  new_starts_close='2026-11-23T04:00:00+00:00',drain='2026-11-23T06:00:00+00:00',gear=2)
    write(tmp_path/'ACCEPTANCE.json',original)
    supplement=dict(approved=True,owner_authorization='approved transfer',
                    acceptance_sha256=digest(tmp_path/'ACCEPTANCE.json'),
                    cpu_limits=dict(A=150,B=50),total_cpu_seconds=200,drain='2099-01-01')
    write(tmp_path/'RESOURCE_REALLOCATION.json',supplement)
    effective=resource_contract(tmp_path)
    assert effective==dict(original,cpu_limits=dict(A=150,B=50))
    plan=dict(pool='A',cpu_cap_seconds=40,wall_cap_seconds=60)
    current=datetime(2026,11,17,tzinfo=timezone.utc)
    assert admissible(effective,dict(A=100,B=20),plan,current)
    assert not admissible(effective,dict(A=120,B=20),plan,current)
    supplement['cpu_limits']['A']=151
    write(tmp_path/'RESOURCE_REALLOCATION.json',supplement,immutable=False)
    with pytest.raises(ValueError,match='total changed'):resource_contract(tmp_path)
