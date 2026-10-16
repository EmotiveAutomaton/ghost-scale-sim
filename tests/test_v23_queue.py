from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
from runners import queue_v23_families as queue
from ghostscale.validation.soundingline.v23.common import write,read,digest


def test_unstarted_admissions_and_burnin_reservations_count_toward_resources(tmp_path):
    write(tmp_path/'ACCEPTANCE.json',dict(max_scientific_workers=3,cpu_seconds=90000,protected_cpu_seconds=18000))
    (tmp_path/'attempts').mkdir()
    plan=dict(cpu_cap_seconds=1000,budget_group='repairs',family='R1',second_burnin_reservation_seconds=4000)
    entries=[(tmp_path/'a',plan,{})]*3
    assert queue.capacity(tmp_path,entries,300)=='NO_SLOT'
    entries=[(tmp_path/'a',plan,dict(state='complete',cpu_seconds=2000))]
    assert queue.capacity(tmp_path,entries,47000)=='REPAIRS_GROUP_CAP'
    assert queue.capacity(tmp_path,entries,47000,second_family='R1')=='AVAILABLE'


def test_prefits_reserve_both_burnins_and_use_their_own_group(tmp_path):
    write(tmp_path/'ACCEPTANCE.json',dict(max_scientific_workers=3,cpu_seconds=324000,protected_cpu_seconds=64800))
    (tmp_path/'attempts').mkdir()
    plan=dict(cpu_cap_seconds=1800,budget_group='n5',family='N5',burnin_reservations_seconds=[4000,4000])
    entries=[(tmp_path/'fits',plan,dict(state='complete',cpu_seconds=1000))]
    assert queue.capacity(tmp_path,entries,2000,budget_group='n5')=='FAMILY_GROUP_CAP'
    write(tmp_path/'packets/v23-n5-batch-1/PLAN.json',{})
    assert queue.capacity(tmp_path,entries,5000,second_family='N5',budget_group='n5')=='AVAILABLE'


def test_terminal_family_card_cannot_launch_or_reinterpret_results(tmp_path,monkeypatch):
    now=datetime.now(timezone.utc)
    write(tmp_path/'ACCEPTANCE.json',dict(report_start=(now+timedelta(days=1)).isoformat()))
    summary=tmp_path/'cal.json';write(summary,dict(kill_threshold=.05,prior=[.08,.05]))
    write(tmp_path/'CALIBRATION_ACCEPTANCE.json',dict(summary=str(summary),summary_sha256=digest(summary),complete_sha256='a'))
    root=tmp_path/'packets/first'
    write(root/'PLAN.json',{})
    write(root/'COMPLETE.json',dict(state='verified'));write(root/'NUMERICAL_REVIEW.json',dict(state='passed'))
    write(root/'STATUS.json',dict(state='complete'))
    write(root/'MACHINE_CARD.json',dict(state='NOMINATED',complete_sha256=digest(root/'COMPLETE.json'),
        numerical_review_sha256=digest(root/'NUMERICAL_REVIEW.json'),calibration_complete_sha256='a'))
    def forbidden(*args,**kwargs):raise AssertionError('terminal family was rerun')
    monkeypatch.setattr(queue,'launch',forbidden);monkeypatch.setattr(queue.family,'decide',forbidden)
    result=queue.step(tmp_path,dict(families={'R1':dict(first_packet=str(root))}))
    assert result['state']=='REVIEW_BOUNDARY'
    assert read(tmp_path/'events/r1-decision-1.json')['kind']=='family_decision'
    write(root/'NUMERICAL_REVIEW.json',dict(state='changed'),immutable=False)
    with pytest.raises(ValueError,match='bindings'):queue.step(tmp_path,dict(families={'R1':dict(first_packet=str(root))}))


def test_new_queue_preserves_old_cost_and_delivers_distinct_boundary(tmp_path,monkeypatch):
    old=tmp_path/'attempts/family-queue.json';write(old,dict(cpu_seconds=157,state='REVIEW_BOUNDARY'))
    config=tmp_path/'config.json';write(config,dict(service_id='repair-queue-2'))
    monkeypatch.setattr(queue,'step',lambda *a:dict(state='REVIEW_BOUNDARY',families={'R4':'NOMINATED'}))
    monkeypatch.setattr(queue.runtime,'below_normal',lambda:None)
    queue.run(tmp_path,tmp_path/'owner',config)
    assert read(old)['cpu_seconds']==157
    assert (tmp_path/'attempts/repair-queue-2.json').exists()
    assert read(tmp_path/'events/repair-queue-2-boundary.json')['evidence']['families']=={'R4':'NOMINATED'}
    with pytest.raises(ValueError,match='already used'):queue.run(tmp_path,tmp_path/'owner',config)


def test_partition_reservation_counts_once_and_children_use_slots(tmp_path):
    write(tmp_path/'ACCEPTANCE.json',dict(max_scientific_workers=3,cpu_seconds=324000,protected_cpu_seconds=64800))
    (tmp_path/'attempts').mkdir()
    parent=dict(cpu_cap_seconds=30000,budget_group='repairs',family='R4',scheduler_only=True)
    child=dict(cpu_cap_seconds=7500,budget_group='repairs',family='R4',logical_parent='parent')
    entries=[(tmp_path/'parent',parent,dict(state='partitioned',cpu_seconds=10000)),
             (tmp_path/'part1',child,dict(state='complete',cpu_seconds=5000)),
             (tmp_path/'part2',child,dict(state='running',cpu_seconds=5000))]
    assert queue.active_count(entries)==1
    assert queue.capacity(tmp_path,entries,20000)=='AVAILABLE'
    assert queue.capacity(tmp_path,entries,21000)=='REPAIRS_GROUP_CAP'
    entries.append((tmp_path/'calibration',dict(cpu_cap_seconds=100,budget_group='setup'),dict(state='complete',cpu_seconds=50)))
    assert queue.capacity(tmp_path,entries,20000)=='AVAILABLE'
