from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v23.common import write,read,digest
from runners import partition_v23,queue_v23_families as queue,queue_v23_addendum as addendum


def setup(campaign,parallel=3):
    write(campaign/'ACCEPTANCE.json',dict(max_scientific_workers=3,report_start='9999-12-31'))
    source=campaign/'source';source.mkdir();(source/'scientific.py').write_text('frozen=1')
    root=campaign/'packets/logical'
    plan=dict(task='family-batch',family='N7',batch=1,worlds=16,cases=64,budget_group='n7',
        scheduler_only=True,max_parallel_parts=parallel,cpu_cap_seconds=4300,
        source_archive=str(source.with_suffix('.zip')),sources={'scientific.py':digest(source/'scientific.py')},
        partition_layout=dict(chunks=[list(range(i,i+4)) for i in range(0,16,4)],part_cpu_cap=1000,assembly_cpu_cap=300))
    write(root/'PLAN.json',plan)
    return root,plan


def test_parallel_dispatch_is_slot_bounded_and_keeps_frozen_world_chunks(tmp_path,monkeypatch):
    root,plan=setup(tmp_path);launched=[]
    monkeypatch.setattr(queue,'launch',lambda part,*args:launched.append(part))
    for _ in range(4):partition_v23.step(tmp_path,root)
    assert len(launched)==3 and queue.active_count(queue.inventory(tmp_path))==3
    for index,part in enumerate(launched):
        child=read(part/'PLAN.json')
        assert child['world_indices']==plan['partition_layout']['chunks'][index]
        assert child['sources']==plan['sources'] and child['logical_plan_sha256']==digest(root/'PLAN.json')
    # Finish the middle partition first: no skip, overlap or duplicate launch.
    write(launched[1]/'COMPLETE.json',{})
    write(launched[1]/'STATUS.json',dict(state='complete',cpu_seconds=200))
    partition_v23.step(tmp_path,root)
    assert len(launched)==4 and read(launched[3]/'PLAN.json')['world_indices']==[12,13,14,15]
    partition_v23.step(tmp_path,root)
    assert len(launched)==4


def test_original_sequential_packet_keeps_one_part_at_a_time(tmp_path,monkeypatch):
    root,_=setup(tmp_path,parallel=1);launched=[]
    monkeypatch.setattr(queue,'launch',lambda part,*args:launched.append(part))
    partition_v23.step(tmp_path,root);partition_v23.step(tmp_path,root)
    assert len(launched)==1


def test_failure_drains_existing_parts_without_freeing_budget_or_refilling(tmp_path,monkeypatch):
    root,_=setup(tmp_path);launched=[]
    monkeypatch.setattr(queue,'launch',lambda part,*args:launched.append(part))
    partition_v23.step(tmp_path,root);partition_v23.step(tmp_path,root)
    write(launched[0]/'STATUS.json',dict(state='failed',cpu_seconds=100))
    write(launched[1]/'STATUS.json',dict(state='running',cpu_seconds=75))
    partition_v23.step(tmp_path,root)
    status=read(root/'STATUS.json');assert status['state']=='partitioned' and status['phase']=='failure-draining'
    assert status['cpu_seconds']==175 and len(launched)==2
    write(launched[1]/'STATUS.json',dict(state='complete',cpu_seconds=120),immutable=False)
    write(launched[1]/'COMPLETE.json',{})
    partition_v23.step(tmp_path,root)
    assert read(root/'STATUS.json')['state']=='failed' and read(root/'STATUS.json')['cpu_seconds']==220


def test_assembly_claim_counts_as_one_scientist(tmp_path):
    entries=[(tmp_path,dict(scheduler_only=True),dict(state='partitioned',uses_scientific_slot=True))]
    assert queue.active_count(entries)==1


def test_watchdog_does_not_treat_dependencies_or_busy_workers_as_ready():
    waiting=dict(families={'N7':'WAITING_FOR_SLOT'},active_scientists=2,worker_limit=3)
    since,trigger=addendum.idle_watchdog(None,waiting,0)
    assert since==0 and not trigger
    assert addendum.idle_watchdog(since,waiting,3600)==(0,True)
    for result in (dict(waiting,active_scientists=3),dict(waiting,families={'T1':'DEPENDENCY_HELD'}),
                   dict(waiting,newly_admitted=['bounded-packet'])):
        assert addendum.idle_watchdog(since,result,3600)==(None,False)


def test_addendum_dispatch_orders_preparation_before_ready_continuation(tmp_path,monkeypatch):
    write(tmp_path/'ACCEPTANCE.json',dict(report_start='9999-12-31'))
    monkeypatch.setattr(addendum.resource_policy,'binding',lambda *a:{'bound':'policy'})
    monkeypatch.setattr(addendum.resource_policy,'worker_limit',lambda *a:3)
    monkeypatch.setattr(queue,'inventory',lambda *a:[])
    calls=[]
    def prep(*args):calls.append('N4/N7');return dict(state='RUNNING',families={'N4':'PILOT_RUNNING','N7':'WAITING_FOR_SLOT'})
    def continuation(*args):calls.append('R3');return dict(state='RUNNING',families={'R3':'RUNNING_OR_ADMITTED'})
    monkeypatch.setattr(addendum.stock_v23,'step',prep);monkeypatch.setattr(queue,'step',continuation)
    config=dict(resource_binding={'bound':'policy'},preparation=dict(families={'N4':{},'N7':{}}),continuation=dict(families={'R3':{}}))
    assert addendum.step(tmp_path,tmp_path/'state',config)['state']=='RUNNING'
    assert calls==['N4/N7','R3']
    config['preparation']['families']['N4c']={}
    with pytest.raises(ValueError,match='unreviewed'):addendum.step(tmp_path,tmp_path/'state',config)
