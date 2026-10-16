from datetime import datetime, timezone
from pathlib import Path
import zipfile
import pytest
from ghostscale.validation.soundingline.v23 import resource_policy as policy, partition
from ghostscale.validation.soundingline.v23.common import write, read, digest, canonical
from runners import queue_v23_families as queue
from runners.admit_v23_repair import budget_limits


def install(campaign):
    write(campaign/'ACCEPTANCE.json', dict(cpu_seconds=90*3600, protected_cpu_seconds=18*3600,
        max_scientific_workers=3, deadline='2026-10-22T05:04:15+00:00', report_start='2026-10-22T03:04:15+00:00'))
    source=campaign/'source.md';source.write_text('Explicit resource authorization fixture.')
    write(campaign/'RESOURCE_REALLOCATION.json', dict(total_cpu_hours=90,protected_cpu_hours=18,
        max_scientific_workers=3,authority='owner-approved-2026-10-15',group_caps_cpu_hours=dict(n2=24,n3=24,repairs=32)))
    row=dict(schema='v23-resource-supplement-v1',acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),
        scope='resources_and_unambiguous_engineering_only',authority=dict(user_message='Implement independent engineering.'),
        source_document=dict(path=str(source),sha256=digest(source)),cpu_seconds=285*3600,
        protected_cpu_seconds=18*3600,packet_cpu_cap_seconds=14400,
        deadline='2026-10-22T05:04:15+00:00',report_start='2026-10-22T03:04:15+00:00',
        group_caps_cpu_hours=dict(n7=45,n4=44,twins=40,repairs=48,confirmation=40),
        worker_schedule=dict(timezone='America/Los_Angeles',day_start=8,night_start=22,day_workers=8,
            night_workers=10,restricted_date='2026-10-21',restricted_workers=3),unverified_interference_worker_ceiling=3)
    write(campaign/policy.SUPPLEMENT,row)
    bound=dict(file=policy.SUPPLEMENT,sha256=digest(campaign/policy.SUPPLEMENT))
    write(campaign/'ACTIVE_RESOURCE_POLICY.json',dict(schema='v23-resource-policy-pointer-v1',acceptance_supplement=bound))
    (campaign/'attempts').mkdir()
    return row,bound


def test_new_budget_is_bound_while_old_packet_keeps_original(tmp_path):
    row,bound=install(tmp_path)
    original=digest(tmp_path/'ACCEPTANCE.json')
    assert policy.resolve(tmp_path)['cpu_seconds']==285*3600
    assert policy.resolve(tmp_path,{})['cpu_seconds']==90*3600
    assert policy.resolve(tmp_path,{policy.BINDING:bound})['cpu_seconds']==285*3600
    assert budget_limits(tmp_path)['n3']==24 and budget_limits(tmp_path)['n7']==45
    assert digest(tmp_path/'ACCEPTANCE.json')==original
    row['cpu_seconds']+=1;write(tmp_path/policy.SUPPLEMENT,row,immutable=False)
    with pytest.raises(ValueError,match='hash changed'):policy.resolve(tmp_path)
    # Existing jobs do not become dependent on a later document.
    assert policy.resolve(tmp_path,{})['cpu_seconds']==90*3600


def test_new_policy_never_weakens_original_reallocation_validation(tmp_path):
    install(tmp_path)
    path=tmp_path/'RESOURCE_REALLOCATION.json';row=read(path);row['total_cpu_hours']=285
    write(path,row,immutable=False)
    with pytest.raises(ValueError,match='unchanged total'):budget_limits(tmp_path)


def test_retained_partition_recovery_cannot_use_budget_its_old_runtime_lacks(tmp_path):
    install(tmp_path)
    write(tmp_path/'attempts/charged.json',dict(cpu_seconds=70*3600))
    assert queue.capacity(tmp_path,[],3*3600,needs_slot=False)=='AVAILABLE'
    assert queue.capacity(tmp_path,[],3*3600,needs_slot=False,policy_plan={})=='GLOBAL_CPU_RESERVE'


@pytest.mark.parametrize('family,group,cap',[('N2','n2',18000),('N3','n3',37932),('R3','repairs',27512)])
def test_pending_successors_fit_new_total_without_releasing_live_work(tmp_path,family,group,cap):
    install(tmp_path);write(tmp_path/'attempts/retained.json',dict(cpu_seconds=43.53*3600))
    entries=[(tmp_path/'active',dict(cpu_cap_seconds=23.08*3600,budget_group='n4',family='N4',scheduler_only=True),
              dict(state='partitioned',cpu_seconds=0))]
    assert queue.capacity(tmp_path,entries,cap,second_family=family,budget_group=group,needs_slot=False)=='AVAILABLE'
    assert queue.capacity(tmp_path,entries,250*3600,needs_slot=False)=='GLOBAL_CPU_RESERVE'


@pytest.mark.parametrize('utc,expected',[
    ('2026-10-17T04:59:59+00:00',8),('2026-10-17T05:00:00+00:00',10),
    ('2026-10-17T14:59:59+00:00',10),('2026-10-17T15:00:00+00:00',8),
    ('2026-10-21T07:00:00+00:00',3),('2026-10-22T06:59:59+00:00',3)])
def test_clock_edges_and_rehearsal_day(tmp_path,utc,expected):
    row,_=install(tmp_path);at=datetime.fromisoformat(utc)
    assert policy.scheduled_workers(row,at)==expected
    assert policy.worker_limit(tmp_path,at=at)==3 # Missing interference evidence cannot authorize expansion.


def test_fewer_slots_drain_without_cancelling_live_plans(tmp_path):
    install(tmp_path)
    plan=dict(cpu_cap_seconds=100,budget_group='repairs',family='R3')
    entries=[(tmp_path/str(i),plan,dict(state='running')) for i in range(4)]
    before=canonical(entries[0][1])
    assert queue.capacity(tmp_path,entries,100)=='NO_SLOT'
    assert canonical(entries[0][1])==before


def test_successor_runtime_migration_preserves_all_science(tmp_path):
    install(tmp_path);repo=Path(__file__).resolve().parents[1];source=tmp_path/'old'
    relative='ghostscale/validation/soundingline/v23/runtime.py'
    target=source/relative;target.parent.mkdir(parents=True);target.write_text('# old runtime')
    (source/'science.py').write_text('immutable_science = 1\n')
    archive=source.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x') as z:
        z.write(target,relative);z.write(source/'science.py','science.py')
    plan=dict(source_archive=str(archive),source_archive_sha256=digest(archive),
        sources={relative:digest(target),'science.py':digest(source/'science.py')},fit_hashes={'fit':'unchanged'},canary_key='same')
    result=policy.rebind_successor(tmp_path,'v23-r3-batch-2',plan,repo)
    assert result['fit_hashes']==plan['fit_hashes'] and result['canary_key']==plan['canary_key']
    assert result['sources']['science.py']==plan['sources']['science.py']
    assert digest(target)==plan['sources'][relative] and result['sources'][relative]==digest(repo/relative)
    assert policy.resolve(tmp_path,result)['cpu_seconds']==285*3600


def test_smaller_execution_parts_preserve_all_worlds_and_auxiliary_cost():
    layout=partition.layout(16,66000,600,min_worlds=1)
    assert [w for chunk in layout['chunks'] for w in chunk]==list(range(16))
    assert max(map(len,layout['chunks']))==2 and layout['part_cpu_cap']<=14400
    assert layout['total_cpu_cap']>=66000+600*(len(layout['chunks'])-1)
    with pytest.raises(ValueError,match='individual packet'):partition.layout(16,16*15000,min_worlds=1)


def test_confirmation_projection_exposes_serial_critical_path():
    start=datetime(2026,10,21,5,tzinfo=timezone.utc)
    end=policy.projected_finish(0,3,start,serial_cpu_seconds=40*3600)
    assert end>datetime(2026,10,22,3,4,15,tzinfo=timezone.utc)


def test_clock_quantile_canonical_and_alias_replay_identically():
    from ghostscale.validation.soundingline.v23 import twin_chess,registry
    variant=registry.variants('T2')[0]
    def sample(**columns):
        return canonical(twin_chess.player(variant,'clock-contract-test',0,0,games=1,
                         clock_quantiles=dict(probabilities=[0,1],**columns)))
    assert sample(values=[1,120])==sample(seconds=[1,120])==sample(values=[1,120],seconds=[1,120])
    with pytest.raises(ValueError,match='conflicting'):sample(values=[1,120],seconds=[2,120])
