from copy import deepcopy
from pathlib import Path
import os
import pytest
from ghostscale.validation.soundingline.v23 import twin_partition,twin_pipeline as pipe,registry,ledger,runtime
from ghostscale.validation.soundingline.v23.common import read,write,digest,interval


def test_twin_partitions_retain_whole_worlds_and_reconstruct_complete_evidence(tmp_path,monkeypatch):
    # Four worlds is an engineering fixture, not a changed 16-world admission.
    variant=deepcopy(registry.variants('T4')[0]);variant['settings'].update(makers=2,items=2)
    monkeypatch.setattr(registry,'variants',lambda family:[variant])
    schema,_=ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    shape=dict(raters=2,items=2,ratings_per_rater=[2,2],raters_per_item=[2,2],
        rating_semantics='nominal_yes_no_unsure',rating_marginals=[.2,.3,.5])
    reference=dict(source=dict(revision='a'*40,sha256='b'*64,citation='engineering fixture only'),aggregates=shape)
    root=tmp_path/'logical';plan=dict(task='twin-batch',family='T4',batch=1,worlds=4,
        scheduler_only=True,partition_layout=dict(chunks=[[0,1],[2,3]]),fit_hashes={},fit_directory=str(tmp_path/'fits'),
        canary_key='ab'*32,sources={},empirical_reference_sha256='f'*64)
    write(root/'PLAN.json',plan);parts=[];whole=[]
    for index,worlds in enumerate(([0,1],[2,3])):
        part=root.with_name(root.name+f'-part-{index+1}')
        child=dict(plan,scheduler_only=False,world_indices=worlds,logical_parent=str(root.resolve()),
            logical_plan_sha256=digest(root/'PLAN.json'))
        write(part/'PLAN.json',child)
        for label in ('original','replay'):
            results=[pipe.evaluate_world(part/label/variant['id']/f'world-{world:04d}',variant,'engineering-partition-only',world,
                reference,schema,plan['canary_key'],lambda **kw:None) for world in worlds]
            gain=interval([row['world_gain'] for row in results],[variant['id'],'detectability'])
            write(part/label/'SUMMARY.json',dict(state='TWIN_BATCH_CHECKED',family='T4',variants=[dict(variant=variant,worlds=results,
                gain=gain,detected=bool(gain['finite'] and gain['lower']>.02))]))
            if label=='original':whole.extend(results)
        write(part/'REPLAY.json',runtime.compare(part/'original',part/'replay'))
        write(part/'COMPLETE.json',dict(state='verified',plan_sha256=digest(part/'PLAN.json'),
            replay_sha256=digest(part/'REPLAY.json'),summary_sha256=digest(part/'original/SUMMARY.json')))
        parts.append(part)
    receipt=twin_partition.assemble(root,parts)
    write(root/'COMPLETE.json',receipt)
    combined=read(root/'original/SUMMARY.json')['variants'][0]
    assert combined['worlds']==whole
    assert combined['gain']==interval([row['world_gain'] for row in whole],[variant['id'],'detectability'])
    assert pipe.reconstruct(root/'original',plan)['variants'][variant['id']]['worlds']==4
    assert len(list((root/'original'/variant['id']).glob('world-*/WORLD_GRAPH.json')))==4
    # A changed/missing whole world cannot produce a complete result.
    summary=read(root/'original/SUMMARY.json');summary['variants'][0]['worlds'].pop()
    write(root/'original/SUMMARY.json',summary,immutable=False)
    with pytest.raises(ValueError,match='world coverage'):pipe.reconstruct(root/'original',plan)


def test_twin_fits_reserve_partition_overhead_and_never_split_a_maker(tmp_path,monkeypatch):
    from ghostscale.validation.soundingline.v23 import twin_admission
    variant=registry.variants('T4')[0]
    monkeypatch.setattr(registry,'variants',lambda family:[variant])
    row=dict(profile=[dict(variant=variant['id'],cpu_seconds=1000)],
        training=[dict(training_generation_cpu_seconds=1,training_labels=128,fit_cpu_seconds=1,fit_epochs=2)])
    for label in ('original','replay'):write(tmp_path/label/'pilot-cost.json',row)
    cost=twin_admission.estimate(tmp_path,'T4',partitioned=True)
    assert cost['batch']>48000 and cost['partition_layout']['part_cpu_cap']<=14400
    assert [w for chunk in cost['partition_layout']['chunks'] for w in chunk]==list(range(16))
    row['profile'][0]['cpu_seconds']=6000
    for label in ('original','replay'):write(tmp_path/label/'pilot-cost.json',row,immutable=False)
    with pytest.raises(ValueError,match='individual packet limit'):twin_admission.estimate(tmp_path,'T4',partitioned=True)


def test_logical_twin_admission_binds_policy_and_never_launches_an_oversize_worker(tmp_path,monkeypatch):
    from test_v23_resource_policy import install
    from ghostscale.validation.soundingline.v23 import twin_admission,partition,resource_policy
    from runners import admit_v23_extension as admission
    install(tmp_path)
    acceptance=read(tmp_path/'ACCEPTANCE.json');acceptance['opened']='2020-01-01'
    # Install a complete fixture with matching immutable acceptance binding.
    write(tmp_path/'ACCEPTANCE.json',acceptance,immutable=False)
    supplement=read(tmp_path/resource_policy.SUPPLEMENT);supplement['acceptance_sha256']=digest(tmp_path/'ACCEPTANCE.json')
    write(tmp_path/resource_policy.SUPPLEMENT,supplement,immutable=False)
    pointer=read(tmp_path/'ACTIVE_RESOURCE_POLICY.json');pointer['acceptance_supplement']['sha256']=digest(tmp_path/resource_policy.SUPPLEMENT)
    write(tmp_path/'ACTIVE_RESOURCE_POLICY.json',pointer,immutable=False)
    (tmp_path/'packets').mkdir();write(tmp_path/'ENVIRONMENT.json',{})
    write(tmp_path/'SCHEMA_PIN.json',dict(committed_files_verified=True,directory='fixture-only'))
    monkeypatch.setattr(admission.runtime,'environment',lambda:{})
    monkeypatch.setattr(admission.ledger,'canonical_schema',lambda *args:(None,{}))
    monkeypatch.setattr(admission,'prerequisites',lambda *args:'a'*64)
    split=partition.layout(16,50000,min_worlds=1)
    monkeypatch.setattr(twin_admission,'validate',lambda *args:(split['total_cpu_cap'],0))
    monkeypatch.setattr(twin_admission,'estimate',lambda *args,**kw:dict(partition_layout=split))
    def forbidden(*args):raise AssertionError('logical oversized batch launched as one scientist')
    monkeypatch.setattr(admission.queue,'launch',forbidden)
    engine=tmp_path/'engine';write(engine/'PLAN.json',dict(external_source_files={},extra_pythonpath=[]))
    extras=dict(partitioned_twin=True,pilot_packet='fixture-only',engine_reference_packet=str(engine),batch=1,worlds=16)
    root=admission.admit(tmp_path,'v23-t1-batch-1','twin-batch','T1',split['total_cpu_cap'],extras,group='twins')
    plan=read(root/'PLAN.json')
    assert plan['scheduler_only'] and plan['partition_layout']==split and plan['max_parallel_parts']==3
    assert plan['acceptance_supplement']==pointer['acceptance_supplement']
    assert not (root/'LAUNCH.json').exists()
    with pytest.raises(ValueError,match='override admission authority'):
        admission.admit(tmp_path,'v23-t1-bad','twin-batch','T1',split['total_cpu_cap'],dict(extras,max_parallel_parts=100))
