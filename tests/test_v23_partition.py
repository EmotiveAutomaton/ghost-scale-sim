from pathlib import Path
import os
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import partition,registry,ledger,runtime,family
from ghostscale.validation.soundingline.v23.common import Direct,read,write,digest


def test_partitions_refuse_overlap_missing_worlds_and_oversize():
    with pytest.raises(ValueError,match='overlap'):partition.merge_inputs([dict(worlds=[0,1]),dict(worlds=[1,3])],4)
    with pytest.raises(ValueError,match='individual packet'):partition.layout(16,100000)
    layout=partition.layout(16,30379,284)
    assert layout['total_cpu_cap']>30379 and layout['part_cpu_cap']<=14400
    with pytest.raises(ValueError,match='preverified'):partition.world_indices(dict(worlds=16,world_indices=[0,1]))


@pytest.mark.parametrize('name',['N3','R4'])
def test_complete_partition_assembly_matches_unsplit_science(tmp_path,name):
    from ghostscale.validation.soundingline.v23 import pipeline,pipeline_new,dispatch,repairs
    runner=pipeline_new if name.startswith('N') else pipeline
    generator=dispatch if name.startswith('N') else repairs
    directory=os.environ['V23_SCHEMA_DIRECTORY'];schema,hashes=ledger.canonical_schema(directory)
    v=registry.variants(name)[0];p,t=generator.sample(v,'engineering-partition-fit',0,0)
    x=np.array([runner.features(v,p)]*4);y=np.array([t['y']]*4)
    fitroot=tmp_path/'fits';fitroot.mkdir();model=Direct(x,y,x,y,4 if name=='N3' else 2,'partition-fixture',0,max_epochs=1)
    model.save(fitroot/'fixture.npz');write(fitroot/'FITS.json',{'fixture':model.metadata})
    root=tmp_path/'logical';plan=dict(family=name,batch=1,variants=[v['id']],worlds=4,cases=2,
        schema_directory=directory,schema_pin=dict(files=hashes,schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash()),
        fit_directory=str(fitroot),fit_hashes={p.name:digest(p) for p in fitroot.iterdir()},canary_key='ab'*32,
        sources={},scheduler_only=True,partition_layout=dict(chunks=[[0,1],[2,3]]))
    write(root/'PLAN.json',plan);parts=[]
    runner.run_batch(tmp_path/'whole',plan,lambda **kw:None)
    for index,worlds in enumerate(([0,1],[2,3])):
        part=root.with_name(root.name+'-part-'+str(index+1));pp=dict(plan,scheduler_only=False,world_indices=worlds,
            logical_parent=str(root.resolve()),logical_plan_sha256=digest(root/'PLAN.json'))
        write(part/'PLAN.json',pp)
        for label in ('original','replay'):runner.run_batch(part/label,pp,lambda **kw:None)
        write(part/'REPLAY.json',runtime.compare(part/'original',part/'replay'))
        write(part/'COMPLETE.json',dict(state='verified',plan_sha256=digest(part/'PLAN.json'),
            replay_sha256=digest(part/'REPLAY.json'),summary_sha256=digest(part/'original/SUMMARY.json')))
        parts.append(part)
    with pytest.raises(ValueError,match='partial worlds'):family.decide(name,[parts[0]],{})
    completion=partition.assemble(root,parts)
    assert not (root/'COMPLETE.json').exists()
    write(root/'COMPLETE.json',completion)
    assert family.reconstruct(root)['state']=='passed'
    assert read(root/'original/SUMMARY.json')==read(tmp_path/'whole/SUMMARY.json')
    for stream in ('public','truth','prediction','posterior','ledger'):
        path=v['id']+'/'+stream+'_points.jsonl.gz'
        assert list(family.rows(root/'original'/path))==list(family.rows(tmp_path/'whole'/path))


def test_retained_partition_requires_original_failure_and_unchanged_science(tmp_path):
    original=tmp_path/'original';part=tmp_path/'part';root=tmp_path/'repaired'
    keys=dict(family='R3',batch=1,worlds=4,cases=2,variants=['R3-0000'],
        fit_hashes={'fit':'a'},canary_key='ab',sources={'source':'b'})
    write(original/'PLAN.json',keys);write(original/'STATUS.json',dict(state='failed'))
    write(part/'COMPLETE.json',dict(state='verified'))
    plan=dict(keys,repair_parent=str(original),repair_parent_plan_sha256=digest(original/'PLAN.json'),
        repair_parent_status_sha256=digest(original/'STATUS.json'),partition_layout=dict(chunks=[[0,1],[2,3]]),
        retained_parts={'0':dict(path=str(part),complete_sha256=digest(part/'COMPLETE.json'))})
    child=dict(logical_parent=str(original.resolve()),logical_plan_sha256=digest(original/'PLAN.json'),world_indices=[0,1])
    partition.verify_part_binding(root,plan,part,child,0)
    assert partition.part_paths(root,plan)==[part,root.with_name(root.name+'-part-2')]
    plan['cases']=3
    with pytest.raises(ValueError,match='scientific definition'):partition.verify_part_binding(root,plan,part,child,0)
