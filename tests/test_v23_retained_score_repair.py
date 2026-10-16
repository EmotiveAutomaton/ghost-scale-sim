import os
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v23 import adaptive_numerics,dispatch,family,ledger,n2_rescore,pipeline_new,registry,runtime
from ghostscale.validation.soundingline.v23.common import Direct,digest,read,write


def test_retained_score_repair_changes_no_prediction_or_draw(tmp_path,monkeypatch):
    v=registry.variants('N2')[79];p,t=dispatch.sample(v,'score-repair-engineering-fit',0,0)
    x=np.array([pipeline_new.features(v,p)]*4);y=np.array([t['y']]*4)
    model=Direct(x,y,x,y,2,'score-repair-engineering',0,max_epochs=1)
    fitroot=tmp_path/'fits';fitroot.mkdir();model.save(fitroot/'fixture.npz');write(fitroot/'FITS.json',{'fixture':model.metadata})
    directory=os.environ['V23_SCHEMA_DIRECTORY'];schema,hashes=ledger.canonical_schema(directory)
    source=tmp_path/'source';plan=dict(task='family-batch',family='N2',batch=1,worlds=2,cases=2,variants=[v['id']],
        schema_directory=directory,schema_pin=dict(files=hashes,schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash()),
        fit_directory=str(fitroot),fit_hashes={p.name:digest(p) for p in fitroot.iterdir()},canary_key='bb'*32)
    write(source/'PLAN.json',plan)
    for label in ('original','replay'):pipeline_new.run_batch(source/label,plan,lambda **kw:None)
    write(source/'REPLAY.json',runtime.compare(source/'original',source/'replay'))
    write(source/'COMPLETE.json',dict(state='verified',plan_sha256=digest(source/'PLAN.json'),replay_sha256=digest(source/'REPLAY.json'),summary_sha256=digest(source/'original/SUMMARY.json')))
    repaired=tmp_path/'repaired';new=dict(plan,task='n2-rescore',rescore_source=str(source),rescore_source_complete_sha256=digest(source/'COMPLETE.json'),
        numerical_validation_packet='engineering-only',numerical_validation_complete_sha256='engineering-only')
    write(repaired/'PLAN.json',new);monkeypatch.setattr(adaptive_numerics,'validate',lambda p:None)
    for label in ('original','replay'):n2_rescore.run(repaired/label,new,lambda **kw:None)
    write(repaired/'REPLAY.json',runtime.compare(repaired/'original',repaired/'replay'))
    write(repaired/'COMPLETE.json',dict(state='verified',plan_sha256=digest(repaired/'PLAN.json'),replay_sha256=digest(repaired/'REPLAY.json'),summary_sha256=digest(repaired/'original/SUMMARY.json')))
    assert family.reconstruct(repaired)['state']=='passed'
    for a,b in zip(family.rows(source/'original'/v['id']/'prediction_points.jsonl.gz'),family.rows(repaired/'original'/v['id']/'prediction_points.jsonl.gz'),strict=True):
        assert a['probabilities']==b['probabilities']
    assert read(repaired/'original/REANALYSIS.json')['new_sampling'] is False
    assert read(repaired/'original/SUMMARY.json')['variants'][0]['primary']['se']>0
