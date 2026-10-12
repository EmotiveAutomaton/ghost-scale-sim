"""Complete bounded engineering consumer with an explicitly forced fixture gate."""
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v21 import world as W
from ghostscale.validation.soundingline.v21.studies import data
from ghostscale.validation.soundingline.v21.readers import Direct
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from ghostscale.validation.soundingline.v21.dependence import run
from runners.review_v21 import verify


def preflight(base,soundingline):
    source=base/'engineering-source';source.mkdir(parents=True)
    d=dict(handler='dependence',tier='context',history=4,history_tier='complete',history_observed=True,law_count='exact',
        training_lineages=[71000710,71000711],training_cases=128,validation_lineages=[71000712],validation_cases=64,
        lineages=[71000713],test_cases=8,fit_seeds=[0,1],states=4,iterations=3,soundingline=str(soundingline),export_fixtures=True)
    (source/'training').mkdir();(source/'reader').mkdir();(source/'evaluator').mkdir()
    for fit in (0,1):
        x,y=data(d,'training',fit);vx,vy=data(d,'validation',fit)
        np.savez_compressed(source/'training'/f'{fit}_points.npz',evidence=x,labels=y,validation_evidence=vx,validation_labels=vy)
        model=Direct(x,y,vx,vy,fit_seed=fit,max_epochs=2);model.save(source/'training'/f'{fit}-direct-model_points.npz')
    rows=W.enumerate_world(71000713);chosen=W.sampled(rows,8,'b02-engineering')
    xx,_=W.inputs(rows,chosen,'context',4,'b02-engineering',history_tier='complete',history_observed=True)
    np.savez_compressed(source/'reader/71000713_points.npz',evidence=xx)
    np.savez_compressed(source/'evaluator/71000713_points.npz',truth=chosen[:,10].astype(int),supplied_logp=np.full((8,128),-np.log(128)))
    write(source/'PLAN.json',dict(design=d,engineering_only=True))
    write(source/'SUMMARY.json',dict(headroom=dict(lower=1),fit_records=[dict(early_stopped=True)]*2,scope='Forced engineering gate, not scientific admission.'))
    write(source/'COMPLETE.json',dict(plan_sha256=file_digest(source/'PLAN.json'),files=scientific_files(source)))
    d.update(forecast_root=str(source),forecast_complete_sha256=file_digest(source/'COMPLETE.json'))
    receipts=[]
    for i in (0,1):
        root=base/f'consumer-{i}';root.mkdir();summary=run(root,d,lambda **k:None)
        assert all(summary['controls'].values())
        write(root/'PLAN.json',dict(design=d));write(root/'SUMMARY.json',summary)
        write(root/'COMPLETE.json',dict(plan_sha256=file_digest(root/'PLAN.json'),files=scientific_files(root)))
        receipts.append(verify(root))
    a=read(base/'consumer-0/COMPLETE.json')['files'];b=read(base/'consumer-1/COMPLETE.json')['files'];assert a.keys()==b.keys()
    def strip(v):
        if isinstance(v,dict):return {k:strip(x) for k,x in v.items() if k!='fit_cpu_seconds'}
        if isinstance(v,list):return [strip(x) for x in v]
        return v
    for name in a:
        if a[name]!=b[name]:
            assert name.endswith('.json') and strip(read(base/'consumer-0'/name))==strip(read(base/'consumer-1'/name)),name
    return dict(passed=True,engineering_only=True,complete_consumers=2,files=len(a),scores=receipts,canonical_copying_fixtures=12,
        scope='Tiny engineering fits and forced gate; full scientific design replay is separately required.')
