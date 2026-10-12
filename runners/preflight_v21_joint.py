"""Two complete isolated engineering consumers; never a scientific replay."""
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v21 import world as W,oracle as O
from ghostscale.validation.soundingline.v21.studies import data,retain
from ghostscale.validation.soundingline.v21.joint_study import run
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from runners.review_v21 import verify


def setup(base,training_cases=256,validation_cases=128,test_cases=8):
    design=dict(handler='joint-architectures',card='G21-C02',engineering_only=True,tier='context',history=4,history_tier='complete',history_observed=True,observed=False,regime='original',lineages=[71000727],test_cases=test_cases,training_lineages=[71000728],validation_lineages=[71000729],training_cases=training_cases,validation_cases=validation_cases,fit_seeds=[0,1],parameter_budgets=[4096,8192],max_epochs=4)
    for shift in ('native','changed-tool'):
        root=base/shift;root.mkdir(parents=True)
        write(root/'PLAN.json',dict(design=dict(design,shift=shift),engineering_only=True))
        for fit in design['fit_seeds']:
            x,y=data(design,'training',fit);vx,vy=data(design,'validation',fit)
            (root/'training').mkdir(exist_ok=True);np.savez_compressed(root/'training'/f'{fit}_points.npz',evidence=x,labels=y,validation_evidence=vx,validation_labels=vy)
        for world in design['lineages']:
            rows=W.enumerate_world(world,shift);chosen=W.sampled(rows,test_cases,'c02-fixture',world,shift)
            x,_=W.inputs(rows,chosen,'context',4,('c02-fixture',world,shift),False,'complete',True)
            supplied,_=O.posterior(x,'context',[W.law(world,shift)],history_tier='complete',history_observed=True,shift=shift)
            (root/'reader').mkdir(exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
            np.savez_compressed(root/'reader'/f'{world}_points.npz',evidence=x)
            np.savez_compressed(root/'evaluator'/f'{world}_points.npz',truth=chosen[:,10].astype(int),supplied_logp=supplied)
            for fit in design['fit_seeds']:
                for name in ('converged-direct','naive-bayes','naive-bayes-without-evidence'):retain(root,f'{fit}-{world}-{name}',O.normalize_log(np.where(np.isfinite(supplied),0.,-np.inf)),chosen[:,10].astype(int))
        write(root/'SUMMARY.json',dict(engineering_only=True,screen_admitted=False,fit_records=[]))
        write(root/'COMPLETE.json',dict(plan_sha256=file_digest(root/'PLAN.json'),files=scientific_files(root)))
    design.update(headroom_root=str(base/'native'),shifted_headroom_root=str(base/'changed-tool'),headroom_complete_sha256=file_digest(base/'native/COMPLETE.json'),shifted_headroom_complete_sha256=file_digest(base/'changed-tool/COMPLETE.json'))
    return design


def preflight(base):
    d=setup(base/'inputs');receipts=[]
    for i in range(2):
        root=base/f'consumer-{i}';root.mkdir(parents=True);write(root/'PLAN.json',dict(design=d,engineering_only=True))
        summary=run(root,d,lambda **k:None);assert all(summary['controls'].values())
        write(root/'SUMMARY.json',summary);write(root/'COMPLETE.json',dict(plan_sha256=file_digest(root/'PLAN.json'),files=scientific_files(root)))
        receipts.append(verify(root))
    def strip(v):
        if isinstance(v,dict):return {k:strip(x) for k,x in v.items() if k!='fit_cpu_seconds'}
        if isinstance(v,list):return [strip(x) for x in v]
        return v
    a=read(base/'consumer-0/COMPLETE.json')['files'];b=read(base/'consumer-1/COMPLETE.json')['files'];assert a.keys()==b.keys()
    for name in a:
        if a[name]!=b[name]:assert name.endswith('.json') and strip(read(base/'consumer-0'/name))==strip(read(base/'consumer-1'/name)),name
    return dict(passed=True,engineering_only=True,complete_consumers=2,files=len(a),scores=receipts)
