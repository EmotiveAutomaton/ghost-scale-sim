"""Complete engineering consumers only; scientific replay remains separate."""
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v21.acquisition import run
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from runners.review_v21 import verify


def strip_timing(value):
    if isinstance(value,dict):return {k:strip_timing(v) for k,v in value.items() if not k.endswith('_cpu_seconds')}
    if isinstance(value,list):return [strip_timing(v) for v in value]
    return value


def preflight(base):
    d=dict(handler='acquisition',card='G21-D01',engineering_only=True,lineages=[71000735],test_cases=8,regime='original',shift='native',fees=[0,.01,.05,.2])
    scores=[]
    for i in range(2):
        p=base/f'consumer-{i}';p.mkdir(parents=True);write(p/'PLAN.json',dict(design=d))
        summary=run(p,d,lambda **k:None);assert all(summary['controls'].values())
        write(p/'SUMMARY.json',summary);write(p/'COMPLETE.json',dict(plan_sha256=file_digest(p/'PLAN.json'),files=scientific_files(p)));scores.append(verify(p))
    a=read(base/'consumer-0/COMPLETE.json')['files'];b=read(base/'consumer-1/COMPLETE.json')['files'];assert a.keys()==b.keys()
    for n in a:
        if a[n]!=b[n]:assert n.endswith('.json') and strip_timing(read(base/'consumer-0'/n))==strip_timing(read(base/'consumer-1'/n)),n
    return dict(passed=True,engineering_only=True,complete_consumers=2,scientific_files=len(a),scores=scores)
