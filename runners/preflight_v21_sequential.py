"""Two complete engineering consumers; not the D04 scientific reference replay."""
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v21.sequential import run
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from runners.review_v21 import verify


def strip_timing(x):
    if isinstance(x,dict):return {k:strip_timing(v) for k,v in x.items() if not k.endswith('cpu_seconds')}
    if isinstance(x,list):return [strip_timing(v) for v in x]
    return x


def preflight(base):
    d=dict(handler='sequential',card='G21-D04',engineering_only=True,lineages=[71000740],test_cases=4,regime='original',shift='native',budgets=[1,2,4,8],fees=[0,.05,.2],entropy_threshold=.1)
    scores=[]
    for i in range(2):
        p=base/f'consumer-{i}';p.mkdir(parents=True);write(p/'PLAN.json',dict(design=d));s=run(p,d,lambda **k:None);assert all(s['controls'].values())
        write(p/'SUMMARY.json',s);write(p/'COMPLETE.json',dict(plan_sha256=file_digest(p/'PLAN.json'),files=scientific_files(p)));scores.append(verify(p))
    a=read(base/'consumer-0/COMPLETE.json')['files'];b=read(base/'consumer-1/COMPLETE.json')['files'];assert a.keys()==b.keys()
    for n in a:
        if a[n]!=b[n]:assert n.endswith('.json') and strip_timing(read(base/'consumer-0'/n))==strip_timing(read(base/'consumer-1'/n)),n
    return dict(passed=True,engineering_only=True,complete_consumers=2,scientific_files=len(a),scores=scores)
