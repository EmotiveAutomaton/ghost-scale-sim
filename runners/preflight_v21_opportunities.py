"""Two complete engineering consumers with an independently scored replay."""
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v21.opportunities import run
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from runners.review_v21 import verify
from runners.preflight_v21_sequential import strip_timing

def preflight(base,soundingline=None):
    d=dict(handler='opportunities',card='G21-D06',engineering_only=True,lineages=[71000833],test_cases=4,context_counts=[1,2,4,8]);receipts=[]
    for i in range(2):
        p=base/f'consumer-{i}';p.mkdir(parents=True);write(p/'PLAN.json',dict(design=d));s=run(p,d,lambda **k:None);assert all(s['controls'].values());write(p/'SUMMARY.json',s)
        write(p/'COMPLETE.json',dict(plan_sha256=file_digest(p/'PLAN.json'),files=scientific_files(p)));receipts.append(verify(p))
    a=read(base/'consumer-0/COMPLETE.json')['files'];b=read(base/'consumer-1/COMPLETE.json')['files'];assert a.keys()==b.keys()
    for n in a:
        if a[n]!=b[n]:assert n=='SUMMARY.json' and strip_timing(read(base/'consumer-0'/n))==strip_timing(read(base/'consumer-1'/n))
    return dict(passed=True,engineering_only=True,complete_consumers=2,scientific_files=len(a),scores=receipts)
