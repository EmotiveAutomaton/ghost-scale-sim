"""Complete E01 engineering consumers, separate from scientific replay."""
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v21.selection import run
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from runners.review_v21 import verify
from runners.preflight_v21_sequential import strip_timing

def preflight(base,soundingline):
    d=dict(handler='selection',card='G21-E01',engineering_only=True,lineages=[71000804],test_cases=8,descendants=[2,4,8],cosmetic_edits=[0,1,4,16],export_fixtures=True,soundingline=str(soundingline));receipts=[]
    for i in range(2):
        p=base/f'consumer-{i}';p.mkdir(parents=True);write(p/'PLAN.json',dict(design=d));s=run(p,d,lambda **k:None);assert all(s['controls'].values());write(p/'SUMMARY.json',s)
        write(p/'COMPLETE.json',dict(plan_sha256=file_digest(p/'PLAN.json'),files=scientific_files(p)));receipts.append(verify(p))
    a=read(base/'consumer-0/COMPLETE.json')['files'];b=read(base/'consumer-1/COMPLETE.json')['files'];assert a.keys()==b.keys()
    for name in a:
        if a[name]!=b[name]:assert name=='SUMMARY.json' and strip_timing(read(base/'consumer-0'/name))==strip_timing(read(base/'consumer-1'/name))
    return dict(passed=True,engineering_only=True,complete_consumers=2,scientific_files=len(a),scores=receipts,selection_fixtures=12)
