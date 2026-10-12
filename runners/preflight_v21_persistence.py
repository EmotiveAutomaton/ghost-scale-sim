"""Full B03 engineering consumer, independent score checks and repeat identity."""
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v21.persistence import run
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from runners.review_v21 import verify


def preflight(base):
    design=dict(handler='persistence',card='G21-B03',lineages=[71000723],test_cases=8,history=4,regime='original')
    receipts=[]
    for i in range(2):
        root=base/f'consumer-{i}';root.mkdir(parents=True)
        write(root/'PLAN.json',dict(design=design,engineering_only=True))
        summary=run(root,design,lambda **k:None);assert all(summary['controls'].values())
        write(root/'SUMMARY.json',summary)
        write(root/'COMPLETE.json',dict(plan_sha256=file_digest(root/'PLAN.json'),files=scientific_files(root)))
        receipts.append(verify(root))
    a=read(base/'consumer-0/COMPLETE.json')['files'];b=read(base/'consumer-1/COMPLETE.json')['files'];assert a==b
    return dict(passed=True,engineering_only=True,complete_consumers=2,files=len(a),scores=receipts)
