"""Complete engineering-only consumer and replay for the B07 supplement."""
from pathlib import Path
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from ghostscale.validation.soundingline.v21.studies import headroom
from ghostscale.validation.soundingline.v21 import adaptation, adaptation_controls
from runners.review_v21 import verify


def run(base):
    d=dict(handler='headroom',card='G21-A01',tier='context',history=4,history_tier='complete',history_observed=True,
        observed=False,shift='changed-tool',history_shift='native',regime='original',training_shift='native',
        training_lineages=list(range(71000500,71000508)),validation_lineages=list(range(71000600,71000608)),
        training_cases=128,validation_cases=128,lineages=[71000700],test_cases=8,law_count=2,fit_seeds=[0])
    for label in ('native','shifted'):
        target=base/label;dd=dict(d)
        if label=='native':dd.update(shift='native',history_shift=None)
        summary=headroom(target,dd,lambda **kw:None)
        # Engineering-only forced gate, never scientific admission.
        summary['screen_admitted']=True
        for f in summary['fit_records']:f['early_stopped']=True
        write(target/'SUMMARY.json',summary);write(target/'PLAN.json',dict(design=dd))
        write(target/'COMPLETE.json',dict(plan_sha256=file_digest(target/'PLAN.json'),files=scientific_files(target)))
    original=base/'original';original.mkdir()
    dd=dict(d,handler='adaptation',headroom_root=str(base/'shifted'))
    summary=adaptation.run(original,dd,lambda **kw:None)
    write(original/'SUMMARY.json',summary);write(original/'PLAN.json',dict(design=dd))
    write(original/'COMPLETE.json',dict(plan_sha256=file_digest(original/'PLAN.json'),files=scientific_files(original)))
    design=dict(dd,handler='adaptation-controls',original_root=str(original),native_headroom_root=str(base/'native'),
        original_complete_sha256=file_digest(original/'COMPLETE.json'),native_complete_sha256=file_digest(base/'native/COMPLETE.json'),
        shifted_complete_sha256=file_digest(base/'shifted/COMPLETE.json'))
    receipts=[]
    for repeat in (0,1):
        target=base/f'supplement-{repeat}';target.mkdir()
        summary=adaptation_controls.run(target,design,lambda **kw:None)
        write(target/'SUMMARY.json',summary);write(target/'PLAN.json',dict(design=design))
        write(target/'COMPLETE.json',dict(plan_sha256=file_digest(target/'PLAN.json'),files=scientific_files(target)))
        receipts.append(verify(target))
    a=read(base/'supplement-0/COMPLETE.json')['files'];b=read(base/'supplement-1/COMPLETE.json')['files']
    assert a==b,'full scientific consumer differs'
    result=dict(passed=True,scope='Engineering fixture with explicit forced gate only; no discovery populations.',
        complete_consumer_replays=2,matched_files=len(a),scores=receipts)
    write(base/'VALIDITY.json',result)
    return result
