"""Known-answer engineering consumer and exact replay for noisy-record handling."""
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v21.studies import retain
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from ghostscale.validation.soundingline.v21.reliability import run
from runners.review_v21 import verify


def preflight(base, soundingline):
    source=base/'engineering-source';source.mkdir(parents=True)
    world=71000701;y=np.array([0,4,31,35,63,67,123,127])
    p=np.full((len(y),128),1/128)
    (source/'evaluator').mkdir();np.savez_compressed(source/'evaluator'/f'{world}_points.npz',truth=y)
    for fit in (0,1):retain(source,f'{fit}-{world}-converged-direct',np.log(p),y)
    # Forced complete gate on a known uniform prior; engineering only, never a scientific ceiling.
    write(source/'PLAN.json',dict(design=dict(law_count='exact',engineering_only=True)))
    write(source/'SUMMARY.json',dict(screen_admitted=True,headroom=dict(lower=1),fit_records=[dict(early_stopped=True)]*2,
        scope='Forced engineering consumer gate; synthetic known uniform forecasts; not discovery or exact-law evidence.'))
    write(source/'COMPLETE.json',dict(plan_sha256=file_digest(source/'PLAN.json'),files=scientific_files(source)))
    d=dict(handler='reliability',forecast_root=str(source),forecast_complete_sha256=file_digest(source/'COMPLETE.json'),
        lineages=[world],fit_seeds=[0,1],soundingline=str(soundingline),export_fixtures=True)
    receipts=[]
    for i in (0,1):
        root=base/f'consumer-{i}';root.mkdir()
        summary=run(root,d,lambda **kw:None)
        assert all(summary['controls'].values())
        write(root/'PLAN.json',dict(design=d));write(root/'SUMMARY.json',summary)
        write(root/'COMPLETE.json',dict(plan_sha256=file_digest(root/'PLAN.json'),files=scientific_files(root)))
        receipts.append(verify(root))
    assert read(base/'consumer-0/COMPLETE.json')['files']==read(base/'consumer-1/COMPLETE.json')['files']
    result=dict(passed=True,engineering_only=True,full_consumer_replays=2,
        files=len(read(base/'consumer-0/COMPLETE.json')['files']),scores=receipts,
        canonical_false_context_fixtures=12,scope='Synthetic uniform predictor, forced engineering gate; no discovery inference.')
    write(base/'VALIDITY.json',result);return result
