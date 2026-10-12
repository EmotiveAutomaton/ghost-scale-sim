"""Complete engineering consumers; full scientific family replay is separate."""
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v21.drift import run
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from runners.review_v21 import verify
from runners.preflight_v21_sequential import strip_timing
import re
import numpy as np

def preflight(base):
    d=dict(handler='role-drift',card='G21-B04',engineering_only=True,lineages=[71000761],streams=2,horizons=[32,96],change_positions=[8,24,72])
    scores=[]
    for i in range(2):
        p=base/f'consumer-{i}';p.mkdir(parents=True);write(p/'PLAN.json',dict(design=d));s=run(p,d,lambda **k:None);assert all(s['controls'].values())
        for f in (p/'reader').glob('*.npz'):
            assert re.fullmatch(r'case-[0-9a-f]{16}_points.npz',f.name)
            with np.load(f) as z:assert set(z.files)=={'observations'} and z['observations'].shape[1] in d['horizons']
        for f in (p/'evaluator').glob('*_points.npz'):
            with np.load(f) as z:
                if 'reader_ids' not in z:continue
                for ident,horizon in zip(z['reader_ids'],z['horizons']):
                    with np.load(p/'reader'/f'{ident}_points.npz') as r:assert np.array_equal(r['observations'],z['observations'][:,:horizon])
        write(p/'SUMMARY.json',s);write(p/'COMPLETE.json',dict(plan_sha256=file_digest(p/'PLAN.json'),files=scientific_files(p)));scores.append(verify(p))
    a=read(base/'consumer-0/COMPLETE.json')['files'];b=read(base/'consumer-1/COMPLETE.json')['files'];assert a.keys()==b.keys()
    for n in a:
        if a[n]!=b[n]:assert n.endswith('.json') and strip_timing(read(base/'consumer-0'/n))==strip_timing(read(base/'consumer-1'/n)),n
    return dict(passed=True,engineering_only=True,complete_consumers=2,scientific_files=len(a),scores=scores)
