"""Retained-path audit and source-frozen cost gate for M's one numerical repair."""
from pathlib import Path
import time
import numpy as np
from . import m_remainder_r1 as module
from .common import read,write,digest,run_cell


def validate(setting,dest,pulse=lambda **kw:None):
    from runners.run_v31 import equivalent
    c=Path(setting['campaign']);checks=module.controls(dest/'controls',pulse)
    assert checks['passed'],checks
    rows=[]
    for job in sorted((c/'jobs').glob('M-CORE-*')):
        for lane in ('pilot','study'):
            old=job/f'output/{lane}-original'
            if not (old/'SUMMARY.json').exists() or not (old/'RAW_MANIFEST.json').exists():continue
            for name,sha in read(old/'RAW_MANIFEST.json').items():assert digest(old/name)==sha
            s=read(old/'SUMMARY.json')['setting'];root=dest/f'{job.name}-{lane}'
            run_cell(module,s,root/'original',lane=lane,pulse=pulse)
            assert digest(root/'original/cell_points.npz')==digest(old/'cell_points.npz')
            run_cell(module,s,root/'replay',lane=lane,pulse=pulse)
            equivalent(root/'original',root/'replay',True)
            run_cell(module,s,root/'reconstruct',mode='reconstruct',original=root/'original',lane=lane,pulse=pulse)
            equivalent(root/'original',root/'reconstruct')
            changed=[]
            with np.load(old/'cell_measures.npz',allow_pickle=False) as a,np.load(root/'original/cell_measures.npz',allow_pickle=False) as b:
                for key in a.files:
                    if not np.allclose(a[key],b[key],rtol=1e-10,atol=1e-8,equal_nan=True):changed.append(key)
            assert set(changed)<= {'maker_short_correlation_mean'},changed
            rows.append(dict(job=job.name,lane=lane,raw_sha256=digest(old/'cell_points.npz'),
                             changed_measures=changed,corrected_summary_sha256=digest(root/'original/SUMMARY.json')))
    assert any(x['job']=='M-CORE-0004' and x['lane']=='study' for x in rows)
    probe=dict(setting['probe'],worlds=8,blocks=8);started=time.process_time()
    run_cell(module,probe,dest/'profile-original',lane='excluded-cost-profile',pulse=pulse)
    run_cell(module,probe,dest/'profile-replay',lane='excluded-cost-profile',pulse=pulse)
    equivalent(dest/'profile-original',dest/'profile-replay',True)
    run_cell(module,probe,dest/'profile-reconstruct',mode='reconstruct',original=dest/'profile-original',lane='excluded-cost-profile',pulse=pulse)
    equivalent(dest/'profile-original',dest/'profile-reconstruct')
    used=time.process_time()-started
    return dict(passed=True,controls=checks,retained_paths=rows,full_seed_replay=True,
                independent_reconstruction=True,profile_cpu_seconds=used,profile_worlds=8,
                cpu_cap_seconds=max(120.,used/8*240*1.8),production_module=module.__name__)
