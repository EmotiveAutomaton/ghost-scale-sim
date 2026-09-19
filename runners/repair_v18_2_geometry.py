"""Recompute only constrained geometry cells, preserving fits and original raw data."""
import argparse
import gzip
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time


def repair(root,source,implementation):
    sys.path.insert(0,str(source))
    import numpy as np
    from ghostscale.validation.soundingline.v18_2 import learned as original,model as m,verify as v
    from ghostscale.validation.soundingline.v18_2.runtime import load
    from ghostscale.validation.soundingline.v16.records import read,write,file_digest,now
    cpu=time.process_time();wall=time.monotonic();plan=read(root/'PLAN.json');design=plan['design']
    assert {p:file_digest(source/p) for p in plan['sources']}==plan['sources']
    spec=importlib.util.spec_from_file_location('ghostscale.validation.soundingline.v18_2.geometry_repair_impl',implementation)
    fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)
    options={k:design.get(k,default) for k,default in [('aligned',False),('probe_mode','standard'),('reader_facts',False)] if k in inspect.signature(original.build_data).parameters}
    x,y,_=original.build_data(design['namespace']+'-train','train',1250,lambda:False,lambda **kw:None,**options)
    units=[u for name in read(root/'COMPLETE.json')['blocks'] for u in load(root,name)]
    frames=[]
    for unit in units:
        for tier in m.TIERS:
            for j in range(4):
                payload=m.public_packet(unit['case'],tier,j)
                feature,decode=original.represent(payload,design.get('aligned',False)) if hasattr(original,'represent') else (original.features(payload),list(range(16)))
                frames.append((feature,decode))
    predictions,geometry=fixed.geometry(np.array([f[0] for f in frames]),x,y,None,None,None,design.get('seed',0))
    assert geometry['full_inverse_max_error']<1e-10
    rows=[];cells={};offset=0
    for unit in units:
        case=unit['case'];per={}
        for tier in m.TIERS:
            for j,probe in enumerate(case['probes']):
                truth=m.artifacts(m.policy(case['world'],case['truth']['future_state'],probe['context']))
                for method,values in predictions.items():
                    q=np.zeros(16);q[frames[offset][1]]=values[offset]
                    scores=m.score(q,truth,probe['observed']['artifact'])
                    condition='crossed-heldout' if j==3 else 'base';key='|'.join(['test',condition,tier,method])
                    rows.append(dict(case_id=case['case_id'],tier=tier,probe=j,method=method,condition=condition,probabilities=q.tolist(),scores=scores))
                    for metric,value in scores.items():per.setdefault((key,metric),[]).append(value)
                offset+=1
        for (key,metric),values in per.items():cells.setdefault(key,{}).setdefault(metric,[]).append(float(np.mean(values)))
    raw=root/'GEOMETRY_REPAIR_points.json.gz';raw.write_bytes(gzip.compress(m.canonical(rows),mtime=0))
    (root/'GEOMETRY_REPAIR_SOURCE.py').write_bytes(implementation.read_bytes())
    record=dict(passed=True,at=now(),reason='Fit PCA and supervised axes in the actual nonlinear training representation, not the pre-transform covariance.',
        supersedes='all six original rank-limited geometry cells; original rows retained',unchanged='exact, split and flat predictions; trained weights and test observations',
        discovery_support='same exposed lineages; no new independent sample',units=len(units),rows=len(rows),geometry=geometry,
        source_plan_sha256=file_digest(root/'PLAN.json'),implementation_sha256=file_digest(implementation),raw_sha256=file_digest(raw),
        cells={k:{metric:v.interval(values) for metric,values in metrics.items()} for k,metrics in cells.items()},
        cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
    write(root/'GEOMETRY_REPAIR.json',record)
    write(root.parent/'audit/attempts'/f'{root.name}-geometry.json',dict(cpu_seconds=record['cpu_seconds'],wall_seconds=record['wall_seconds'],child_cpu_seconds=0))
    return dict(batch=root.name,passed=True,rows=len(rows),units=len(units),cpu_seconds=record['cpu_seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--implementation',type=Path,required=True)
    a=p.parse_args();print(json.dumps(repair(a.root,a.source,a.implementation)))
