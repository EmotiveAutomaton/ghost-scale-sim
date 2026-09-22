"""Saved-stream source identity omission under the five frozen filters."""
from collections import defaultdict
import gzip
import json
import math
import time
import numpy as np
from ..v18_3.io import canonical, digest, file_digest, read, write
from . import unknown_change as U, transient_filter as T, change_timing as C
from .crossed_review import population
from .local_world import CONTEXTS


def supplied_observations(observations):
    """Only identity changes: preserve observed contents and both timestamps."""
    T.unique_sources(observations)  # Reject inconsistent original provenance.
    if [r['step'] for r in observations] != list(range(1, len(observations)+1)):
        raise ValueError('nonconsecutive observation times')
    return [dict(r, source_id=f'source-{r["step"]:03d}') for r in observations]


def controls():
    checks=U.controls()
    a=dict(step=1,source_step=1,source_id='source-001',context=0,endpoint=0)
    b=dict(a,step=2)
    obs=supplied_observations([a,b])
    law=np.full((16,4,8),1/8)
    flat=U.filter_checkpoints(law,obs,'unknown-time-type',32,'purpose',[2])[2][0]
    informative=law.copy();informative[:8,0]=[.8,.2,0,0,0,0,0,0]
    informative[8:,0]=[.2,.8,0,0,0,0,0,0]
    post=U.filter_checkpoints(informative,obs,'static',32,'purpose',[2])[2][0]
    checks.update({
        'placebo:neutral_omission':bool(np.allclose(flat,1/16)),
        'positive:duplicate_counted_twice':bool(np.allclose(post[:8],(.8**2/(.8**2+.2**2))/8)),
        'positive:time_and_content_preserved':all({k:v for k,v in x.items() if k!='source_id'}=={k:v for k,v in y.items() if k!='source_id'} for x,y in zip([a,b],obs)),
        'positive:distinct_supplied_ids':len(T.unique_sources(obs))==2,
    })
    return checks


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('source omission controls failed')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input differs')
    cells=[];packets={};reproductions=[];stream_count=observation_count=row_count=0
    for lineage in cfg['lineages']:
        pulse(phase='native-support-and-parent-reproduction',lineage=lineage)
        records=json.loads(gzip.decompress((root/'inputs'/f'lineage-{lineage}_points.json.gz').read_bytes()))
        if len(records)!=cfg['paths_per_lineage']:raise ValueError('path denominator')
        population(records,lineage,'original');table=T.endpoint_law(records)
        write(root/'evaluator'/f'{lineage}-law.json',table.tolist())
        parent=root/'inputs/parent'
        reproductions.append(C.reproduce_parent(parent,lineage,table))
        originals=json.loads(gzip.decompress((parent/'raw'/f'{lineage}-observations_points.json.gz').read_bytes()))
        oldrows=json.loads(gzip.decompress((parent/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
        old={(r['stream'],r['step'],r['arm']):r for r in oldrows}
        expected={(d,m,l,k,s,b) for d in cfg['draws'] for m in range(16) for l in cfg['lengths'] for k in U.FLIPS for s in (False,True) for b in (False,True)}
        actual={tuple(s[k] for k in ('draw','maker','length','kind','switched','duplicates')) for s in originals}
        if actual!=expected or len(actual)!=len(originals):raise ValueError('stream roster differs')
        streams=[dict(s,observations=supplied_observations(s['observations'])) for s in originals]
        raw=root/'raw';raw.mkdir(exist_ok=True)
        (raw/f'{lineage}-observations_points.json.gz').write_bytes(gzip.compress(canonical(streams),mtime=0))
        rows=[];acc=defaultdict(list);joints=defaultdict(list);joint_map=defaultdict(list)
        identities=0
        for si,s in enumerate(streams):
            pulse(phase='source-identity-omission',lineage=lineage,stream=si)
            length=s['length'];kind=s['kind'];steps=T.checkpoints(length)
            stream_count+=1;observation_count+=length
            for arm in U.ARMS:
                start=time.process_time()
                outputs=U.filter_checkpoints(table,s['observations'],arm,length,kind,steps)
                elapsed=time.process_time()-start
                with (root/'TIMING.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(lineage=lineage,stream=si,arm=arm,cpu_seconds=elapsed),sort_keys=True)+'\n')
                for step in steps:
                    current,joint,hs=outputs[step];joint=joint.reshape(-1)
                    truth=int(U.FLIPS[kind][s['maker']]) if s['switched'] and step>length//2 else s['maker']
                    values,forecast=T.score(table,current,truth)
                    if not s['duplicates']:
                        prior=old[si,step,arm]
                        if not np.array_equal(current,prior['posterior']) or not np.array_equal(forecast,prior['forecast']):
                            raise ValueError('independent-stream identity differs')
                        if any(values[m]!=prior[m] for m in T.METRICS):raise ValueError('independent score identity differs')
                        identities+=1
                    key=f'{length}-{kind}-{arm}';ji=len(joints[key])
                    joints[key].append(joint);joint_map[key].append([si,step])
                    types={k:math.fsum(float(w) for (x,t,m),w in zip(hs,joint) if x==k) for k in ('none','purpose','skill')}
                    row=dict(stream=si,draw=s['draw'],initial_maker=s['maker'],actual_maker=truth,length=length,kind=kind,
                        switched=s['switched'],duplicates=s['duplicates'],step=step,arm=arm,
                        supplied_sources=len(T.unique_sources(s['observations'][:step])),
                        true_sources=len(T.unique_sources(originals[si]['observations'][:step])),
                        posterior=current.tolist(),forecast=forecast.tolist(),joint_array=key,joint_row=ji,type_mass=types,**values)
                    rows.append(row);acc[s['draw'],length,kind,s['switched'],s['duplicates'],step,arm].append(row)
            visible=dict(contexts=list(CONTEXTS),observations=s['observations']);ident=digest(visible)
            packets[ident]=dict(input_sha256=ident,inputs=visible)
        np.savez_compressed(raw/f'{lineage}-joint_points.npz',**{k:np.asarray(v,dtype=np.float64) for k,v in sorted(joints.items())})
        mapping={}
        for key in sorted(joints):
            length,kind,arm=key.split('-',2);hs,prior=U.hypotheses(arm,int(length),kind)
            mapping[key]=dict(rows=joint_map[key],hypotheses=hs,prior=prior.tolist())
        write(root/'evaluator'/f'{lineage}-joint-map.json',mapping)
        (raw/f'{lineage}-forecasts_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0));row_count+=len(rows)
        if identities*2!=len(rows):raise ValueError('independent identity denominator')
        reproductions[-1]['independent_identity_rows']=identities
        for key,rr in sorted(acc.items()):
            if len(rr)!=16:raise ValueError('maker denominator')
            cells.append(dict(lineage=lineage,**dict(zip(('draw','length','kind','switched','duplicates','step','arm'),key)),makers=16,
                **{m:math.fsum(r[m] for r in rr)/16 for m in T.METRICS}))
    checks.update({'positive:parent_reproduced':True,'positive:independent_stream_exact_identity':True})
    write(root/'PARENT_REPRODUCTION.json',dict(passed=True,lineages=reproductions))
    write(root/'PUBLIC_PACKET.json',dict(schema='v19.source-omission.reader.1',cases=[packets[k] for k in sorted(packets)],
        role='observed endpoints and supplied distinct identities; source times remain visible; true identity and filters evaluator-only'))
    return dict(controls=checks,cells=cells,streams=stream_count,observations=observation_count,rows=row_count,public_packets=len(packets),fits=0,
        scope='identity-only omission with source times retained; frozen filters do not infer dependence from shared time/content; saved controls not new replicates')
