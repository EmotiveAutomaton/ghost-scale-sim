"""Complete supplied-law mixtures over one unknown change time and factor."""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import time
import numpy as np
from ..v18_3.io import canonical, digest, file_digest, write
from ..v18_3.world import rng
from .local_world import MAKERS, CONTEXTS
from . import transient_filter as T
from .crossed_review import population

ARMS = ('static', 'reset-16', 'known-time-type', 'unknown-time', 'unknown-time-type')
FLIPS = {kind: np.array([MAKERS.index(tuple(1-v if i==axis else v for i,v in enumerate(m))) for m in MAKERS])
         for kind,axis in (('purpose',0),('skill',1))}


def hypotheses(arm, length, kind):
    if kind not in FLIPS or arm not in ARMS: raise ValueError('unknown comparison')
    result = [('none',0,m) for m in range(16)]
    if arm in ('static','reset-16'): return result, np.full(16,1/16)
    times = [length//2] if arm=='known-time-type' else range(8,length-7)
    kinds = tuple(FLIPS) if arm=='unknown-time-type' else (kind,)
    changed = [(k,t,m) for k in kinds for t in times for m in range(16)]
    return result+changed, np.array([.5/16]*16+[.5/len(changed)]*len(changed))


def state_indices(hypotheses, step):
    return np.array([int(FLIPS[k][m]) if k!='none' and step>t else m for k,t,m in hypotheses])


def filter_checkpoints(table, observations, arm, length, kind, steps):
    hs,prior = hypotheses(arm,length,kind)
    if arm in ('static','reset-16'):
        return {step: (*T.posterior(table,observations[:step],arm,length//2,step),hs) for step in steps}
    logweights=np.log(prior); outputs={};seen={}
    for row in observations:
        identity=(row['source_step'],row['context'],row['endpoint']);source=row['source_id']
        if source in seen:
            if seen[source]!=identity: raise ValueError('conflicting duplicate source')
        else:
            seen[source]=identity
            current=state_indices(hs,row['source_step'])
            with np.errstate(divide='ignore'):logweights+=np.log(table[current,row['context'],row['endpoint']])
        step=row['step']
        if step in steps:
            joint=T.normalize(logweights)
            current=np.bincount(state_indices(hs,step),weights=joint,minlength=16)
            outputs[step]=(current,joint,hs)
    return outputs


def make_stream(table,lineage,draw,maker,length,switched,duplicates,kind):
    uniforms=rng('v19-transient-filter',lineage,draw,maker,length).random(length);rows=[]
    for i,u in enumerate(uniforms):
        step=i+1
        if duplicates and step%4==0: rows.append(dict(rows[-1],step=step));continue
        state=int(FLIPS[kind][maker]) if switched and step>length//2 else maker
        context=(i+i//4)%4
        endpoint=min(int(np.searchsorted(np.cumsum(table[state,context]),u,side='right')),7)
        rows.append(dict(step=step,source_step=step,source_id=f'source-{step:03d}',context=context,endpoint=endpoint))
    return rows


def controls():
    neutral=np.full((16,4,8),1/8)
    a=dict(step=1,source_step=1,source_id='a',context=0,endpoint=0)
    hs,prior=hypotheses('unknown-time-type',32,'purpose')
    flat,joint,_=filter_checkpoints(neutral,[a],'unknown-time-type',32,'purpose',[1])[1]
    masses={k:math.fsum(p for (x,t,m),p in zip(hs,prior) if x==k) for k in ('none','purpose','skill')}
    return dict(T.controls(), **{
        'positive:unknown_prior_mass':masses=={'none':.5,'purpose':.25,'skill':.25},
        'positive:complete_hypothesis_count':len(hs)==16*(1+2*17),
        'positive:skill_involution':bool(np.array_equal(FLIPS['skill'][FLIPS['skill']],np.arange(16))),
        'positive:future_switch_stays_initial':bool(np.array_equal(state_indices(hs,8),np.array([h[2] for h in hs]))),
        'placebo:unknown_uniform':bool(np.allclose(flat,1/16,rtol=0,atol=1e-14)),
        'placebo:unknown_prior_retained':bool(np.allclose(joint,prior,rtol=0,atol=1e-14))})


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('unknown change controls failed')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input differs')
    cells=[];packets={};stream_count=observation_count=row_count=0
    for lineage in cfg['lineages']:
        pulse(phase='independent-native-support',lineage=lineage)
        records=json.loads(gzip.decompress((root/'inputs'/f'lineage-{lineage}_points.json.gz').read_bytes()))
        if len(records)!=cfg['paths_per_lineage']:raise ValueError('path denominator')
        population(records,lineage,'original')
        table=T.endpoint_law(records);write(root/'evaluator'/f'{lineage}-law.json',table.tolist())
        streams=[]
        for draw,maker,length,kind,switched,duplicates in product(cfg['draws'],range(16),cfg['lengths'],FLIPS,(False,True),(False,True)):
            streams.append(dict(draw=draw,maker=maker,length=length,kind=kind,switched=switched,duplicates=duplicates,
                observations=make_stream(table,lineage,draw,maker,length,switched,duplicates,kind)))
        raw=root/'raw';raw.mkdir(exist_ok=True)
        (raw/f'{lineage}-observations_points.json.gz').write_bytes(gzip.compress(canonical(streams),mtime=0))
        rows=[];acc=defaultdict(list);joints=defaultdict(list);joint_map=defaultdict(list)
        for si,s in enumerate(streams):
            pulse(phase='unknown-time-filter',lineage=lineage,stream=si)
            length=s['length'];kind=s['kind'];steps=T.checkpoints(length)
            stream_count+=1;observation_count+=length
            for arm in ARMS:
                start=time.process_time()
                outputs=filter_checkpoints(table,s['observations'],arm,length,kind,steps)
                elapsed=time.process_time()-start
                with (root/'TIMING.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(lineage=lineage,stream=si,arm=arm,cpu_seconds=elapsed),sort_keys=True)+'\n')
                for step in steps:
                    current,joint,hs=outputs[step];joint=joint.reshape(-1)
                    actual=int(FLIPS[kind][s['maker']]) if s['switched'] and step>length//2 else s['maker']
                    values,forecast=T.score(table,current,actual)
                    key=f'{length}-{kind}-{arm}'
                    ji=len(joints[key]);joints[key].append(joint);joint_map[key].append([si,step])
                    types={k:math.fsum(float(w) for (x,t,m),w in zip(hs,joint) if x==k) for k in ('none','purpose','skill')}
                    row=dict(stream=si,draw=s['draw'],initial_maker=s['maker'],actual_maker=actual,length=length,kind=kind,
                        switched=s['switched'],duplicates=s['duplicates'],step=step,arm=arm,
                        unique_sources=len(T.unique_sources(s['observations'][:step])),posterior=current.tolist(),forecast=forecast.tolist(),
                        joint_array=key,joint_row=ji,type_mass=types,**values)
                    rows.append(row);acc[s['draw'],length,kind,s['switched'],s['duplicates'],step,arm].append(row)
            visible=dict(contexts=list(CONTEXTS),observations=s['observations']);ident=digest(visible)
            packets[ident]=dict(input_sha256=ident,inputs=visible)
        np.savez_compressed(raw/f'{lineage}-joint_points.npz',**{k:np.asarray(v,dtype=np.float64) for k,v in sorted(joints.items())})
        mapping={}
        for length,kind,arm in product(cfg['lengths'],FLIPS,ARMS):
            key=f'{length}-{kind}-{arm}';hs,prior=hypotheses(arm,length,kind)
            mapping[key]=dict(rows=joint_map[key],hypotheses=hs,prior=prior.tolist())
        write(root/'evaluator'/f'{lineage}-joint-map.json',mapping)
        (raw/f'{lineage}-forecasts_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0));row_count+=len(rows)
        for key,rr in sorted(acc.items()):
            if len(rr)!=16:raise ValueError('maker denominator')
            cells.append(dict(lineage=lineage,**dict(zip(('draw','length','kind','switched','duplicates','step','arm'),key)),makers=16,
                **{m:math.fsum(r[m] for r in rr)/16 for m in T.METRICS}))
    checks['positive:independent_native_support']=True
    write(root/'PUBLIC_PACKET.json',dict(schema='v19.unknown-change.reader.1',cases=[packets[k] for k in sorted(packets)],
        role='observed endpoints and source identities only; filters are evaluator references'))
    return dict(controls=checks,cells=cells,streams=stream_count,observations=observation_count,rows=row_count,public_packets=len(packets),fits=0,
        scope='supplied-law unknown-time/type mixture; planted middle switch; no-change type cells identical; no historical path claim')
