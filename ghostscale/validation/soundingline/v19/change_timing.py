"""Frozen-filter transfer to quarter/three-quarter actual change times."""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import time
import numpy as np
from ..v18_3.io import canonical, digest, file_digest, read, write
from ..v18_3.world import rng
from . import unknown_change as U, transient_filter as T
from .crossed_review import population
from .local_world import CONTEXTS


def hypotheses(arm, length, kind, switch_at):
    if not 8 <= switch_at <= length-8:
        raise ValueError('actual time outside admitted roster')
    hs, prior = U.hypotheses(arm, length, kind)
    if arm == 'known-time-type':
        hs = [(k, switch_at if k != 'none' else t, m) for k,t,m in hs]
    return hs, prior


def filter_checkpoints(table, observations, arm, length, kind, switch_at, steps):
    hs, prior = hypotheses(arm,length,kind,switch_at)
    if arm != 'known-time-type':
        return U.filter_checkpoints(table,observations,arm,length,kind,steps)
    logweights = np.log(prior); outputs = {}; seen = {}
    for row in observations:
        identity = (row['source_step'],row['context'],row['endpoint'])
        source = row['source_id']
        if source in seen:
            if seen[source] != identity:
                raise ValueError('conflicting duplicate source')
        else:
            seen[source] = identity
            current = U.state_indices(hs,row['source_step'])
            with np.errstate(divide='ignore'):
                logweights += np.log(table[current,row['context'],row['endpoint']])
        step = row['step']
        if step in steps:
            joint = T.normalize(logweights)
            current = np.bincount(U.state_indices(hs,step),weights=joint,minlength=16)
            outputs[step] = current,joint,hs
    return outputs


def make_stream(table,lineage,draw,maker,length,duplicates,kind,switch_at):
    uniforms = rng('v19-transient-filter',lineage,draw,maker,length).random(length)
    rows = []
    for i,u in enumerate(uniforms):
        step = i+1
        if duplicates and step%4 == 0:
            rows.append(dict(rows[-1],step=step));continue
        state = int(U.FLIPS[kind][maker]) if step>switch_at else maker
        context = (i+i//4)%4
        endpoint = min(int(np.searchsorted(np.cumsum(table[state,context]),u,side='right')),7)
        rows.append(dict(step=step,source_step=step,source_id=f'source-{step:03d}',
                         context=context,endpoint=endpoint))
    return rows


def near(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('nonfinite parent value or shape mismatch')
    error=float(np.max(abs(a-b)))
    if error>1e-12:raise ValueError('parent reconstruction differs')
    return error


def reproduce_parent(parent,lineage,table):
    """Recombine original scores and means, without refitting or new science."""
    streams=json.loads(gzip.decompress((parent/'raw'/f'{lineage}-observations_points.json.gz').read_bytes()))
    rows=json.loads(gzip.decompress((parent/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
    maximum=near(table,read(parent/'evaluator'/f'{lineage}-law.json'))
    identities=set()
    for s in streams:
        identity=tuple(s[k] for k in ('draw','maker','length','kind','switched','duplicates'))
        if identity in identities:raise ValueError('duplicate parent stream')
        identities.add(identity)
        original=U.make_stream(table,lineage,s['draw'],s['maker'],s['length'],s['switched'],s['duplicates'],s['kind'])
        if s['observations']!=original:raise ValueError('parent stream identity differs')
        if s['switched'] and make_stream(table,lineage,s['draw'],s['maker'],s['length'],s['duplicates'],s['kind'],s['length']//2)!=original:
            raise ValueError('midpoint stream differs')
    acc=defaultdict(list);row_ids=set()
    for row in rows:
        ident=row['stream'],row['arm'],row['step']
        if ident in row_ids:raise ValueError('duplicate parent score')
        row_ids.add(ident)
        s=streams[row['stream']]
        actual=int(U.FLIPS[s['kind']][s['maker']]) if s['switched'] and row['step']>s['length']//2 else s['maker']
        if actual!=row['actual_maker']:raise ValueError('parent actual maker differs')
        values,forecast=T.score(table,np.asarray(row['posterior']),actual)
        maximum=max(maximum,near(forecast,row['forecast']),*(near(values[m],row[m]) for m in T.METRICS))
        key=tuple(row[k] for k in ('draw','length','kind','switched','duplicates','step','arm'))
        acc[key].append(row)
    parent_cells=[x for x in read(parent/'SUMMARY.json')['cells'] if x['lineage']==lineage]
    if len(acc)!=len(parent_cells):raise ValueError('parent cell denominator differs')
    for cell in parent_cells:
        key=tuple(cell[k] for k in ('draw','length','kind','switched','duplicates','step','arm'))
        rr=acc[key]
        if len(rr)!=16 or cell['makers']!=16:raise ValueError('parent maker denominator differs')
        for metric in T.METRICS:maximum=max(maximum,near(math.fsum(r[metric] for r in rr)/16,cell[metric]))
    if len(rows)!=sum(len(T.checkpoints(s['length']))*len(U.ARMS) for s in streams):
        raise ValueError('parent row denominator differs')
    return dict(lineage=lineage,streams=len(streams),rows=len(rows),cells=len(parent_cells),
                maximum_error=maximum,stationary_and_midpoint='identities, not new replicates')


def controls():
    checks=U.controls()
    law=np.full((16,4,8),1/8)
    a=dict(step=8,source_step=8,source_id='a',context=0,endpoint=0)
    for at in (8,24):
        hs,prior=hypotheses('known-time-type',32,'purpose',at)
        cur,joint,_=filter_checkpoints(law,[a],'known-time-type',32,'purpose',at,[8])[8]
        checks[f'placebo:uniform-{at}']=bool(np.allclose(cur,1/16) and np.allclose(joint,prior))
        checks[f'positive:known-time-{at}']=set(t for k,t,m in hs if k!='none')=={at}
        checks[f'positive:no-future-remap-{at}']=bool(np.array_equal(U.state_indices(hs,at),[m for k,t,m in hs]))
    checks['positive:unknown-roster-fixed']=hypotheses('unknown-time-type',32,'skill',8)[0]==hypotheses('unknown-time-type',32,'skill',24)[0]
    return checks


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('change timing controls failed')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input differs')
    cells=[];packets={};reproductions=[];stream_count=observation_count=row_count=0
    for lineage in cfg['lineages']:
        pulse(phase='native-support-and-parent-reproduction',lineage=lineage)
        records=json.loads(gzip.decompress((root/'inputs'/f'lineage-{lineage}_points.json.gz').read_bytes()))
        if len(records)!=cfg['paths_per_lineage']:raise ValueError('path denominator')
        population(records,lineage,'original');table=T.endpoint_law(records)
        write(root/'evaluator'/f'{lineage}-law.json',table.tolist())
        reproductions.append(reproduce_parent(root/'inputs/parent',lineage,table))
        streams=[]
        for draw,maker,length,kind,quarter,duplicates in product(cfg['draws'],range(16),cfg['lengths'],U.FLIPS,cfg['quarters'],(False,True)):
            switch_at=length*quarter//4
            streams.append(dict(draw=draw,maker=maker,length=length,kind=kind,quarter=quarter,
                switch_at=switch_at,duplicates=duplicates,
                observations=make_stream(table,lineage,draw,maker,length,duplicates,kind,switch_at)))
        raw=root/'raw';raw.mkdir(exist_ok=True)
        (raw/f'{lineage}-observations_points.json.gz').write_bytes(gzip.compress(canonical(streams),mtime=0))
        rows=[];acc=defaultdict(list);joints=defaultdict(list);joint_map=defaultdict(list)
        for si,s in enumerate(streams):
            pulse(phase='off-midpoint-filter',lineage=lineage,stream=si)
            length=s['length'];kind=s['kind'];steps=T.checkpoints(length)
            stream_count+=1;observation_count+=length
            for arm in U.ARMS:
                start=time.process_time()
                outputs=filter_checkpoints(table,s['observations'],arm,length,kind,s['switch_at'],steps)
                elapsed=time.process_time()-start
                with (root/'TIMING.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(lineage=lineage,stream=si,arm=arm,cpu_seconds=elapsed),sort_keys=True)+'\n')
                for step in steps:
                    current,joint,hs=outputs[step];joint=joint.reshape(-1)
                    actual=int(U.FLIPS[kind][s['maker']]) if step>s['switch_at'] else s['maker']
                    values,forecast=T.score(table,current,actual)
                    key=f'{length}-{kind}-{s["quarter"]}-{arm}'
                    ji=len(joints[key]);joints[key].append(joint);joint_map[key].append([si,step])
                    types={k:math.fsum(float(w) for (x,t,m),w in zip(hs,joint) if x==k) for k in ('none','purpose','skill')}
                    row=dict(stream=si,draw=s['draw'],initial_maker=s['maker'],actual_maker=actual,
                        length=length,kind=kind,quarter=s['quarter'],switch_at=s['switch_at'],
                        duplicates=s['duplicates'],step=step,arm=arm,
                        unique_sources=len(T.unique_sources(s['observations'][:step])),posterior=current.tolist(),
                        forecast=forecast.tolist(),joint_array=key,joint_row=ji,type_mass=types,**values)
                    rows.append(row);acc[s['draw'],length,kind,s['quarter'],s['duplicates'],step,arm].append(row)
            visible=dict(contexts=list(CONTEXTS),observations=s['observations']);ident=digest(visible)
            packets[ident]=dict(input_sha256=ident,inputs=visible)
        np.savez_compressed(raw/f'{lineage}-joint_points.npz',**{k:np.asarray(v,dtype=np.float64) for k,v in sorted(joints.items())})
        mapping={}
        for length,kind,quarter,arm in product(cfg['lengths'],U.FLIPS,cfg['quarters'],U.ARMS):
            key=f'{length}-{kind}-{quarter}-{arm}';hs,prior=hypotheses(arm,length,kind,length*quarter//4)
            mapping[key]=dict(rows=joint_map[key],hypotheses=hs,prior=prior.tolist())
        write(root/'evaluator'/f'{lineage}-joint-map.json',mapping)
        (raw/f'{lineage}-forecasts_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0));row_count+=len(rows)
        for key,rr in sorted(acc.items()):
            if len(rr)!=16:raise ValueError('maker denominator')
            cells.append(dict(lineage=lineage,**dict(zip(('draw','length','kind','quarter','duplicates','step','arm'),key)),makers=16,
                **{m:math.fsum(r[m] for r in rr)/16 for m in T.METRICS}))
    checks.update({'positive:independent_native_support':True,'positive:parent_scores_and_midpoint_identity':True})
    write(root/'PARENT_REPRODUCTION.json',dict(passed=True,lineages=reproductions))
    write(root/'PUBLIC_PACKET.json',dict(schema='v19.change-timing.reader.1',cases=[packets[k] for k in sorted(packets)],
        role='observed endpoints and source identities only; filters and change timing are evaluator references'))
    return dict(controls=checks,cells=cells,streams=stream_count,observations=observation_count,
        rows=row_count,public_packets=len(packets),fits=0,scope='supplied-law filters; quarter and three-quarter actual switches; fixed checkpoints; saved midpoint and stationary identities are not new replicates')
