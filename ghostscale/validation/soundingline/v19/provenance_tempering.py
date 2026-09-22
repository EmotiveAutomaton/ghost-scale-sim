"""Fixed likelihood powers on retained streams; no fitting or inferred provenance."""
from collections import defaultdict
import gzip
import json
import math
import time
import numpy as np
from ..v18_3.io import canonical, file_digest, read, write
from . import unknown_change as U, transient_filter as T

POWERS = (1., .75, .5)
ARM = 'unknown-time-type'


def filter_powers(table, observations, length, kind, steps):
    hs, prior = U.hypotheses(ARM,length,kind)
    # Keep exponent-one addition order identical to the frozen parent.
    logs = np.broadcast_to(np.log(prior),(len(POWERS),len(prior))).copy()
    out = {}; seen = {}
    for i,row in enumerate(observations,1):
        if row['step'] != i: raise ValueError('nonconsecutive stream')
        identity=(row['source_step'],row['context'],row['endpoint']); source=row['source_id']
        if source in seen:
            if seen[source]!=identity: raise ValueError('source conflict')
        else:
            seen[source]=identity; current=U.state_indices(hs,row['source_step'])
            with np.errstate(divide='ignore'):
                likelihood=np.log(table[current,row['context'],row['endpoint']])
            for j,power in enumerate(POWERS): logs[j]+=power*likelihood
        if i in steps:
            state=U.state_indices(hs,i)
            for j,power in enumerate(POWERS):
                joint=T.normalize(logs[j]); posterior=np.bincount(state,weights=joint,minlength=16)
                out[i,power]=(posterior,joint)
    return out,hs,prior


def controls():
    neutral=np.full((16,4,8),1/8)
    a=dict(step=1,source_step=1,source_id='a',context=0,endpoint=0)
    duplicate=[a,dict(a,step=2)]; omitted=[a,dict(a,step=2,source_id='b')]
    flat,hs,prior=filter_powers(neutral,duplicate,32,'purpose',[2])
    law=neutral.copy();law[:8,:,0]=.3;law[:8,:,1:]=.7/7;law[8:,:,0]=.1;law[8:,:,1:]=.9/7
    aware,_,_=filter_powers(law,duplicate,32,'purpose',[2])
    missing,_,_=filter_powers(law,omitted,32,'purpose',[2])
    scalar=[]
    for power in POWERS:
        expected=prior*np.array([law[m,0,0]**(2*power) for _,_,m in hs]);expected/=expected.sum()
        scalar.append(np.allclose(missing[2,power][1],expected,rtol=0,atol=1e-14))
    old=U.filter_checkpoints(law,omitted,ARM,32,'purpose',[2])[2]
    return {'placebo:uniform_preserves_prior':all(np.allclose(v[1],prior,rtol=0,atol=1e-14) for v in flat.values()),
        'positive:independent_scalar_powers':bool(all(scalar)),
        'positive:parent_one_identity':bool(np.array_equal(old[0],missing[2,1.][0]) and np.array_equal(old[1],missing[2,1.][1])),
        'live:omitted_copy_changes_weight':bool(np.max(abs(aware[2,1.][0]-missing[2,1.][0]))>1e-3),
        'positive:half_two_copies_equals_one':bool(np.allclose(missing[2,.5][1],aware[2,1.][1],rtol=0,atol=1e-14))}


def gz(path):return json.loads(gzip.decompress(path.read_bytes()))


def run(root,plan,pulse):
    cfg=plan['design']; checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('tempering controls failed')
    if cfg['powers']!=list(POWERS):raise ValueError('power roster changed')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input differs')
    cells=[];rows_count=0;identities=0;streams_count=0
    for lineage in cfg['lineages']:
        pulse(phase='retained-stream-admission',lineage=lineage)
        table=np.asarray(read(root/'inputs/aware/evaluator'/f'{lineage}-law.json'))
        if table.shape!=(16,4,8) or not np.allclose(table.sum(-1),1,rtol=0,atol=1e-12):raise ValueError('invalid supplied law')
        allrows=[];joints=defaultdict(list);maps=defaultdict(list);acc=defaultdict(list)
        original=gz(root/'inputs/aware/raw'/f'{lineage}-observations_points.json.gz')
        expected={(d,m,l,k,s,b) for d in cfg['draws'] for m in range(16) for l in (32,128) for k in ('purpose','skill') for s in (False,True) for b in (False,True)}
        actual={tuple(s[k] for k in ('draw','maker','length','kind','switched','duplicates')) for s in original}
        if actual!=expected or len(actual)!=len(original):raise ValueError('stream roster')
        for evidence in ('aware','omitted'):
            parent=root/'inputs'/evidence
            if not np.array_equal(table,read(parent/'evaluator'/f'{lineage}-law.json')):raise ValueError('law differs')
            ss=gz(parent/'raw'/f'{lineage}-observations_points.json.gz')
            if len(ss)!=len(original):raise ValueError('paired streams')
            oldrows=gz(parent/'raw'/f'{lineage}-forecasts_points.json.gz')
            index={(r['stream'],r['step']):r for r in oldrows if r['arm']==ARM}
            mapping=read(parent/'evaluator'/f'{lineage}-joint-map.json')
            with np.load(parent/'raw'/f'{lineage}-joint_points.npz') as saved:
                # Only this retained arm is loaded; the other four remain immutable.
                arrays={k:saved[k] for k in saved.files if k.endswith('-'+ARM)}
            for si,s in enumerate(ss):
                pulse(phase='fixed-likelihood-powers',lineage=lineage,evidence=evidence,stream=si)
                base=original[si]
                if {k:v for k,v in s.items() if k!='observations'}!={k:v for k,v in base.items() if k!='observations'}:raise ValueError('stream pairing')
                for x,y in zip(s['observations'],base['observations']):
                    if {k:v for k,v in x.items() if k!='source_id'}!={k:v for k,v in y.items() if k!='source_id'}:raise ValueError('observations changed')
                if len(s['observations'])!=s['length']:raise ValueError('observation denominator')
                steps=T.checkpoints(s['length']);start=time.process_time()
                forecasts,hs,prior=filter_powers(table,s['observations'],s['length'],s['kind'],steps)
                with (root/'TIMING.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(lineage=lineage,evidence=evidence,stream=si,cpu_seconds=time.process_time()-start),sort_keys=True)+'\n')
                streams_count+=1
                for step in steps:
                    old=index.pop((si,step));oldkey=old['joint_array']
                    if mapping[oldkey]['hypotheses']!=[list(h) for h in hs] or mapping[oldkey]['prior']!=prior.tolist():raise ValueError('hypothesis/prior differs')
                    for power in POWERS:
                        posterior,joint=forecasts[step,power]
                        truth=int(U.FLIPS[s['kind']][s['maker']]) if s['switched'] and step>s['length']//2 else s['maker']
                        values,prediction=T.score(table,posterior,truth)
                        if power==1.:
                            if not np.array_equal(joint,arrays[oldkey][old['joint_row']]) or not np.array_equal(posterior,old['posterior']) or not np.array_equal(prediction,old['forecast']):raise ValueError('parent forecast identity')
                            if any(values[k]!=old[k] for k in T.METRICS):raise ValueError('parent score identity')
                            identities+=1
                        key=f'{evidence}-{s["length"]}-{s["kind"]}-{power:g}'
                        ji=len(joints[key]);joints[key].append(joint);maps[key].append([si,step])
                        row=dict(stream=si,draw=s['draw'],initial_maker=s['maker'],actual_maker=truth,length=s['length'],kind=s['kind'],switched=s['switched'],duplicates=s['duplicates'],step=step,evidence=evidence,power=power,posterior=posterior.tolist(),forecast=prediction.tolist(),joint_array=key,joint_row=ji,**values)
                        allrows.append(row);acc[s['draw'],s['length'],s['kind'],s['switched'],s['duplicates'],step,evidence,power].append(row)
            if index:raise ValueError('unmatched parent rows')
        raw=root/'raw';raw.mkdir(exist_ok=True)
        np.savez_compressed(raw/f'{lineage}-joint_points.npz',**{k:np.asarray(v) for k,v in sorted(joints.items())})
        write(root/'evaluator'/f'{lineage}-joint-map.json',dict(rows=dict(maps),hypothesis_definition='unchanged unknown-time-type roster/prior from paired retained inputs'))
        (raw/f'{lineage}-forecasts_points.json.gz').write_bytes(gzip.compress(canonical(allrows),mtime=0));rows_count+=len(allrows)
        for key,rows in sorted(acc.items()):
            if len(rows)!=16:raise ValueError('maker count')
            cells.append(dict(lineage=lineage,**dict(zip(('draw','length','kind','switched','duplicates','step','evidence','power'),key)),makers=16,**{m:math.fsum(r[m] for r in rows)/16 for m in T.METRICS}))
    checks['positive:all_exponent_one_parent_identities']=True
    write(root/'EVIDENCE_ROLES.json',dict(reader='No new evidence; reuse the two parent allowlisted public packets unchanged',scientific='tempered posteriors, forecasts, settings, times and scores',evaluator='laws, true maker/source identities and full hypothesis weights; never reader inputs'))
    return dict(controls=checks,cells=cells,rows=rows_count,paired_streams=streams_count,parent_identity_rows=identities,fits=0,
        scope='supplied-law fixed tempering, eight reused lineages/two draws; no inferred provenance, calibrated-confidence guarantee or historical correspondence')
