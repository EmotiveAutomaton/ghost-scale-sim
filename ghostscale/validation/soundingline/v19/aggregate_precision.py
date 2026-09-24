"""Frozen storage precision diagnostic for a single uniform-source report."""
import numpy as np
from . import aggregate_report_state as base
from .retrospective_source import ALPHAS, transitions
TOL = base.TOL

DTYPES = ('float64', 'float32', 'float16')
MODES = ('direct', 'mass-preserving')
VARIANTS = tuple((dtype, mode) for dtype in DTYPES for mode in MODES)


def reconstruct(q, joint, mode):
    """Query in float64; a lost positive joint row invalidates reconstruction."""
    q = np.asarray(q, dtype=np.float64).copy()
    joint = np.asarray(joint, dtype=np.float64).copy()
    if mode not in MODES or q.ndim != 1 or joint.shape != (len(q), 8):
        raise ValueError('shape/mode')
    if not np.isfinite(q).all() or not np.isfinite(joint).all() or (q < 0).any() or (joint < 0).any():
        raise ValueError('mass')
    failed = False
    if mode == 'mass-preserving':
        sums = joint.sum(-1)
        failed = bool(q.sum() <= 0 or np.any((q > 0) & (sums == 0)))
        if not failed:
            q /= q.sum()
            joint *= np.divide(q, sums, out=np.zeros_like(q), where=sums > 0)[:, None]
    return q, joint, failed


def query(q, joint, counts):
    a = np.asarray(ALPHAS)[:, None, None]
    numerator = a*(counts/counts.sum())[None, :, None]*q[None, None, :] + (1-a)*joint.T[None, :, :]
    probability = numerator.sum(-1)
    possible = probability > 0
    posterior = np.divide(numerator, probability[..., None], out=np.zeros_like(numerator), where=possible[..., None])
    return probability, possible, posterior


def evaluate(w, st, law, ids, operator=None):
    q, joint, counts, _ = base.acquire(w, st, law, ids)
    probability, possible, exact = base.update(q, joint, counts)
    raw = dict(group_mass=q, joint_report_mass=joint, copy_counts=counts,
               report_probability=probability, reference_possible=possible,
               source_rows=np.asarray(ids, dtype=np.int32))
    failures=[]; predictions=[]; supports=[]; posteriors=[]
    drift=[]; underflow=[]
    for dtype in DTYPES:
        sq=q.astype(dtype); sj=joint.astype(dtype)
        # The exact masses are the float64 stored state. Other dtypes are
        # retained byte-for-byte; repaired query arrays are deterministic.
        if dtype != 'float64':
            raw['stored_group_'+dtype]=sq; raw['stored_joint_'+dtype]=sj
        for mode in MODES:
            uq, uj, failed = reconstruct(sq, sj, mode)
            pp, mm, post = query(uq, uj, counts)
            failures.append(failed)
            predictions.append(pp); supports.append(mm); posteriors.append(post)
            drift.append([abs(sq.astype(float).sum()-1), np.max(abs(sj.astype(float).sum(-1)-sq.astype(float)))])
            underflow.append([np.count_nonzero((q>0)&(sq==0)), np.count_nonzero((joint>0)&(sj==0))])
    failures=np.asarray(failures); supports=np.asarray(supports)
    defined=possible[None,:,:]&supports&~failures[:,None,None]
    delta=np.asarray(posteriors)-exact[None,:,:,:]
    op=transitions(dict(future=st['signatures'])) if operator is None else operator
    increments=(delta.reshape(-1,len(q))@op).reshape(6,5,8,st['future'].shape[1],16)
    future=(increments.cumsum(axis=-2)@np.asarray(law).reshape(16,32)).reshape(6,5,8,-1,4,8)
    maximum=abs(future).max(axis=(-3,-2,-1))
    squared=(future**2).sum(-1).mean(axis=(-2,-1))
    tv=.5*abs(delta).sum(-1)
    strata=np.where(probability==0,-1,np.where(probability<1e-6,0,np.where(probability<1e-3,1,2)))
    raw.update(reconstruction_failed=failures,
        approximate_report_probability=np.asarray(predictions),approximate_possible=supports,defined=defined,
        max_future_probability_error=np.where(defined,maximum,np.nan),future_squared_error=np.where(defined,squared,np.nan),
        updated_group_total_variation=np.where(defined,tv,np.nan),pre_repair_drift=np.asarray(drift),underflow_counts=np.asarray(underflow),
        report_strata=strata,state_bytes=np.array([np.dtype(d).itemsize*9*len(q)+counts.nbytes for d,m in VARIANTS]),
        shared_law_bytes=np.array(np.asarray(law,dtype=np.float64).nbytes),shared_schedule_bytes=np.array(st['signatures'].size*4))
    return raw


def summarize(raw):
    p=raw['report_probability']; possible=raw['reference_possible']; approx=raw['approximate_possible']; failed=raw['reconstruction_failed']
    if p.shape!=(5,8) or not np.isfinite(p).all() or (p<0).any() or np.max(abs(p.sum(-1)-1))>base.TOL or not np.array_equal(possible,p>0):
        raise ValueError('reference support')
    if approx.shape!=(6,5,8) or failed.shape!=(6,) or not np.array_equal(approx,raw['approximate_report_probability']>0):raise ValueError('approximate support')
    defined=possible[None,:,:]&approx&~failed[:,None,None]
    if not np.array_equal(defined,raw['defined']):raise ValueError('defined mask')
    q,joint,counts=raw['group_mass'],raw['joint_report_mass'],raw['copy_counts']
    rebuilt,mask,_=base.update(q,joint,counts)
    if not np.array_equal(mask,possible) or np.max(abs(rebuilt-p))>TOL:raise ValueError('reference state')
    if not np.array_equal(counts,np.bincount(raw['source_rows'][:,2],minlength=8)):raise ValueError('copy histogram')
    for vi,(dtype,mode) in enumerate(VARIANTS):
        sq,sj=(q,joint) if dtype=='float64' else (raw['stored_group_'+dtype],raw['stored_joint_'+dtype])
        if sq.dtype!=np.dtype(dtype) or sj.dtype!=np.dtype(dtype) or not np.array_equal(sq,q.astype(dtype)) or not np.array_equal(sj,joint.astype(dtype)):raise ValueError('stored cast')
        uq,uj,bad=reconstruct(sq,sj,mode);rp,rm,_=query(uq,uj,counts)
        if bad!=failed[vi] or not np.array_equal(rm,approx[vi]) or not np.array_equal(rp,raw['approximate_report_probability'][vi]):raise ValueError('reconstruction')
        if raw['state_bytes'][vi]!=np.dtype(dtype).itemsize*9*len(q)+32:raise ValueError('storage')
    result={
        'reconstruction_failed':np.broadcast_to(failed[:,None],(6,5)).astype(float),
        'lost_report_probability':(p[None,:,:]*(possible[None,:,:]&~approx)).sum(-1),
        'unusable_report_probability':(p[None,:,:]*(possible[None,:,:]&~defined)).sum(-1),
        'gained_support_count':(~possible[None,:,:]&approx).sum(-1).astype(float),
        'max_report_probability_error':abs(raw['approximate_report_probability']-p[None,:,:]).max(-1),
        'report_total_drift':abs(raw['approximate_report_probability'].sum(-1)-1),
    }
    for key,values in [('state_bytes',raw['state_bytes']),('group_underflows',raw['underflow_counts'][:,0]),('joint_underflows',raw['underflow_counts'][:,1]),('group_mass_drift_before_repair',raw['pre_repair_drift'][:,0]),('joint_row_drift_before_repair',raw['pre_repair_drift'][:,1])]:
        result[key]=np.broadcast_to(values[:,None],(6,5)).astype(float)
    for field in ('max_future_probability_error','future_squared_error','updated_group_total_variation'):
        v=raw[field]
        if v.shape!=(6,5,8) or not np.array_equal(np.isnan(v),~defined) or not np.isfinite(v[defined]).all() or (v[defined]<0).any():raise ValueError('undefined/error mask')
        result['defined_weighted_'+field]=(p[None,:,:]*np.nan_to_num(v)).sum(-1)
    strata=np.where(p==0,-1,np.where(p<1e-6,0,np.where(p<1e-3,1,2)))
    if not np.array_equal(strata,raw['report_strata']):raise ValueError('strata')
    for s in range(3):
        mass=p*(strata==s)
        result[f'stratum_{s}_mass']=np.broadcast_to(mass.sum(-1),(6,5))
        result[f'stratum_{s}_defined_mass']=(mass[None,:,:]*defined).sum(-1)
        result[f'stratum_{s}_squared_error_sum']=(mass[None,:,:]*np.nan_to_num(raw['future_squared_error'])).sum(-1)
    result['exact_zero_reports']=np.broadcast_to((p==0).sum(-1),(6,5)).astype(float)
    return result


def controls():
    from .reachable_retrospective import prepare,fixture
    st=prepare(fixture(),[.25,.25,.5]);w=np.array([.6,.1,.3]);ids=np.array([[1,0,0],[2,1,1]])
    law=np.full((16,4,8),1/8);flat=evaluate(w,st,law,ids)
    law[0,:,:2]=[.25,.75];law[0,:,2:]=0;live=evaluate(w,st,law,ids)
    _,_,failed=reconstruct(np.array([1.,2e-8]).astype('float16'),np.array([[.125]*8,[2e-8/8]*8]).astype('float16'),'mass-preserving')
    return {'live:rounding_detected':bool(live['pre_repair_drift'][-1].max()>0),
        'positive:float64_self':bool(np.nanmax(live['max_future_probability_error'][:2])<base.TOL),
        'placebo:constant_law':bool(np.nanmax(flat['max_future_probability_error'])<base.TOL),
        'positive:zero_group_not_reintroduced':not failed}


import gzip
import json
import time
from itertools import product
from ..v18_3.io import read, write, canonical, file_digest
from .reachable_retrospective import prepare


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['alphas']!=list(ALPHAS) or cfg['report_state']!='precision-uniform-source-joint-group-endpoint' or cfg['storage_dtypes']!=list(DTYPES) or cfg['reconstruction_modes']!=list(MODES):raise ValueError('controls/design')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    if not read(root/'inputs/PARENT_REVIEW.json')['numerical_acceptance']:raise ValueError('parent acceptance')
    specs=read(root/'inputs/SCHEDULES.json');structures={}
    for key,spec in specs.items():
        hs=spec['hypotheses'];st=prepare(spec,[.5/16 if h[0]=='none' else .5/(len(hs)-16) for h in hs]);structures[key]=(st,transitions(dict(future=st['signatures'])))
    rows=[];unavailable=[];source_total=0;paired={};(root/'raw').mkdir(exist_ok=True)
    for lineage in cfg['lineages']:
        law=np.asarray(read(root/'inputs/aware/evaluator'/f'{lineage}-law.json'))
        for evidence in ('aware','omitted'):
            base=root/'inputs'/evidence
            if not np.array_equal(law,read(base/'evaluator'/f'{lineage}-law.json')):raise ValueError('law pairing')
            maps=read(base/'evaluator'/f'{lineage}-joint-map.json')
            with np.load(base/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as z:arrays={n:z[n] for n in z.files if n.endswith('-unknown-time-type')}
            for key,spec in specs.items():
                length,cp=spec['length'],spec['checkpoint'];prefix=f'{lineage}-{evidence}-{key}'
                binding_path=root/'inputs/bindings'/(prefix+'-bindings.json')
                if not binding_path.exists():
                    if (length,cp) not in ((128,8),(128,17),(128,20)):raise ValueError('missing checkpoint')
                    unavailable.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp));continue
                binding=read(binding_path);st,op=structures[key];chosen=binding['rows']
                ids=[tuple(r[k] for k in ('draw','initial_maker','kind','switched','duplicates')) for r in chosen]
                expected=set(product(cfg['draws'],range(16),('purpose','skill'),(False,True),(False,True)))
                if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('paired roster')
                byrow={i:[] for i in range(len(chosen))}
                for i,t,ctx,e,source_id in binding['sources']:byrow[i].append((t,ctx,e,source_id))
                start=time.process_time();chunks={}
                for index,(r,identity) in enumerate(zip(chosen,ids)):
                    if index%cfg['batch_rows']==0:
                        pulse(phase='aggregate-precision',lineage=lineage,evidence=evidence,checkpoint=cp,row=index)
                    name,j=r['joint_array'],r['joint_row'];bound=maps[name]
                    if bound['hypotheses']!=spec['hypotheses'] or bound['rows'][j]!=[r['stream'],cp]:raise ValueError('posterior binding')
                    sources=byrow[index]
                    if len(sources)!=r['report_sources'] or len({s[3] for s in sources})!=len(sources):raise ValueError('source roster')
                    pair=(lineage,key,*identity);witness=(r['stream'],sources)
                    if evidence=='aware':paired[pair]=witness
                    elif paired[pair]!=witness:raise ValueError('source pairing')
                    w=arrays[name][j]
                    baseline=(np.bincount(st['future'][:,0],weights=w,minlength=16)@law.reshape(16,32)).reshape(4,8)
                    if not np.allclose(baseline,r['forecast'],atol=TOL,rtol=0):raise ValueError('parent forecast identity')
                    raw=evaluate(w,st,law,np.asarray([s[:3] for s in sources]),op);metrics=summarize(raw)
                    for k,v in raw.items():chunks[f'{index:03d}__'+k]=v
                    for vi,(dtype,mode) in enumerate(VARIANTS):
                        for ai,alpha in enumerate(ALPHAS):rows.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,alpha=alpha,storage_dtype=dtype,reconstruction=mode,**{k:r[k] for k in ('draw','initial_maker','kind','switched','duplicates','stream','report_sources')},**{k:float(v[vi,ai]) for k,v in metrics.items()}))
                    source_total+=len(sources)
                    if (index+1)%cfg['batch_rows']==0 or index==len(chosen)-1:
                        np.savez_compressed(root/'raw'/f'{prefix}-{index//cfg["batch_rows"]:03d}_points.npz',**chunks);chunks={}
                with (root/'TIMING.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,posterior_rows=len(chosen),sources=sum(map(len,byrow.values())),cpu_seconds=time.process_time()-start))+'\n')
    (root/'raw/aggregate_precision_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader input',evaluator='supplied fixed uniform source mixture,law,posterior and full source identities;quantized aggregate joint future-group/report table;integer copy histogram;pre-repair drift and underflow;undefined reports remain NaN;precision/reconstruction failures separate',scope='single report under fixed source prior/content/law;not arbitrary reweighting,disclosed source,two dependent reports,minimal state or optimized compact latency'))
    return dict(controls=checks,posterior_rows=len(rows)//(len(ALPHAS)*len(VARIANTS)),rows=len(rows),sources=source_total,report_queries=len(rows)*8,unavailable_checkpoints=unavailable,numerical_acceptance=False,scope='fixed float64/32/16 storage and direct/mass-preserving reconstruction;all endpoint reports and future coordinates;no fitted model,observation or protected lineage')
