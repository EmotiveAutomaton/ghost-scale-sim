"""Fixed-law aggregate joint state when a report also discloses its context."""
import numpy as np
from .aggregate_report_state import acquire
from .reachable_retrospective import group, TOL
from .retrospective_source import ALPHAS, transitions


def acquire_context(w, st, law, ids):
    q, _, _, likelihood = acquire(w, st, law, ids)
    ids = np.asarray(ids); N = len(ids); G = len(q)
    joint = np.zeros((G, 4, 8)); counts = np.zeros((4, 8), dtype=np.int32)
    full_joint = np.zeros((4, 8, len(w)))
    for c in range(4):
        selected = ids[:, 1] == c
        # Both include the probability of seeing this context, never a new prior.
        if selected.any():
            average = likelihood[selected].sum(0)/N
            joint[:, c] = group((w[:, None]*average).T, st).T
            full_joint[c] = np.einsum('h,she->eh', w, likelihood[selected])/N
            counts[c] = np.bincount(ids[selected, 2], minlength=8)
    return q, joint, counts, full_joint


def update(q, joint, counts):
    q = np.asarray(q, float); joint = np.asarray(joint, float); counts = np.asarray(counts)
    if q.ndim != 1 or not np.isfinite(q).all() or (q < 0).any() or abs(q.sum()-1) > TOL: raise ValueError('group mass')
    if joint.shape != (len(q), 4, 8) or not np.isfinite(joint).all() or (joint < 0).any() or np.max(abs(joint.sum((1, 2))-q)) > TOL: raise ValueError('context joint mass')
    if counts.shape != (4, 8) or counts.dtype.kind not in 'iu' or (counts < 0).any() or counts.sum() <= 0: raise ValueError('context copy counts')
    if np.max(abs(joint.sum((0, 2))-counts.sum(1)/counts.sum())) > TOL: raise ValueError('context probability')
    a = np.asarray(ALPHAS)[:, None, None, None]
    numerator = a*(counts/counts.sum())[None, :, :, None]*q[None, None, None, :] + (1-a)*joint.transpose(1, 2, 0)[None]
    probability = numerator.sum(-1); possible = probability > 0
    posterior = np.divide(numerator, probability[..., None], out=np.zeros_like(numerator), where=possible[..., None])
    return probability, possible, posterior


def evaluate(w, st, law, ids, operator=None):
    w = np.asarray(w, float); law = np.asarray(law, float); ids = np.asarray(ids)
    q, joint, counts, independent = acquire_context(w, st, law, ids)
    prob, possible, post = update(q, joint, counts)
    a = np.asarray(ALPHAS)[:, None, None, None]
    full = a*(counts/counts.sum())[None, :, :, None]*w[None, None, None, :] + (1-a)*independent[None]
    ref = full.sum(-1); support = ref > 0
    if not np.array_equal(possible, support): raise ValueError('context support')
    fullpost = np.divide(full, ref[..., None], out=np.zeros_like(full), where=support[..., None])
    truth = group(fullpost, st); delta = post-truth
    op = transitions(dict(future=st['signatures'])) if operator is None else operator
    def forecast(d):
        shape = d.shape[:-1]
        increments = (d.reshape(-1, len(q)) @ op).reshape(*shape, st['future'].shape[1], 16)
        return (increments.cumsum(axis=-2) @ law.reshape(16, 32)).reshape(*shape, -1, 4, 8)
    # Project joint masses once, before mixing the five copy probabilities.
    # Linearity preserves every coordinate while avoiding repeated projections.
    base_future = forecast(q)
    independent_future = forecast(joint.transpose(1, 2, 0))
    reference_independent_future = forecast(group(independent, st))
    aa = np.asarray(ALPHAS)[:, None, None, None, None, None]
    copied = (counts/counts.sum())[None, :, :, None, None, None]*base_future
    forecast_numer = aa*copied+(1-aa)*independent_future[None]
    reference_numer = aa*copied+(1-aa)*reference_independent_future[None]
    future = np.divide(forecast_numer, prob[..., None, None, None], out=np.zeros_like(forecast_numer), where=possible[..., None, None, None])
    reference_future = np.divide(reference_numer, ref[..., None, None, None], out=np.zeros_like(reference_numer), where=support[..., None, None, None])
    error = future-reference_future
    maximum = abs(error).max(axis=(-3, -2, -1)); squared = (error**2).sum(-1).mean(axis=(-2, -1)); tv = .5*abs(delta).sum(-1)
    if max(float(maximum.max()), float(tv.max()), float(abs(prob-ref).max())) > TOL: raise ValueError('context sufficiency')
    endpoint_prob = prob.sum(1)
    endpoint_mass = forecast_numer.sum(1)
    endpoint_future = np.divide(endpoint_mass, endpoint_prob[..., None, None, None], out=np.zeros_like(endpoint_mass), where=endpoint_prob[..., None, None, None] > 0)
    direction = future-endpoint_future[:, None]
    regret = (direction**2).sum(-1).mean(axis=(-2, -1))
    return dict(group_mass=q, joint_context_report_mass=joint, context_copy_counts=counts, source_rows=ids.astype(np.int32),
        report_probability=prob, reference_probability=ref, possible=possible, endpoint_report_probability=endpoint_prob,
        max_future_probability_error=np.where(possible, maximum, np.nan), future_squared_error=np.where(possible, squared, np.nan),
        updated_group_total_variation=np.where(possible, tv, np.nan), endpoint_only_squared_regret=np.where(possible, regret, np.nan),
        context_float64_count=np.array(33*len(q)), context_int32_count=np.array(32),
        endpoint_float64_count=np.array(9*len(q)), endpoint_int32_count=np.array(8),
        full_hypothesis_float64_count=np.array(len(w)), shared_law_float64_count=np.array(512),
        shared_group_schedule_int32_count=np.array(st['signatures'].size), full_schedule_int32_count=np.array(st['future'].size+st['past'].size))


def summarize(raw):
    q, joint, counts = raw['group_mass'], raw['joint_context_report_mass'], raw['context_copy_counts']
    p, possible, _ = update(q, joint, counts)
    if not np.array_equal(raw['possible'], possible) or np.max(abs(p-raw['report_probability'])) > TOL: raise ValueError('report state')
    hist = np.zeros((4, 8), dtype=np.int32)
    for _, c, e in raw['source_rows']: hist[c, e] += 1
    if not np.array_equal(hist, counts): raise ValueError('context histogram')
    if np.max(abs(p.sum(1)-raw['endpoint_report_probability'])) > TOL: raise ValueError('endpoint marginal')
    if raw['context_float64_count'] != 33*len(q) or raw['context_int32_count'] != 32 or raw['endpoint_float64_count'] != 9*len(q) or raw['endpoint_int32_count'] != 8: raise ValueError('state storage')
    result = dict(possible_reports=possible.sum((1,2)).astype(float), supported_contexts=(p.sum(2) > 0).sum(1).astype(float), max_report_probability_error=abs(p-raw['reference_probability']).max((1,2)))
    for field in ('max_future_probability_error', 'future_squared_error', 'updated_group_total_variation', 'endpoint_only_squared_regret'):
        value = raw[field]
        if value.shape != p.shape or not np.array_equal(np.isnan(value), ~possible) or not np.isfinite(value[possible]).all() or (value[possible] < 0).any(): raise ValueError('undefined/error mask')
        result['expected_'+field] = (p*np.nan_to_num(value)).sum((1,2))
        result['maximum_'+field] = np.nanmax(value, axis=(1,2))
    for field in raw:
        if field.endswith('_count'): result[field] = np.full(5, raw[field], dtype=float)
    result['context_state_bytes'] = 8*result['context_float64_count']+4*result['context_int32_count']
    result['endpoint_state_bytes'] = 8*result['endpoint_float64_count']+4*result['endpoint_int32_count']
    result['full_weight_state_bytes'] = 8*result['full_hypothesis_float64_count']
    return result


def controls():
    from .reachable_retrospective import prepare
    spec = dict(hypotheses=[['none',0,i] for i in range(16)],length=2,checkpoint=2,signatures=[[i] for i in range(16)],membership=list(range(16)))
    w = np.full(16, 1/16); st = prepare(spec,w); law = np.full((16,4,8),1/8)
    flat = summarize(evaluate(w,st,law,np.array([[1,0,0],[2,1,0]])))
    law[:8,0] = np.eye(8)[0]; law[8:,0] = np.eye(8)[1]
    law[:8,1] = np.eye(8)[1]; law[8:,1] = np.eye(8)[0]
    live = summarize(evaluate(w,st,law,np.array([[1,0,0],[2,1,0]])))
    single = summarize(evaluate(w,st,law,np.array([[1,0,0],[2,0,0]])))
    return {'live:informative_context': bool(live['expected_endpoint_only_squared_regret'][0] > .01),
        'positive:context_sufficiency': bool(live['maximum_max_future_probability_error'].max() < TOL),
        'placebo:constant_law': bool(flat['expected_endpoint_only_squared_regret'].max() < TOL),
        'placebo:single_context': bool(single['expected_endpoint_only_squared_regret'].max() < TOL),
        'placebo:certain_copy': bool(live['expected_endpoint_only_squared_regret'][-1] < TOL)}

import gzip
import json
import time
from itertools import product
from ..v18_3.io import read, write, canonical, file_digest
from .reachable_retrospective import prepare


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['alphas']!=list(ALPHAS) or cfg['report_state']!='fixed-uniform-source-joint-group-context-endpoint':raise ValueError('controls/design')
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
                        pulse(phase='aggregate-context-state',lineage=lineage,evidence=evidence,checkpoint=cp,row=index)
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
                    for ai,alpha in enumerate(ALPHAS):rows.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,alpha=alpha,**{k:r[k] for k in ('draw','initial_maker','kind','switched','duplicates','stream','report_sources')},**{k:float(v[ai]) for k,v in metrics.items()}))
                    source_total+=len(sources)
                    if (index+1)%cfg['batch_rows']==0 or index==len(chosen)-1:
                        np.savez_compressed(root/'raw'/f'{prefix}-{index//cfg["batch_rows"]:03d}_points.npz',**chunks);chunks={}
                with (root/'TIMING.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,posterior_rows=len(chosen),sources=sum(map(len,byrow.values())),cpu_seconds=time.process_time()-start))+'\n')
    (root/'raw/aggregate_context_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader input',evaluator='supplied fixed uniform source mixture,law,posterior and full source identities;aggregate joint future-group/context/report table;context-specific copy histogram;all future-coordinate errors;undefined reports remain NaN',scope='single report under fixed source prior/content/law;not arbitrary reweighting,disclosed source,two dependent reports,minimal state or optimized compact latency'))
    return dict(controls=checks,posterior_rows=len(rows)//len(ALPHAS),rows=len(rows),sources=source_total,report_queries=len(rows)*32,unavailable_checkpoints=unavailable,numerical_acceptance=False,scope='aggregate fixed context/report law versus full hypotheses and endpoint-only state;all32context/endpoint reports and future coordinates;no fitted model,observation or protected lineage')
