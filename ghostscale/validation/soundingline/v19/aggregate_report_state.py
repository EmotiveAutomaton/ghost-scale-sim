"""Exact aggregate state for one fixed uniform unknown-source report law.

The table is tied to the supplied law and source mixture. It does not answer
source disclosure, a changed source prior, or a second dependent report.
"""
import numpy as np
from .reachable_retrospective import group, TOL
from .retrospective_source import ALPHAS, transitions


def acquire(w, st, law, ids):
    w = np.asarray(w, float); law = np.asarray(law, float); ids = np.asarray(ids)
    if w.shape != (len(st['mapping']),) or not np.isfinite(w).all() or (w < 0).any() or abs(w.sum()-1) > TOL:
        raise ValueError('weights')
    if law.shape != (16, 4, 8) or not np.isfinite(law).all() or (law < 0).any() or np.max(abs(law.sum(-1)-1)) > TOL:
        raise ValueError('law')
    if ids.ndim != 2 or ids.shape[1] != 3 or not len(ids) or ids.dtype.kind not in 'iu':
        raise ValueError('sources')
    t, c, old = ids.T
    if (t < 1).any() or (t > len(st['past'])).any() or len(set(t)) != len(t) or (c < 0).any() or (c > 3).any() or (old < 0).any() or (old > 7).any():
        raise ValueError('source bounds')
    likelihood = law[st['past'][t-1], c[:, None], :]
    if np.any(np.einsum('h,sh->s', w, likelihood[np.arange(len(t)), :, old]) <= 0):
        raise ValueError('old endpoint unsupported')
    # Contract the source dimension first; retain no separate source table.
    average_likelihood = likelihood.mean(axis=0)
    q = group(w, st)
    joint = group((w[:, None]*average_likelihood).T, st).T
    counts = np.bincount(old, minlength=8).astype(np.int32)
    return q, joint, counts, likelihood


def update(q, joint, counts):
    q = np.asarray(q, float); joint = np.asarray(joint, float); counts = np.asarray(counts)
    if q.ndim != 1 or not np.isfinite(q).all() or (q < 0).any() or abs(q.sum()-1) > TOL:
        raise ValueError('group mass')
    if joint.shape != (len(q), 8) or not np.isfinite(joint).all() or (joint < 0).any() or np.max(abs(joint.sum(-1)-q)) > TOL:
        raise ValueError('joint mass')
    if counts.shape != (8,) or counts.dtype.kind not in 'iu' or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError('copy counts')
    a = np.asarray(ALPHAS)[:, None, None]
    numerator = a*(counts/counts.sum())[None, :, None]*q[None, None, :] + (1-a)*joint.T[None, :, :]
    probability = numerator.sum(-1)
    possible = probability > 0
    posterior = np.divide(numerator, probability[..., None], out=np.zeros_like(numerator), where=possible[..., None])
    return probability, possible, posterior


def evaluate(w, st, law, ids, operator=None):
    w = np.asarray(w, float); law = np.asarray(law, float); ids = np.asarray(ids)
    q, joint, counts, likelihood = acquire(w, st, law, ids)
    probability, possible, post = update(q, joint, counts)
    # Independent order of contraction for full-hypothesis Bayes: weight and
    # sum each source's report numerator before projecting onto future groups.
    a = np.asarray(ALPHAS)[:, None, None]
    full_independent = np.einsum('h,she->eh', w, likelihood)/len(ids)
    numerator = a*(counts/counts.sum())[None, :, None]*w[None, None, :] + (1-a)*full_independent[None, :, :]
    reference_probability = numerator.sum(-1)
    reference_possible = reference_probability > 0
    full_post = np.divide(numerator, reference_probability[..., None], out=np.zeros_like(numerator), where=reference_possible[..., None])
    reference_group = group(full_post, st)
    if not np.array_equal(possible, reference_possible):
        raise ValueError('support disagreement')
    delta = post-reference_group
    op = transitions(dict(future=st['signatures'])) if operator is None else operator
    increments = (delta.reshape(-1, len(q)) @ op).reshape(5, 8, st['future'].shape[1], 16)
    future_delta = (increments.cumsum(axis=-2) @ law.reshape(16, 32)).reshape(5, 8, -1, 4, 8)
    maximum = abs(future_delta).max(axis=(-3, -2, -1))
    squared = (future_delta**2).sum(-1).mean(axis=(-2, -1))
    tv = .5*abs(delta).sum(-1)
    if max(float(maximum.max()), float(tv.max()), float(abs(probability-reference_probability).max())) > TOL:
        raise ValueError('aggregate sufficiency')
    if '_aggregate_costs' not in st:
        costs = []
        for past in st['past']:
            cells = np.unique(st['mapping']*16+past)
            sizes = np.bincount(cells//16, minlength=len(q))
            costs.append((int(sizes[sizes > 1].sum()), len(cells)))
        st['_aggregate_costs'] = np.asarray(costs, dtype=np.int64)
    mixed, cells = st['_aggregate_costs'][ids[:, 0]-1].sum(0)
    return dict(group_mass=q, joint_report_mass=joint, copy_counts=counts,
                source_rows=ids.astype(np.int32), report_probability=probability,
                reference_probability=reference_probability, possible=possible,
                max_future_probability_error=np.where(possible, maximum, np.nan),
                future_squared_error=np.where(possible, squared, np.nan),
                updated_group_total_variation=np.where(possible, tv, np.nan),
                aggregate_float64_count=np.array(9*len(q)), aggregate_int32_count=np.array(8),
                source_table_float64_count=np.array(len(q)+mixed),
                source_table_int32_count=np.array(8+3*len(ids)+2*cells),
                full_hypothesis_float64_count=np.array(len(w)),
                shared_law_float64_count=np.array(512),
                shared_group_schedule_int32_count=np.array(st['signatures'].size),
                full_schedule_int32_count=np.array(st['future'].size+st['past'].size))


def summarize(raw):
    p, possible = raw['report_probability'], raw['possible']
    if p.shape != (5, 8) or not np.isfinite(p).all() or (p < 0).any() or np.max(abs(p.sum(-1)-1)) > TOL or not np.array_equal(possible, p > 0):
        raise ValueError('report support')
    q, joint, counts = raw['group_mass'], raw['joint_report_mass'], raw['copy_counts']
    rebuilt, support, _ = update(q, joint, counts)
    if not np.array_equal(support, possible) or np.max(abs(rebuilt-p)) > TOL:
        raise ValueError('state report mismatch')
    if not np.array_equal(counts, np.bincount(raw['source_rows'][:, 2], minlength=8)):
        raise ValueError('copy histogram')
    result = dict(possible_reports=possible.sum(-1).astype(float),
                  max_report_probability_error=abs(p-raw['reference_probability']).max(-1))
    for field in ('max_future_probability_error', 'future_squared_error', 'updated_group_total_variation'):
        value = raw[field]
        if value.shape != p.shape or not np.array_equal(np.isnan(value), ~possible) or not np.isfinite(value[possible]).all() or (value[possible] < 0).any():
            raise ValueError('undefined/error mask')
        result['expected_'+field] = (p*np.nan_to_num(value)).sum(-1)
        result['maximum_'+field] = np.nanmax(value, axis=-1)
    for field in ('aggregate_float64_count', 'aggregate_int32_count', 'source_table_float64_count', 'source_table_int32_count', 'full_hypothesis_float64_count', 'shared_law_float64_count', 'shared_group_schedule_int32_count', 'full_schedule_int32_count'):
        result[field] = np.full(5, raw[field], dtype=float)
    if raw['aggregate_float64_count'] != 9*len(q) or raw['aggregate_int32_count'] != 8:
        raise ValueError('aggregate storage')
    result['aggregate_state_bytes'] = 8*result['aggregate_float64_count']+4*result['aggregate_int32_count']
    result['source_table_state_bytes'] = 8*result['source_table_float64_count']+4*result['source_table_int32_count']
    result['full_weight_state_bytes'] = 8*result['full_hypothesis_float64_count']
    return result


def controls():
    from .reachable_retrospective import prepare, fixture
    st = prepare(fixture(), [.25, .25, .5]); w = np.array([.6, .1, .3])
    law = np.full((16, 4, 8), 1/8); ids = np.array([[1, 0, 0], [2, 1, 1]])
    flat = evaluate(w, st, law, ids)
    law[0, :, :2] = [.25, .75]; law[0, :, 2:] = 0
    live = evaluate(w, st, law, ids)
    _, _, posterior = update(live['group_mass'], live['joint_report_mass'], live['copy_counts'])
    return {'live:informative_report': bool(abs(posterior[0]-live['group_mass']).max() > .001),
            'positive:exact_aggregate': bool(np.nanmax(live['max_future_probability_error']) < TOL),
            'placebo:constant_law': bool(np.max(abs(flat['joint_report_mass']-flat['group_mass'][:, None]/8)) < TOL),
            'placebo:certain_copy': bool(np.max(abs(posterior[-1, live['possible'][-1]]-live['group_mass'])) < TOL)}


import gzip
import json
import time
from itertools import product
from ..v18_3.io import read, write, canonical, file_digest
from .reachable_retrospective import prepare


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['alphas']!=list(ALPHAS) or cfg['report_state']!='fixed-uniform-source-joint-group-endpoint':raise ValueError('controls/design')
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
                        pulse(phase='aggregate-report-state',lineage=lineage,evidence=evidence,checkpoint=cp,row=index)
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
    (root/'raw/aggregate_report_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader input',evaluator='supplied fixed uniform source mixture,law,posterior and full source identities;aggregate joint future-group/report table;copy histogram;all future-coordinate errors;undefined reports remain NaN',scope='single report under fixed source prior/content/law;not arbitrary reweighting,disclosed source,two dependent reports,minimal state or optimized compact latency'))
    return dict(controls=checks,posterior_rows=len(rows)//len(ALPHAS),rows=len(rows),sources=source_total,report_queries=len(rows)*8,unavailable_checkpoints=unavailable,numerical_acceptance=False,scope='aggregate fixed report law versus full hypotheses and per-source state;all endpoint reports and future coordinates;no fitted model,observation or protected lineage')
