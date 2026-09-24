"""Fixed source-prior basis for one later mixture/report update.

Joint numerators mix before normalization. The law, basis and mixture menu are
supplied; neither learned access nor arbitrary source reweighting is tested.
"""
import numpy as np
from .aggregate_report_state import acquire
from .reachable_retrospective import group, TOL
from .retrospective_source import ALPHAS, transitions

PRIORS = ('uniform', 'time-proportional', 'reciprocal-time')
MIXTURES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                     [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
                     [1/3, 1/3, 1/3]])


def source_priors(ids):
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.shape[1] != 3 or not len(ids) or ids.dtype.kind not in 'iu' or (ids[:, 0] < 1).any():
        raise ValueError('source times')
    t = ids[:, 0].astype(float)
    pi = np.array([np.ones(len(t)), t, 1/t])
    return pi/pi.sum(-1, keepdims=True)


def acquire_basis(w, st, law, ids):
    q, _, _, likelihood = acquire(w, st, law, ids)
    pi = source_priors(ids)
    average = np.einsum('ks,she->keh', pi, likelihood)
    joint = group(average*np.asarray(w), st)
    copied = np.array([np.bincount(np.asarray(ids)[:, 2], weights=p, minlength=8) for p in pi])
    return q, joint, copied, pi, likelihood


def update(q, joint, copied, mixtures=MIXTURES):
    q = np.asarray(q, float); joint = np.asarray(joint, float)
    copied = np.asarray(copied, float); mixtures = np.asarray(mixtures, float)
    if q.ndim != 1 or not np.isfinite(q).all() or (q < 0).any() or abs(q.sum()-1) > TOL:
        raise ValueError('group mass')
    if joint.shape != (3, 8, len(q)) or not np.isfinite(joint).all() or (joint < 0).any() or np.max(abs(joint.sum(1)-q)) > TOL:
        raise ValueError('basis joint mass')
    if copied.shape != (3, 8) or not np.isfinite(copied).all() or (copied < 0).any() or np.max(abs(copied.sum(1)-1)) > TOL:
        raise ValueError('basis copied mass')
    if mixtures.ndim != 2 or mixtures.shape[1] != 3 or not len(mixtures) or not np.isfinite(mixtures).all() or (mixtures < 0).any() or np.max(abs(mixtures.sum(1)-1)) > TOL:
        raise ValueError('convex mixture')
    a = np.asarray(ALPHAS)[None, :, None, None]
    j = np.einsum('mk,keg->meg', mixtures, joint)
    copied_mix = mixtures@copied
    numerator = a*copied_mix[:, None, :, None]*q + (1-a)*j[:, None]
    probability = numerator.sum(-1); possible = probability > 0
    posterior = np.divide(numerator, probability[..., None], out=np.zeros_like(numerator), where=possible[..., None])
    return probability, possible, posterior


def evaluate(w, st, law, ids, operator=None):
    w = np.asarray(w, float); law = np.asarray(law, float); ids = np.asarray(ids)
    q, joint, copied, pi, likelihood = acquire_basis(w, st, law, ids)
    prob, possible, post = update(q, joint, copied)
    # Independent source-level Bayes: mix source probabilities, then contract
    # each full hypothesis's likelihood. Do not mix normalized posteriors.
    mixed_pi = MIXTURES@pi
    independent = np.einsum('ms,she,h->meh', mixed_pi, likelihood, w)
    ref_joint = group(independent, st)
    ref_copy = np.array([np.bincount(ids[:, 2], weights=p, minlength=8) for p in mixed_pi])
    a = np.asarray(ALPHAS)[None, :, None, None]
    ref_num = a*ref_copy[:, None, :, None]*w + (1-a)*independent[:, None]
    reference_probability = ref_num.sum(-1); support = reference_probability > 0
    if not np.array_equal(possible, support): raise ValueError('basis report support')
    ref_post = group(np.divide(ref_num, reference_probability[..., None], out=np.zeros_like(ref_num), where=support[..., None]), st)
    tv = .5*abs(post-ref_post).sum(-1)
    op = transitions(dict(future=st['signatures'])) if operator is None else operator
    def forecast(mass):
        shape = mass.shape[:-1]
        increments = (mass.reshape(-1, len(q))@op).reshape(*shape, st['future'].shape[1], 16)
        return increments.cumsum(-2)@law.reshape(16, 32)
    # All future coordinates are projected once per joint basis table and
    # independent reference table, before mixing the five copy probabilities.
    future_q = forecast(q); future_joint = forecast(joint)
    future_ref = forecast(ref_joint)
    mixed_future = np.einsum('mk,ketv->metv', MIXTURES, future_joint)
    mixed_copy = MIXTURES@copied
    maximum = np.zeros_like(prob); squared = np.zeros_like(prob)
    rival_maximum = np.zeros_like(prob); rival_squared = np.zeros_like(prob)
    vertex_prob, vertex_possible, _ = update(q, joint, copied, np.eye(3))
    for ai, alpha in enumerate(ALPHAS):
        numerator = alpha*mixed_copy[:, :, None, None]*future_q+(1-alpha)*mixed_future
        reference = alpha*ref_copy[:, :, None, None]*future_q+(1-alpha)*future_ref
        prediction = np.divide(numerator, prob[:, ai, :, None, None], out=np.zeros_like(numerator), where=possible[:, ai, :, None, None])
        truth = np.divide(reference, reference_probability[:, ai, :, None, None], out=np.zeros_like(reference), where=support[:, ai, :, None, None])
        vertex_num = alpha*copied[:, :, None, None]*future_q+(1-alpha)*future_joint
        vertex_forecast = np.divide(vertex_num, vertex_prob[:, ai, :, None, None], out=np.zeros_like(vertex_num), where=vertex_possible[:, ai, :, None, None])
        # All three source priors have strictly positive weights. Their report
        # supports must therefore coincide; no zero posterior is a rival input.
        if not np.all(vertex_possible[:, ai] == possible[0, ai]): raise ValueError('vertex support')
        rival = np.einsum('mk,ketv->metv', MIXTURES, vertex_forecast)
        error = prediction-truth; rival_error = rival-truth
        maximum[:, ai] = abs(error).max((-2, -1))
        squared[:, ai] = (error**2).sum(-1).mean(-1)/4
        rival_maximum[:, ai] = abs(rival_error).max((-2, -1))
        rival_squared[:, ai] = (rival_error**2).sum(-1).mean(-1)/4
    if max(float(maximum.max()), float(tv.max()), float(abs(prob-reference_probability).max())) > TOL:
        raise ValueError('prior basis sufficiency')
    return dict(group_mass=q, joint_basis_report_mass=joint, basis_copy_probability=copied,
        source_rows=ids.astype(np.int32), basis_source_probability=pi, mixture_weights=MIXTURES.copy(),
        report_probability=prob, reference_probability=reference_probability, possible=possible,
        max_future_probability_error=np.where(possible, maximum, np.nan),
        future_squared_error=np.where(possible, squared, np.nan),
        updated_group_total_variation=np.where(possible, tv, np.nan),
        posterior_mixture_max_future_error=np.where(possible, rival_maximum, np.nan),
        posterior_mixture_squared_regret=np.where(possible, rival_squared, np.nan),
        basis_float64_count=np.array(25*len(q)+24),
        single_prior_float64_count=np.array(9*len(q)+8),
        full_hypothesis_float64_count=np.array(len(w)),
        shared_mixture_float64_count=np.array(MIXTURES.size),
        shared_law_float64_count=np.array(512),
        shared_group_schedule_int32_count=np.array(st['signatures'].size),
        full_schedule_int32_count=np.array(st['future'].size+st['past'].size))


def summarize(raw):
    q, joint, copied = raw['group_mass'], raw['joint_basis_report_mass'], raw['basis_copy_probability']
    if not np.array_equal(raw['mixture_weights'], MIXTURES): raise ValueError('mixture menu')
    p, possible, _ = update(q, joint, copied)
    if not np.array_equal(raw['possible'], possible) or np.max(abs(p-raw['report_probability'])) > TOL:
        raise ValueError('report state')
    if np.max(abs(p.sum(-1)-1)) > TOL: raise ValueError('report normalization')
    pi = source_priors(raw['source_rows'])
    if not np.allclose(pi, raw['basis_source_probability'], atol=TOL, rtol=0): raise ValueError('source prior')
    copied_check = np.array([np.bincount(raw['source_rows'][:, 2], weights=x, minlength=8) for x in pi])
    if np.max(abs(copied-copied_check)) > TOL: raise ValueError('copied probability')
    if raw['basis_float64_count'] != 25*len(q)+24 or raw['single_prior_float64_count'] != 9*len(q)+8:
        raise ValueError('basis storage')
    if raw['reference_probability'].shape != p.shape or not np.isfinite(raw['reference_probability']).all(): raise ValueError('reference probability')
    result = dict(possible_reports=possible.sum(-1).astype(float), max_report_probability_error=abs(p-raw['reference_probability']).max(-1))
    for field in ('max_future_probability_error','future_squared_error','updated_group_total_variation','posterior_mixture_max_future_error','posterior_mixture_squared_regret'):
        value = raw[field]
        if value.shape != p.shape or not np.array_equal(np.isnan(value), ~possible) or not np.isfinite(value[possible]).all() or (value[possible] < 0).any(): raise ValueError('undefined/error mask')
        result['expected_'+field] = (p*np.nan_to_num(value)).sum(-1)
        result['maximum_'+field] = np.nanmax(value, axis=-1)
    for field in raw:
        if field.endswith('_count'): result[field] = np.full((7, 5), raw[field], dtype=float)
    result['basis_state_bytes'] = 8*result['basis_float64_count']
    result['single_prior_state_bytes'] = 8*result['single_prior_float64_count']
    result['full_weight_state_bytes'] = 8*result['full_hypothesis_float64_count']
    return result


def controls():
    from .reachable_retrospective import prepare, fixture
    st=prepare(fixture(), [.25,.25,.5]);w=np.array([.6,.1,.3]);ids=np.array([[1,0,0],[2,1,1]])
    law=np.full((16,4,8),1/8);flat=summarize(evaluate(w,st,law,ids))
    law[0,:,:2]=[.25,.75];law[0,:,2:]=0
    live=summarize(evaluate(w,st,law,ids))
    return {'live:mixture_normalization_matters':bool(live['expected_posterior_mixture_squared_regret'][3:,0].max()>1e-10),
        'positive:joint_basis_sufficiency':bool(live['maximum_max_future_probability_error'].max()<TOL),
        'placebo:vertex_identity':bool(live['maximum_posterior_mixture_max_future_error'][:3].max()<TOL),
        'placebo:constant_law':bool(flat['maximum_posterior_mixture_max_future_error'].max()<TOL),
        'placebo:certain_copy':bool(live['maximum_posterior_mixture_max_future_error'][:,-1].max()<TOL)}


import gzip
import json
import time
from itertools import product
from ..v18_3.io import read, write, canonical, file_digest
from .reachable_retrospective import prepare


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['alphas']!=list(ALPHAS) or cfg['report_state']!='three-source-prior-basis-seven-mixtures' or cfg['source_priors']!=list(PRIORS) or cfg['mixtures']!=MIXTURES.tolist():raise ValueError('controls/design')
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
                        pulse(phase='aggregate-prior-basis',lineage=lineage,evidence=evidence,checkpoint=cp,row=index)
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
                    for mi in range(len(MIXTURES)):
                        for ai,alpha in enumerate(ALPHAS):rows.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,mixture=mi,alpha=alpha,**{k:r[k] for k in ('draw','initial_maker','kind','switched','duplicates','stream','report_sources')},**{k:float(v[mi,ai]) for k,v in metrics.items()}))
                    source_total+=len(sources)
                    if (index+1)%cfg['batch_rows']==0 or index==len(chosen)-1:
                        np.savez_compressed(root/'raw'/f'{prefix}-{index//cfg["batch_rows"]:03d}_points.npz',**chunks);chunks={}
                with (root/'TIMING.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,posterior_rows=len(chosen),sources=sum(map(len,byrow.values())),cpu_seconds=time.process_time()-start))+'\n')
    (root/'raw/aggregate_prior_basis_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader input',evaluator='supplied law,posterior,source identities and three fixed source-prior tables;seven predeclared mixtures;copied-endpoint distributions;joint-numerator Bayes;normalized-posterior mixture rival;all future-coordinate errors;undefined reports remain NaN',scope='single report under fixed source-prior convex hull/content/law;not arbitrary reweighting,disclosed source,two dependent reports,minimal state or optimized compact latency'))
    return dict(controls=checks,posterior_rows=len(rows)//(len(ALPHAS)*len(MIXTURES)),rows=len(rows),sources=source_total,report_queries=len(rows)*8,unavailable_checkpoints=unavailable,numerical_acceptance=False,scope='aggregate three-prior basis versus source-level Bayes and normalized-posterior mixtures;all endpoint reports and future coordinates;no fitted model,observation or protected lineage')
