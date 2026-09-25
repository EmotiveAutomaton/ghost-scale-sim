"""Fixed retention under supplied uniform, earlier and later source priors.

The reference executor uses full hypotheses to evaluate the compact sufficient
statistics algebraically. Storage counts describe that materializable state;
execution timing is not a claim about an optimized compact implementation.
"""
import gzip
import json
import time
from itertools import product
import numpy as np
from ..v18_3.io import read, write, canonical, file_digest
from .reachable_retrospective import prepare, group, TOL
from .retrospective_source import transitions, ALPHAS

PRIORS = ('uniform', 'recency', 'early')

WINDOWS = ('full', 'recent_half', 'recent_quarter', 'spaced_half', 'spaced_quarter', 'priority_half', 'priority_quarter')


def spaced_mask(ids, count):
    """Midpoint ranks in the sorted-time roster; no endpoints or contexts used."""
    if not 0 <= count <= len(ids):
        raise ValueError('selection count')
    mask = np.zeros(len(ids), bool)
    if count:
        order = np.argsort(ids[:, 0], kind='stable')
        ranks = ((2*np.arange(count)+1)*len(ids))//(2*count)
        mask[order[ranks]] = True
    return mask


def retention(ids, checkpoint, source_prior="uniform"):
    recent = [ids[:, 0] > checkpoint//2, ids[:, 0] > (3*checkpoint)//4]
    if source_prior not in PRIORS: raise ValueError('source prior')
    t = ids[:, 0]
    weight = np.ones(len(t)) if source_prior == 'uniform' else t.astype(float) if source_prior == 'recency' else 1/t
    order = np.lexsort((-t, -weight))
    priority = []
    for k in recent:
        mask = np.zeros(len(ids), bool); mask[order[:int(k.sum())]] = True; priority.append(mask)
    return np.stack([np.ones(len(ids), bool), *recent,
                     *(spaced_mask(ids, int(k.sum())) for k in recent), *priority])


def evaluate(weight, st, law, source_rows, operator=None, source_prior="uniform"):
    w = np.asarray(weight, float); law = np.asarray(law, float); ids = np.asarray(source_rows)
    if w.shape != (len(st['mapping']),) or not np.isfinite(w).all() or (w < 0).any() or abs(w.sum()-1) > TOL: raise ValueError('weights')
    if law.shape != (16, 4, 8) or not np.isfinite(law).all() or (law < 0).any() or np.max(abs(law.sum(-1)-1)) > TOL: raise ValueError('law')
    if ids.ndim != 2 or ids.shape[1] != 3 or not len(ids) or ids.dtype.kind not in 'iu': raise ValueError('sources')
    t, c, old = ids.T; cp = len(st['past']); count = len(ids)
    if (t < 1).any() or (t > cp).any() or len(set(t)) != count or (c < 0).any() or (c > 3).any() or (old < 0).any() or (old > 7).any(): raise ValueError('source bounds/identity')
    if source_prior not in PRIORS: raise ValueError('source prior')
    unnormalized = np.ones(count) if source_prior == 'uniform' else t.astype(float) if source_prior == 'recency' else 1/t
    normalizer = float(unnormalized.sum())
    probabilities = unnormalized/normalizer
    likelihood = law[st['past'][t-1], c[:, None], :]
    independent = np.einsum('h,she->se', w, likelihood)
    if (independent[np.arange(count), old] <= 0).any(): raise ValueError('old endpoint unsupported')
    # Original maker prior, before history; no conditioning on the future group.
    prior_endpoint = law.mean(axis=0)
    keep = retention(ids, cp, source_prior)
    histogram = probabilities @ np.eye(8)[old]
    effective = np.stack([np.einsum('s,she->he', probabilities[k], likelihood[k]) + (probabilities[~k] @ prior_endpoint[c[~k]])[None, :] for k in keep])
    a = np.asarray(ALPHAS)
    numer = effective.transpose(0, 2, 1)*w[None, None, :]
    independent_mass = numer.sum(-1)
    independent_post = np.divide(numer, independent_mass[..., None], out=np.zeros_like(numer), where=independent_mass[..., None]>0)
    report = a[None, :, None]*histogram[None, None, :] + (1-a[None, :, None])*independent_mass[:, None, :]
    possible = report > 0
    beta = np.divide((1-a[None, :, None])*independent_mass[:, None, :], report, out=np.zeros_like(report), where=possible)
    if np.max(abs(report.sum(-1)-1)) > TOL: raise ValueError('mass preservation')
    # Copying adds a state-independent factor. Each alpha therefore moves on the
    # same prior-to-independent-report segment; propagate its direction once.
    direction = independent_post-w[None, None, :]
    op = transitions(st) if operator is None else operator
    increments = (direction.reshape(-1, len(w)) @ op).reshape(len(WINDOWS), 8, st['future'].shape[1], 16)
    future_direction = (np.cumsum(increments, axis=-2) @ law.reshape(16, 32)).reshape(len(WINDOWS), 8, -1, 4, 8)
    # Exact quadratic factorization avoids materializing all alpha-by-future
    # squared differences. Maximum errors still enumerate every future coordinate.
    f = future_direction.reshape(len(WINDOWS), 8, -1)
    divisor = future_direction.shape[-3]*future_direction.shape[-2]
    norm = np.einsum('wei,wei->we', f, f)/divisor
    cross = np.einsum('wei,ei->we', f, f[0])/divisor
    squared = beta*beta*norm[:,None] + beta[0:1]*beta[0:1]*norm[0:1,None] - 2*beta*beta[0:1]*cross[:,None]
    squared = np.maximum(squared, 0.)  # Roundoff only; the identity is nonnegative.
    squared[0] = 0.
    maximum = np.zeros_like(beta); tv = np.zeros_like(beta)
    gd = group(direction, st)
    for ai in range(len(ALPHAS)-1):
        delta = f[1:]*beta[1:,ai,:,None]
        delta -= f[0:1]*beta[0,ai,:,None]
        maximum[1:,ai] = np.max(abs(delta),axis=-1)
        change = gd[1:]*beta[1:,ai,:,None]
        change -= gd[0:1]*beta[0,ai,:,None]
        tv[1:,ai] = .5*abs(change).sum(-1)
    both = possible & possible[0:1]
    if np.any(maximum[both] > tv[both]+TOL): raise ValueError('total variation bound')
    # Per-source mixed joint cells need floats; deterministic group/past mappings
    # reuse group mass. Structural indices are shared, and counted separately.
    if '_forgetting_counts' not in st:
        cells=[]; mixed=[]
        for past in st['past']:
            unique = np.unique(st['mapping']*16+past); counts=np.bincount(unique//16, minlength=len(st['signatures']))
            cells.append(len(unique)); mixed.append(int(counts[counts>1].sum()))
        st['_forgetting_counts'] = (np.asarray(cells), np.asarray(mixed))
    cells, mixed = st['_forgetting_counts']
    floats = 13 + len(st['signatures']) + np.array([mixed[t[k]-1].sum() for k in keep], dtype=np.int64)
    # Thirteen floats: eight copy masses, four remainder-context masses and one
    # source-weight normalizer. Retained weights reconstruct from times and rule.
    integer_counts = np.array([3*int(k.sum())+2*int(cells[t[k]-1].sum()) for k in keep], dtype=np.int64)
    return dict(source_rows=ids.astype(np.int32), source_probability=probabilities, source_normalizer=np.array(normalizer), retained=keep, retained_sources=keep.sum(-1),
                forgotten_mass=np.array([probabilities[~k].sum() for k in keep]), copy_histogram=histogram,
                forgotten_context_mass=np.stack([np.bincount(c[~k], weights=probabilities[~k], minlength=4) for k in keep]),
                report_probability=report, possible=possible, comparable=both,
                future_squared_regret=np.where(both, squared, np.nan),
                max_future_probability_error=np.where(both, maximum, np.nan),
                updated_group_total_variation=np.where(both, tv, np.nan),
                compact_float64_count=floats, compact_int32_count=integer_counts,
                full_hypothesis_float64_count=np.full(len(WINDOWS), len(w), dtype=np.int64),
                shared_prior_float64_count=np.array(32, dtype=np.int64))


def summarize(raw):
    p=raw['report_probability']; possible=raw['possible']; truth=p[0]; out={}
    if np.max(abs(p.sum(-1)-1)) > TOL: raise ValueError('report mass')
    count=len(raw['source_rows'])
    if not np.array_equal(raw['retained'].sum(-1),raw['retained_sources']): raise ValueError('retained count')
    if np.max(abs(raw['forgotten_mass']-np.sum(raw['source_probability'][None,:]*~raw['retained'],axis=-1))) > TOL: raise ValueError('remainder mass')
    if abs(raw['source_probability'].sum()-1)>TOL or (raw['source_probability']<0).any(): raise ValueError('source probability')
    if np.max(abs(raw['forgotten_context_mass'].sum(-1)-raw['forgotten_mass'])) > TOL: raise ValueError('remainder histogram')
    for wi,label in enumerate(WINDOWS):
        out[label+'_unsupported_mass']=(truth*(~possible[wi])).sum(-1)
        out[label+'_supported_mass']=(truth*possible[wi]).sum(-1)
        out[label+'_expected_squared_regret']=(truth*np.nan_to_num(raw['future_squared_regret'][wi])).sum(-1)
        out[label+'_expected_max_probability_error']=(truth*np.nan_to_num(raw['max_future_probability_error'][wi])).sum(-1)
        out[label+'_expected_group_total_variation']=(truth*np.nan_to_num(raw['updated_group_total_variation'][wi])).sum(-1)
        for key in ('retained_sources','forgotten_mass','compact_float64_count','compact_int32_count','full_hypothesis_float64_count'):
            out[label+'_'+key]=np.full(5,raw[key][wi])
    return out


def controls():
    spec=dict(hypotheses=[['none',0,i] for i in range(16)],length=4,checkpoint=4,signatures=[[i] for i in range(16)],membership=list(range(16)))
    st=prepare(spec,np.full(16,1/16));w=np.full(16,1/16);law=np.full((16,4,8),1/8)
    ids=[[1,1,0],[2,0,0],[3,1,0],[4,1,0]]
    flat=summarize(evaluate(w,st,law,ids))
    law[:8,0]=np.eye(8)[0];law[8:,0]=np.eye(8)[1]
    raw=evaluate(w,st,law,ids);live=summarize(raw)
    return {'live:older_informative_time':bool(live['spaced_half_expected_squared_regret'][0]<live['recent_half_expected_squared_regret'][0]-.001),
            'placebo:constant_law':bool(abs(flat['spaced_half_expected_squared_regret']).max()<TOL),
            'positive:full_retention_identity':bool(abs(live['full_expected_squared_regret']).max()<TOL),
            'placebo:certain_copy_identity':bool(all(abs(live[k][-1])<TOL for k in live if k.endswith('expected_squared_regret'))),
            'positive:matched_source_counts':bool(np.array_equal(raw['retained_sources'][1:3],raw['retained_sources'][3:5]))}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['alphas']!=list(ALPHAS) or cfg['selection_rule']!='highest-source-weight-then-recent-time' or cfg['source_priors']!=list(PRIORS):raise ValueError('controls/design')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    if not read(root/'inputs/PARENT_REVIEW.json')['numerical_acceptance']:raise ValueError('parent acceptance')
    specs=read(root/'inputs/SCHEDULES.json');structures={}
    for key,spec in specs.items():
        hs=spec['hypotheses'];st=prepare(spec,[.5/16 if h[0]=='none' else .5/(len(hs)-16) for h in hs]);structures[key]=(st,transitions(st))
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
                        pulse(phase='retention-mass-priority',lineage=lineage,evidence=evidence,checkpoint=cp,row=index)
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
                    for prior in PRIORS:
                        source_ids=np.asarray([s[:3] for s in sources]);tag=f'{index:03d}__'
                        raw=evaluate(w,st,law,source_ids,op,prior);metrics=summarize(raw)
                        for k,v in raw.items():chunks[f'{index:03d}__'+prior+'__'+k]=v
                        for ai,alpha in enumerate(ALPHAS):rows.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,alpha=alpha,source_prior=prior,**{k:r[k] for k in ('draw','initial_maker','kind','switched','duplicates','stream','report_sources')},**{k:float(v[ai]) for k,v in metrics.items()}))
                    source_total+=len(sources)
                    if (index+1)%cfg['batch_rows']==0 or index==len(chosen)-1:
                        np.savez_compressed(root/'raw'/f'{prefix}-{index//cfg["batch_rows"]:03d}_points.npz',**chunks);chunks={}
                with (root/'TIMING.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,posterior_rows=len(chosen),sources=sum(map(len,byrow.values())),cpu_seconds=time.process_time()-start))+'\n')
    (root/'raw/retention_mass_priority_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader input',evaluator='supplied uniform,time-proportional and reciprocal-time source priors,laws,complete posteriors and source identities;matched-count highest-source-weight, recency and evenly spaced sorted-time retention; explicit prior remainder;all future coordinates;undefined errors retain NaN',scope='time-balanced source retention;no learned provenance or historical correspondence'))
    return dict(controls=checks,posterior_rows=len(rows)//(len(ALPHAS)*len(PRIORS)),rows=len(rows),sources=source_total,report_queries=len(rows)*8,unavailable_checkpoints=unavailable,numerical_acceptance=False,scope='full,recent half/quarter,spaced half/quarter,highest-weight half/quarter;8endpoint reports;three fixed source priors;no new observations,models or protected lineages')
