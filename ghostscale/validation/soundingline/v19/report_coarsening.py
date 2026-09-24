"""Fixed endpoint-bit report coarsening under unknown source identity."""
import gzip
import json
import time
from itertools import product
import numpy as np
from ..v18_3.io import read, write, canonical, file_digest
from .reachable_retrospective import prepare, TOL
from .retrospective_source import transitions, ALPHAS

# First eight columns are complete endpoints; then (bit0=0,bit0=1), etc.
CONTENT = np.concatenate([np.eye(8), *[
    np.eye(2)[(np.arange(8) >> bit) & 1] for bit in range(3)]], axis=1)


def evaluate(weight, st, law, source_rows, operator=None):
    w = np.asarray(weight, float); law = np.asarray(law, float); ids = np.asarray(source_rows)
    if w.shape != (len(st['mapping']),) or not np.isfinite(w).all() or (w < 0).any() or abs(w.sum()-1) > TOL: raise ValueError('weights')
    if law.shape != (16, 4, 8) or not np.isfinite(law).all() or (law < 0).any() or np.max(abs(law.sum(-1)-1)) > TOL: raise ValueError('law')
    if ids.ndim != 2 or ids.shape[1] != 3 or not len(ids) or ids.dtype.kind not in 'iu': raise ValueError('sources')
    t, c, old = ids.T
    if (t < 1).any() or (t > len(st['past'])).any() or len(set(t)) != len(t) or (c < 0).any() or (c >= 4).any() or (old < 0).any() or (old >= 8).any(): raise ValueError('source bounds/identity')
    likelihood = law[st['past'][t-1], c[:, None], :]
    independent = np.einsum('h,she->se', w, likelihood)
    if (independent[np.arange(len(ids)), old] <= 0).any(): raise ValueError('old endpoint unsupported')
    alpha = np.asarray(ALPHAS)
    fine_source = alpha[:, None, None]*np.eye(8)[old][None] + (1-alpha)[:, None, None]*independent[None]
    source_report = fine_source @ CONTENT
    report = source_report.mean(axis=1); valid = report > 0
    source_post = np.divide(source_report/len(ids), report[:, None, :], out=np.zeros_like(source_report), where=report[:, None, :] > 0)
    fine_likelihood = alpha[:, None, None]*np.eye(8)[old].mean(axis=0)[None, :, None] + (1-alpha)[:, None, None]*likelihood.mean(axis=0).T[None]
    report_likelihood = np.einsum('aeh,er->arh', fine_likelihood, CONTENT)
    posterior = np.divide(w[None, None, :]*report_likelihood, report[..., None], out=np.zeros_like(report_likelihood), where=valid[..., None])
    if np.max(abs(posterior.sum(-1)-valid)) > TOL: raise ValueError('posterior normalization')
    op = transitions(st) if operator is None else operator
    def forecast(weights):
        inc = (weights.reshape(-1, len(w)) @ op).reshape(*weights.shape[:-1], st['future'].shape[1], 16)
        return (np.cumsum(inc, axis=-2) @ law.reshape(16, 32)).reshape(*weights.shape[:-1], -1, 4, 8)
    future = forecast(posterior); base = forecast(w)
    square_loss = 1-(future*future).sum(-1).mean(axis=(-2, -1))
    base_loss = np.full_like(square_loss, 1-(base*base).sum(-1).mean())
    improvement = ((future-base)**2).sum(-1).mean(axis=(-2, -1))
    entropy = -np.sum(np.where(source_post > 0, source_post*np.log(np.maximum(source_post, np.finfo(float).tiny)), 0), axis=1)
    # Fine-to-coarse excess loss, one value for each fine report and each bit.
    fine_coarse = np.empty((5, 3, 8))
    for bit in range(3):
        coarse = future[:, 8+2*bit+((np.arange(8) >> bit) & 1)]
        fine_coarse[:, bit] = ((future[:, :8]-coarse)**2).sum(-1).mean(axis=(-2, -1))
    return dict(source_rows=ids.astype(np.int32), fine_source_probability=fine_source,
                report_probability=report, source_posterior=source_post, possible=valid,
                source_entropy=np.where(valid, entropy, np.nan),
                future_optimal_squared_loss=np.where(valid, square_loss, np.nan),
                no_report_squared_loss=base_loss,
                future_squared_gain=np.where(valid, improvement, np.nan),
                fine_to_coarse_squared_regret=np.where(valid[:, None, :8], fine_coarse, np.nan))


def summarize(raw):
    p = raw['report_probability']; valid = raw['possible']; out = {}
    for label, lo, hi in [('fine', 0, 8), ('bit0', 8, 10), ('bit1', 10, 12), ('bit2', 12, 14)]:
        mass = p[:, lo:hi]; ok = valid[:, lo:hi]
        out[label+'_supported_mass'] = mass.sum(-1)
        out[label+'_impossible_reports'] = (~ok).sum(-1)
        for metric in ('source_entropy', 'future_optimal_squared_loss', 'future_squared_gain'):
            out[label+'_expected_'+metric] = (mass*np.nan_to_num(raw[metric][:, lo:hi])).sum(-1)
        regret = raw['no_report_squared_loss'][:, 0]-out[label+'_expected_future_optimal_squared_loss']
        if np.max(abs(regret-out[label+'_expected_future_squared_gain'])) > TOL: raise ValueError('proper-loss gain identity')
        if np.max(abs(mass.sum(-1)-1)) > TOL: raise ValueError('report mass')
    for bit in range(3):
        gap = out['fine_expected_future_squared_gain']-out[f'bit{bit}_expected_future_squared_gain']
        explicit = (p[:, :8]*np.nan_to_num(raw['fine_to_coarse_squared_regret'][:, bit])).sum(-1)
        entropy_gap = out[f'bit{bit}_expected_source_entropy']-out['fine_expected_source_entropy']
        if np.max(abs(gap-explicit)) > TOL or np.min(gap) < -TOL or np.min(entropy_gap) < -TOL: raise ValueError('coarsening identity')
        out[f'bit{bit}_lost_future_gain'] = gap
        out[f'bit{bit}_fine_to_coarse_regret'] = explicit
        out[f'bit{bit}_added_source_entropy'] = entropy_gap
    return out


def controls():
    spec = dict(hypotheses=[['none', 0, i] for i in range(16)], length=3, checkpoint=2, signatures=[[i, i] for i in range(16)], membership=list(range(16)))
    st = prepare(spec, np.full(16, 1/16)); w = np.full(16, 1/16); law = np.full((16, 4, 8), 1/8)
    flat = summarize(evaluate(w, st, law, [[1, 0, 0], [2, 1, 1]]))
    law[:8, :, :] = np.eye(8)[0]; law[8:, :, :] = np.eye(8)[2]
    live = summarize(evaluate(w, st, law, [[1, 0, 0], [2, 1, 2]]))
    return {'live:discard_informative_bit': bool(live['bit0_lost_future_gain'][0] > .1),
            'placebo:constant_law': bool(abs(flat['fine_expected_future_squared_gain']).max() < TOL),
            'positive:retained_bit_sufficient': bool(abs(live['bit1_lost_future_gain']).max() < TOL),
            'placebo:certain_copy_no_state_gain': bool(abs(live['fine_expected_future_squared_gain'][-1]) < TOL)}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['alphas']!=list(ALPHAS):raise ValueError('controls/design')
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
                        pulse(phase='report-content-coarsening',lineage=lineage,evidence=evidence,checkpoint=cp,row=index)
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
    (root/'raw/report_coarsening_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader input',evaluator='supplied equal source prior,laws,complete posteriors and source identities;fine endpoint and three fixed binary reports;all future coordinates;undefined errors retain NaN',scope='fixed report-content coarsening;no learned provenance or historical correspondence'))
    return dict(controls=checks,posterior_rows=len(rows)//len(ALPHAS),rows=len(rows),sources=source_total,report_queries=len(rows)*14,unavailable_checkpoints=unavailable,numerical_acceptance=False,scope='eight-endpoint versus three fixed binary reports;equal source prior;no new observations,models or protected lineages')
