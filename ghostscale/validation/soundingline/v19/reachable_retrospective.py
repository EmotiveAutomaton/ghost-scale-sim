"""Complete retained-posterior retrospective reports, with lossless factors.

The prior-conditional rival receives future-group weights and a fixed prior
within each group. Exact posterior conditionals are a privileged identity
control. All reports and future coordinates are evaluated, never sampled.
"""
from itertools import product
import gzip
import json
import math
import time
import numpy as np
from scipy.sparse import csr_matrix
from ..v18_3.io import read, write, canonical, file_digest

TOL = 1e-12


def prepare(spec, prior):
    hs = spec['hypotheses']; cp = spec['checkpoint']; horizon = spec['length']
    if not 1 <= cp <= horizon: raise ValueError('time bounds')
    def state(h, t):
        kind, change, maker = h
        if kind not in ('none', 'purpose', 'skill') or not 0 <= maker < 16: raise ValueError('hypothesis')
        return maker ^ ({'purpose': 8, 'skill': 4}.get(kind, 0) if t > change else 0)
    future = np.array([[state(h, t) for t in range(cp, horizon+1)] for h in hs], np.int32)
    past = np.array([[state(h, t) for h in hs] for t in range(1, cp+1)], np.int32)
    mapping = np.asarray(spec['membership'], np.int32); sig = np.asarray(spec['signatures'], np.int32)
    if mapping.shape != (len(hs),) or set(mapping) != set(range(len(sig))): raise ValueError('mapping range')
    if len(set(map(tuple, sig))) != len(sig) or not np.array_equal(sig[mapping], future): raise ValueError('schedule mapping')
    prior = np.asarray(prior, float)
    if prior.shape != mapping.shape or (prior <= 0).any() or not np.isfinite(prior).all() or abs(math.fsum(prior)-1)>TOL: raise ValueError('prior')
    order = np.argsort(mapping, kind='stable'); starts = np.r_[0, np.flatnonzero(np.diff(mapping[order]))+1]
    mass = np.add.reduceat(prior[order], starts)
    return dict(past=past, signatures=sig, mapping=mapping, order=order, starts=starts,
                prior=prior, prior_mass=mass, counts=np.bincount(mapping), future=future)


def group(values, st):
    return np.add.reduceat(values[..., st['order']], st['starts'], axis=-1)


def forecast_array(weights, signatures, law):
    """Lossless sparse transition increments, then every future coordinate."""
    groups, times = signatures.shape
    row=list(range(groups));col=signatures[:,0].tolist();data=[1.]*groups
    g,offset=np.nonzero(signatures[:,1:]!=signatures[:,:-1]);offset=offset+1
    row.extend(g.tolist());col.extend((16*offset+signatures[g,offset]).tolist());data.extend([1.]*len(g))
    row.extend(g.tolist());col.extend((16*offset+signatures[g,offset-1]).tolist());data.extend([-1.]*len(g))
    transitions=csr_matrix((data,(row,col)),shape=(groups,times*16))
    increments=(weights.reshape(-1,groups)@transitions).reshape(*weights.shape[:-1],times,16)
    return np.cumsum(increments,axis=-2)@law.reshape(16,32)


def evaluate(weights, st, law, pulse=lambda **kw: None, retain_full=False):
    W = np.asarray(weights, float); law = np.asarray(law, float)
    if W.ndim != 2 or W.shape[1] != len(st['mapping']) or not np.isfinite(W).all() or (W<0).any() or np.max(abs(W.sum(-1)-1))>TOL: raise ValueError('weights')
    if law.shape != (16,4,8) or not np.isfinite(law).all() or (law<0).any() or np.max(abs(law.sum(-1)-1))>TOL: raise ValueError('law')
    G = group(W, st); mixed = np.flatnonzero(st['counts']>1)
    prior_likelihoods=[]; mixed_numerators=[]; outputs=[]; all_full=[]
    for t, past in enumerate(st['past'], 1):
        pulse(phase='all-retrospective-reports',past_time=t)
        likelihood = law[past].reshape(len(past),32).T
        numer = group(W[:,None,:]*likelihood[None,:,:],st)
        prior_lik = group(st['prior'][None,:]*likelihood,st)/st['prior_mass'][None,:]
        rival_numer = G[:,None,:]*prior_lik[None,:,:]
        p = numer.sum(-1); q = rival_numer.sum(-1)
        exact = np.divide(numer,p[...,None],out=np.zeros_like(numer),where=p[...,None]>0)
        rival = np.divide(rival_numer,q[...,None],out=np.zeros_like(numer),where=q[...,None]>0)
        conditional = np.divide(numer,G[:,None,:],out=np.zeros_like(numer),where=G[:,None,:]>0)
        privileged = conditional*G[:,None,:]
        privileged = np.divide(privileged,privileged.sum(-1)[...,None],out=np.zeros_like(privileged),where=privileged.sum(-1)[...,None]>0)
        if np.max(abs(privileged-exact))>TOL: raise ValueError('posterior conditional identity')
        possible=p>0; both=possible&(q>0)
        # A positive native prior gives support wherever the posterior does.
        if np.any(possible & (q==0)): raise ValueError('rival support failure')
        delta=exact-rival; update=.5*np.abs(delta).sum(-1)
        differences=forecast_array(delta,st['signatures'],law)
        maximum=np.abs(differences).max(axis=(-2,-1))
        mean_tv=.5*np.abs(differences.reshape(*p.shape,-1,4,8)).sum(-1).mean(axis=(-2,-1))
        update[~both]=np.nan;maximum[~both]=np.nan;mean_tv[~both]=np.nan
        if np.max(abs(p.reshape(-1,4,8).sum(-1)-1))>TOL: raise ValueError('report denominator')
        prior_likelihoods.append(prior_lik); mixed_numerators.append(numer[...,mixed])
        outputs.append(dict(report_probability=p,rival_report_probability=q,possible=possible,
                            quotient_tv=update,max_forecast_difference=maximum,mean_future_forecast_tv=mean_tv,
                            privileged_max_error=np.max(abs(privileged-exact),axis=-1)))
        if retain_full:all_full.append(dict(exact=exact,rival=rival,differences=differences))
    raw={k:np.stack([x[k] for x in outputs],axis=1) for k in outputs[0]}
    raw.update(initial_group_weights=G,prior_conditional_likelihood=np.stack(prior_likelihoods),
               mixed_groups=mixed,mixed_exact_numerator=np.stack(mixed_numerators,axis=1))
    return raw, all_full


def copied_report(group_weights):
    """Known-source reports contribute no second likelihood factor."""
    return np.asarray(group_weights).copy()


def fixture():
    return dict(hypotheses=[['none',0,0],['purpose',1,8],['none',0,1]],length=3,checkpoint=2,
                signatures=[[0,0],[1,1]],membership=[0,0,1])


def controls():
    st=prepare(fixture(),[.25,.25,.5]);W=np.array([[.6,.1,.3]])
    law=np.full((16,4,8),1/8);flat,_=evaluate(W,st,law)
    law[0]=[.75,.25,0,0,0,0,0,0];law[8]=[.25,.75,0,0,0,0,0,0];law[1]=[.5,.5,0,0,0,0,0,0]
    varied,_=evaluate(W,st,law)
    return {'live:prior_conditional_rival_can_differ':float(np.nanmax(varied['max_forecast_difference']))>0,
            'placebo:constant_law':float(np.nanmax(flat['max_forecast_difference']))<TOL,
            'positive:posterior_conditional_identity':float(varied['privileged_max_error'].max())<TOL,
            'placebo:same_source_copy':np.array_equal(copied_report(group(W,st)),group(W,st)),
            'positive:impossible_reports':bool((~varied['possible']).any())}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for name,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name)!=h:raise ValueError('input binding')
    specs=read(root/'inputs/SCHEDULES.json');all_rows=[];unavailable=[];paired={}
    (root/'raw').mkdir(exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
    write(root/'evaluator/SCHEDULES.json',specs)
    for lineage in cfg['lineages']:
        law=np.asarray(read(root/'inputs/aware/evaluator'/f'{lineage}-law.json'))
        for evidence in ('aware','omitted'):
            base=root/'inputs'/evidence
            if not np.array_equal(law,np.asarray(read(base/'evaluator'/f'{lineage}-law.json'))):raise ValueError('paired law')
            mapping=read(base/'evaluator'/f'{lineage}-joint-map.json')
            rows=json.loads(gzip.decompress((base/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
            rows=[r for r in rows if r['arm']=='unknown-time-type']
            with np.load(base/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as z:arrays={n:z[n] for n in z.files if n.endswith('-unknown-time-type')}
            for key,spec in specs.items():
                cp=spec['checkpoint'];length=spec['length']
                chosen=[r for r in rows if r['step']==cp and r['length']==length]
                if not chosen:
                    if (length,cp) not in ((128,8),(128,17),(128,20)):raise ValueError('unexpected missing checkpoint')
                    unavailable.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp));continue
                keys=[tuple(r[k] for k in ('draw','initial_maker','kind','switched','duplicates')) for r in chosen]
                expected=set(product(cfg['draws'],range(16),('purpose','skill'),(False,True),(False,True)))
                if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('complete paired population')
                hs=spec['hypotheses'];prior=np.array([.5/16 if h[0]=='none' else .5/(len(hs)-16) for h in hs]);st=prepare(spec,prior)
                W=[]
                for r,identity in zip(chosen,keys):
                    n,j=r['joint_array'],r['joint_row'];bound=mapping[n]
                    if bound['rows'][j]!=[r['stream'],cp] or bound['hypotheses']!=hs:raise ValueError('parent mapping')
                    pair_key=(lineage,key,*identity);witness=(r['stream'],r['actual_maker'])
                    if evidence=='aware':paired[pair_key]=witness
                    elif paired[pair_key]!=witness:raise ValueError('source-condition pairing')
                    W.append(arrays[n][j])
                W=np.asarray(W);prefix=f'{lineage}-{evidence}-{key}';parts=[];cpu=time.process_time()
                baseline=forecast_array(group(W,st),st['signatures'],law)[:,0].reshape(-1,4,8)
                if not np.allclose(baseline,np.asarray([r['forecast'] for r in chosen]),atol=TOL,rtol=0):raise ValueError('parent forecast identity')
                for first in range(0,len(W),cfg['batch_rows']):
                    raw,_=evaluate(W[first:first+cfg['batch_rows']],st,law,pulse)
                    raw['full_hypothesis_weights']=W[first:first+cfg['batch_rows']]
                    np.savez_compressed(root/'raw'/f'{prefix}-{first:03d}_points.npz',**raw)
                    for offset,r in enumerate(chosen[first:first+cfg['batch_rows']]):
                        prob=raw['report_probability'][offset];valid=raw['possible'][offset]
                        metrics={}
                        for field in ('quotient_tv','max_forecast_difference','mean_future_forecast_tv'):
                            values=raw[field][offset]
                            metrics['expected_'+field]=float(np.where(valid,prob*np.nan_to_num(values),0).sum()/(4*cp))
                            metrics['max_'+field]=float(np.nanmax(values))
                        all_rows.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,
                                             **{k:r[k] for k in ('draw','initial_maker','kind','switched','duplicates','stream','actual_maker','joint_array','joint_row')},
                                             reports=cp*32,possible_reports=int(valid.sum()),impossible_reports=int((~valid).sum()),
                                             privileged_max_error=float(raw['privileged_max_error'][offset].max()),**metrics))
                with (root/'TIMING.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,rows=len(W),cpu_seconds=time.process_time()-cpu))+'\n')
    (root/'raw/reachable_summary_points.json.gz').write_bytes(gzip.compress(canonical(all_rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader input',evaluator='all retained full posterior factors,exact mixed-group numerators,prior conditional likelihoods,group weights,report denominators and forecast-error coordinates;full forecasts reconstruct losslessly from supplied laws and complete schedules',unavailable='no regeneration of missing checkpoints'))
    return dict(controls=checks,rows=len(all_rows),reports=sum(r['reports'] for r in all_rows),unavailable_checkpoints=unavailable,
                scope='supplied-law reachable posterior comparison against frozen prior-conditional quotient rival;not all possible quotient algorithms;not process correspondence or human intent',numerical_acceptance=False)
