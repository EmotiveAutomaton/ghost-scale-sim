"""Unknown report-source identity: marginalize likelihoods before normalization.

All retained distinct sources and report endpoints are enumerated. Supplied law
and true candidate-source roster are evaluator access, never learned provenance.
"""
import gzip
import json
import time
from itertools import product
import numpy as np
from ..v18_3.io import read, write, canonical, file_digest
from .reachable_retrospective import prepare, group, TOL
from .retrospective_source import transitions, ALPHAS


def evaluate(weight, st, law, source_rows, operator=None):
    """One posterior and all its distinct sources (time, context, old endpoint).

    Uniform-source rival averages normalized posteriors over compatible sources.
    Impossible source/report pairs are excluded explicitly, with their fraction
    retained. A most-recent-source rival is undefined on unsupported reports.
    Denominators, uniform averaging coefficients and source posterior weights
    losslessly reconstruct all three hypothesis posteriors from bound inputs.
    """
    w=np.asarray(weight,float); law=np.asarray(law,float); ids=np.asarray(source_rows)
    if w.shape!=(len(st['mapping']),) or not np.isfinite(w).all() or (w<0).any() or abs(w.sum()-1)>TOL:raise ValueError('weights')
    if law.shape!=(16,4,8) or not np.isfinite(law).all() or (law<0).any() or np.max(abs(law.sum(-1)-1))>TOL:raise ValueError('law')
    if ids.ndim!=2 or ids.shape[1]!=3 or not len(ids) or ids.dtype.kind not in 'iu':raise ValueError('sources')
    t,c,old=ids.T
    if (t<1).any() or (t>len(st['past'])).any() or len(set(t))!=len(t) or (c<0).any() or (c>=4).any() or (old<0).any() or (old>=8).any():raise ValueError('source bounds/identity')
    likelihood=law[st['past'][t-1],c[:,None],:]
    independent=np.einsum('h,she->se',w,likelihood)
    if (independent[np.arange(len(ids)),old]<=0).any():raise ValueError('old endpoint unsupported')
    a=np.asarray(ALPHAS)[:,None,None]; indicator=np.eye(8)[old]
    probabilities=a*indicator[None]+(1-a)*independent[None]
    source_possible=probabilities>0;counts=source_possible.sum(axis=1)
    report_probability=probabilities.mean(axis=1);possible=report_probability>0
    source_posterior=np.divide(probabilities,probabilities.sum(axis=1,keepdims=True),out=np.zeros_like(probabilities),where=probabilities.sum(axis=1,keepdims=True)>0)
    coefficients=np.divide(1.,probabilities*counts[:,None,:],out=np.zeros_like(probabilities),where=source_possible)
    mean_likelihood=likelihood.mean(axis=0).T
    numerator=w[None,None,:]*(np.asarray(ALPHAS)[:,None,None]*indicator.mean(axis=0)[None,:,None]+(1-np.asarray(ALPHAS))[:,None,None]*mean_likelihood[None])
    exact=np.divide(numerator,report_probability[...,None],out=np.zeros_like(numerator),where=possible[...,None])
    uniform_likelihood=np.einsum('ase,she->aeh',coefficients,likelihood,optimize=True)
    uniform=w[None,None,:]*(np.asarray(ALPHAS)[:,None,None]*(coefficients*indicator[None]).sum(axis=1)[...,None]+(1-np.asarray(ALPHAS))[:,None,None]*uniform_likelihood)
    last=int(np.argmax(t));last_p=probabilities[:,last];last_possible=last_p>0
    last_num=w[None,None,:]*(np.asarray(ALPHAS)[:,None,None]*indicator[last][None,:,None]+(1-np.asarray(ALPHAS))[:,None,None]*likelihood[last].T[None])
    recent=np.divide(last_num,last_p[...,None],out=np.zeros_like(last_num),where=last_possible[...,None])
    for x,valid in ((exact,possible),(uniform,possible),(recent,last_possible)):
        if np.max(abs(x.sum(-1)-valid),initial=0)>TOL:raise ValueError('posterior normalization')
    out=dict(source_rows=ids.astype(np.int32),independent_source_probability=independent,source_report_probability=probabilities,
             source_possible=source_possible,report_probability=report_probability,possible=possible,
             source_posterior=source_posterior,uniform_coefficients=coefficients,
             compatible_source_count=counts,most_recent_possible=last_possible,
             source_entropy=-np.sum(np.where(source_posterior>0,source_posterior*np.log(np.maximum(source_posterior,np.finfo(float).tiny)),0),axis=1))
    op=transitions(st) if operator is None else operator
    for arm,rival,valid in (('uniform',uniform,possible),('recent',recent,possible&last_possible)):
        delta=exact-rival
        d=group(delta,st);tv=.5*abs(d).sum(-1)
        increments=(delta.reshape(-1,len(w))@op).reshape(len(ALPHAS),8,st['future'].shape[1],16)
        forecasts=np.cumsum(increments,axis=-2)@law.reshape(16,32)
        maximum=abs(forecasts).max(axis=(-2,-1));mean=.5*abs(forecasts.reshape(len(ALPHAS),8,-1,4,8)).sum(-1).mean(axis=(-2,-1))
        if np.any(maximum[valid]>tv[valid]+TOL):raise ValueError('forecast contraction')
        for key,value in (('group_tv',tv),('max_future_difference',maximum),('mean_future_tv',mean)):
            # NaN marks genuinely unsupported queries, never a zero error.
            out[arm+'_'+key]=np.where(valid,value,np.nan)
    return out


def summarize(raw):
    p=raw['report_probability'];possible=raw['possible'];ns=len(raw['source_rows']);out={}
    out['impossible_reports']=(~possible).sum(-1)
    out['expected_source_entropy']=(p*raw['source_entropy']).sum(-1)
    out['expected_incompatible_source_fraction']=(p*(1-raw['compatible_source_count']/ns)).sum(-1)
    for arm in ('uniform','recent'):
        valid=possible if arm=='uniform' else possible&raw['most_recent_possible']
        out[arm+'_support_failure_mass']=np.where(possible&~valid,p,0).sum(-1)
        out[arm+'_supported_mass']=np.where(valid,p,0).sum(-1)
        for metric in ('group_tv','max_future_difference','mean_future_tv'):
            v=raw[arm+'_'+metric]
            out[arm+'_supported_error_mass_'+metric]=np.where(valid,p*np.nan_to_num(v),0).sum(-1)
            out[arm+'_max_supported_'+metric]=np.max(np.where(valid,v,0),axis=-1)
    return out


def controls():
    spec=dict(hypotheses=[['none',0,i] for i in range(16)],length=3,checkpoint=2,signatures=[[i,i] for i in range(16)],membership=list(range(16)))
    st=prepare(spec,np.full(16,1/16));w=np.full(16,1/16);law=np.full((16,4,8),1/8)
    flat=evaluate(w,st,law,[[1,0,0],[2,1,1]])
    law[:8,0,:2]=[.65,.05];law[:8,0,2:]=.05
    law[8:,0,:2]=[.05,.65];law[8:,0,2:]=.05
    law[:,1,:2]=[.3,.4];law[:,1,2:]=.05
    live=evaluate(w,st,law,[[1,0,0],[2,1,1]])
    single=evaluate(w,st,law,[[1,0,0]])
    return {'live:normalizing_each_source_can_change_forecast':bool(np.nanmax(live['uniform_max_future_difference'])>1e-4),
            'placebo:constant_law':bool(np.nanmax(flat['uniform_max_future_difference'])<TOL),
            'placebo:single_source':bool(np.nanmax(single['uniform_group_tv'])<TOL),
            'positive:copy_only_source_support_failure':bool(summarize(live)['recent_support_failure_mass'][-1]>.1)}


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
                        pulse(phase='unknown-retrospective-source',lineage=lineage,evidence=evidence,checkpoint=cp,row=index)
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
    (root/'raw/source_identity_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader input',evaluator='supplied candidate source identities,full hypothesis posteriors and laws;raw report probabilities,source posteriors,uniform coefficients and support masks reconstruct all posteriors;all future coordinates evaluated;undefined recent-source reports have NaN errors',scope='uniform candidate-source prior;uniform rival averages compatible source-specific posteriors;source roster supplied;no learned provenance or historical correspondence'))
    return dict(controls=checks,posterior_rows=len(rows)//len(ALPHAS),rows=len(rows),sources=source_total,report_queries=len(rows)*8,unavailable_checkpoints=unavailable,numerical_acceptance=False,scope='unknown source identity with supplied law and true candidate-source roster;no new observations,models or protected lineages')
