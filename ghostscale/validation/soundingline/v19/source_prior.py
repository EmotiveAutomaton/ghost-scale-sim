"""Complete source-prior mismatch comparison; no learned provenance."""
import gzip
import json
import time
from itertools import product
import numpy as np
from ..v18_3.io import read,write,canonical,file_digest
from .reachable_retrospective import prepare,group,TOL
from .retrospective_source import transitions,ALPHAS

PRIORS=('uniform','recency','early')


def evaluate(weight,st,law,source_rows,operator=None):
    w=np.asarray(weight,float);law=np.asarray(law,float);ids=np.asarray(source_rows)
    if w.shape!=(len(st['mapping']),) or not np.isfinite(w).all() or (w<0).any() or abs(w.sum()-1)>TOL:raise ValueError('weights')
    if law.shape!=(16,4,8) or not np.isfinite(law).all() or (law<0).any() or np.max(abs(law.sum(-1)-1))>TOL:raise ValueError('law')
    if ids.ndim!=2 or ids.shape[1]!=3 or not len(ids) or ids.dtype.kind not in 'iu':raise ValueError('sources')
    t,c,old=ids.T
    if (t<1).any() or (t>len(st['past'])).any() or len(set(t))!=len(t) or (c<0).any() or (c>=4).any() or (old<0).any() or (old>=8).any():raise ValueError('source bounds/identity')
    priors=np.stack([np.ones(len(ids)),t.astype(float),1/t]);priors/=priors.sum(-1,keepdims=True)
    likelihood=law[st['past'][t-1],c[:,None],:]
    independent=np.einsum('h,she->se',w,likelihood)
    if (independent[np.arange(len(ids)),old]<=0).any():raise ValueError('old endpoint unsupported')
    a=np.asarray(ALPHAS);indicator=np.eye(8)[old]
    p=a[:,None,None]*indicator[None]+(1-a)[:,None,None]*independent[None]
    report=np.einsum('rs,ase->rae',priors,p)
    source_mass=priors[:,None,:,None]*p[None]
    source_post=np.divide(source_mass,report[:,:,None,:],out=np.zeros_like(source_mass),where=report[:,:,None,:]>0)
    averaged=np.einsum('rs,she->reh',priors,likelihood,optimize=True)
    oldmass=priors@indicator
    numer=w[None,None,None,:]*(a[None,:,None,None]*oldmass[:,None,:,None]+(1-a)[None,:,None,None]*averaged[:,None])
    possible=report>0
    post=np.divide(numer,report[...,None],out=np.zeros_like(numer),where=possible[...,None])
    if not np.array_equal(possible,np.broadcast_to(possible[0],possible.shape)):raise ValueError('positive prior support identity')
    if np.max(abs(post.sum(-1)-possible))>TOL:raise ValueError('posterior normalization')
    op=transitions(st) if operator is None else operator
    increments=(post.reshape(-1,len(w))@op).reshape(3,5,8,st['future'].shape[1],16)
    forecasts=(np.cumsum(increments,axis=-2)@law.reshape(16,32)).reshape(3,5,8,-1,4,8)
    out=dict(source_rows=ids.astype(np.int32),source_prior=priors,source_report_probability=p,report_probability=report,source_posterior=source_post,possible=possible,
             source_entropy=-np.sum(np.where(source_post>0,source_post*np.log(np.maximum(source_post,np.finfo(float).tiny)),0),axis=2))
    groups=group(post,st)
    for i,name in enumerate(PRIORS[1:],1):
        truth=forecasts[i];prediction=forecasts[0];delta=prediction-truth
        square=(delta*delta).sum(-1).mean(axis=(-2,-1))
        excess=((prediction*prediction-2*prediction*truth).sum(-1)+(truth*truth).sum(-1)).mean(axis=(-2,-1))
        if np.max(abs(square[possible[i]]-excess[possible[i]]),initial=0)>TOL:raise ValueError('proper score identity')
        tv=.5*abs(groups[i]-groups[0]).sum(-1);maximum=abs(delta).max(axis=(-3,-2,-1));mean=.5*abs(delta).sum(-1).mean(axis=(-2,-1))
        if np.any(maximum[possible[i]]>tv[possible[i]]+TOL):raise ValueError('contraction')
        for key,value in dict(group_tv=tv,max_future_difference=maximum,mean_future_tv=mean,squared_forecast_difference=square,brier_regret=excess).items():out[name+'_'+key]=np.where(possible[i],value,np.nan)
    return out


def summarize(raw):
    out={}
    for i,name in enumerate(PRIORS[1:],1):
        p=raw['report_probability'][i];valid=raw['possible'][i]
        out[name+'_impossible_reports']=(~valid).sum(-1)
        out[name+'_supported_mass']=np.where(valid,p,0).sum(-1)
        out[name+'_expected_source_entropy']=(p*raw['source_entropy'][i]).sum(-1)
        for metric in ('group_tv','max_future_difference','mean_future_tv','squared_forecast_difference','brier_regret'):
            v=raw[name+'_'+metric]
            out[name+'_expected_'+metric]=(p*np.nan_to_num(v)).sum(-1)
            out[name+'_max_supported_'+metric]=np.max(np.where(valid,v,0),axis=-1)
    return out


def controls():
    spec=dict(hypotheses=[['none',0,i] for i in range(16)],length=3,checkpoint=2,signatures=[[i,i] for i in range(16)],membership=list(range(16)))
    st=prepare(spec,np.full(16,1/16));w=np.full(16,1/16);law=np.full((16,4,8),1/8)
    flat=summarize(evaluate(w,st,law,[[1,0,0],[2,1,1]]))
    law[:8,0,:2]=[.65,.05];law[:8,0,2:]=.05;law[8:,0,:2]=[.05,.65];law[8:,0,2:]=.05
    live=summarize(evaluate(w,st,law,[[1,0,0],[2,1,1]]));single=summarize(evaluate(w,st,law,[[1,0,0]]))
    return {'live:source_prior_changes_forecasts':bool(live['recency_expected_squared_forecast_difference'][0]>1e-6),
            'placebo:constant_law':bool(flat['recency_expected_squared_forecast_difference'].max()<TOL),
            'placebo:single_source':bool(single['early_expected_squared_forecast_difference'].max()<TOL),
            'positive:proper_score_identity':bool(np.max(abs(live['recency_expected_squared_forecast_difference']-live['recency_expected_brier_regret']))<TOL)}


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
                        pulse(phase='retrospective-source-prior',lineage=lineage,evidence=evidence,checkpoint=cp,row=index)
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
    (root/'raw/source_prior_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader input',evaluator='supplied source priors,laws,full hypothesis posteriors and true source identities;all endpoints and future coordinates;undefined errors retain NaN',scope='source-prior mismatch;no learned provenance or historical correspondence'))
    return dict(controls=checks,posterior_rows=len(rows)//len(ALPHAS),rows=len(rows),sources=source_total,report_queries=len(rows)*8*2,unavailable_checkpoints=unavailable,numerical_acceptance=False,scope='two nonuniform supplied source priors versus uniform-prior inference;no new observations,models or protected lineages')
