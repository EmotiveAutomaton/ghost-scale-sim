"""Value of revealing a retrospective report's source, with supplied laws."""
import gzip
import json
import time
from itertools import product
import numpy as np
from ..v18_3.io import read, write, canonical, file_digest
from .reachable_retrospective import prepare, group, TOL
from .retrospective_source import ALPHAS, transitions


def future_design(st, law):
    """Lossless quotient of identical future-time state maps, with multiplicity."""
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(st['future'],axis=1),axis=0))+1]
    counts = np.diff(np.r_[starts,st['future'].shape[1]])
    inverse = np.repeat(np.arange(len(starts)),counts)
    states = st['future'][:,starts]
    if not np.array_equal(states[:,inverse], st['future']):
        raise ValueError('future time reconstruction')
    operator = transitions(dict(future=states))
    return operator, inverse, counts


def evaluate(weight, st, law, source_rows, design=None):
    w = np.asarray(weight, float); law = np.asarray(law, float); ids = np.asarray(source_rows)
    if w.shape != (len(st['mapping']),) or not np.isfinite(w).all() or (w < 0).any() or abs(w.sum()-1) > TOL: raise ValueError('weights')
    if law.shape != (16,4,8) or not np.isfinite(law).all() or (law < 0).any() or np.max(abs(law.sum(-1)-1)) > TOL: raise ValueError('law')
    if ids.ndim != 2 or ids.shape[1] != 3 or not len(ids) or ids.dtype.kind not in 'iu': raise ValueError('sources')
    t, c, old = ids.T
    if (t < 1).any() or (t > len(st['past'])).any() or len(set(t)) != len(t) or (c < 0).any() or (c >= 4).any() or (old < 0).any() or (old >= 8).any(): raise ValueError('source bounds/identity')
    operator, inverse, counts = future_design(st, law) if design is None else design
    likelihood = law[st['past'][t-1], c[:,None], :].transpose(0,2,1)
    numer = likelihood*w
    prob = numer.sum(-1)
    if (prob[np.arange(len(ids)),old] <= 0).any(): raise ValueError('old endpoint unsupported')
    increments = (numer.reshape(-1,len(w)) @ operator).reshape(len(ids),8,len(counts),16)
    independent_forecast = (np.cumsum(increments,axis=-2) @ law.reshape(16,32)).reshape(len(ids),8,-1)
    baseline = (np.cumsum((w @ operator).reshape(len(counts),16),axis=0) @ law.reshape(16,32)).reshape(-1)
    base_group = group(w, st); independent_group = group(numer, st)
    indicator = np.eye(8)[old]
    result = dict(source_rows=ids.astype(np.int32), future_time_class=inverse,
                  future_time_multiplicity=counts, independent_source_probability=prob)
    saved = {k:[] for k in ('source_report_probability','possible','report_probability','source_posterior','source_entropy',
                            'squared_forecast_change','brier_improvement','group_tv','max_future_difference','mean_future_tv')}
    for alpha in ALPHAS:
        p = alpha*indicator+(1-alpha)*prob; joint = p/len(ids); total = joint.sum(0)
        ok = p > 0
        source = np.divide(joint,total[None],out=np.zeros_like(joint),where=total[None]>0)
        forecast_numer = alpha*indicator[...,None]*baseline+(1-alpha)*independent_forecast
        disclosed = np.divide(forecast_numer,p[...,None],out=np.zeros_like(forecast_numer),where=ok[...,None])
        undisclosed = np.divide(forecast_numer.mean(0),total[:,None],out=np.zeros_like(forecast_numer[0]),where=total[:,None]>0)
        if np.max(abs(disclosed.reshape(len(ids),8,-1,4,8).sum(-1)-ok[...,None,None])) > TOL: raise ValueError('forecast mass')
        delta = (disclosed-undisclosed).reshape(len(ids),8,-1,4,8)
        # Multiclass squared loss sums endpoints, then averages contexts and all
        # future times. Repeated time maps retain their exact multiplicities.
        square = np.sum(delta*delta,axis=-1).mean(-1) @ counts / counts.sum()
        truth = disclosed.reshape(delta.shape); prediction = undisclosed.reshape(8,-1,4,8)[None]
        before = np.sum(prediction*prediction-2*prediction*truth,axis=-1)+1
        after = 1-np.sum(truth*truth,axis=-1)
        improvement = (before-after).mean(-1) @ counts / counts.sum()
        if np.max(abs(square[ok]-improvement[ok]),initial=0) > TOL: raise ValueError('proper score variance identity')
        gn = alpha*indicator[...,None]*base_group+(1-alpha)*independent_group
        gp = np.divide(gn,p[...,None],out=np.zeros_like(gn),where=ok[...,None])
        gu = np.divide(gn.mean(0),total[:,None],out=np.zeros_like(gn[0]),where=total[:,None]>0)
        tv = .5*abs(gp-gu).sum(-1)
        maximum = abs(delta).max(axis=(-3,-2,-1))
        mean_tv = (.5*abs(delta).sum(-1)).mean(-1) @ counts / counts.sum()
        if np.any(maximum[ok] > tv[ok]+TOL): raise ValueError('forecast contraction')
        values = dict(source_report_probability=p, possible=ok, report_probability=total, source_posterior=source,
                      source_entropy=-np.sum(np.where(source>0,source*np.log(np.maximum(source,np.finfo(float).tiny)),0),axis=0))
        for k,v in dict(squared_forecast_change=square,brier_improvement=improvement,group_tv=tv,max_future_difference=maximum,mean_future_tv=mean_tv).items(): values[k]=np.where(ok,v,np.nan)
        for k,v in values.items(): saved[k].append(v)
    result.update({k:np.stack(v) for k,v in saved.items()})
    return result


def summarize(raw):
    joint=raw['source_report_probability']/len(raw['source_rows'])
    out={'source_alphabet':np.full(len(ALPHAS),len(raw['source_rows'])),
         'impossible_source_report_pairs':(~raw['possible']).sum(axis=(1,2)),
         'expected_source_entropy_reduction':(raw['report_probability']*raw['source_entropy']).sum(-1)}
    for metric in ('squared_forecast_change','brier_improvement','group_tv','max_future_difference','mean_future_tv'):
        out['expected_'+metric]=np.sum(joint*np.nan_to_num(raw[metric]),axis=(1,2))
        out['max_supported_'+metric]=np.max(np.where(raw['possible'],raw[metric],0),axis=(1,2))
    return out


def controls():
    spec=dict(hypotheses=[['none',0,i] for i in range(16)],length=3,checkpoint=2,signatures=[[i,i] for i in range(16)],membership=list(range(16)))
    st=prepare(spec,np.full(16,1/16));w=np.full(16,1/16);law=np.full((16,4,8),1/8)
    flat=evaluate(w,st,law,[[1,0,0],[2,1,1]])
    law[:8,0,:2]=[.65,.05];law[:8,0,2:]=.05;law[8:,0,:2]=[.05,.65];law[8:,0,2:]=.05
    varied=summarize(evaluate(w,st,law,[[1,0,0],[2,1,1]]))
    single=summarize(evaluate(w,st,law,[[1,0,0]]))
    return {'live:source_disclosure_changes_forecast':bool(varied['expected_squared_forecast_change'][0]>1e-5),
            'placebo:constant_law':bool(summarize(flat)['expected_squared_forecast_change'].max()<TOL),
            'placebo:single_source':bool(single['expected_squared_forecast_change'].max()<TOL),
            'positive:proper_score_identity':bool(np.max(abs(varied['expected_squared_forecast_change']-varied['expected_brier_improvement']))<TOL)}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['alphas']!=list(ALPHAS):raise ValueError('controls/design')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    if not read(root/'inputs/PARENT_REVIEW.json')['numerical_acceptance']:raise ValueError('parent acceptance')
    specs=read(root/'inputs/SCHEDULES.json');structures={}
    for key,spec in specs.items():
        hs=spec['hypotheses'];st=prepare(spec,[.5/16 if h[0]=='none' else .5/(len(hs)-16) for h in hs]);structures[key]=st
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
                binding=read(binding_path);st=structures[key];op=future_design(st,law);chosen=binding['rows']
                ids=[tuple(r[k] for k in ('draw','initial_maker','kind','switched','duplicates')) for r in chosen]
                expected=set(product(cfg['draws'],range(16),('purpose','skill'),(False,True),(False,True)))
                if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('paired roster')
                byrow={i:[] for i in range(len(chosen))}
                for i,t,ctx,e,source_id in binding['sources']:byrow[i].append((t,ctx,e,source_id))
                start=time.process_time();chunks={}
                for index,(r,identity) in enumerate(zip(chosen,ids)):
                    if index%cfg['batch_rows']==0:
                        pulse(phase='retrospective-source-disclosure',lineage=lineage,evidence=evidence,checkpoint=cp,row=index)
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
    (root/'raw/source_disclosure_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader input',evaluator='supplied candidate source identities,full hypothesis posteriors and laws;uniform true source prior;all source/report pairs and all future coordinates retained;lossless future-time quotient with multiplicity;undefined pairs retain NaN',scope='value of source identity given report content;not learned provenance or historical correspondence'))
    return dict(controls=checks,posterior_rows=len(rows)//len(ALPHAS),rows=len(rows),sources=source_total,source_report_queries=source_total*len(ALPHAS)*8,unavailable_checkpoints=unavailable,numerical_acceptance=False,scope='supplied-law source disclosure value;no new observations,models or protected lineages')
