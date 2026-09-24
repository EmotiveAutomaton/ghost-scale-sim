"""Retrospective copied/independent source mixture, with lossless endpoint factors."""
import gzip
import json
import time
from itertools import product
import numpy as np
from scipy.sparse import csr_matrix
from ..v18_3.io import read, write, canonical, file_digest
from .reachable_retrospective import prepare, group, TOL

ALPHAS = (0., .25, .5, .75, 1.)


def transitions(st):
    sig=st['future']; count,times=sig.shape
    row=list(range(count)); col=sig[:,0].tolist(); value=[1.]*count
    h,t=np.nonzero(sig[:,1:]!=sig[:,:-1]); t=t+1
    row.extend(h.tolist()); col.extend((16*t+sig[h,t]).tolist()); value.extend([1.]*len(h))
    row.extend(h.tolist()); col.extend((16*t+sig[h,t-1]).tolist()); value.extend([-1.]*len(h))
    return csr_matrix((value,(row,col)),shape=(count,16*times))


def sources(observations, checkpoint):
    """True retained source identities, including original source time and content."""
    if [r['step'] for r in observations] != list(range(1,len(observations)+1)):
        raise ValueError('observation times')
    seen={}; result=[]
    for r in observations[:checkpoint]:
        key=r['source_id']; value=(r['source_step'],r['context'],r['endpoint'])
        if not 1<=value[0]<=r['step'] or not 0<=value[1]<4 or not 0<=value[2]<8:
            raise ValueError('source bounds')
        if key in seen:
            if seen[key]!=value: raise ValueError('source identity content')
        else:
            if r['source_step']!=r['step']: raise ValueError('missing original source')
            seen[key]=value; result.append((key,*value))
    return result


def evaluate(W, st, law, source_rows, operator=None):
    """One row per source: posterior row, original time, context and old endpoint.

    All eight endpoints and all five alpha values are retained. For a different
    endpoint, the correct update is exactly the independent update (alpha<1),
    while the copy rival has zero support. At the old endpoint the correct
    posterior is a convex mixture of independent and copy posteriors. Thus a
    single signed future-forecast difference losslessly determines every valid
    contrast; all future times and all 32 coordinates are actually evaluated.
    """
    W=np.asarray(W,float); law=np.asarray(law,float); ids=np.asarray(source_rows,int)
    if W.ndim!=2 or W.shape[1]!=len(st['mapping']) or not np.isfinite(W).all() or (W<0).any() or np.max(abs(W.sum(-1)-1))>TOL: raise ValueError('weights')
    if law.shape!=(16,4,8) or not np.isfinite(law).all() or (law<0).any() or np.max(abs(law.sum(-1)-1))>TOL: raise ValueError('law')
    if ids.ndim!=2 or ids.shape[1]!=4 or not len(ids): raise ValueError('sources shape')
    ri,t,c,e=ids.T
    if (ri<0).any() or (ri>=len(W)).any() or (t<1).any() or (t>len(st['past'])).any() or (c<0).any() or (c>=4).any() or (e<0).any() or (e>=8).any():raise ValueError('source bounds')
    selected=W[ri]; past=st['past'][t-1]
    likelihood=law[past,c[:,None],:]
    p=np.einsum('nh,nhe->ne',selected,likelihood)
    if np.max(abs(p.sum(-1)-1))>TOL: raise ValueError('report mass')
    oldp=p[np.arange(len(ids)),e]
    if (oldp<=0).any(): raise ValueError('observed source unsupported')
    oldlik=likelihood[np.arange(len(ids))[:,None],np.arange(len(st['mapping']))[None,:],e[:,None]]
    delta=selected*(oldlik/oldp[:,None]-1.)
    group_delta=group(delta,st); group_tv=.5*abs(group_delta).sum(-1)
    operator=transitions(st) if operator is None else operator
    increments=(delta@operator).reshape(len(ids),st['future'].shape[1],16)
    differences=np.cumsum(increments,axis=1)@law.reshape(16,32)
    maximum=abs(differences).max(axis=(1,2))
    mean_tv=.5*abs(differences.reshape(len(ids),-1,4,8)).sum(-1).mean(axis=(1,2))
    if np.any(maximum>group_tv+TOL):raise ValueError('forecast contraction')
    alpha=np.asarray(ALPHAS)[None,:,None]; indicator=np.eye(8)[e,None,:]
    correct=alpha*indicator+(1-alpha)*p[:,None,:]
    copy_fraction=np.divide(alpha*indicator,correct,out=np.zeros_like(correct),where=correct>0)
    return dict(source_rows=ids.astype(np.int32),independent_report_probability=p,
                correct_report_probability=correct,copy_fraction=copy_fraction,
                independent_possible=p>0,copy_possible=indicator[:,0,:].astype(bool),
                correct_possible=correct>0,independent_vs_copy_group_tv=group_tv,
                independent_vs_copy_max_future_difference=maximum,
                independent_vs_copy_mean_future_tv=mean_tv)


def summarize(raw):
    """Metrics per source/alpha/rival. Unsupported mass stays separate."""
    p=raw['correct_report_probability']; valid=raw['correct_possible']; f=raw['copy_fraction']
    e=raw['source_rows'][:,3]; old=np.eye(8)[e,None,:]
    answer={}
    for arm,supported,scale in [('independent',raw['independent_possible'][:,None,:],f),
                                ('copy',raw['copy_possible'][:,None,:],(1-f)*old)]:
        ok=valid & supported
        answer[arm+'_support_failure_mass']=np.where(valid & ~supported,p,0).sum(-1)
        answer[arm+'_supported_mass']=np.where(ok,p,0).sum(-1)
        for name in ('group_tv','max_future_difference','mean_future_tv'):
            err=scale*raw['independent_vs_copy_'+name][:,None,None]
            answer[arm+'_supported_error_mass_'+name]=np.where(ok,p*err,0).sum(-1)
            answer[arm+'_max_supported_'+name]=np.where(ok,err,0).max(-1)
    return answer


def controls():
    spec=dict(hypotheses=[['none',0,i] for i in range(16)],checkpoint=2,length=3,
              signatures=[[i,i] for i in range(16)],membership=list(range(16)))
    st=prepare(spec,np.full(16,1/16));W=np.full((1,16),1/16);law=np.zeros((16,4,8))
    law[:8,:,:2]=[.8,.2];law[8:,:,:2]=[.2,.8]
    raw=evaluate(W,st,law,[[0,1,0,0]]);m=summarize(raw)
    flat=evaluate(W,st,np.full((16,4,8),1/8),[[0,1,0,0]])
    return {'live:source_mixture_changes_forecasts':bool(raw['independent_vs_copy_max_future_difference'][0]>0),
            'placebo:constant_law':bool(flat['independent_vs_copy_max_future_difference'].max()<TOL),
            'placebo:alpha_zero_independent':bool(m['independent_supported_error_mass_group_tv'][:,0].max()==0),
            'placebo:alpha_one_copy':bool(m['copy_supported_error_mass_group_tv'][:,-1].max()==0),
            'positive:copy_support_failure':bool(m['copy_support_failure_mass'][0,0]>.1),
            'positive:impossible_reports':bool((~raw['correct_possible']).any())}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['alphas']!=list(ALPHAS):raise ValueError('controls/design')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    specs=read(root/'inputs/SCHEDULES.json');all_rows=[];unavailable=[];source_count=0;paired={}
    (root/'raw').mkdir(exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
    write(root/'evaluator/SCHEDULES.json',specs)
    structures={}
    for key,spec in specs.items():
        hs=spec['hypotheses'];prior=[.5/16 if h[0]=='none' else .5/(len(hs)-16) for h in hs]
        st=prepare(spec,prior);structures[key]=(st,transitions(st))
    for lineage in cfg['lineages']:
        law=np.asarray(read(root/'inputs/aware/evaluator'/f'{lineage}-law.json'))
        streams=json.loads(gzip.decompress((root/'inputs/aware/raw'/f'{lineage}-observations_points.json.gz').read_bytes()))
        for evidence in ('aware','omitted'):
            base=root/'inputs'/evidence
            if not np.array_equal(law,read(base/'evaluator'/f'{lineage}-law.json')):raise ValueError('law pairing')
            mapping=read(base/'evaluator'/f'{lineage}-joint-map.json')
            rows=json.loads(gzip.decompress((base/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
            rows=[r for r in rows if r['arm']=='unknown-time-type']
            with np.load(base/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as z:arrays={n:z[n] for n in z.files if n.endswith('-unknown-time-type')}
            for key,spec in specs.items():
                length,cp=spec['length'],spec['checkpoint'];st,op=structures[key]
                chosen=[r for r in rows if r['length']==length and r['step']==cp]
                if not chosen:
                    if (length,cp) not in ((128,8),(128,17),(128,20)):raise ValueError('missing checkpoint')
                    unavailable.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp));continue
                ids=[tuple(r[k] for k in ('draw','initial_maker','kind','switched','duplicates')) for r in chosen]
                expected=set(product(cfg['draws'],range(16),('purpose','skill'),(False,True),(False,True)))
                if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('paired roster')
                W=[];bindings=[];source_ids=[]
                for row,(r,identity) in enumerate(zip(chosen,ids)):
                    n,j=r['joint_array'],r['joint_row'];s=streams[r['stream']]
                    if mapping[n]['hypotheses']!=spec['hypotheses'] or mapping[n]['rows'][j]!=[r['stream'],cp]:raise ValueError('posterior binding')
                    if tuple(s[k] for k in ('draw','maker','kind','switched','duplicates'))!=identity or s['length']!=length:raise ValueError('stream binding')
                    selected=sources(s['observations'],cp);pair=(lineage,key,*identity)
                    witness=(r['stream'],r['actual_maker'],selected)
                    if evidence=='aware':paired[pair]=witness
                    elif paired[pair]!=witness:raise ValueError('source pairing')
                    W.append(arrays[n][j]);bindings.append(dict(**r,report_sources=len(selected)))
                    for source_id,t,ctx,e in selected:source_ids.append((row,t,ctx,e,source_id))
                W=np.asarray(W);numeric=np.asarray([s[:4] for s in source_ids],int)
                current=st['future'][:,0]
                state_mass=np.stack([W[:,current==s].sum(-1) for s in range(16)],axis=-1)
                baseline=(state_mass@law.reshape(16,32)).reshape(-1,4,8)
                if not np.allclose(baseline,[r['forecast'] for r in chosen],atol=TOL,rtol=0):raise ValueError('parent forecast identity')
                write(root/'evaluator'/f'{lineage}-{evidence}-{key}-bindings.json',dict(rows=bindings,sources=source_ids,role='evaluator identities;true provenance used for report design in both posterior conditions'))
                accum={};start=time.process_time()
                for first in range(0,len(numeric),cfg['source_batch_size']):
                    pulse(phase='retrospective-source-mixture',lineage=lineage,evidence=evidence,checkpoint=key,first_source=first)
                    raw=evaluate(W,st,law,numeric[first:first+cfg['source_batch_size']],op);metrics=summarize(raw)
                    np.savez_compressed(root/'raw'/f'{lineage}-{evidence}-{key}-{first:05d}_points.npz',**raw)
                    for off,index in enumerate(raw['source_rows'][:,0]):
                        if int(index) not in accum:accum[int(index)]={k:np.zeros(len(ALPHAS)) for k in metrics}
                        for k,v in metrics.items():
                            if '_max_supported_' in k:accum[int(index)][k]=np.maximum(accum[int(index)][k],v[off])
                            else:accum[int(index)][k]+=v[off]
                for row,r in enumerate(bindings):
                    for ai,alpha in enumerate(ALPHAS):
                        all_rows.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,alpha=alpha,
                            **{k:r[k] for k in ('draw','initial_maker','kind','switched','duplicates','stream','actual_maker','joint_array','joint_row','report_sources')},
                            **{k:float(v[ai] if '_max_supported_' in k else v[ai]/r['report_sources']) for k,v in accum[row].items()}))
                source_count+=len(source_ids)
                with (root/'TIMING.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,posterior_rows=len(W),sources=len(source_ids),cpu_seconds=time.process_time()-start))+'\n')
    (root/'raw/source_summary_points.json.gz').write_bytes(gzip.compress(canonical(all_rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='retained full posteriors,laws,source identities and lossless factorization of all endpoint and alpha contrasts;no new samples or fitted models',factors='for e!=old:correct equals independent when possible,copy unsupported;for e==old:correct is copy_fraction*old_posterior+(1-copy_fraction)*independent_update;all future differences scale the fully evaluated independent-minus-copy difference',metrics='support failure probability and supported error mass separate;error conditional on support obtained by dividing by supported mass,undefined when zero;source mean within posterior before paired law summaries'))
    return dict(controls=checks,rows=len(all_rows),posterior_rows=len(all_rows)//len(ALPHAS),sources=source_count,report_queries=source_count*8*len(ALPHAS),unavailable_checkpoints=unavailable,numerical_acceptance=False,scope='supplied source-mixture likelihood;retained posterior conditions;all true source identities,endpoints and future coordinates;no historical correspondence or human intent')
