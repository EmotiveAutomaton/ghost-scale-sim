"""Joint future-group/past-state mass: exact one-report sufficiency and cost."""
import gzip
import json
import time
from itertools import product
import numpy as np
from ..v18_3.io import read,write,canonical,file_digest
from .reachable_retrospective import prepare,group,copied_report,fixture,TOL


def support(st):
    result=[]
    for past in st['past']:
        pairs, mapping=np.unique(np.column_stack((st['mapping'],past)),axis=0,return_inverse=True)
        order=np.argsort(mapping,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(mapping[order]))+1]
        groups=pairs[:,0];group_starts=np.r_[0,np.flatnonzero(np.diff(groups))+1]
        mixed=st['counts'][groups]>1
        result.append(dict(pairs=pairs,mapping=mapping.astype(np.int32),order=order,starts=starts,group_starts=group_starts,mixed=mixed))
    return result


def evaluate(W,st,law,pulse=lambda **kw:None):
    W=np.asarray(W,float);law=np.asarray(law,float)
    if W.ndim!=2 or W.shape[1]!=len(st['mapping']) or not np.isfinite(W).all() or (W<0).any() or np.max(abs(W.sum(-1)-1))>TOL:raise ValueError('weights')
    if law.shape!=(16,4,8) or not np.isfinite(law).all() or (law<0).any() or np.max(abs(law.sum(-1)-1))>TOL:raise ValueError('law')
    G=group(W,st);structures=support(st);raw={'group_weights':G};metrics=[]
    # Singleton groups contain exactly one hypothesis: their two numerator
    # expressions have identical operands. Share that exact computation.
    single=np.flatnonzero(st['counts'][st['mapping']]==1)
    mixed_h=np.flatnonzero(st['counts'][st['mapping']]>1)
    mixed_order=mixed_h[np.argsort(st['mapping'][mixed_h],kind='stable')]
    mixed_starts=np.r_[0,np.flatnonzero(np.diff(st['mapping'][mixed_order]))+1] if len(mixed_h) else np.array([],int)
    for t,(past,s) in enumerate(zip(st['past'],structures),1):
        pulse(phase='complete-joint-report-state',past_time=t)
        joint=np.add.reduceat(W[:,s['order']],s['starts'],axis=-1)
        # First expression factors via group/state cells; second directly weights
        # every full hypothesis. Both evaluate ALL32 reports and ALLgroups.
        pairs=s['pairs'][s['mixed']];joint_mixed=joint[:,s['mixed']]
        if len(pairs):
            starts=np.r_[0,np.flatnonzero(np.diff(pairs[:,0]))+1]
            local=joint_mixed[:,None,:]*law[pairs[:,1]].reshape(-1,32).T[None,:,:]
            num=np.add.reduceat(local,starts,axis=-1)
            direct=np.add.reduceat(W[:,None,mixed_order]*law[past[mixed_order]].reshape(-1,32).T[None,:,:],mixed_starts,axis=-1)
        else:num=direct=np.empty((len(W),32,0))
        singleton_total=W[:,single]@law[past[single]].reshape(len(single),32)
        p=singleton_total+num.sum(-1);q=singleton_total+direct.sum(-1)
        if not np.array_equal(p>0,q>0):raise ValueError('report support')
        a=np.divide(num,p[...,None],out=np.zeros_like(num),where=p[...,None]>0)
        b=np.divide(direct,q[...,None],out=np.zeros_like(direct),where=q[...,None]>0)
        inverse_p=np.divide(1.,p,out=np.zeros_like(p),where=p>0)
        inverse_q=np.divide(1.,q,out=np.zeros_like(q),where=q>0)
        tv=.5*(abs(a-b).sum(-1)+singleton_total*abs(inverse_p-inverse_q))
        if np.max(tv)>TOL or np.max(abs(p-q))>TOL:raise ValueError('joint sufficiency')
        # Every remaining endpoint is a [0,1]-valued function of future group.
        # Therefore abs(E_a f-E_b f) <= TV(a,b), without enumerating duplicate
        # contractions. This is an upper bound, NOT a measured forecast error.
        raw[f'{t}-mixed_joint_mass']=joint[:,s['mixed']]
        raw[f'{t}-report_probability']=p
        raw[f'{t}-possible']=p>0
        raw[f'{t}-updated_group_tv']=tv
        raw[f'{t}-report_probability_error']=abs(p-q)
        metrics.append(dict(time=t,full_joint_cells=len(s['pairs']),mixed_joint_cells=int(s['mixed'].sum()),
                            shared_support_int32_bytes=int(4*(s['pairs'].size+s['mapping'].size)),
                            max_update_tv=float(tv.max()),max_report_probability_error=float(abs(p-q).max())))
    return raw,structures,metrics


def controls():
    st=prepare(fixture(),[.25,.25,.5]);law=np.full((16,4,8),1/8);W=np.array([[.6,.1,.3]])
    raw,_,m=evaluate(W,st,law)
    law[0]=[.75,.25,0,0,0,0,0,0];law[8]=[.25,.75,0,0,0,0,0,0];law[1]=[.5,.5,0,0,0,0,0,0]
    varied,_,vm=evaluate(W,st,law)
    return {'live:joint_report_sufficiency':max(x['max_update_tv'] for x in vm)<TOL,
            'placebo:constant_law':max(x['max_update_tv'] for x in m)<TOL,
            'placebo:copied_source_identity':np.array_equal(copied_report(raw['group_weights']),raw['group_weights']),
            'positive:impossible_reports':bool((~varied['1-possible']).any()),
            'positive:nontrivial_mixed_state':varied['1-mixed_joint_mass'].shape[-1]>varied['2-mixed_joint_mass'].shape[-1]}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    specs=read(root/'inputs/SCHEDULES.json');rows_out=[];unavailable=[];paired={}
    (root/'raw').mkdir(exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
    write(root/'evaluator/SCHEDULES.json',specs)
    for key,spec in specs.items():
        hs=spec['hypotheses'];prior=[.5/16 if h[0]=='none' else .5/(len(hs)-16) for h in hs];st=prepare(spec,prior)
        write(root/'evaluator'/f'{key}-support.json',[dict(pairs=s['pairs'].tolist(),mapping=s['mapping'].tolist(),mixed=s['mixed'].tolist()) for s in support(st)])
    for lineage in cfg['lineages']:
        law=np.asarray(read(root/'inputs/aware/evaluator'/f'{lineage}-law.json'))
        for evidence in ('aware','omitted'):
            base=root/'inputs'/evidence
            if not np.array_equal(law,np.asarray(read(base/'evaluator'/f'{lineage}-law.json'))):raise ValueError('law pairing')
            mapping=read(base/'evaluator'/f'{lineage}-joint-map.json')
            rows=json.loads(gzip.decompress((base/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()));rows=[r for r in rows if r['arm']=='unknown-time-type']
            with np.load(base/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as z:arrays={n:z[n] for n in z.files if n.endswith('-unknown-time-type')}
            for key,spec in specs.items():
                length,cp=spec['length'],spec['checkpoint'];chosen=[r for r in rows if r['length']==length and r['step']==cp]
                if not chosen:
                    if (length,cp) not in ((128,8),(128,17),(128,20)):raise ValueError('missing checkpoint')
                    unavailable.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp));continue
                keys=[tuple(r[k] for k in ('draw','initial_maker','kind','switched','duplicates')) for r in chosen]
                expected=set(product(cfg['draws'],range(16),('purpose','skill'),(False,True),(False,True)))
                if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('complete roster')
                hs=spec['hypotheses'];prior=[.5/16 if h[0]=='none' else .5/(len(hs)-16) for h in hs];st=prepare(spec,prior);W=[]
                for r,identity in zip(chosen,keys):
                    n,j=r['joint_array'],r['joint_row']
                    if mapping[n]['rows'][j]!=[r['stream'],cp] or mapping[n]['hypotheses']!=hs:raise ValueError('parent mapping')
                    pk=(lineage,key,*identity);witness=(r['stream'],r['actual_maker'])
                    if evidence=='aware':paired[pk]=witness
                    elif paired[pk]!=witness:raise ValueError('paired evidence')
                    W.append(arrays[n][j])
                W=np.asarray(W);G=group(W,st)
                baseline=np.stack([G[:,st['signatures'][:,0]==s].sum(-1) for s in range(16)],axis=-1)@law.reshape(16,32)
                if not np.allclose(baseline.reshape(-1,4,8),[r['forecast'] for r in chosen],atol=TOL,rtol=0):raise ValueError('parent forecast identity')
                for first in range(0,len(W),cfg['batch_rows']):
                    t=time.process_time();raw,structures,metrics=evaluate(W[first:first+cfg['batch_rows']],st,law,pulse)
                    elapsed=time.process_time()-t;prefix=f'{lineage}-{evidence}-{key}-{first:03d}'
                    np.savez_compressed(root/'raw'/f'{prefix}_points.npz',**raw)
                    for off,r in enumerate(chosen[first:first+cfg['batch_rows']]):
                        rows_out.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,
                            **{k:r[k] for k in ('draw','initial_maker','kind','switched','duplicates','stream','actual_maker','joint_array','joint_row')},
                            full_weight_values=len(hs),group_weight_values=len(G[0]),full_joint_values=sum(x['full_joint_cells'] for x in metrics),
                            factored_joint_values=len(G[0])+sum(x['mixed_joint_cells'] for x in metrics),
                            shared_support_int32_bytes=sum(x['shared_support_int32_bytes'] for x in metrics),
                            shared_future_schedule_int32_bytes=int(st['signatures'].size*4),
                            shared_hypothesis_to_group_int32_bytes=int(st['mapping'].size*4),
                            reports=32*cp,possible_reports=sum(int(raw[f'{i}-possible'][off].sum()) for i in range(1,cp+1)),
                            max_update_tv=max(float(raw[f'{i}-updated_group_tv'][off].max()) for i in range(1,cp+1)),
                            all_future_coordinate_error_upper_bound=max(float(raw[f'{i}-updated_group_tv'][off].max()) for i in range(1,cp+1)),
                            max_report_probability_error=max(float(raw[f'{i}-report_probability_error'][off].max()) for i in range(1,cp+1))))
                    with (root/'TIMING.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,first=first,rows=len(raw['group_weights']),reports=32*cp,cpu_seconds=elapsed))+'\n')
    (root/'raw/sufficient_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows_out),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='full saved posterior inputs;complete joint group/past-state support maps;numeric mass including zero cells;every report denominator and update discrepancy;all-future forecast error bounded by update total variation,not measured forecast maxima',storage='float64 values;shared int32 support/maps/schedules counted separately;singleton masses reuse their group weight;all mixed cells retained at every past time;not a minimality claim'))
    return dict(controls=checks,rows=len(rows_out),reports=sum(r['reports'] for r in rows_out),unavailable_checkpoints=unavailable,
                max_update_tv=max(r['max_update_tv'] for r in rows_out),numerical_acceptance=False,
                scope='supplied-law single-report joint-state sufficiency and storage;all-future bound via total-variation contraction;not minimal state,learned access,process correspondence or human intent')
