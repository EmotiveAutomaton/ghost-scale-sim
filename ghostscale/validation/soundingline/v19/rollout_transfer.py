"""Frozen forward-table transfer to changed tools, with explicit zero-mass scores."""
from pathlib import Path
import gzip
import time
import numpy as np
from ..v18_3.io import canonical,write,file_digest
from ..v18_3.world import rng
from . import local_world as L,rollout as R,missing_tool as M

ARMS=R.ARMS+('retrieval','known-rule-oracle')


def oracle(q,rule):
    a=b=R.ARTIFACTS[q[2]]
    for op in q[3:]:b,a=a,M.execute(a,b,L.OPERATIONS[op],(0,q[0],q[1],0),rule)
    return R.code(a)


def retrieval(direct,q):
    if not direct:return np.ones(8)/8
    keys=sorted(direct);dist=np.sum(np.asarray(keys)!=np.asarray(q),axis=1);best=dist.min()
    return R.normalized(sum((direct[k] for k,d in zip(keys,dist) if d==best),np.zeros(8)))


def score(p,target):
    p=np.asarray(p)
    if p.shape!=(8,) or (p<0).any() or not np.isclose(p.sum(),1):raise ValueError('invalid predictive distribution')
    return dict(infinite_loss_mass=float(p[target]<=0),finite_loss_contribution=float(-np.log(p[target])) if p[target]>0 else 0.,
        squared_error=float(np.sum((p-np.eye(8)[target])**2)),true_probability=float(p[target]))


def controls():
    q=(1,0,2,3,4,4);p=np.eye(8)[oracle(q,'original')]
    unchanged=(1,0,2,4,4,4)
    return {'live:tool_shift':oracle(q,'original')==6 and oracle(q,'presentation-tool')==3,
        'placebo:no_tool_stays':oracle(unchanged,'original')==oracle(unchanged,'presentation-tool'),
        'positive:hard_wrong_law':score(p,3)['infinite_loss_mass']==1 and score(p,3)['finite_loss_contribution']==0,
        'positive:proper_score':score(np.ones(8)/8,3)['finite_loss_contribution']==float(np.log(8)),
        'positive:retrieval_seen':bool(np.array_equal(retrieval({q:np.arange(1.,9.)},q),np.arange(1.,9.)/36)),
        'placebo:retrieval_empty':bool(np.array_equal(retrieval({},q),np.ones(8)/8))}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls()
    if not all(checks.values()) or cfg['arms']!=list(ARMS) or cfg['rules']!=list(M.RULES):raise ValueError('transfer admission failed')
    retained=root/'inputs'
    for name,sha in cfg['input_files'].items():
        if file_digest(retained/name)!=sha:raise ValueError('frozen F1 input changed')
    write(root/'CONTROLS.json',checks)
    for folder in ('raw','forecasts','reader'):(root/folder).mkdir()
    populations={};queries=set();enumeration=[]
    for lineage in cfg['lineages']:
        for rule in M.RULES:
            pulse(phase='changed-tool-populations',lineage=lineage,rule=rule)
            rr=M.enumerate_rule(L.law(lineage),rule);groups={}
            for r in rr:
                q=R.query(r);t=R.code(r['final']);queries.add(q)
                if oracle(q,rule)!=t:raise ValueError('rule endpoint reference mismatch')
                if q not in groups:groups[q]=[t,0.]
                if groups[q][0]!=t:raise ValueError('query insufficient')
                groups[q][1]+=r['probability']
            if abs(sum(w for t,w in groups.values())-1)>1e-10:raise ValueError('population mass')
            populations[(lineage,rule)]=groups
            (root/'raw'/f'{lineage}-{rule}_points.json.gz').write_bytes(gzip.compress(canonical(rr),mtime=0))
            enumeration.append(dict(lineage=lineage,rule=rule,paths=len(rr),queries=len(groups)))
    qs=sorted(queries);public=[R.visible(q) for q in qs]
    for p in public:R.validate_visible(p)
    write(root/'reader/QUERIES.json',public);write(root/'ENUMERATION.json',enumeration)
    write(root/'QUERY_TRUTH.json',[dict(query=list(q),targets={rule:oracle(q,rule) for rule in M.RULES}) for q in qs])
    cells=[];timing=[];total_rows=0
    for draw in cfg['training_draws']:
        for budget in cfg['budgets']:
            pulse(phase='frozen-forecast-transfer',draw=draw,budget=budget);start=time.process_time()
            with np.load(retained/'models'/f'{draw}-{budget}.npz') as data:
                table=R.normalized(data['transition_counts']);direct={tuple(map(int,k)):v for k,v in zip(data['direct_keys'],data['direct_counts'])}
            arrays={arm:[] for arm in R.ARMS+('retrieval',)};uniforms=[]
            for q in qs:
                u=rng('v19-rollout-query',draw,*q).uniform(size=(16,3));uniforms.append(u)
                for arm,p in dict(R.forecasts(table,direct,q,u),retrieval=retrieval(direct,q)).items():arrays[arm].append(p)
            arrays={a:np.array(ps) for a,ps in arrays.items()}
            for rule in M.RULES:arrays['known-rule-oracle:'+rule]=np.eye(8)[[oracle(q,rule) for q in qs]]
            np.savez(root/'forecasts'/f'{draw}-{budget}.npz',queries=np.array(qs),uniforms=np.array(uniforms),**arrays)
            rows=[]
            for (lineage,rule),groups in populations.items():
                for arm in ARMS:
                    entries=[]
                    for i,q in enumerate(qs):
                        if q not in groups:continue
                        target,mass=groups[q];p=arrays['known-rule-oracle:'+rule if arm=='known-rule-oracle' else arm][i]
                        entries.append(dict(draw=draw,budget=budget,lineage=lineage,rule=rule,arm=arm,query_index=i,
                            probability_mass=mass,changed=oracle(q,'original')!=oracle(q,'presentation-tool'),**score(p,target)))
                    rows.extend(entries)
                    for subset in ('all','changed','stay'):
                        chosen=[r for r in entries if subset=='all' or r['changed']==(subset=='changed')];mass=sum(r['probability_mass'] for r in chosen)
                        if not mass:continue
                        cells.append(dict(draw=draw,budget=budget,lineage=lineage,rule=rule,arm=arm,subset=subset,queries=len(chosen),population_mass=mass,
                            **{k:float(sum(r['probability_mass']*r[k] for r in chosen)/mass) for k in
                               ('infinite_loss_mass','finite_loss_contribution','squared_error','true_probability')}))
            (root/'raw'/f'{draw}-{budget}-transfer_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0));total_rows+=len(rows)
            timing.append(dict(draw=draw,budget=budget,cpu_seconds=time.process_time()-start,new_fits=0,retrieval_visited_keys=len(direct)))
    write(root/'TIMING.jsonl',dict(measurements=timing,accounting='new query/scoring work; historical F1 fitting retained and not recharged'))
    return dict(controls=checks,cells=cells,rows=total_rows,paths=sum(r['paths'] for r in enumeration),queries=len(qs),fits=0,
        scope='frozen learned mechanics transfer with supplied skill/belief and operations; constructed method; no inverse process or capacity match claim')
