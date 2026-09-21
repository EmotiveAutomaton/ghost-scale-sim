"""Training support and a fixed action-composition holdout with common smoothing."""
import gzip,time
import numpy as np
from ..v18_3.io import canonical,read,write,file_digest
from ..v18_3.world import rng
from . import rollout as R,rollout_transfer as T,local_world as L

ARMS=('direct','retrieval','learned-exact','rollout-1','rollout-4','rollout-16','oracle-exact')
EPSILON=1/32

def withheld(q):return q[3:5]==(3,1)
def smooth(p):return (1-EPSILON)*np.asarray(p)+EPSILON/8
def empirical(endpoints):return np.bincount(endpoints,minlength=8)/len(endpoints)

def support(counts,direct,q):
    a=b=q[2];visited=[]
    for op in q[3:]:
        visited.append(bool(counts[q[0],q[1],a,b,op].sum()>8))
        n=R.code(L.execute(R.ARTIFACTS[a],R.ARTIFACTS[b],L.OPERATIONS[op],(0,q[0],q[1],0)));b,a=a,n
    return dict(query_seen=q in direct,all_primitives_seen=all(visited),visited_primitives=sum(visited),withheld_composition=withheld(q))

def controls():
    exact=np.eye(8)[2]
    return {'live:heldout_pair':withheld((1,0,2,3,1,4)),
        'placebo:reversed_pair_retained':not withheld((1,0,2,1,3,4)),
        'positive:common_smoothing':bool(np.array_equal(smooth(empirical([2])),smooth(empirical([2]*16)))),
        'positive:strict_positive':bool((smooth(exact)>0).all() and np.isclose(smooth(exact).sum(),1)),
        'positive:empty_support':not support(np.ones((2,2,8,8,6,8)),{},(1,0,2,4,4,4))['all_primitives_seen']}

def run(root,plan,pulse):
    cfg=plan['design'];checks=controls()
    if not all(checks.values()) or cfg['arms']!=list(ARMS) or cfg['epsilon']!=EPSILON:raise ValueError('support admission failed')
    inputs=root/'inputs'
    for n,h in cfg['input_files'].items():
        if file_digest(inputs/n)!=h:raise ValueError('frozen F1 inputs changed')
    for folder in ('raw','models','forecasts','reader'):(root/folder).mkdir()
    write(root/'CONTROLS.json',checks);populations={};queries=set()
    for lineage in cfg['lineages']:
        pulse(phase='support-development',lineage=lineage);rr=L.enumerate_world(L.law(lineage));groups={}
        for r in rr:
            q=R.query(r);target=R.code(r['final']);queries.add(q)
            if R.oracle(q)!=target:raise ValueError('reference mismatch')
            groups.setdefault(q,[target,0.])[1]+=r['probability']
        if abs(sum(v[1] for v in groups.values())-1)>1e-10:raise ValueError('native mass')
        populations[lineage]=groups
        (root/'raw'/f'{lineage}_points.json.gz').write_bytes(gzip.compress(canonical(rr),mtime=0))
    qs=sorted(queries);public=[R.visible(q) for q in qs]
    for p in public:R.validate_visible(p)
    write(root/'reader/QUERIES.json',public);write(root/'QUERY_TRUTH.json',[dict(query=list(q),endpoint=R.oracle(q)) for q in qs])
    cells=[];timing=[];total_rows=0
    for draw in cfg['training_draws']:
        selected=read(inputs/'auxiliary'/f'selection-{draw}.json')['paths']
        original=selected[:2048];filtered=[r for r in original if not withheld(R.query(r))]
        if not filtered or not len(filtered)<len(original):raise ValueError('empty or inactive composition holdout')
        write(root/'models'/f'{draw}-holdout-selection.json',dict(retained_indices=[i for i,r in enumerate(original) if not withheld(R.query(r))],original_count=len(original),retained_count=len(filtered)))
        for mode in ('original','composition-holdout'):
            start=time.process_time();pulse(phase='support-model',draw=draw,mode=mode)
            if mode=='original':
                with np.load(inputs/'models'/f'{draw}-2048.npz') as data:
                    counts=data['transition_counts'];direct={tuple(map(int,k)):v for k,v in zip(data['direct_keys'],data['direct_counts'])}
                new_fits=0
            else:counts,direct=R.fit(filtered);new_fits=1
            table=R.normalized(counts);keys=sorted(direct)
            np.savez(root/'models'/f'{draw}-{mode}.npz',transition_counts=counts,direct_keys=np.array(keys),direct_counts=np.array([direct[q] for q in keys]))
            timing.append(dict(draw=draw,mode=mode,phase='fit-or-load',cpu_seconds=time.process_time()-start,new_fits=new_fits,training_paths=len(original) if mode=='original' else len(filtered)))
            predictions={a:[] for a in ARMS};diagnostics=[];uniforms=[]
            for q in qs:
                pulse(phase='support-queries',draw=draw,mode=mode)
                u=rng('v19-rollout-query',draw,*q).uniform(size=(16,3));uniforms.append(u);diagnostics.append(support(counts,direct,q))
                for arm in ARMS:
                    start=time.process_time()
                    if arm=='direct':p=R.normalized(direct.get(q,np.ones(8)))
                    elif arm=='retrieval':p=T.retrieval(direct,q)
                    elif arm=='learned-exact':p=R.propagate(table,q)
                    elif arm=='oracle-exact':p=np.eye(8)[R.oracle(q)]
                    else:p=empirical(R.sampled_paths(table,q,u[:int(arm.split('-')[1])]))
                    predictions[arm].append(smooth(p));timing.append(dict(draw=draw,mode=mode,phase='query',arm=arm,cpu_seconds=time.process_time()-start))
            arrays={a:np.array(ps) for a,ps in predictions.items()}
            np.savez(root/'forecasts'/f'{draw}-{mode}.npz',queries=np.array(qs),uniforms=np.array(uniforms),**arrays)
            write(root/'forecasts'/f'{draw}-{mode}-support.json',diagnostics);rows=[]
            for lineage,groups in populations.items():
                for arm in ARMS:
                    entries=[]
                    for i,q in enumerate(qs):
                        if q not in groups:continue
                        target,mass=groups[q];p=arrays[arm][i]
                        entries.append(dict(lineage=lineage,draw=draw,mode=mode,arm=arm,query_index=i,probability_mass=mass,loss=float(-np.log(p[target])),squared_error=float(np.sum((p-np.eye(8)[target])**2)),**diagnostics[i]))
                    rows.extend(entries)
                    subsets={'all':entries,'query-seen':[r for r in entries if r['query_seen']],
                        'query-unseen':[r for r in entries if not r['query_seen']],
                        'heldout-composition':[r for r in entries if r['withheld_composition']],
                        'heldout-seen-primitives':[r for r in entries if r['withheld_composition'] and r['all_primitives_seen']]}
                    for subset,rr in subsets.items():
                        mass=sum(r['probability_mass'] for r in rr)
                        if not mass:continue
                        cells.append(dict(lineage=lineage,draw=draw,mode=mode,arm=arm,subset=subset,queries=len(rr),population_mass=mass,
                            **{k:float(sum(r['probability_mass']*r[k] for r in rr)/mass) for k in ('loss','squared_error')}))
            (root/'raw'/f'{draw}-{mode}-support_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0));total_rows+=len(rows)
    write(root/'TIMING.jsonl',dict(measurements=timing,accounting='native charge includes new holdout fits, all separately timed arm queries and scoring; old F1 fitting retained'))
    return dict(controls=checks,cells=cells,rows=total_rows,fits=len(cfg['training_draws']),queries=len(qs),
        scope='support diagnostic and predeclared action-composition holdout; constructed method; fewer retained training paths after holdout; no capacity match or historical intent claim')
