"""Finite forward-table pilot, with explicit mechanics metadata and compute controls."""
from collections import defaultdict
from itertools import product
import gzip
import time
import numpy as np
from ..v18_3.io import canonical,write
from ..v18_3.world import rng
from . import local_world as L

ARTIFACTS=tuple(product(range(2),repeat=3))
ARMS=('direct','direct-mc16','rollout-1','rollout-4','rollout-16','learned-exact','wrong-law-16','oracle-exact')

def code(a):return 4*a[0]+2*a[1]+a[2]
def query(r):return (r['maker'][1],r['maker'][2],code(r['initial']),*(L.OPERATIONS.index(e['operation']) for e in r['steps']))
def visible(q):return dict(skill=q[0],belief_error=q[1],initial=list(ARTIFACTS[q[2]]),operations=[L.OPERATIONS[i] for i in q[3:]])
def validate_visible(p):
    if set(p)!={'skill','belief_error','initial','operations'}:raise ValueError('invalid reader fields')
    if p['skill'] not in (0,1) or p['belief_error'] not in (0,1) or tuple(p['initial']) not in ARTIFACTS or len(p['operations'])!=3 or not set(p['operations'])<=set(L.OPERATIONS):raise ValueError('invalid reader values')
    if not p['skill'] and 'accept-tool' in p['operations']:raise ValueError('unavailable tool')
    return True

def fit(records):
    transitions=np.ones((2,2,8,8,6,8));direct={}
    for r in records:
        q=query(r);direct.setdefault(q,np.ones(8))[code(r['final'])]+=1
        for e in r['steps']:transitions[q[0],q[1],code(e['before']),code(e['undo_buffer']),L.OPERATIONS.index(e['operation']),code(e['after'])]+=1
    return transitions,direct

def normalized(x):return x/x.sum(axis=-1,keepdims=True)

def propagate(table,q):
    state=np.zeros((8,8));state[q[2],q[2]]=1
    for op in q[3:]:
        following=np.zeros_like(state)
        for a,b in zip(*np.nonzero(state)):
            following[:,a]+=state[a,b]*table[q[0],q[1],a,b,op]
        state=following
    return state.sum(axis=1)

def sample(p,u):return min(int(np.searchsorted(np.cumsum(p),u,side='right')),7)
def sampled_paths(table,q,uniforms,wrong=False):
    endpoints=[]
    for us in uniforms:
        a=b=q[2]
        for op,u in zip(q[3:],us):
            n=sample(table[q[0],q[1],a,b,op],u)
            if wrong:n^=1
            b,a=a,n
        endpoints.append(a)
    return endpoints
def estimate(endpoints):return (np.bincount(endpoints,minlength=8)+.125)/(len(endpoints)+1)
def oracle(q):
    a=b=ARTIFACTS[q[2]];maker=(0,q[0],q[1],0)
    for op in q[3:]:b,a=a,L.execute(a,b,L.OPERATIONS[op],maker)
    return code(a)
def forecasts(table,direct,q,uniforms):
    d=normalized(direct.get(q,np.ones(8)));ends=sampled_paths(table,q,uniforms)
    exact=np.zeros(8);exact[oracle(q)]=1
    return dict(zip(ARMS,[d,estimate([sample(d,u) for u in uniforms[:,-1]]),
        *[estimate(ends[:n]) for n in (1,4,16)],propagate(table,q),estimate(sampled_paths(table,q,uniforms,True)),exact]))

def controls():
    q=(1,0,2,4,4,4);table=np.zeros((2,2,8,8,6,8))
    for skill,belief,a,b,op in product(range(2),range(2),range(8),range(8),range(6)):
        if not skill and op==3:table[skill,belief,a,b,op]=.125
        else:table[skill,belief,a,b,op,code(L.execute(ARTIFACTS[a],ARTIFACTS[b],L.OPERATIONS[op],(0,skill,belief,0)))]=1
    u=np.full((16,3),.5);prediction=propagate(table,q);wrong=sampled_paths(table,q,u,True)
    return {'live:known_execution':bool(oracle(q)==2 and prediction[2]==1),
        'positive:undo_buffer':oracle((1,0,2,0,5,4))==2,
        'placebo:unseen_uniform':bool(np.all(normalized(np.ones(8))==.125)),
        'positive:normalized_propagation':bool(abs(propagate(np.ones_like(table)/8,q).sum()-1)<1e-12),
        'positive:shared_prefixes':sampled_paths(table,q,u)[:4]==sampled_paths(table,q,u[:4]),
        'live:wrong_law':wrong==[3]*16,
        'positive:finite_mc':bool(np.isclose(estimate([2])[2],.5625) and np.isclose(estimate([2]).sum(),1))}

def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['arms']!=list(ARMS):raise ValueError('rollout admission failed')
    for folder in ('raw','models','forecasts','reader','auxiliary'):(root/folder).mkdir()
    training={};timing=[]
    for l in cfg['train_lineages']:
        pulse(phase='training-populations',lineage=l);rr=L.enumerate_world(L.law(l));training[l]=rr
        (root/'raw'/f'train-{l}_points.json.gz').write_bytes(gzip.compress(canonical(rr),mtime=0))
    populations={};queries={}
    for l in cfg['development_lineages']:
        pulse(phase='evaluation-populations',lineage=l);rr=L.enumerate_world(L.law(l));groups={}
        for r in rr:
            q=query(r);target=code(r['final'])
            if q in groups:
                if groups[q][0]!=target:raise ValueError('query is not mechanics sufficient')
                groups[q][1]+=r['probability']
            else:groups[q]=[target,r['probability']]
            if oracle(q)!=target:raise ValueError('known-law endpoint failure')
            queries[q]=target
        if abs(sum(v[1] for v in groups.values())-1)>1e-12:raise ValueError('population mass')
        populations[l]=groups;(root/'raw'/f'development-{l}_points.json.gz').write_bytes(gzip.compress(canonical(rr),mtime=0))
    qs=sorted(queries);public=[visible(q) for q in qs]
    for p in public:validate_visible(p)
    write(root/'reader/QUERIES.json',public);write(root/'QUERY_TRUTH.json',[dict(query=list(q),endpoint=queries[q]) for q in qs])
    rows=[];cells=[]
    for draw in cfg['training_draws']:
        selected=[];perline={}
        for l,rr in training.items():
            random=rng('v19-rollout-training',draw,l);weights=np.array([r['probability'] for r in rr]);weights/=weights.sum()
            perline[l]=random.choice(len(rr),size=cfg['paths_per_lineage'],p=weights).tolist()
        for index in range(cfg['paths_per_lineage']):
            for l in cfg['train_lineages']:selected.append(training[l][perline[l][index]])
        write(root/'auxiliary'/f'selection-{draw}.json',dict(indices=perline,paths=selected,role='complete transition and endpoint training supervision'))
        for budget in cfg['budgets']:
            pulse(phase='count-fitting',draw=draw,budget=budget);began=time.process_time();counts,direct=fit(selected[:budget]);table=normalized(counts)
            direct_keys=sorted(direct)
            np.savez(root/'models'/f'{draw}-{budget}.npz',transition_counts=counts,direct_keys=np.array(direct_keys),direct_counts=np.array([direct[q] for q in direct_keys]))
            fitted=time.process_time();predictions={arm:[] for arm in ARMS};uniforms=[]
            for q in qs:
                u=rng('v19-rollout-query',draw,*q).uniform(size=(16,3));uniforms.append(u)
                for arm,p in forecasts(table,direct,q,u).items():
                    if not np.isclose(p.sum(),1) or p.min()<0:raise ValueError('invalid forecast')
                    predictions[arm].append(p)
            arrays={a:np.array(ps) for a,ps in predictions.items()}
            np.savez(root/'forecasts'/f'{draw}-{budget}.npz',queries=np.array(qs),uniforms=np.array(uniforms),**arrays)
            for l,groups in populations.items():
                for arm in ARMS:
                    losses=[];squared=[];mass=[]
                    for i,q in enumerate(qs):
                        if q not in groups:continue
                        t,w=groups[q];p=arrays[arm][i];truth=np.eye(8)[t]
                        if p[t]<=0:raise ValueError('zero true endpoint mass')
                        loss=float(-np.log(p[t]));brier=float(np.sum((p-truth)**2))
                        rows.append(dict(draw=draw,budget=budget,lineage=l,query_index=i,arm=arm,probability_mass=w,loss=loss,squared_error=brier))
                        losses.append(loss);squared.append(brier);mass.append(w)
                    cells.append(dict(draw=draw,budget=budget,lineage=l,arm=arm,loss=float(np.dot(mass,losses)),squared_error=float(np.dot(mass,squared)),queries=len(mass)))
            timing.append(dict(draw=draw,budget=budget,fit_cpu_seconds=fitted-began,query_and_score_cpu_seconds=time.process_time()-fitted,
                visited_transition_rows=int(np.sum(counts.sum(-1)>8)),visited_direct_rows=len(direct),transition_count_parameters=int(counts.size),direct_visited_parameters=len(direct)*8))
    (root/'raw/rollout_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'TIMING.jsonl',dict(measurements=timing,accounting='included in native charge'))
    return dict(controls=checks,cells=cells,rows=len(rows),queries=len(qs),fits=len(cfg['budgets'])*len(cfg['training_draws']),
        scope='forward endpoint prediction with supplied skill/belief mechanics metadata; distinct from local-process recovery; miniature — architecture untested')
