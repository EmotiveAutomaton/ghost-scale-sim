"""Finite contribution generator. Policy weights are stipulated, not psychology.

32 makers x 8 contexts x (1 or 2 routes) x 2**6 episode choices = 24576
weighted trajectories. Three substantive edits: selection, revision and repair.
Proposal and optional inspection are separate observed events, not effort units.
"""
from itertools import product
from functools import lru_cache
import hashlib
import numpy as np

TIERS=('artifact','context','sparse','complete')
MAKERS=tuple(product(range(4),range(2),range(2),range(2)))
CONTEXTS=tuple(product(range(2),repeat=3))
BITS=tuple(product(range(2),repeat=7))
LABELS=np.array(BITS,dtype=np.int8)  # route, select, revise, order, inspect, aims A/B
FEATURES=('claim','evidence','presentation','initial','request','available','route','selected','revised','order','inspection','context_cue','repeat_cue')

def seed(*items):
    return int.from_bytes(hashlib.sha256(repr(items).encode()).digest()[:8],'little')

def law(lineage,shift='native'):
    rng=np.random.default_rng(seed('v20-law',int(lineage)))
    return dict(lineage=int(lineage),route=int(rng.integers(2,7)),select=int(rng.integers(2,7)),
        revise=int(rng.integers(2,7)),skill=int(rng.integers(5,9)),goal=int(rng.integers(2,7)),
        inspect=int(rng.integers(2,7)),presentation=int(rng.integers(2)),shift=shift)

def execute(initial,proposal,selected,revised,order,precision,goal,shift='native'):
    state=[initial,initial,0]; undo=initial
    if selected:state[0]=proposal
    for op in (('revise','repair') if order==0 else ('repair','revise')):
        if op=='revise' and revised:
            state[0]=(goal&1) if precision else undo
        if op=='repair':state[1]=state[0] if shift!='changed-tool' else 1-state[0]
    if revised:state[2]=((goal>>1)&1) if shift!='presentation-shift' else 1-((goal>>1)&1)
    return tuple(state)

@lru_cache(maxsize=4)
def enumerate_world(lineage,shift='native'):
    w=law(lineage,shift); records=[]
    for mi,(goal,skill,pref,review) in enumerate(MAKERS):
        for ci,(initial,request,available) in enumerate(CONTEXTS):
            maker_weight=(w['goal'] if (goal&1)==request else 10-w['goal'])/160.
            for route in range(available+1):
                route_p=1. if not available else (w['route']/10 if route==pref else 1-w['route']/10)
                for proposal,selected,inspection,revised,order,precision in product(range(2),repeat=6):
                    select_p=(w['select'] if proposal==(goal&1) else 10-w['select'])/10
                    revise_p=(w['revise'] if selected else 10-w['revise'])/10
                    inspect_p=(w['inspect'] if review else 10-w['inspect'])/10
                    precision_p=w['skill']/10 if skill else (10-w['skill'])/10
                    p=maker_weight/8*route_p*.5*.5
                    for b,q in ((selected,select_p),(revised,revise_p),(inspection,inspect_p),(precision,precision_p)):
                        p*=q if b else 1-q
                    y=route*64+selected*32+revised*16+order*8+inspection*4+(goal&1)*2+(goal>>1)
                    final=execute(initial,proposal,selected,revised,order,precision,goal,shift)
                    records.append((mi,ci,route,proposal,selected,inspection,revised,order,precision,goal,y,*final,p))
    a=np.array(records);a[:,-1]/=a[:,-1].sum()
    return a

def project(rows,tier,review_observed=False):
    if tier not in TIERS:raise ValueError('unknown evidence tier')
    x=np.full((len(rows),len(FEATURES)),-1,dtype=np.int8)
    x[:,:3]=rows[:,11:14]
    if tier!='artifact':x[:,3:6]=np.asarray(CONTEXTS)[rows[:,1].astype(int)]
    if tier in ('sparse','complete'):x[:,6]=rows[:,2]
    if tier=='complete':
        x[:,7:10]=rows[:,[4,6,7]]
        if review_observed:x[:,10]=rows[:,5]
    return x

def sampled(rows,n,*identity):
    rng=np.random.default_rng(seed('sample',*identity))
    return rows[rng.choice(len(rows),size=n,p=rows[:,-1])]

def group_reference(rows,tier,review_observed=False):
    x=project(rows,tier,review_observed);keys,inv=np.unique(x,axis=0,return_inverse=True)
    masses=np.zeros((len(keys),128));np.add.at(masses,(inv,rows[:,10].astype(int)),rows[:,-1])
    return keys,masses,inv

def events(row,inspection_visible=False):
    """Observed ProcessEvent ABI only. Private aims never become recorded facts."""
    def event(i,operation,actor,target,parents=(),payload=None):
        return dict(event_id='event-'+str(i),order=i,actor_id=actor,operation=operation,target=target,
            parent_event_ids=list(parents),primary_goal_id=None,secondary_goal_candidates=[],constraint_ids=[],alternatives=[],
            perceptual_access=None,noticed=None,visible_in_final='unknown',ground_truth_source='construction',payload=payload or {})
    result=[event(0,'propose','tool' if row[2] else 'human','content-0',payload={'proposal':int(row[3])}),
        event(1,'select' if row[4] else 'reject','human','content-0',('event-0',))]
    if inspection_visible and row[5]:result.append(event(2,'perceive','human','content-0',('event-1',),{'recorded_scope':'inspection; construction only'}))
    for op in (('revise','repair') if row[7]==0 else ('repair','revise')):
        if op=='revise' and not row[6]:op='retain'
        result.append(event(len(result),op,'human','content-1' if op=='repair' else 'content-0',(result[-1]['event_id'],)))
    return result
