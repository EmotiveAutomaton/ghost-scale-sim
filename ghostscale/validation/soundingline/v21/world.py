"""V20 transition regression with separate current/history observation channels."""
from functools import lru_cache
from itertools import product
import numpy as np
from ..v20 import world as old

LABELS=old.LABELS
MAKERS=old.MAKERS
CONTEXTS=old.CONTEXTS
TIERS=old.TIERS
seed=old.seed
project=old.project
sampled=old.sampled

def law(lineage,shift='native',regime='original'):
    rng=np.random.default_rng(seed('v21-law',int(lineage)))
    result={k:int(rng.integers(2,7)) for k in ('route','select','revise','goal','inspect')}
    result.update(skill=int(rng.integers(5,9)),presentation=0,lineage=int(lineage),shift=shift)
    if regime=='wide':
        result.update(route=int(rng.integers(1,10)),inspect=int(rng.integers(1,10)))
    elif regime!='original':raise ValueError('unregistered world regime')
    return result

@lru_cache(maxsize=3)
def skeleton(shift):
    # All law coefficients affect probability, not the enumerated transitions.
    return old.enumerate_world(987654321,shift).copy()

def weights(rows,w):
    mi=rows[:,0].astype(int);ci=rows[:,1].astype(int)
    goal,skill,pref,review=np.asarray(MAKERS)[mi].T
    initial,request,available=np.asarray(CONTEXTS)[ci].T
    route,proposal,selected,inspection,revised,order,precision=rows[:,2:9].T
    p=np.where((goal&1)==request,w['goal'],10-w['goal'])/1280
    p*=np.where(available,np.where(route==pref,w['route'],10-w['route'])/10,1)/4
    for bit,q in ((selected,np.where(proposal==(goal&1),w['select'],10-w['select'])),
                  (revised,np.where(selected,w['revise'],10-w['revise'])),
                  (inspection,np.where(review,w['inspect'],10-w['inspect'])),
                  (precision,np.where(skill,w['skill'],10-w['skill']))):
        p*=np.where(bit,q,10-q)/10
    return p/p.sum()

def enumerate_law(w):
    rows=skeleton(w['shift']).copy();rows[:,-1]=weights(rows,w);return rows

def enumerate_world(lineage,shift='native',regime='original'):
    return enumerate_law(law(lineage,shift,regime))

def all_laws(shift='native',regime='original'):
    r=range(1,10) if regime=='wide' else range(2,7)
    for values in product(r,range(2,7),range(2,7),range(5,9),range(2,7),r):
        yield dict(zip(('route','select','revise','skill','goal','inspect'),values),shift=shift,lineage=-1,presentation=0)

def inputs(rows,chosen,tier,history,identity,review_observed=False,history_tier=None,history_observed=False,
           persistence=1.,duplicate_rate=0.):
    rng=np.random.default_rng(seed('v21-history',identity));parts=[project(chosen,tier,review_observed)];indices=[]
    groups=[np.flatnonzero(rows[:,0]==m) for m in range(32)];maker=chosen[:,0].astype(int)
    for step in range(history):
        owners=np.where(rng.random(len(chosen))<persistence,maker,rng.integers(32,size=len(chosen)))
        inds=np.empty(len(chosen),int)
        for m,rr in enumerate(groups):
            pos=np.flatnonzero(owners==m)
            if len(pos):inds[pos]=rng.choice(rr,len(pos),p=rows[rr,-1]/rows[rr,-1].sum())
        if step and duplicate_rate:
            copied=rng.random(len(chosen))<duplicate_rate;inds[copied]=indices[0][copied]
        indices.append(inds);parts.append(project(rows[inds],history_tier or tier,history_observed))
    return np.concatenate(parts,axis=1),indices
