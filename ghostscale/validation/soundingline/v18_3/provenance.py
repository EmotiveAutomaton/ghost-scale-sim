"""Finite copy networks with uncertain, partial and mistaken source provenance."""
import math
import numpy as np
from . import world as W

MODES=('independent','copied','shared-error','selected','partial','wrong-provenance')


def root_probability(truth,flip=0,selected=False):
    p=.8 if (truth ^ flip) else .2
    if selected:p=p*.95/(p*.95+(1-p)*.25)
    return p


def likelihood(values,groups,truth,shared=False,selected=False):
    total=0.
    for flip,mass in ((0,.8),(1,.2)) if shared else ((0,1.),):
        p=root_probability(truth,flip,selected)
        term=mass
        for group in sorted(set(groups)):
            seen=[v for v,g in zip(values,groups) if g==group]
            p1=math.prod(.97 if v else .03 for v in seen)
            p0=math.prod(.03 if v else .97 for v in seen)
            term*=p*p1+(1-p)*p0
        total+=term
    return total


def partitions(n):
    result=set()
    for roots in sorted(set((1,2,4,n))):
        if roots>n:continue
        result.add(tuple(min(roots-1,i*roots//n) for i in range(n)))
        result.add(tuple(i%roots for i in range(n)))
    return sorted(result)


def posterior(values,candidates,shared=False,selected=False,links=()):
    candidates=[g for g in candidates if all((g[a]==g[b])==same for a,b,same in links)]
    if not candidates:return None
    weights=np.array([sum(likelihood(values,g,z,shared,selected) for g in candidates) for z in (0,1)])
    return weights/weights.sum() if weights.sum() else None


def unit(index,roots=2,copies=3,mode='copied',split='test'):
    r=W.rng('source',split,index,roots,mode)
    truth=int(r.integers(2));shared=mode=='shared-error';selected=mode=='selected'
    groups=list(range(roots*copies)) if mode=='independent' else [i for i in range(roots) for _ in range(copies)]
    flip=int(shared and r.random()<.2)
    root_values={g:int(W.rng('source-root',split,index,roots,mode,g).random()<root_probability(truth,flip,selected)) for g in set(groups)}
    reports=[root_values[g] ^ int(W.rng('copy-noise',split,index,roots,mode,g,sum(h==g for h in groups[:i])).random()<.03) for i,g in enumerate(groups)]
    links=[]
    if mode in ('partial','wrong-provenance'):
        for a in range(0,len(groups)-1,3):
            links.append([a,a+1,groups[a]==groups[a+1]])
        if mode=='wrong-provenance' and len(groups)>1:
            links=[[0,len(groups)-1,True]]
    correction=[truth ^ int(W.rng('independent-correction',split,index,roots,mode,j).random()>.9) for j in range(2)]
    n=len(reports);candidates=partitions(n)
    methods={
        'independent':([tuple(range(n))],False,False,()),
        'known-graph':([tuple(groups)],shared,selected,()),
        'unknown-graph':(candidates,False,False,()),
        'partial-graph':(candidates,False,False,links),
        'cautious-mixture':(candidates,True,False,links),
        'selection-aware':(candidates,True,True,links),
    }
    rows=[]
    for name,(graphs,bias,selection,metadata) in methods.items():
        weights=posterior(reports,graphs,bias,selection,metadata)
        if weights is None:
            rows.append(dict(method=name,instrument='incompatible-provenance',posterior=None));continue
        initial=weights.copy()
        after=[]
        for value in correction:
            weights*=np.array([.9 if z==value else .1 for z in (0,1)]);weights/=weights.sum()
            after.append(weights.tolist())
        rows.append(dict(method=name,instrument='valid',posterior=initial.tolist(),after_correction=after,
                         initial_loss=-math.log(initial[truth]),final_loss=-math.log(weights[truth]),
                         initial_brier=float((initial[1]-truth)**2),final_brier=float((weights[1]-truth)**2),
                         false_confidence=bool(initial[1-truth]>.95),corrected_false_confidence=bool(weights[1-truth]>.95),
                         credible_coverage=bool(initial[truth]>=.05),root_hypotheses=len(graphs)))
    return dict(family='C',index=index,roots=roots,copies=copies,mode=mode,
                public=dict(reports=reports,links=links,independent_correction=correction),
                evaluator=dict(truth=truth,groups=groups,root_values=root_values,shared_flip=flip,
                               selected_roots=selected,independent_roots=len(set(groups))),rows=rows)
