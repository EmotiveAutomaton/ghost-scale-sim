"""Exhaustive finite codebooks; no iterative optimum is mislabeled as exact."""
from functools import lru_cache
from itertools import product,combinations
import numpy as np
from . import world as W


@lru_cache(maxsize=8)
def partitions(n):
    def extend(prefix):
        if len(prefix)==n:
            yield tuple(prefix);return
        for value in range(max(prefix,default=-1)+2):
            yield from extend(prefix+[value])
    return tuple(extend([]))


@lru_cache(maxsize=1)
def codebooks():
    codes=np.array(partitions(8),dtype=np.int16)
    masks=np.array([codes==i for i in range(8)]).transpose(1,0,2).astype(float)
    return codes,masks,codes.max(axis=1)+1


def finite_histories(w):
    states=[i for i,s in enumerate(W.STATES) if s[0]>0 and s[3]==0]
    queries=(W.context(),W.context(goal=0),W.context(signal=0,budget=1))
    predicates=(lambda p: w['groups'][0][0] in p,
                lambda p: w['groups'][1][0] in p,
                lambda p: len(p)>=2)
    probabilities=np.array([W.matrix(w,q)[states][:,[pred(p) for p in W.PROGRAMS]].sum(axis=1)
                            for q,pred in zip(queries,predicates)]).T
    histories=list(product((0,1),repeat=3))
    independent=np.array([np.prod(np.where(np.array(h)[None,:],probabilities,1-probabilities),axis=1) for h in histories])
    # Shared observation noise preserves each native marginal, but couples reports.
    shared=np.array([np.maximum(0,np.min(np.where(np.array(h)[None,:],probabilities,1.),axis=1)-
                                  np.max(np.where(np.array(h)[None,:],0.,probabilities),axis=1)) for h in histories])
    joint=(.5*independent+.5*shared if w['shared'] else independent)/len(states)
    ph=joint.sum(axis=1);post=joint/ph[:,None]
    old=post@W.artifact_matrix(w,W.context())[states]
    new=np.concatenate([post@W.artifact_matrix(w,q)[states] for q in (W.context(goal=1),W.context(signal=1),W.context(goal=0,signal=1))],axis=1)/3
    return histories,ph,post,old,new


def code_losses(masks,ph,predictions):
    mass=np.einsum('pch,h->pc',masks,ph)
    joint=np.einsum('pch,h,hy->pcy',masks,ph,predictions)
    prob=np.divide(joint,mass[:,:,None],out=np.ones_like(joint),where=mass[:,:,None]>0)
    terms=np.zeros_like(joint);np.log(prob,out=terms,where=prob>0)
    losses=-np.sum(joint*terms,axis=(1,2))
    logs=np.zeros_like(mass);np.log(mass,out=logs,where=mass>0)
    rates=-np.sum(mass*logs,axis=1)
    return losses,rates


def reference_loss(code,ph,predictions):
    result=0.
    for z in set(code):
        indices=[i for i,c in enumerate(code) if c==z]
        total=sum(ph[i] for i in indices)
        q=sum(ph[i]*predictions[i] for i in indices)/total
        for i in indices:result+=ph[i]*W.cross_entropy(predictions[i],q)
    return float(result)


def unit(index,cell=0,split='test',rule=None):
    w=W.make_world(cell,index+30000)
    if rule is not None:
        if rule!='lexicographic':raise ValueError('undeclared holdout rule')
        w['rule']=rule
    histories,ph,post,old,new=finite_histories(w)
    codes,masks,cardinality=codebooks()
    oldloss,rates=code_losses(masks,ph,old);newloss,_=code_losses(masks,ph,new)
    rows=[]
    for k in (1,2,4,8):
        candidates=np.flatnonzero(cardinality==k)
        best=int(candidates[np.argmin(oldloss[candidates])])
        selected_bits=int(np.log2(k));product_codes=[]
        for axes in combinations(range(3),selected_bits):
            values=[sum(h[axis]<<j for j,axis in enumerate(axes)) for h in histories]
            # Canonical relabeling makes a codebook's class identity explicit.
            lookup={};labels=[lookup.setdefault(v,len(lookup)) for v in values]
            match=np.flatnonzero(np.all(codes==labels,axis=1))
            assert len(match)==1;product_codes.append(int(match[0]))
        product_best=min(product_codes,key=lambda i:(oldloss[i],i))
        for method,i in (('exhaustive-flat',best),('observation-product',product_best)):
            if abs(reference_loss(codes[i],ph,old)-oldloss[i])>1e-12:raise ValueError('compression reference mismatch')
            rows.append(dict(method=method,cardinality=k,code=codes[i].tolist(),rate_nats=float(rates[i]),
                             storage_bits=float(np.log2(k)),old_loss=float(oldloss[i]),new_loss=float(newloss[i]-np.log(3)),
                             new_query_identity_entropy=float(np.log(3)),
                             enumerated_candidates=len(candidates) if method=='exhaustive-flat' else len(product_codes),
                             optimization_target='old query family only'))
    return dict(family='H',index=index,cell=cell,**({'rule':rule} if rule else {}),world=w,histories=histories,history_probabilities=ph.tolist(),
                old_predictions=old.tolist(),new_predictions=new.tolist(),partitions_enumerated=len(codes),rows=rows,
                scope='exact optimum within all deterministic partitions of eight declared histories at fixed cardinality')
