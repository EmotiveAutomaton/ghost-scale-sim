"""Independent scalar ledger likelihood, retaining prefix class evidence."""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def sigmoid(z,floor):
    if z<=math.log(floor/(1-floor)):return floor
    if z>=math.log((1-floor)/floor):return 1-floor
    return 1/(1+math.exp(-z))

@njit(cache=True)
def lse(x):
    if len(x)==0:return -math.inf
    top=np.max(x)
    if top==-math.inf:return top
    return top+math.log(np.exp(x-top).sum())

@njit(cache=True)
def kernel(records,params,initials,model,logprior,base,classes,slope,floor):
    worlds,steps,_=records.shape;size=len(params);ng=len(classes)
    evidence=np.empty((worlds,steps,ng));final=np.empty((worlds,len(base)))
    frozen_trace=np.zeros((worlds,steps))
    for w in range(worlds):
        streak=initials.copy();frozen=np.zeros(size,np.bool_);shutdown=np.zeros(size,np.int64)
        hardship=np.zeros(size,np.int64);breached=np.zeros(size,np.bool_)
        for j in range(size):frozen[j]=params[j,0]==4 and initials[j]>=params[j,2]
        ll=logprior.copy()
        for t in range(steps):
            money,need,stock,offer,price,accept,after,spent=records[w,t,:8]
            for j in range(size):
                typ=int(params[j,0]);weight=params[j,1];duration=int(params[j,2])
                shutdown[j]=max(0,shutdown[j]-1)
                hardship[j]=hardship[j]+1 if money==0 and need==6 else 0
                prob=0.
                if offer==-1:
                    gain=price/2+min(price,max(0,need-money))
                    restricted=typ==1 or typ==2 or typ==3 or (typ==4 and frozen[j]) or (typ>=5 and not breached[j])
                    prob=floor if restricted else sigmoid(slope*(gain-weight),floor)
                    if typ==6 and breached[j]:prob=floor/4
                    if typ>=5 and not breached[j] and hardship[j]>=8:prob=1.
                elif offer==1 and money>=price:
                    cost=price/2+max(0,min(price,need-money+price))
                    prob=1-floor if typ==2 else sigmoid(slope*(weight-cost),floor)
                if typ==3 and shutdown[j]>0:prob=0.
                chosen=prob if accept else 1-prob
                ll[j]+=math.log(chosen) if chosen>0 else -math.inf
                if offer==-1:streak[j]=0 if accept else streak[j]+1
                if typ==4 and streak[j]>=duration:frozen[j]=True
                if typ==3 and offer==-1 and price>=4:shutdown[j]=duration+1
                if typ>=5 and offer==-1 and accept and hardship[j]>=8:breached[j]=True
                meet=0.
                if after>0 and need>0:meet=sigmoid(slope*(min(need,3)-(weight if typ==2 else .5)),floor)
                chosen=meet if spent else 1-meet
                ll[j]+=math.log(chosen) if chosen>0 else -math.inf
            for g in range(ng):
                selected=params[:,0]==classes[g]
                evidence[w,t,g]=lse(ll[selected])-math.log(np.sum(base[:,0]==classes[g]))
            freeze=params[:,0]==4
            if np.any(freeze):
                denom=lse(ll[freeze]);numer=lse(ll[freeze & frozen])
                frozen_trace[w,t]=math.exp(numer-denom)
        for j in range(len(base)):final[w,j]=lse(ll[model==j])
    return evidence,final,frozen_trace


def likelihood(raw,s,frozen=False,breach=False):
    # Grid declarations are shared; every probability, state update and summation is independent.
    from . import ledger
    if frozen:
        from .x2 import expand
        base,params,initials,model,prior=expand(s.get('start',0),breach)
    else:
        base=ledger.candidates();params=base;initials=np.zeros(len(base),int)
        model=np.arange(len(base));prior=np.zeros(len(base))
    start=s.get('start',0)
    return kernel(raw['records'][:,start:start+s['steps']],params,initials,model,prior,base,
                  np.unique(base[:,0]),s.get('slope',1.5),s.get('tremble',.02))
