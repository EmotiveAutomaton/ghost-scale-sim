"""Independent scalar ride law and finite table filter for Q3."""
import math
import numpy as np
from numba import njit
from .ledger_reference import lse

@njit(cache=True)
def filter_table(states,rides,eligible,table,classes,true):
    n,steps=states.shape;m=table.shape[1];ll=np.zeros((n,m));last=np.full((n,3),-1);final=np.zeros((n,3));visit=np.zeros((n,216),np.int64);same=np.ones((n,3),np.bool_)
    for w in range(n):
        for t in range(steps):
            i=states[w,t]
            if eligible[w,t]:
                visit[w,i]+=1
                for j in range(m):ll[w,j]+=math.log(table[i,j] if rides[w,t] else 1-table[i,j])
            evidence=np.zeros(3)
            for k in range(3):evidence[k]=lse(ll[w,classes==k])-math.log(np.sum(classes==k))
            for k in range(3):
                final[w,k]=evidence[true]-evidence[k]
                if final[w,k]<3:last[w,k]=t
                if abs(final[w,k])>=1e-12:same[w,k]=False
        same[w,true]=False
    return last+2,final,ll,visit,same.astype(np.float64)

def infer(raw,s):
    from .q3 import candidates
    params,scaled,median=candidates(s['profile']);table=np.empty((216,len(params)));slope=s.get('slope',1.5);floor=s.get('tremble',0.)
    for i in range(216):
        for j,(kind,rate,theta) in enumerate(params):table[i,j]=rate*min(1-floor,max(floor,1/(1+math.exp(-slope*(theta-scaled[i,int(kind)])))))
    pair,odds,ll,visits,same=filter_table(raw['states'],raw['rides'],raw['eligible'],table,params[:,0].astype(np.int64),s['truth'])
    divergence=np.empty((216,3))
    for i in range(216):
        probabilities=[s['rate']*min(1-floor,max(floor,1/(1+math.exp(-slope*(median[k]-scaled[i,k]))))) for k in range(3)]
        p=probabilities[s['truth']]
        for k,q in enumerate(probabilities):divergence[i,k]=p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))
    return dict(same_probability=same,pair_separation=pair,final_class_odds=odds,final_likelihood=ll,visits=visits,divergence=divergence,informative_share=(visits@(divergence.max(1)>.01))/np.maximum(1,visits.sum(1)))
