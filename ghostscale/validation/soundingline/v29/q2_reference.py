"""Independent scalar sufficient-statistic Q2 reconstruction; no producer likelihood calls."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def lse(x):
    maximum=np.max(x)
    return maximum+math.log(np.exp(x-maximum).sum())


@njit(cache=True)
def kernel(deficits, actions, successes, watching, true, slope, floor):
    worlds,steps=deficits.shape
    result=np.zeros((worlds,11)); lls=np.zeros((worlds,8));same=np.ones((worlds,8))
    for w in range(worlds):
        won=0.;lost=0.;last=-1;lp=-1;lk=-1
        for t in range(steps):
            if watching[t]:
                won+=.5*successes[w,t];lost+=.5*(1-successes[w,t])
            else:
                probs=np.zeros(8)
                for j in range(8):
                    a=(1. if j<4 else 4.)+won;b=(4. if j<4 else 1.)+lost
                    k=(.25,.5,1.,2.)[j%4]
                    z=slope*(a/(a+b)*k*deficits[w,t]-1.)
                    prob=1./(1.+math.exp(-z));prob=min(1-floor,max(floor,prob))
                    probs[j]=prob
                    lls[w,j]+=math.log(prob if actions[w,t] else 1-prob)
                for j in range(8):
                    if probs[j]!=probs[true]:same[w,j]=0
                won+=successes[w,t];lost+=actions[w,t]-successes[w,t]
                if t and not watching[t-1] and successes[w,t-1] and deficits[w,t]==deficits[w,t-1]:
                    result[w,7]+=actions[w,t]-actions[w,t-1];result[w,8]+=1
            best=-math.inf
            for j in range(8):
                if j!=true:best=max(best,lls[w,j])
            if lls[w,true]-best<3:last=t
            prior=np.array([lse(lls[w,:4]),lse(lls[w,4:])]);sens=np.zeros(4)
            for j in range(4):sens[j]=lse(np.array([lls[w,j],lls[w,j+4]]))
            if prior[true//4]-prior[1-true//4]<3:lp=t
            best=-math.inf
            for j in range(4):
                if j!=true%4:best=max(best,sens[j])
            if sens[true%4]-best<3:lk=t
        result[w,9]=int(prior[true//4]-prior[1-true//4]>=3);result[w,10]=int(prior[true//4]-prior[1-true//4]<=-3)
        result[w,0]=last+2;result[w,1]=lp+2;result[w,2]=lk+2
        result[w,3]=int(np.argmax(prior)==true//4);result[w,4]=int(np.argmax(sens)==true%4)
        result[w,5]=actions[w].sum()/max(1,(watching==0).sum())
    same[:,true]=0
    return result,lls,same


def infer(raw,s):
    out,ll,same=kernel(raw['deficits'],raw['actions'],raw['successes'],raw['watching'],
                  s['truth'],s.get('slope',2.),s.get('tremble',.01))
    return dict(same_probability=same,separation=out[:,0],prior_separation=out[:,1],sensitivity_separation=out[:,2],
                final_likelihood=ll,prior_read_right=out[:,9],prior_read_wrong=out[:,10],prior_correct=out[:,3],sensitivity_correct=out[:,4],
                push_rate=out[:,5],success_response_sum=out[:,7],success_response_count=out[:,8])
