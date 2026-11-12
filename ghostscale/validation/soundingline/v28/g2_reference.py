"""Scalar compiled G2 reconstruction, independently stated from the packet."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def _solve(stocks,actions,parameters,true_parameter):
    worlds,steps=stocks.shape;n=len(parameters)
    sep=np.empty((worlds,9));bad=np.full((worlds,9),-1);odds=np.empty((worlds,9))
    fits=np.empty((worlds,2));loss=np.empty(worlds);theta12=np.empty(worlds)
    for w in range(worlds):
        likelihood=np.zeros(n+95);reference=np.empty(n+95);streak=np.zeros(n+95)
        params=np.empty((n+95,4));params[:n]=parameters
        k=n
        for theta in range(4,23):
            for precision in (.25,.5,1.,2.,4.):
                params[k]=np.array([1.,float(theta),precision,0.]);k+=1
        for h in range(n+95):reference[h]=params[h,1]
        for t in range(steps):
            stock=float(stocks[w,t]);action=actions[w,t]
            for h in range(n+95):
                typ=int(params[h,0]);theta=params[h,1];pi=params[h,2];rate=params[h,3]
                if typ==3 and t>0:reference[h]=min(20.,max(2.,reference[h]+rate*(stocks[w,t-1]-reference[h])))
                condition=(typ==4 and stock<theta-2) or (typ==5 and abs(stock-theta)<=2)
                streak[h]=streak[h]+1 if condition else 0
                if typ==4:pi*=math.exp(-rate*streak[h])
                if typ==5:pi*=min(3.,1+rate*streak[h])
                lower=reference[h] if typ==3 else theta
                upper=theta+rate if typ==2 else theta
                restore=math.exp(pi*max(0.,lower-stock)-3.)
                spend=math.exp(pi*max(0.,stock-upper)-3.) if typ==1 or typ==2 else 0.
                selected=1. if action==0 else (restore if action==1 else spend)
                likelihood[h]+=math.log(selected/(1+restore+spend)) if selected else -math.inf
            evidence=np.empty(9)
            for group in range(9):
                high=-math.inf;count=0
                for h in range(n):
                    included=int(params[h,0])==group if group<6 else int(params[h,0])==group-3 and params[h,3]==.1
                    if included:
                        high=max(high,likelihood[h]);count+=1
                total=0.
                if math.isfinite(high):
                    for h in range(n):
                        included=int(params[h,0])==group if group<6 else int(params[h,0])==group-3 and params[h,3]==.1
                        if included:total+=math.exp(likelihood[h]-high)
                    evidence[group]=high+math.log(total/count)
                else:evidence[group]=-math.inf
            for group in range(9):
                odds[w,group]=evidence[int(true_parameter[0])]-evidence[group]
                if odds[w,group]<1:bad[w,group]=t
        for group in range(9):sep[w,group]=math.inf if odds[w,group]<1 else bad[w,group]+1
        best=n
        for h in range(n+1,n+95):
            if likelihood[h]>likelihood[best]:best=h
        fits[w,0]=params[best,1];fits[w,1]=params[best,2]
        true_ll=0.
        for h in range(n):
            if np.all(params[h]==true_parameter):true_ll=likelihood[h]
        loss[w]=(true_ll-likelihood[best])/steps
        high=-math.inf
        for h in range(n):
            if params[h,0]==0:high=max(high,likelihood[h])
        num=0.;den=0.
        for h in range(n):
            if params[h,0]==0:
                value=math.exp(likelihood[h]-high) if math.isfinite(high) else 0.
                den+=value
                if params[h,1]==12:num+=value
        theta12[w]=num/den if den else math.nan
    return sep,bad,odds,fits,loss,theta12


def infer(raw,setting):
    parameters=[]
    for name in range(6):
        widths=(4.,8.) if name==2 else ((.02,.1) if name in (3,4,5) else (0.,))
        for theta in (8,10,12,14,16,18,20):
            for precision in (.5,1.,2.):
                for extra in widths:parameters.append((name,theta,precision,extra))
    result=_solve(raw['stocks'],raw['actions'],np.asarray(parameters,float),np.asarray(setting['truth'],float))
    sep,bad,odds,fits,loss,post=result
    return dict(separation=sep,last_below=bad,final_odds=odds,fitted_theta=fits[:,0],
        fitted_pi=fits[:,1],loss_per_step=loss,one_theta12_posterior=post)
