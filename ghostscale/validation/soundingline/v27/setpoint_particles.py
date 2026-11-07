"""Prepared K3 alternative: bootstrap particle likelihood, NOT admitted by default.

An explicit protocol choice and resolution receipt are required before use.
Exact static-grid controls remain separate. No probability or likelihood floor
is applied. A particle population losing all exact-observation support raises
an instrument failure instead of falsely excluding that model.
"""
import math
import numpy as np
from numba import njit
from .common import rng


@njit
def step_weights(stock,edge,pi,streak,action,size,shock,law,width,beta,lapse):
    n,needs=stock.shape;weight=np.empty(n)
    for k in range(n):
        urgency=np.empty(needs);top=1.
        for j in range(needs):
            if shock==j:stock[k,j]=0.
            streak[k,j]=min(20,streak[k,j]+1) if stock[k,j]<edge[k,j] else 0
            if law==1 and stock[k,j]>=edge[k,j] and (width==0 or stock[k,j]<=edge[k,j]+width):pi[k,j]=min(10.,pi[k,j]*1.03)
            if streak[k,j]>=20:
                if law==2:pi[k,j]=max(.1,pi[k,j]*.97)
                if law==3:edge[k,j]=max(stock[k,j],edge[k,j]-.1)
                if law==4:pi[k,j]=min(10.,pi[k,j]*1.03)
            if law in (0,1):streak[k,j]=0
            urgency[j]=min(5.,max(0.,edge[k,j]-stock[k,j])*pi[k,j]);top=max(top,urgency[j])
        denom=math.exp(1-top)
        for j in range(needs):denom+=math.exp(urgency[j]-top)
        selected=1. if action==needs else urgency[action]
        weight[k]=(1-lapse)*math.exp(selected-top)/denom+lapse/(needs+1)
        if beta>=0 and action<needs and abs(size-5-beta*urgency[action])>1e-10:weight[k]=0.
    return weight


def reference_weights(stock,edge,pi,streak,action,size,shock,law,width,beta,lapse):
    if shock>=0:stock[:,shock]=0.
    below=stock<edge;streak[:]=np.where(below,np.minimum(20,streak+1),0)
    if law==1:
        inside=(stock>=edge)&((stock<=edge+width) if width else True);pi[:]=np.where(inside,np.minimum(10,pi*1.03),pi)
    if law==2:pi[:]=np.where(streak>=20,np.maximum(.1,pi*.97),pi)
    if law==3:edge[:]=np.where(streak>=20,np.maximum(stock,edge-.1),edge)
    if law==4:pi[:]=np.where(streak>=20,np.minimum(10,pi*1.03),pi)
    if law in (0,1):streak[:]=0
    urgency=np.minimum(5,np.maximum(0,edge-stock)*pi);u=np.column_stack([urgency,np.ones(len(stock))]);u-=u.max(axis=1)[:,None]
    probability=np.exp(u);probability/=probability.sum(axis=1)[:,None];weight=(1-lapse)*probability[:,action]+lapse/probability.shape[1]
    if beta>=0 and action<stock.shape[1]:weight=np.where(np.abs(size-5-beta*urgency[:,action])<=1e-10,weight,0.)
    return weight


@njit
def reference_ancestors(weights,uniform):
    n=len(weights);out=np.empty(n,np.int64);index=0;cum=weights[0]
    for j in range(n):
        target=(j+uniform)/n
        while target>cum and index<n-1:index+=1;cum+=weights[index]
        out[j]=index
    return out


def likelihood(raw,s,point,dynamic,width,beta,independent=False,particles=8192,replicate=0):
    if 'inference_key' not in raw:raise ValueError('particle seed identity not recorded')
    needs=s['needs'];r=rng('particle-engineering',raw['inference_key'],list(point),dynamic,width,beta,particles,replicate)
    stock=r.integers(0,21,(particles,needs)).astype(float);edge=np.tile(np.array(point)-width/2,(particles,1));pi=np.ones_like(stock);streak=np.zeros_like(stock,dtype=int)
    code=('P0','P+','P-','S','G').index(dynamic);logs=[];acc=0.;minimum_ess=particles
    for t,row in enumerate(raw['records']):
        kernel=reference_weights if independent else step_weights
        weight=kernel(stock,edge,pi,streak,row['action'],row['size'],row['shock'],code,width,-1. if beta is None else beta,s['lapse'])
        total=math.fsum(weight) if independent else float(weight.sum())
        if total<=0:raise ArithmeticError(f'particle support collapse at record {t+1}; increase/guided proposal required, never floor')
        acc+=math.log(total/particles);logs.append(acc);weight/=total
        minimum_ess=min(minimum_ess,1/float(weight@weight));uniform=float(r.random())
        if independent:indices=reference_ancestors(weight,uniform)
        else:indices=np.searchsorted(np.cumsum(weight),(np.arange(particles)+uniform)/particles,side='left');indices=np.minimum(indices,particles-1)
        stock=stock[indices].copy();edge=edge[indices].copy();pi=pi[indices].copy();streak=streak[indices].copy()
        draws=r.random((particles,needs));loss=np.where(draws<.45,0.,np.where(draws<.95,1.,5.))*s['consumption']
        stock-=loss
        if row['action']<needs:stock[:,row['action']]+=row['size']
        np.clip(stock,0,20,out=stock)
    return np.array(logs),dict(particles=particles,minimum_effective_particles=minimum_ess,replicate=replicate)


def resolution(raw,s,point,dynamic,width,beta,counts=(2048,8192,32768)):
    rows=[]
    for count in counts:
        values=[likelihood(raw,s,point,dynamic,width,beta,particles=count,replicate=k)[0] for k in range(4)]
        rows.append(dict(particles=count,replicates=4,final_log_likelihoods=[float(v[-1]) for v in values],
                         largest_prefix_standard_deviation=float(np.std(values,axis=0,ddof=1).max())))
    return dict(resolutions=rows,passed=False,
        admission='Requires owner interpretation and a separately frozen posterior/margin stability assessment over all registered dynamic hypotheses; these likelihood diagnostics alone do not admit science.')
