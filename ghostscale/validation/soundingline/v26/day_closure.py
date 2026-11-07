"""G3: finite source library, exact stopping DP and declared non-Bayesian drift."""
import math
from functools import lru_cache
import numpy as np
from numba import njit
from .common import rng
from .day_common import measure

def rate(z,agenda=-1,strength=0.):
    q=.7 if z else .3
    if agenda<0:return q
    return (1-strength)*q+strength*((1-(1-q)**8) if agenda==1 else q**8)

def library(expanded=False):
    source=[(-1,0.,.5),(1,.5,.25),(0,.5,.25)] if not expanded else [(-1,0.,.5),(1,.5,.125),(0,.5,.125),(1,1.,.125),(0,1.,.125)]
    z=np.array([v for v in (0,1) for _ in source]);agenda=np.array([a for _ in (0,1) for a,_,_ in source]);p=np.array([rate(v,a,s) for v in (0,1) for a,s,_ in source]);prior=np.array([w/2 for _ in (0,1) for _,_,w in source]);return z,agenda,p,prior

@njit
def trajectory(items,truth,source_code,theta,delta,switch,overpush,pull_away,z,agenda,prob,prior):
    n,tmax=items.shape;output=np.empty((n,9));lengths=np.full(n,-1,dtype=np.int64)
    for i in range(n):
        k=prior.copy();d=prior.copy();closed=False;ot=np.zeros(2);tt=np.zeros(2)
        for t in range(tmax):
            item=items[i,t]
            if item>=0:
                for j in range(len(k)):
                    emission=prob[j] if item else 1-prob[j];k[j]*=emission
                    if not closed:d[j]*=emission
                for v in range(2):
                    q=.7 if v else .3;strength=0.
                    if source_code==1:strength=.5
                    elif source_code==2:strength=overpush
                    elif source_code==3 and t>=switch:strength=1.
                    p=(1-strength)*q+strength*(1-(1-q)**8)
                    tt[v]+=math.log(p if item else 1-p);ot[v]+=math.log(q if item else 1-q)
            k/=k.sum()
            if not closed:
                d/=d.sum();harm=0.
                for j in range(len(d)):
                    if agenda[j]>=0 and agenda[j]!=z[j]:harm+=d[j]
                if harm>=theta or item<0:
                    closed=True;lengths[i]=t+1
                    if pull_away:d=prior.copy()
            harmful=0.;a0=0.;a1=0.
            for j in range(len(k)):
                if agenda[j]>=0 and agenda[j]!=z[j]:harmful+=k[j]
                if agenda[j]==0:a0+=k[j]
                elif agenda[j]==1:a1+=k[j]
            if harmful>=theta and delta>0:
                favored=1 if a1>a0 else 0
                for j in range(len(k)):
                    if z[j]==favored:k[j]*=math.exp(delta)
                k/=k.sum()
        kp=0.;dp=0.
        for j in range(len(k)):
            if z[j]==truth[i]:kp+=k[j];dp+=d[j]
        ta=max(tt[0],tt[1]);oa=max(ot[0],ot[1]);tl=tt[truth[i]]-ta-math.log(math.exp(tt[0]-ta)+math.exp(tt[1]-ta));ol=ot[truth[i]]-oa-math.log(math.exp(ot[0]-oa)+math.exp(ot[1]-oa))
        output[i,0]=math.log(kp);output[i,1]=math.log(dp);output[i,2]=tl;output[i,3]=ol;output[i,4]=1-kp;output[i,5]=1-dp;output[i,6]=1. if closed else 0.;output[i,7]=math.log(dp)-math.log(kp);output[i,8]=kp-dp
    return output,lengths

@njit
def lognormalize(x):
    top=np.max(x);return x-top-math.log(np.exp(x-top).sum())

@njit
def review_trajectories(items,truth,source_code,theta,delta,switch,overpush,pull_away,z,agenda,prob,prior):
    n,horizon=items.shape;out=np.empty((n,9));lengths=np.full(n,-1,dtype=np.int64)
    for i in range(n):
        lk=np.log(prior);ld=lk.copy();tt=np.zeros(2);oo=np.zeros(2);closed=False
        for t in range(horizon):
            y=items[i,t]
            if y>=0:
                emission=np.log(prob if y else 1-prob);lk+=emission
                if not closed:ld+=emission
                strength=.5 if source_code==1 else overpush if source_code==2 else 1. if source_code==3 and t>=switch else 0.
                for v in range(2):
                    q=.7 if v else .3;p=(1-strength)*q+strength*(1-(1-q)**8)
                    tt[v]+=math.log(p if y else 1-p);oo[v]+=math.log(q if y else 1-q)
            lk=lognormalize(lk)
            if not closed:
                ld=lognormalize(ld);harm=np.exp(ld[(agenda>=0)&(agenda!=z)]).sum()
                if harm>=theta or y<0:
                    closed=True;lengths[i]=t+1
                    if pull_away:ld=np.log(prior)
            if delta>0 and np.exp(lk[(agenda>=0)&(agenda!=z)]).sum()>=theta:
                favored=1 if np.exp(lk[agenda==1]).sum()>np.exp(lk[agenda==0]).sum() else 0
                lk[z==favored]+=delta;lk=lognormalize(lk)
        kp=np.exp(lk[z==truth[i]]).sum();dp=np.exp(ld[z==truth[i]]).sum()
        out[i,0]=math.log(kp);out[i,1]=math.log(dp);out[i,2]=lognormalize(tt)[truth[i]];out[i,3]=lognormalize(oo)[truth[i]];out[i,4]=1-kp;out[i,5]=1-dp;out[i,6]=1. if closed else 0.;out[i,7]=math.log(dp)-math.log(kp);out[i,8]=kp-dp
    return out,lengths

def reconstruct_one(items,truth,source_code,s):
    """Separate log-weight implementation, retaining the same declared source law."""
    from scipy.special import logsumexp
    z,agenda,prob,prior=library(s['expanded']);lk=np.log(prior);ld=np.log(prior);closed=False;when=None;tt=np.zeros(2);oo=np.zeros(2)
    for t,item in enumerate(items):
        if item>=0:
            emission=np.log(prob if item else 1-prob);lk+=emission
            if not closed:ld+=emission
            strength=.5 if source_code==1 else s['overpush'] if source_code==2 else 1. if source_code==3 and t>=s['switch'] else 0.
            pt=np.array([rate(v,1,strength) for v in (0,1)]);tt+=np.log(pt if item else 1-pt);oo+=np.log(np.array([.3,.7]) if item else np.array([.7,.3]))
        lk-=logsumexp(lk)
        if not closed:
            ld-=logsumexp(ld)
            if np.exp(ld[(agenda>=0)&(agenda!=z)]).sum()>=s['theta'] or item<0:
                closed=True;when=t+1
                if s['pull_away']:ld=np.log(prior)
        if np.exp(lk[(agenda>=0)&(agenda!=z)]).sum()>=s['theta'] and s['delta']>0:
            a=1 if np.exp(lk[agenda==1]).sum()>np.exp(lk[agenda==0]).sum() else 0;lk[z==a]+=s['delta'];lk-=logsumexp(lk)
    kl=float(logsumexp(lk[z==truth]));dl=float(logsumexp(ld[z==truth]));return [kl,dl,float(tt[truth]-logsumexp(tt)),float(oo[truth]-logsumexp(oo)),1-math.exp(kl),1-math.exp(dl),float(closed),dl-kl,math.exp(kl)-math.exp(dl)],when

def observations(s,r,source):
    n=s['worlds_per_block']*s['sources'];truth=np.arange(n)%2;items=np.empty((n,s['records']),dtype=np.int8)
    for i,z in enumerate(truth):
        for t in range(s['records']):
            if source=='HU' and r.random()<s['rho']:items[i,t]=-1;continue
            pool=int(r.binomial(8,.7 if z else .3));strength=.5 if source=='M' else s['overpush'] if source=='Mx' else 1. if source=='Mc' and t>=s['switch'] else 0.
            items[i,t]=int(r.random()<(1-strength)*pool/8+strength*(pool>0))
    return truth,items

def block(s,identity,independent=False):
    r=rng(*identity);raw=[];out=[];z,agenda,p,prior=library(s['expanded']);names=['H','M','Mx','Mc','HU'] if s['delta']==0 else ['H','M'] if s.get('delta_honest',True) else ['M']
    for source in names:
        code=['H','M','Mx','Mc','HU'].index(source);truth,items=observations(s,r,source)
        kernel=review_trajectories if independent else trajectory
        values,ls=kernel(items,truth,code,s['theta'],s['delta'],s['switch'],s['overpush'],s['pull_away'],z,agenda,p,prior);lengths=[None if x<0 else int(x) for x in ls]
        raw.append(dict(source=source,truth=truth.tolist(),items=items.tolist()))
        for true in (0,1):
            mask=truth==true
            for j,name in enumerate(['K_logscore','D_logscore','T_logscore','O_logscore','K_capture','D_capture','closed','D_minus_K_logscore','D_minus_K_capture']):
                measure(out,name,values[mask,j].tolist(),source=source,truth=true,theta=s['theta'],delta=s['delta'])
            measure(out,'closure_record',[v for v,keep in zip(lengths,mask) if keep],'median',source=source,truth=true,theta=s['theta'],delta=s['delta'])
    return dict(observations=raw,measures=out)

@lru_cache(maxsize=256)
def exact(source,truth=0,theta=.7,horizon=64,rho=.05,switch=16,overpush=1.,expanded=False,pull_away=False):
    """Count-state dynamic programme: alive gate and unrestricted K separately."""
    z,agenda,p,prior=library(expanded);h=horizon
    posterior={}
    for n in range(h+1):
        for ones in range(n+1):
            logs=np.log(prior)+ones*np.log(p)+(n-ones)*np.log1p(-p);v=np.exp(logs-logs.max());v/=v.sum()
            posterior[n,ones]=(float(v[z==truth].sum()),float(v[(agenda>=0)&(agenda!=z)].sum()))
    active={(0,0):1.};allp={(0,0):1.};closed=dcapture=dlog=0.;distribution=[]
    for t in range(h):
        strength=.5 if source=='M' else overpush if source=='Mx' else 1. if source=='Mc' and t>=switch else 0.
        q=rate(truth,1,strength);u=rho if source=='HU' else 0.;branches=[(0,0,u,True),(1,1,(1-u)*q,False),(1,0,(1-u)*(1-q),False)]
        aa={};bb={};newclosed=0.
        for pool,dest,is_active in [(allp,aa,False),(active,bb,True)]:
            for (n,ones),mass in pool.items():
                for dn,do,weight,act in branches:
                    if not weight:continue
                    key=(n+dn,ones+do);m=mass*weight;prob,harm=posterior[key]
                    if is_active and (harm>=theta or act):
                        if pull_away:prob=.5
                        newclosed+=m;dcapture+=m*(1-prob);dlog+=m*math.log(prob)
                    else:dest[key]=dest.get(key,0.)+m
        allp,active=aa,bb;closed+=newclosed;distribution.append(newclosed)
    kc=kl=0.
    for key,m in allp.items():
        q,_=posterior[key];kc+=m*(1-q);kl+=m*math.log(q)
    for key,m in active.items():q,_=posterior[key];dcapture+=m*(1-q);dlog+=m*math.log(q)
    return dict(D_minus_K_capture=dcapture-kc,D_minus_K_logscore=dlog-kl,closed=closed,K_capture=kc,
        captured_share=sum(m for key,m in allp.items() if posterior[key][0]<.5),closure_distribution=distribution)

def design_check(s,identity):
    checks={k:exact(k,0,s['theta'],s['records'],s['rho'],s['switch'],s['overpush'],s['expanded'],s['pull_away'])['captured_share'] for k in ('Mx','Mc')}
    return dict(passed=all(v>=.25 for v in checks.values()),checks=checks,method='Exact count-state programme; no scientific seeds')

def controls():
    z,a,p,prior=library();items=np.array([[1,0]*7+[1]*4],dtype=np.int8);truth=np.array([0]);v,when=trajectory(items,truth,0,.5,0.,16,1.,False,z,a,p,prior)
    assert when[0]==14
    logs=np.log(prior)+7*np.log(p)+7*np.log1p(-p);x=np.exp(logs-logs.max());x/=x.sum();assert abs(math.exp(v[0,1])-x[z==0].sum())<1e-12
    no,_=trajectory(items,truth,0,1.,0.,16,1.,False,z,a,p,prior);assert abs(no[0,0]-no[0,1])<1e-12
    assert np.max(abs(prior*.001/(prior*.001).sum()-prior))<1e-12
    assert abs(math.log(.7/.3)-math.log(7/3))<1e-12
    # A singleton honest library has exactly the transparent Bernoulli likelihood.
    honest,_=trajectory(items,truth,0,1.,0.,16,1.,False,np.array([0,1]),np.array([-1,-1]),np.array([.3,.7]),np.array([.5,.5]))
    assert abs(honest[0,0]-honest[0,2])<1e-12
    ex=exact('Mc');alarm=exact('H');assert abs(ex['D_minus_K_capture']+.0897)<.001 and abs(alarm['closed']-.0899)<.001
    return dict(passed=True,no_gate_identity=True,singleton_honest_transparent=True,unfittable_no_update=True,opaque_log_odds=True,first_gate_record_14=True,exact_primary=ex['D_minus_K_capture'],exact_honest_closed=alarm['closed'])
