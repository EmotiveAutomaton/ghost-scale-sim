"""L texture: two independent show channels sharing event arrivals and persistence."""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from numba import njit
from . import l
from .common import rng,sustained,audit_controls,write

def settings():
    return [dict(part="two_channels",target=target,show=list(show),detector=det,truth=truth,
        mixed=False,ratio=ratio,steps=1500,worlds=200,blocks=8,resolution=2048)
        for target in (.16,.08,.32) for show in ((1.,0.),(.6,.05))
        for det in ("none","middle","exact") for truth in range(3) for ratio in (.5,2.)]

def generate(s,lane="study",pulse=lambda **kw:None):
    ss=dict(s,part="filter");raw=l.generate(ss,"L-pair-"+lane,pulse)
    rr=rng("L-pair-second-show",lane,s);n=s["worlds"];T=s["steps"]
    parameters=raw["parameters"];activation=np.zeros(n);shows=np.zeros((n,T),bool)
    sh,sf=s["show"]
    for t in range(T):
        activation=parameters[:,1]*raw["event"][:,t]*raw["size"][:,t]+parameters[:,2]*activation
        shows[:,t]=rr.random(n)<-np.expm1(-s["ratio"]*activation)
    second=rr.random((n,T))<np.where(shows,sh,sf)
    raw["observed_pair"]=np.stack((raw["observed_show"],second),axis=-1)
    raw["second_show"]=shows
    return raw

@njit
def forward(obs,events,has_detector,sh,sf,eh,ef,ratio,nodes,lo,hi,fraction,cause,reference):
    W,T,_=obs.shape;M=len(nodes);ll=np.zeros((W,T));ep=np.zeros((W,T))
    for w in range(W):
        mass=np.zeros(M);mass[0]=1.;total=0.
        for t in range(T):
            nxt=np.zeros(M);event_mass=np.zeros(M)
            if reference:
                # Independent source-state scatter with emission applied before summing.
                for j in range(M):
                    if mass[j]==0:continue
                    for c in range(4):
                        d=eh if c else ef
                        d=(d if events[w,t] else 1-d) if has_detector else 1.
                        for side in range(2):
                            dest=hi[c,j] if side else lo[c,j]
                            spread=fraction[c,j] if side else 1-fraction[c,j]
                            emission=1.
                            for k in range(2):
                                p=sf+(sh-sf)*(-np.expm1(-nodes[dest]*(ratio if k else 1.)))
                                emission*=p if obs[w,t,k] else 1-p
                            value=mass[j]*cause[c]*d*spread*emission
                            nxt[dest]+=value
                            if c:event_mass[dest]+=value
            else:
                for c in range(4):
                    d=eh if c else ef
                    d=(d if events[w,t] else 1-d) if has_detector else 1.
                    for j in range(M):
                        value=mass[j]*cause[c]*d
                        if value==0:continue
                        a=lo[c,j];b=hi[c,j];f=fraction[c,j]
                        nxt[a]+=value*(1-f);nxt[b]+=value*f
                        if c:
                            event_mass[a]+=value*(1-f);event_mass[b]+=value*f
                for j in range(M):
                    emission=1.
                    for k in range(2):
                        p=sf+(sh-sf)*(-np.expm1(-nodes[j]*(ratio if k else 1.)))
                        emission*=p if obs[w,t,k] else 1-p
                    nxt[j]*=emission;event_mass[j]*=emission
            z=nxt.sum()
            if z<=0:raise ArithmeticError("two-channel zero likelihood")
            mass=nxt/z;total+=np.log(z);ll[w,t]=total;ep[w,t]=event_mass.sum()/z
    return ll,ep

def exact_observed(raw,parameter,s,reference=False):
    p,g,rho=parameter;obs=raw["observed_pair"];W,T,_=obs.shape
    sh,sf=s["show"];out=np.zeros((W,T));a=np.zeros(W);total=np.zeros(W)
    if reference:
        import math
        for w in range(W):
            activation=0.;value=0.
            for t in range(T):
                event=raw["observed_event"][w,t]
                activation=rho*activation+g*raw["size"][w,t]*event
                value+=math.log(p/3 if event else 1-p)
                for k,scale in enumerate((1.,s["ratio"])):
                    chance=sf+(sh-sf)*(-math.expm1(-scale*activation))
                    value+=math.log(max(1e-300,chance if obs[w,t,k] else 1-chance))
                out[w,t]=value
    else:
        for t in range(T):
            event=raw["observed_event"][:,t]
            a=rho*a+g*raw["size"][:,t]*event
            total+=np.where(event,np.log(p/3),np.log1p(-p))
            for k,scale in enumerate((1.,s["ratio"])):
                chance=sf+(sh-sf)*(-np.expm1(-scale*a))
                total+=np.log(np.maximum(1e-300,np.where(obs[:,t,k],chance,1-chance)))
            out[:,t]=total
    return out,raw["observed_event"].astype(float)

def filter_one(raw,parameter,s,reference=False,resolution=None):
    if s["detector"]=="exact":return exact_observed(raw,parameter,s,reference)
    nodes=l.grid(parameter,resolution or s["resolution"])
    lo,hi,fraction,cause=l.transitions(parameter,nodes);eh,ef=l.detector(s);sh,sf=s["show"]
    return forward(raw["observed_pair"],raw["observed_event"],s["detector"]!="none",
        sh,sf,eh,ef,s["ratio"],nodes,lo,hi,fraction,cause,reference)

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    LL=[];EP=[]
    for i,parameter in enumerate(l.makers(s["target"])):
        a,b=filter_one(raw,parameter,s,reference);LL.append(a);EP.append(b);pulse(candidate=i+1)
    ll=np.stack(LL,axis=-1);weights=np.exp(ll-logsumexp(ll,axis=-1,keepdims=True))
    ep=np.sum(weights*np.stack(EP,axis=-1),axis=-1)
    odds=ll[...,s["truth"]]-np.max(np.delete(ll,s["truth"],axis=-1),axis=-1)
    selected=raw["observed_pair"].any(-1);count=selected.sum(1)
    brier=np.divide((((ep-raw["event"])**2)*selected).sum(1),count,
        out=np.zeros(len(count)),where=count>0)
    auc=np.array([l.auc(y[m],p[m]) for y,p,m in zip(raw["event"],ep,selected)])
    return dict(separation=sustained(odds),final_odds=odds[:,-1],brier=brier,
        _auc=auc,_show_count=count,_event_probability=ep,_ll=ll)

summarize=l.summarize

def enumerated(raw,parameter,s):
    p,g,rho=parameter;eh,ef=l.detector(s);sh,sf=s["show"]
    obs=raw["observed_pair"];W,T,_=obs.shape;ll=np.zeros((W,T));ep=np.zeros((W,T))
    for w in range(W):
        states=np.array([0.]);mass=np.array([1.]);total=0.
        for t in range(T):
            states=(rho*states[None,:]+g*np.array([0.,.5,1.,1.5])[:,None]).ravel()
            mass=(np.array([1-p,p/3,p/3,p/3])[:,None]*mass).ravel()
            causes=np.repeat(np.arange(4),len(states)//4)>0
            if s["detector"]!="none":
                d=np.where(causes,eh,ef);mass*=d if raw["observed_event"][w,t] else 1-d
            for k,scale in enumerate((1.,s["ratio"])):
                chance=sf+(sh-sf)*(-np.expm1(-scale*states))
                mass*=chance if obs[w,t,k] else 1-chance
            z=mass.sum();ep[w,t]=mass[causes].sum()/z;mass/=z
            total+=np.log(z);ll[w,t]=total
    return ll,ep

def controls(dest=None,pulse=lambda **kw:None):
    s=dict(settings()[0],worlds=8,blocks=8,steps=8,detector="middle",show=[.6,.05])
    raw=generate(s,"known-answer");parameter=l.makers(.16)[0]
    a,b=filter_one(raw,parameter,s);c,d=filter_one(raw,parameter,s,True)
    x,y=enumerated(raw,parameter,s)
    checks=[dict(name="independent two-channel forward arithmetic",
        passed=bool(np.max(abs(a-c))<1e-10 and np.max(abs(b-d))<1e-10)),
        dict(name="exact short-path enumeration",passed=bool(np.max(abs(a-x))<.01 and np.max(abs(b-y))<.005),
             max_log_error=float(np.max(abs(a-x))),max_event_error=float(np.max(abs(b-y))))]
    chance=dict(s,detector="none");altered=dict(raw,observed_event=~raw["observed_event"])
    a,b=filter_one(raw,parameter,chance);c,d=filter_one(altered,parameter,chance)
    checks.append(dict(name="unseen event detector cannot enter pair reader",passed=bool(np.array_equal(a,c) and np.array_equal(b,d))))
    # With a chance show reader, all observed two-channel outcomes have probability 1/4.
    null=dict(s,show=[.5,.5],detector="none");a,b=filter_one(raw,parameter,null)
    checks.append(dict(name="two chance channels erase activation evidence",
        passed=bool(np.allclose(a,np.arange(1,9)*np.log(.25),atol=1e-10))))
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
