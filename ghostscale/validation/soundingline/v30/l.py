"""L activation filtering; text-first exact controls and explicit grid approximation."""
import itertools
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import rankdata
from numba import njit
from .common import rng,mean_blocks,censored_blocks,sustained,audit_controls,write

def stationary_show(p,g,rho):
    product=1.
    for k in range(100000):
        amount=g*rho**k
        product*=1-p+p*np.exp(-amount*np.array([.5,1.,1.5])).mean()
        if amount<1e-14:break
    else:raise ArithmeticError("stationary product did not converge")
    return 1-product

@lru_cache(None)
def makers(target, mixed=False):
    p=brentq(lambda v:stationary_show(v,1,.3)-target,1e-9,.999)
    g=brentq(lambda v:stationary_show(.1,v,.3)-target,1e-9,1e5)
    rho=brentq(lambda v:stationary_show(.1,1,v)-target,0.,.999)
    result=[(p,1.,.3),(.1,g,.3),(.1,1.,rho)]
    if mixed:result += [(.1,(1+g)/2,(.3+rho)/2),((.1+p)/2,1.,(.3+rho)/2)]
    return tuple(result)

def settings():
    result=[]
    for target in (.16,.08,.32):
        for show in ((1.,0.),(.6,.05),(.5,.02)):
            for detector in ("none","middle","good","exact"):
                for truth in range(5 if target==.16 else 3):
                    result.append(dict(part="filter",target=target,show=list(show),detector=detector,
                        truth=truth,mixed=target==.16,steps=1500,worlds=200,blocks=8,resolution=512))
    result.append(dict(part="correlation",steps=3000,worlds=2000,blocks=8,detector="exact"))
    return result

def detector(s):
    return {"none":(.5,.5),"middle":(.7,.05),"good":(.9,.02),"exact":(1.,0.)}[s["detector"]]

def generate(s,lane="study",pulse=lambda **kw:None):
    r=rng("L",lane,s);n=s["worlds"];T=s["steps"]
    if s["part"]=="correlation":
        parameters=np.column_stack((r.uniform(.05,.25,n),r.uniform(.5,3,n),r.uniform(.1,.7,n)))
        sh,sf=1.,0.
    else:
        parameters=np.broadcast_to(makers(s["target"],s.get("mixed",False))[s["truth"]],(n,3)).copy()
        sh,sf=s["show"]
    eh,ef=detector(s)
    event=r.random((n,T))<parameters[:,0,None]
    size=r.choice(np.array([.5,1.,1.5]),size=(n,T))
    activation=np.zeros(n);shows=np.zeros((n,T),bool)
    for t in range(T):
        activation=parameters[:,1]*event[:,t]*size[:,t]+parameters[:,2]*activation
        shows[:,t]=r.random(n)<-np.expm1(-activation)
    observed_show=r.random((n,T))<np.where(shows,sh,sf)
    observed_event=r.random((n,T))<np.where(event,eh,ef)
    return dict(event=event,size=size,show=shows,observed_show=observed_show,
                observed_event=observed_event,parameters=parameters)

def grid(parameter,resolution):
    p,g,rho=parameter
    maximum=1.5*g/(1-rho)
    # Quadratic spacing resolves low activations; endpoints include full support.
    return maximum*np.linspace(0.,1.,resolution+1)**2

def transitions(parameter,nodes):
    p,g,rho=parameter
    targets=rho*nodes[None,:]+g*np.array([0.,.5,1.,1.5])[:,None]
    hi=np.minimum(np.searchsorted(nodes,targets,side="right"),len(nodes)-1)
    lo=np.maximum(hi-1,0)
    fraction=np.divide(targets-nodes[lo],nodes[hi]-nodes[lo],
                       out=np.zeros_like(targets),where=nodes[hi]!=nodes[lo])
    return lo,hi,np.clip(fraction,0,1),np.array([1-p,p/3,p/3,p/3])

@njit
def grid_forward(show,event,has_detector,sh,sf,eh,ef,nodes,lo,hi,fraction,cause):
    worlds,T=show.shape;M=len(nodes)
    ll=np.zeros((worlds,T));posterior_event=np.zeros((worlds,T))
    obs_prob=sf+(sh-sf)*(1-np.exp(-nodes))
    for world in range(worlds):
        current=np.zeros(M);current[0]=1.;total=0.
        for t in range(T):
            following=np.zeros(M);event_part=np.zeros(M)
            for c in range(4):
                d=(eh if c else ef) if event[world,t] else (1-eh if c else 1-ef)
                if not has_detector:d=1.
                for j in range(M):
                    mass=current[j]*cause[c]*d
                    if mass==0:continue
                    a=lo[c,j];b=hi[c,j];f=fraction[c,j]
                    following[a]+=mass*(1-f);following[b]+=mass*f
                    if c:
                        event_part[a]+=mass*(1-f);event_part[b]+=mass*f
            z=0.;ze=0.
            for j in range(M):
                likelihood=obs_prob[j] if show[world,t] else 1-obs_prob[j]
                following[j]*=likelihood;z+=following[j];ze+=event_part[j]*likelihood
            if z<=0:raise ValueError("zero likelihood")
            current=following/z;total+=np.log(z);ll[world,t]=total
            posterior_event[world,t]=ze/z
    return ll,posterior_event

@njit
def reference_forward(show,event,has_detector,sh,sf,eh,ef,nodes,lo,hi,fraction,cause):
    # Independent flattened scatter, joint observation/cause multiplication first.
    W,T=show.shape;M=len(nodes);ll=np.zeros((W,T));pe=np.zeros((W,T))
    for w in range(W):
        mass=np.zeros(M);mass[0]=1.;logprob=0.
        for t in range(T):
            nxt=np.zeros(M);ev=0.
            for j in range(M):
                if mass[j]==0:continue
                for c in range(4):
                    z=eh if c>0 else ef
                    d=(z if event[w,t] else 1-z) if has_detector else 1.
                    for side in range(2):
                        dest=hi[c,j] if side else lo[c,j]
                        spread=fraction[c,j] if side else 1-fraction[c,j]
                        emission=sf+(sh-sf)*(-np.expm1(-nodes[dest]))
                        emission=emission if show[w,t] else 1-emission
                        value=mass[j]*cause[c]*d*spread*emission
                        nxt[dest]+=value
                        if c>0:ev+=value
            normalizer=np.sum(nxt)
            if normalizer<=0:raise ValueError("reference zero likelihood")
            mass=nxt/normalizer;logprob+=np.log(normalizer);ll[w,t]=logprob;pe[w,t]=ev/normalizer
    return ll,pe

@njit
def observed_reference(events,sizes,shows,p,g,rho,hit,false_alarm):
    W,T=events.shape;ll=np.zeros((W,T))
    for w in range(W):
        activation=0.;total=0.
        for t in range(T):
            activation=rho*activation
            if events[w,t]:activation+=g*sizes[w,t]
            show=false_alarm+(hit-false_alarm)*(-math.expm1(-activation))
            likelihood=show if shows[w,t] else 1-show
            total+=np.log(max(1e-300,likelihood))
            total+=np.log(p/3) if events[w,t] else np.log(1-p)
            ll[w,t]=total
    return ll

def exact_observed(raw,parameter,s,reference=False):
    p,g,rho=parameter;sh,sf=s["show"]
    if reference:
        return observed_reference(raw["observed_event"],raw["size"],raw["observed_show"],p,g,rho,sh,sf),raw["observed_event"].astype(float)
    shape=raw["observed_show"].shape
    a=np.zeros(shape[0]);ll=np.zeros(shape);total=np.zeros(shape[0])
    # Only observable events and their sizes enter. Hidden show labels never enter.
    for t in range(shape[1]):
        e=raw["observed_event"][:,t];a=g*raw["size"][:,t]*e+rho*a
        prob=sf+(sh-sf)*(-np.expm1(-a))
        value=np.where(raw["observed_show"][:,t],prob,1-prob)
        total+=np.log(np.maximum(value,1e-300))+np.where(e,np.log(p/3),np.log1p(-p))
        ll[:,t]=total
    return ll,raw["observed_event"].astype(float)

def filter_one(raw,parameter,s,reference=False,resolution=None):
    if s["detector"]=="exact":return exact_observed(raw,parameter,s,reference)
    nodes=grid(parameter,resolution or s["resolution"]);lo,hi,fraction,cause=transitions(parameter,nodes)
    sh,sf=s["show"];eh,ef=detector(s)
    return (reference_forward if reference else grid_forward)(raw["observed_show"],raw["observed_event"],
             s["detector"]!="none",sh,sf,eh,ef,nodes,lo,hi,fraction,cause)

def auc(y,score):
    y=np.asarray(y,bool)
    if y.sum()==0 or y.sum()==len(y):return np.nan
    return float((rankdata(score)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(len(y)-y.sum())))

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    if s["part"]=="correlation":
        shows=raw["observed_show"];events=raw["observed_event"]
        counts=shows.sum(1);fraction=np.divide((shows&~events).sum(1),counts,
            out=np.full(len(counts),np.nan),where=counts>0)
        return dict(_fraction=fraction,_parameters=raw["parameters"],show_share=shows.mean(1))
    roster=makers(s["target"],s.get("mixed",False));LL=[];EP=[]
    for k,parameter in enumerate(roster):
        a,b=filter_one(raw,parameter,s,reference);LL.append(a);EP.append(b);pulse(candidate=k+1)
    likelihood=np.stack(LL,axis=-1);weights=np.exp(likelihood-logsumexp(likelihood,axis=-1,keepdims=True))
    event_probability=np.sum(weights*np.stack(EP,axis=-1),axis=-1)
    odds=likelihood[...,s["truth"]]-np.max(np.delete(likelihood,s["truth"],axis=-1),axis=-1)
    # L.b conditions on observed shows and records event counts, including absent cases.
    selected=raw["observed_show"];count=selected.sum(1)
    brier=np.divide((((event_probability-raw["event"])**2)*selected).sum(1),count,
                     out=np.zeros(len(count)),where=count>0)
    AUC=np.array([auc(y[m],p[m]) for y,p,m in zip(raw["event"],event_probability,selected)])
    return dict(separation=sustained(odds),final_odds=odds[:,-1],brier=brier,
                _auc=AUC,_show_count=count,_event_probability=event_probability,_ll=likelihood)

def corr(x,y):
    return float(np.corrcoef(x,y)[0,1])

def partial(x,y,controls):
    design=np.column_stack((np.ones(len(x)),controls))
    return corr(x-design@np.linalg.lstsq(design,x,rcond=None)[0],
                y-design@np.linalg.lstsq(design,y,rcond=None)[0])

def summarize(s,m):
    rows=[]
    if s["part"]=="correlation":
        f=m["_fraction"];p,g,rho=m["_parameters"].T
        values=dict(rho=corr(f,rho),g=corr(f,g),p=corr(f,p),
                    rho_given_g=partial(f,rho,g),rho_given_g_p=partial(f,rho,np.column_stack((g,p))),
                    g_given_p_rho=partial(f,g,np.column_stack((p,rho))),
                    p_given_g_rho=partial(f,p,np.column_stack((g,rho))))
        draws=rng("L-c-whole-block-bootstrap").integers(s["blocks"],size=(2000,s["blocks"]))
        indices=np.arange(len(f)).reshape(s["blocks"],-1)
        samples={key:[] for key in values}
        for draw in draws:
            ix=indices[draw].ravel();ff=f[ix];pp=p[ix];gg=g[ix];rr=rho[ix]
            sample=dict(rho=corr(ff,rr),g=corr(ff,gg),p=corr(ff,pp),
                rho_given_g=partial(ff,rr,gg),rho_given_g_p=partial(ff,rr,np.column_stack((gg,pp))),
                g_given_p_rho=partial(ff,gg,np.column_stack((pp,rr))),
                p_given_g_rho=partial(ff,pp,np.column_stack((gg,rr))))
            for key,value in sample.items():samples[key].append(value)
        for key,value in values.items():rows.append(dict(measure="correlation_"+key,estimate=value,
            interval95=np.quantile(samples[key],[.025,.975]).tolist(),blocks=s["blocks"],worlds=s["worlds"]))
        return rows
    for name in ("separation","final_odds"):
        value=censored_blocks(m[name],s["steps"],s["blocks"]) if name=="separation" else mean_blocks(m[name],s["blocks"])
        rows.append(dict(measure=name,**value))
    from .common import conditional_blocks
    valid=np.isfinite(m["_auc"])
    rows.append(dict(measure="event_auc",**conditional_blocks(np.nan_to_num(m["_auc"]),valid,s["blocks"]),
                     undefined_worlds=int((~valid).sum())))
    rows.append(dict(measure="brier_per_world_with_observed_show",
                     **conditional_blocks(m["brier"],m["_show_count"]>0,s["blocks"]),
                     observed_shows=int(m["_show_count"].sum())))
    return rows

def controls(dest=None,pulse=lambda **kw:None):
    checks=[]
    for target in (.08,.16,.32):
        for p,g,rho in makers(target):
            value=stationary_show(p,g,rho)
            checks.append(dict(name="retuned stationary show rate",target=target,value=value,
                               passed=bool(abs(value-target)<1e-10)))
    s=dict(part="filter",target=.16,show=[.6,.05],detector="middle",truth=0,mixed=False,
           steps=12,worlds=8,blocks=8,resolution=256)
    raw=generate(s,"known-answer");a,b=filter_one(raw,makers(.16)[0],s)
    c,d=filter_one(raw,makers(.16)[0],s,True)
    checks.append(dict(name="independent grid reconstruction",passed=bool(
        np.max(abs(a-c))<1e-10 and np.max(abs(b-d))<1e-10)))
    # Chance event detector must cancel from the likelihood calculation.
    c0=dict(s,detector="none");r2={**raw,"observed_event":~raw["observed_event"]}
    x,y=filter_one(raw,makers(.16)[0],c0);xx,yy=filter_one(r2,makers(.16)[0],c0)
    checks.append(dict(name="unseen event bit cannot influence inference",passed=bool(np.array_equal(x,xx) and np.array_equal(y,yy))))
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
