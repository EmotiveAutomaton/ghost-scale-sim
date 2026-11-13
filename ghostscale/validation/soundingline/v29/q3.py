"""Queue reading from exact stationary states; built before supplied pre-flight."""
from functools import lru_cache
from itertools import product
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .common import rng, logistic, class_evidence, sustained, mean_blocks, censored_blocks, write, digest, audit_controls

PROFILES={'even':([.15,.15,.15],[1.,1.,1.]),
          'skewed':([.2,.15,.1],[1.,1.5,3.]),
          'one-big':([.05,.05,.3],[1.,1.,3.])}
STATES=np.array(list(product(range(6),repeat=3)),dtype=int)
READINGS=('COUNT','TOTAL','LOUDEST')


def index(d):return d[...,0]*36+d[...,1]*6+d[...,2]


@lru_cache(None)
def stationary(profile):
    arrivals,a=map(np.asarray,PROFILES[profile]);p=np.zeros((216,216))
    for i,d in enumerate(STATES):
        winner=int(np.argmax(a*d))
        served=d.copy();served[winner]=max(0,served[winner]-1)
        for bits in product((0,1),repeat=3):
            inc=np.array(bits);chance=float(np.prod(np.where(inc,arrivals,1-arrivals)))
            p[i,index(np.minimum(5,d+inc))]+=.1*chance
            p[i,index(np.minimum(5,served+inc))]+=.9*chance
    mass=np.full(216,1/216.)
    for iteration in range(20000):
        new=mass@p
        if abs(new-mass).max()<1e-15:break
        mass=new
    else:raise ArithmeticError('stationary distribution not converged')
    values=np.column_stack([(STATES>0).sum(1),STATES@a,(STATES*a).max(1)]).astype(float)
    center=mass@values;sd=np.sqrt(mass@((values-center)**2))
    scaled=values/sd
    quantiles=np.empty((3,5));median=np.empty(3)
    for j in range(3):
        order=np.argsort(scaled[:,j],kind='stable');cdf=np.cumsum(mass[order])
        quantiles[j]=[scaled[order[min(215,np.searchsorted(cdf,q))],j] for q in np.linspace(.3,.7,5)]
        median[j]=scaled[order[min(215,np.searchsorted(cdf,.5))],j]
    return mass,scaled,quantiles,median,float(abs(mass@p-mass).max())


def settings():
    return [dict(part='Q3',truth=truth,profile=profile,rate=rate,steps=2048,worlds=200,blocks=8)
            for truth in range(3) for profile in PROFILES for rate in (.1,.3)]


def candidates(profile):
    _,scaled,q,median,_=stationary(profile)
    # Quantile duplicates remain members, as the declared five-point prior requires.
    params=np.array([(j,r,theta) for j in range(3) for r in (.05,.1,.2,.3) for theta in q[j]])
    return params,scaled,median


def generate(s,lane='study',worlds=None):
    n=worlds or s['worlds'];steps=s['steps'];r=rng('Q3',lane,s)
    mass,scaled,_,median,_=stationary(s['profile']);arrivals,a=map(np.asarray,PROFILES[s['profile']])
    d=STATES[r.choice(216,n,p=mass)].copy();y=np.zeros(n,int)
    states=np.empty((n,steps),np.int16);rides=np.zeros((n,steps),np.int8);eligible=np.zeros_like(rides)
    actions=np.zeros((n,steps),np.int8)
    for t in range(steps):
        i=index(d);states[:,t]=i;y=np.maximum(0,y-1);eligible[:,t]=y==0
        p=s['rate']*logistic(s.get('slope',1.5)*(median[s['truth']]-scaled[i,s['truth']]),0.)
        ride=(y==0)&(r.random(n)<p);rides[:,t]=ride;y+=2*ride
        pick=(a*d).argmax(1);serve=(r.random(n)<.9)&(d.sum(1)>0);actions[:,t]=np.where(serve,pick+1,0)
        d[np.arange(n),pick]-=serve
        d=np.minimum(5,d+(r.random((n,3))<arrivals))
    return dict(states=states,rides=rides,eligible=eligible,actions=actions)


def infer(raw,s):
    params,scaled,_=candidates(s['profile']);classes=params[:,0].astype(int);n,steps=raw['states'].shape
    table=params[None,:,1]*logistic(s.get('slope',1.5)*(params[None,:,2]-scaled[:,classes]),0.)
    ll=np.zeros((n,len(params)));last=np.full((n,3),-1);visit=np.zeros((n,216),int)
    for t in range(steps):
        p=table[raw['states'][:,t]];act=raw['rides'][:,t,None]
        ll+=raw['eligible'][:,t,None]*np.where(act,np.log(p),np.log1p(-p))
        evidence=class_evidence(ll,classes);odds=evidence[:,s['truth'],None]-evidence
        last=np.where(odds<3,t,last)
        visit[np.arange(n),raw['states'][:,t]]+=raw['eligible'][:,t]
    # Truth-vs-rival probabilities at each truth's own median, matched rate.
    _,_,median=candidates(s['profile'])
    probabilities=s['rate']*logistic(s.get('slope',1.5)*(median[None,:]-scaled),0.)
    p=probabilities[:,s['truth'],None];q=probabilities
    kl=p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
    return dict(pair_separation=last+2,final_class_odds=odds,final_likelihood=ll,visits=visit,
                divergence=kl,informative_share=((visit@(kl.max(1)>.01))/np.maximum(1,visit.sum(1))))


def controls(destination,pulse=lambda **kw:None):
    checks=[]
    for profile in PROFILES:
        mass,scaled,q,median,residual=stationary(profile)
        checks.append(dict(name=profile+' stationary balance',passed=residual<1e-12,maximum_residual=residual))
        sd=np.sqrt(mass@((scaled-mass@scaled)**2))
        checks.append(dict(name=profile+' unit standard deviations',passed=bool(np.allclose(sd,1,atol=1e-12))))
    checks.append(dict(name='state index bijection',passed=bool(np.array_equal(index(STATES),np.arange(216)))))
    p=np.array([.01,.1,.3]);kl=p*np.log(p/p)+(1-p)*np.log((1-p)/(1-p))
    checks.append(dict(name='equal readings zero divergence',passed=bool(np.array_equal(kl,np.zeros(3)))))
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result


def selection_probabilities(d,a,costs,rule,lam=1.):
    """Q3.b: the action probability is identical; only the selected need changes."""
    score=a*d if rule=='loudest' else a*(d-.5)-lam*costs
    score=np.where(d>0,score,-np.inf)
    selected=score.argmax(axis=-1)
    return selected,(np.sum(d,axis=-1)>0)*.9
