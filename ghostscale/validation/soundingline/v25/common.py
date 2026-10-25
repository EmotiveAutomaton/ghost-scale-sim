"""Two numerical routes and deterministic identities; no imported campaign entry point."""
import hashlib,math
import numpy as np
from scipy.special import logsumexp
from ..v23.common import canonical,read,write,digest

def rng(*parts):return np.random.default_rng(int.from_bytes(hashlib.sha256(canonical(['v25',*parts])).digest()[:8],'little'))

def softmax(x,independent=False):
    x=np.asarray(x,float)
    if not independent:return np.exp(x-logsumexp(x,axis=-1,keepdims=True))
    flat=x.reshape(-1,x.shape[-1]);result=[]
    for row in flat:
        largest=max(row);values=[math.exp(float(v-largest)) for v in row];total=math.fsum(values)
        if total<=0:raise ValueError('empty support')
        result.append([v/total for v in values])
    return np.array(result).reshape(x.shape)

def posterior(log_likelihood,prior=None,independent=False):
    x=np.asarray(log_likelihood,float)
    if prior is not None:
        with np.errstate(divide='ignore'):x=x+np.log(prior)
    return softmax(x,independent)

def supported_posterior(log_likelihood,independent=False):
    return posterior(log_likelihood,independent=independent) if np.isfinite(log_likelihood).any() else None

def scores(p,index,prior=None):
    p=np.asarray(p,float);initial=1/len(p) if prior is None else float(prior[index]);truth=float(p[index])
    # Zero support is retained explicitly; no score flooring.
    return dict(truth_probability=truth,log_gain=None if truth==0 or initial==0 else math.log(truth/initial),
        half_brier=float((np.sum(p*p)-2*truth+1)/2),
        accuracy=float(abs(truth-p.max())<=1e-12)/int(np.sum(abs(p-p.max())<=1e-12)))

def information(p,laws):
    p=np.asarray(p);laws=np.asarray(laws);marginal=p@laws
    valid=laws>0
    terms=np.zeros_like(laws);terms[valid]=laws[valid]*np.log((laws/np.maximum(marginal,1e-300))[valid])
    return float(p@terms.sum(-1))

def simplex(denominator=4,interior=False):
    low=1 if interior else 0
    return np.array([(a,b,denominator-a-b) for a in range(low,denominator+1) for b in range(low,denominator+1-a) if denominator-a-b>=low],float)/denominator

def close(a,b,path='root',tol=1e-10):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('keys differ '+path)
        for k in a:close(a[k],b[k],path+'.'+k,tol)
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('length differs '+path)
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+f'[{i}]',tol)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not math.isclose(a,b,abs_tol=tol,rel_tol=tol):raise ValueError(f'numerical disagreement {path}: {a} versus {b}')
    elif a!=b:raise ValueError('value differs '+path)

def record(metrics,**labels):return dict(labels=labels,metrics=metrics)
