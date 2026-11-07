"""Frozen block inference and explicit censoring for the theory-day miniatures."""
import math
import numpy as np
from numba import njit
from scipy.special import logsumexp
from scipy.stats import t as student_t
from .common import rng, softmax

@njit
def scalar_posteriors(logs, prior):
    out=np.empty_like(logs)
    for i in range(len(logs)):
        top=-np.inf
        for j in range(len(prior)):
            if prior[j]>0: top=max(top,logs[i,j]+math.log(prior[j]))
        total=0.
        for j in range(len(prior)):
            out[i,j]=math.exp(logs[i,j]+math.log(prior[j])-top) if prior[j]>0 else 0.
            total+=out[i,j]
        for j in range(len(prior)): out[i,j]/=total
    return out

def posteriors(logs,prior,independent=False):
    a=np.asarray(logs,float);p=np.asarray(prior,float)
    return scalar_posteriors(a,p) if independent else np.exp(a+np.log(p)-logsumexp(a+np.log(p),axis=1,keepdims=True))

def class_prior(classes):
    c=np.asarray(classes);types=np.unique(c)
    return np.array([1/len(types)/np.sum(c==x) for x in c])

def marginals(p, ids, n):return np.stack([p[:,np.asarray(ids)==j].sum(1) for j in range(n)],axis=1)

def margin(logs,prior,a,b):
    z=np.asarray(logs)+np.log(prior)
    return logsumexp(z[:,a],axis=1)-logsumexp(z[:,b],axis=1)

def sustained(path,bar=1.):
    x=np.asarray(path);bad=np.flatnonzero(x<bar)
    return (int(bad[-1])+2 if len(bad) else 1) if len(x) and x[-1]>=bar else None

def scores(p,truth):
    v=float(p[truth]);ties=np.isclose(p,p.max(),atol=1e-12,rtol=0)
    return dict(posterior=v,half_brier=float((np.sum(p*p)-2*v+1)/2),accuracy=float(ties[truth])/int(ties.sum()))

def measure(out,name,values,statistic='mean',**labels):
    out.append(dict(metric=name,labels=labels,statistic=statistic,values=values))

def choices(utilities,precision=3.,lapse=.02,mask=None,independent=False):
    z=np.asarray(utilities,float)*precision
    if mask is not None:z=np.where(mask,z,-np.inf)
    return (1-lapse)*softmax(z,independent)+lapse/z.shape[-1]

def draw_menu(r,n=4,share=.25):
    menu=r.integers(0,3,(n,3))/2
    for i in range(n):
        if r.random()<share:
            menu[i]=0.;menu[i,int(r.integers(3))]=1.
    return menu

def safe_number(x):return float(x) if np.isfinite(x) else None

def statistic(values,kind):
    if kind=='mean':return float(np.mean(values)) if len(values) else math.nan
    if kind=='ratio_of_sums':
        a=np.asarray(values,float).sum(0)
        return float(a[0]/a[1]) if a[1] else math.nan
    if kind=='median':return float(np.median([math.inf if v is None else v for v in values])) if values else math.nan
    if kind in ('median_difference','median_ratio'):
        a=statistic([v[0] for v in values],'median');b=statistic([v[1] for v in values],'median')
        if math.isinf(a) and math.isinf(b):return math.nan
        return a-b if kind=='median_difference' else a/b
    raise ValueError(kind)

def summarize(block_values,kind,identity,independent=False):
    vals=[x for block in block_values for x in block]
    center=statistic(vals,kind);n=len(block_values)
    if kind=='mean':
        bm=[math.fsum(map(float,b))/len(b) for b in block_values if b]
        center=math.fsum(bm)/len(bm) if bm else math.nan
        se=float(np.std(bm,ddof=1)/math.sqrt(len(bm))) if len(bm)>1 else math.nan
        half=student_t.ppf(.975,len(bm)-1)*se if len(bm)>1 else math.nan
        low,high=center-half,center+half
        method='Pointwise Student t interval across independent whole-world blocks'
    elif n<2:
        low=high=se=math.nan;method='Excluded single-block probe; no uncertainty interval'
    else:
        rr=rng('day-block-bootstrap',identity);draws=[]
        for picks in rr.integers(0,n,(2000,n)):
            draws.append(statistic([v for i in picks for v in block_values[i]],kind))
        # Inverted empirical CDF preserves infinite censor bounds, unlike linear interpolation.
        valid=sorted(x for x in draws if not math.isnan(x))
        if len(valid)!=len(draws):low=high=math.nan
        else:low,high=(valid[max(0,math.ceil(q*len(valid))-1)] for q in (.025,.975))
        se=math.nan;method='Pointwise percentile bootstrap, 2000 resamples of entire blocks; unresolved crossings are right-censored'
    return dict(estimate=safe_number(center),interval95=[safe_number(low),safe_number(high)],
        estimate_censored=math.isinf(center),lower_censored=math.isinf(low),upper_censored=math.isinf(high),
        unavailable=math.isnan(center),interval_unavailable=math.isnan(low) or math.isnan(high),
        standard_error=safe_number(se),blocks=n,observations=len(vals),method=method)

def bar_verdict(row,bar,direction):
    lo,hi=row['interval95']
    if row.get('interval_unavailable'):return 'unresolved at this sample size'
    if row.get('lower_censored'):lo=math.inf
    if row.get('upper_censored'):hi=math.inf
    if lo is None or hi is None:return 'unresolved at this sample size'
    if direction=='at_most':return 'met' if hi<=bar else 'not met' if lo>bar else 'unresolved at this sample size'
    if direction=='at_least':return 'met' if lo>=bar else 'not met' if hi<bar else 'unresolved at this sample size'
    if direction=='above':return 'met' if lo>bar else 'not met' if hi<=bar else 'unresolved at this sample size'
    if direction=='below':return 'met' if hi<bar else 'not met' if lo>=bar else 'unresolved at this sample size'
    raise ValueError(direction)
