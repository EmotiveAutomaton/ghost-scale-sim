"""Fixed seeds, world/block uncertainty and censored descriptive summaries."""
import hashlib
import json
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t, binom, norm
from ..v27.common import read, write, digest


def rng(*parts):
    data=json.dumps(['v28',*parts],sort_keys=True,separators=(',',':')).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(data).digest()[:8],'little'))


def softmax(x,axis=-1):
    return np.exp(x-logsumexp(x,axis=axis,keepdims=True))


def mean(x):
    a=np.asarray(x,dtype=float)
    if a.ndim!=1 or len(a)<2 or not np.isfinite(a).all():
        raise ValueError('mean requires finite independent units')
    value=float(np.mean(a));se=float(np.std(a,ddof=1)/np.sqrt(len(a)))
    width=float(t.ppf(.975,len(a)-1))*se
    return dict(estimate=value,interval95=[value-width,value+width],standard_error=se,units=len(a))


def encoded(x):
    return 'infinite' if np.isinf(x) else float(x)


def median(x):
    a=np.sort(np.asarray(x,dtype=float));n=len(a)
    if not n or np.isnan(a).any():raise ValueError('invalid censored sample')
    # Order-statistic CI for a population median, retaining infinite endpoints.
    lower=max(0,int(binom.ppf(.025,n,.5))-1)
    upper=min(n-1,int(binom.ppf(.975,n,.5)))
    value=float(np.median(a))
    return dict(estimate=encoded(value),interval95=[encoded(a[lower]),encoded(a[upper])],
        units=n,censored=int(np.isinf(a).sum()),censored_share=float(np.isinf(a).mean()),
        finite_horizon_aliased=bool(np.isinf(a).mean()>.5),
        interval_method='binomial order-statistic interval; infinite values retained')


def proportion(x):
    a=np.asarray(x,dtype=bool);n=len(a);p=float(a.mean());z=float(norm.ppf(.975))
    den=1+z*z/n;center=(p+z*z/(2*n))/den
    width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(estimate=p,interval95=[center-width,center+width],units=n,
        interval_method='Wilson interval over independent worlds')


def row(setting,measure,stat,scale_only=True,**labels):
    return dict(setting=setting,measure=measure,scale_only=scale_only,**labels,**stat)


def auc_counts(scores,labels):
    s=np.asarray(scores).ravel();y=np.asarray(labels,dtype=bool).ravel()
    order=np.argsort(s,kind='stable');s=s[order];y=y[order]
    _,first,counts=np.unique(s,return_index=True,return_counts=True)
    pos=np.add.reduceat(y.astype(float),first);neg=counts-pos
    below=np.cumsum(neg)-neg;above=y.sum()-(np.cumsum(pos))
    u=float(np.sum(pos*(below+.5*neg)))
    return u,int(y.sum()),int((~y).sum()),order,first,counts,below+.5*neg,above+.5*pos


def auc_blocks(scores,labels):
    """Pooled midrank AUC, uncertainty by deleting whole library blocks."""
    scores=np.asarray(scores);labels=np.asarray(labels,dtype=bool)
    blocks=len(scores)
    u,p,n,order,first,counts,pc,nc=auc_counts(scores,labels)
    if not p or not n:raise ValueError('AUC needs both source classes')
    block=np.broadcast_to(np.arange(blocks).reshape((-1,)+(1,)*(scores.ndim-1)),scores.shape).ravel()[order]
    truth=labels.ravel()[order]
    positive=np.bincount(block,weights=np.repeat(pc,counts)*truth,minlength=blocks)
    negative=np.bincount(block,weights=np.repeat(nc,counts)*(~truth),minlength=blocks)
    leave=[]
    for b in range(blocks):
        inside,bp,bn,*_=auc_counts(scores[b],labels[b])
        leave.append((u-positive[b]-negative[b]+inside)/((p-bp)*(n-bn)))
    value=u/(p*n);leave=np.asarray(leave)
    se=float(np.sqrt((blocks-1)/blocks*np.sum((leave-leave.mean())**2)))
    width=float(t.ppf(.975,blocks-1))*se
    return dict(estimate=value,interval95=[max(0.,value-width),min(1.,value+width)],
        standard_error=se,units=blocks,interval_method='delete-one-library-block jackknife of pooled midrank AUC',
        positive_tokens=p,negative_tokens=n)


def separation(last_below,final_odds):
    # Literal packet statistic: zero-based last-below step plus one. Zero means
    # the odds were already above the bar at the first recorded step, not that
    # zero observations identified the class. Keep this distinct from a hitting time.
    return np.where(np.asarray(final_odds)<1.,np.inf,np.asarray(last_below)+1.)
