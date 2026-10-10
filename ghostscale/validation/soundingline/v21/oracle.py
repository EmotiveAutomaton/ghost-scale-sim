"""Evidence-weighted law mixture, log-domain history, supplied-law diagnostic.

Crucial: mix joint evidence masses, never equally average normalized posteriors.
"""
import numpy as np
from . import world as W

def logsumexp(a,axis):
    m=np.max(a,axis=axis,keepdims=True)
    with np.errstate(invalid='ignore',divide='ignore'):
        out=m+np.log(np.exp(np.where(np.isfinite(m),a-m,-np.inf)).sum(axis=axis,keepdims=True))
    return np.squeeze(out,axis=axis)

def normalize_log(a):return a-logsumexp(a,axis=1)[:,None]

class Layout:
    def __init__(self,x,tier,history_tier=None,observed=False,history_observed=False,shift='native',history_shift=None):
        self.x=x;self.rows=W.skeleton(shift);self.mi=self.rows[:,0].astype(int);self.y=self.rows[:,10].astype(int)
        self.current,self.cix=self.indices(W.project(self.rows,tier,observed),x[:,:13])
        h=x[:,13:].reshape(len(x),-1,13)
        self.hist=[]
        if h.shape[1]:
            self.hinv,self.hidx=self.indices(W.project(W.skeleton(history_shift or shift),history_tier or tier,history_observed),h.reshape(-1,13))
            self.hidx=self.hidx.reshape(len(x),-1)
        self.ncurrent=int(self.current.max())+1
    @staticmethod
    def indices(full,requested):
        keys,inv=np.unique(full,axis=0,return_inverse=True);lookup={tuple(k):i for i,k in enumerate(keys)}
        return inv,np.array([lookup[tuple(k)] for k in requested])
    def log_joint(self,w):
        weights=W.weights(self.rows,w)
        j=np.bincount((self.current*32+self.mi)*128+self.y,weights,minlength=self.ncurrent*32*128).reshape(-1,32,128)
        with np.errstate(divide='ignore'):
            lj=np.log(j[self.cix]);lh=np.zeros((len(self.x),32))
            if self.x.shape[1]>13:
                hm=np.bincount(self.hinv*32+self.mi,weights,minlength=(int(self.hinv.max())+1)*32).reshape(-1,32)
                hm/=np.bincount(self.mi,weights,minlength=32)[None,:]
                for step in range(self.hidx.shape[1]):lh+=np.log(hm[self.hidx[:,step]])
        return logsumexp(lj+lh[:,:,None],axis=1)

def posterior(x,tier,laws,history_tier=None,observed=False,history_observed=False,shift='native',pulse=lambda **kw:None,history_shift=None):
    layout=Layout(x,tier,history_tier,observed,history_observed,shift,history_shift)
    acc=np.full((len(x),128),-np.inf);count=0
    for count,w in enumerate(laws,1):
        acc=np.logaddexp(acc,layout.log_joint(w))
        if count%100==0:pulse(phase='law-marginal',laws=count)
    if not count:raise ValueError('empty law prior')
    lp=normalize_log(acc)
    if not np.isfinite(logsumexp(acc,1)).all():raise ValueError('zero evidence under declared law mixture')
    return lp,dict(laws=count,exact=count in (12500,40500),aggregation='joint evidence weighted',history_arithmetic='log-domain')
