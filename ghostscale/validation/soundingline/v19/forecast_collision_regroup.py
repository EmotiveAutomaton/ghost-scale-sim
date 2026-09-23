"""Third regroup: forecast cross-moments and conditional law-mean dispersion."""
import numpy as np


def resample(saved, counts, chunk=128):
    """Rebuild means under every law multiplicity; expand centered group bias.

    This consumes saved scalar sufficient records only after their independent
    reconstruction. It does not call producer/reviewer resampling functions.
    """
    n=np.asarray(counts,float);mass=np.asarray(saved['mass']);sums=np.asarray(saved['target_sum'])
    laws,groups=mass.shape
    if n.ndim!=2 or n.shape[1]!=laws or np.any(n<0) or not np.all(n.sum(1)==laws):raise ValueError('counts')
    forecast=np.asarray(saved['group_forecast']).reshape(groups,9)
    law_second=np.divide(sums*sums,mass[:,:,None],out=np.zeros_like(sums),where=mass[:,:,None]>0).sum(1)
    forecast_square=(mass[:,:,None]*forecast[None]**2).sum(1)
    cross=(sums*forecast[None]).sum(1)
    same=np.array_equal(mass,np.broadcast_to(mass[0],mass.shape))
    if same:
        gram=np.empty((9,laws,laws))
        for k in range(9):
            a=sums[:,:,k];scaled=np.divide(a,mass,out=np.zeros_like(a),where=mass>0)
            gram[k]=scaled@a.T
    out=np.empty((len(n),4,9))
    for start in range(0,len(n),chunk):
        w=n[start:start+chunk]/laws
        if same:
            pooled=np.stack([np.sum(w*(w@g),axis=1) for g in gram],axis=1)
        else:
            m=w@mass;t=(w@sums.reshape(laws,-1)).reshape(len(w),groups,9)
            pooled=np.divide(t*t,m[:,:,None],out=np.zeros_like(t),where=m[:,:,None]>0).sum(1)
        bias=w@(forecast_square-2*cross)+pooled
        across=w@law_second-pooled
        within=w@saved['within'];loss=w@saved['loss']
        out[start:start+len(w)]=np.stack([loss,bias,within,across],axis=1)
    if not np.isfinite(out).all() or np.min(out)<-1e-12:raise ValueError('decomposition')
    if not np.allclose(out[:,0],out[:,1:].sum(1),atol=1e-10,rtol=0):raise ValueError('conservation')
    return out.reshape(-1,4,3,3)
