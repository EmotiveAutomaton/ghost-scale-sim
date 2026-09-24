"""Independent 27-coordinate group cross-moments for law regrouping."""
import numpy as np


def resample(saved, counts, chunk=128):
    mass=np.asarray(saved['mass']);sums=np.asarray(saved['target_sum']);laws,groups=mass.shape
    counts=np.asarray(counts,float)
    if sums.shape!=(laws,groups,27) or counts.ndim!=2 or counts.shape[1]!=laws or np.any(counts<0) or not np.all(counts.sum(1)==laws):raise ValueError('shape/counts')
    left,right=np.triu_indices(laws)
    cross=(sums[left]*sums[right]).sum(2)*(1+(left!=right))[:,None]
    same=np.array_equal(mass,np.broadcast_to(mass[0],mass.shape))
    if same:reduced=np.divide(cross,mass[0],out=np.zeros_like(cross),where=mass[0]>0).sum(1)
    law_second=np.divide(sums*sums,mass[:,:,None],out=np.zeros_like(sums),where=mass[:,:,None]>0).sum((1,2))
    out=np.empty((len(counts),3))
    for start in range(0,len(counts),chunk):
        w=counts[start:start+chunk]/laws;products=w[:,left]*w[:,right]
        if same:pooled=products@reduced
        else:
            numerators=products@cross;masses=w@mass
            pooled=np.divide(numerators,masses,out=np.zeros_like(numerators),where=masses>0).sum(1)
        total=w@saved['target_square']-pooled;within=w@(saved['target_square']-law_second)
        out[start:start+len(w)]=np.stack([total,within,w@law_second-pooled],axis=1)
    if not np.isfinite(out).all() or np.min(out)<-1e-12:raise ValueError('variation')
    return out
