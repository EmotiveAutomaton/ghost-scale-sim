"""Diagnose exact-path leading-claim disagreements before any R repair."""
import itertools
import numpy as np
from . import r
from .common import read,write,digest

def validate(s,dest,pulse):
    setting=s["setting"];order=tuple(s["order"]);a=np.tile(r.prior(setting),(1,1));like=np.ones((1,4));paths=[()]
    reports=[]
    for probe in order:
        if probe=="CLAIM":
            b=like*r.prior(setting);b/=b.sum(1,keepdims=True)
            ca=np.argmax(np.round(a,12),axis=1);cb=np.argmax(np.round(b,12),axis=1)
            bad=np.flatnonzero(ca!=cb)
            reports.extend(dict(path=paths[int(i)],normalized=a[i].tolist(),product=b[i].tolist(),
                normalized_rounded=np.round(a[i],12).tolist(),product_rounded=np.round(b[i],12).tolist(),
                chosen_normalized=int(ca[i]),chosen_product=int(cb[i])) for i in bad)
            break
        ea=r.emission(setting,probe);eb=r.emission(setting,probe,reference=True);out=ea.shape[1]
        aa=a[:,:,None]*ea;den=aa.sum(1);aa=np.divide(aa,den[:,None,:],out=np.zeros_like(aa),where=den[:,None,:]>0)
        a=aa.transpose(0,2,1).reshape(-1,4);like=(like[:,:,None]*eb).transpose(0,2,1).reshape(-1,4)
        paths=[p+(j,) for p in paths for j in range(out)]
        alive=like.sum(1)>0;a=a[alive];like=like[alive];paths=[p for p,yes in zip(paths,alive) if yes]
    result=dict(passed=True,purpose="diagnostic only",scientific_draws=0,claim_disagreements=reports,
      max_probability_difference=float(np.max(abs(a-b))))
    write(dest/"DIAGNOSIS.json",result);return result
