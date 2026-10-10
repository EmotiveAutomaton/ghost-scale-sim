"""Uncapped log score and full multiclass Brier; no repeat normalization."""
import math
import numpy as np
from .world import LABELS

def scores(logp,y):
    p=np.exp(logp);q=p[np.arange(len(y)),y];mode=p.argmax(1);bits=p@LABELS
    if not np.allclose(p.sum(1),1,atol=1e-10) or np.isnan(logp).any():raise ValueError('invalid forecast')
    return dict(log_loss=-logp[np.arange(len(y)),y],capped_log_loss=-np.log(np.maximum(q,1e-12)),
        brier=(p*p).sum(1)-2*q+1,correct=(mode==y).astype(float),
        zero_mass=(q==0).astype(float),below_floor=((q>0)&(q<1e-12)).astype(float),
        confident_error=((p.max(1)>=.9)&(mode!=y)).astype(float),
        bit_brier=(bits-LABELS[y])**2,bit_correct=((bits>=.5)==LABELS[y]).astype(float),
        top8=np.any(np.argsort(p,axis=1)[:,-8:]==y[:,None],axis=1).astype(float))

def check_scores(logp,y,stored):
    # Separate scalar score implementation; use the exact saved log probabilities.
    for i,(row,target) in enumerate(zip(logp,y)):
        p=[math.exp(float(v)) for v in row];q=p[int(target)]
        if abs(math.fsum(p)-1)>1e-9:raise ValueError('invalid normalized mass')
        truth=-float(row[target]);b=math.fsum(v*v for v in p)-2*q+1
        for name,value in [('log_loss',truth),('brier',b),('capped_log_loss',-math.log(max(q,1e-12)))]:
            if not math.isclose(value,float(stored[name][i]),abs_tol=1e-10,rel_tol=1e-10):raise ValueError('corrupt '+name)
        mode=max(range(len(p)),key=p.__getitem__)
        extra=dict(correct=float(mode==target),zero_mass=float(q==0),below_floor=float(0<q<1e-12),
            confident_error=float(max(p)>=.9 and mode!=target),
            top8=float(target in np.argsort(np.asarray(p))[-8:]))
        for name,value in extra.items():
            if name in stored and value!=stored[name][i]:raise ValueError('corrupt '+name)
        if len(p)==128:
            marginal=[math.fsum(prob*((label>>(6-bit))&1) for label,prob in enumerate(p)) for bit in range(7)]
            for bit,prob in enumerate(marginal):
                target_bit=(int(target)>>(6-bit))&1
                if 'bit_brier' in stored and not math.isclose((prob-target_bit)**2,stored['bit_brier'][i,bit],abs_tol=1e-10):raise ValueError('corrupt bit Brier')
                # Ties within roundoff cannot support a discrete per-bit claim.
                if 'bit_correct' in stored and abs(prob-.5)>1e-12 and float((prob>=.5)==bool(target_bit))!=stored['bit_correct'][i,bit]:raise ValueError('corrupt bit accuracy')
    return True

def mean_scores(a):
    def safe(value):
        if isinstance(value,list):return [safe(v) for v in value]
        return value if np.isfinite(value) else None
    out={k:safe(np.mean(v,axis=0).tolist()) for k,v in a.items()}
    out['infinite_log_loss_fraction']=float(np.isinf(a['log_loss']).mean())
    return out

def bootstrap(values,seed=21):
    values=np.asarray(values,float);rng=np.random.default_rng(seed)
    samples=values[rng.integers(len(values),size=(4096,len(values)))].mean(1)
    return dict(mean=float(values.mean()),lower=float(np.quantile(samples,.025)),upper=float(np.quantile(samples,.975)),worlds=len(values))
