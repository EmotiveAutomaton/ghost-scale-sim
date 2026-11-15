"""E context-dependent extinction, with explicit matched learner choices."""
from pathlib import Path
import numpy as np
from scipy.special import expit
from .common import rng,mean_blocks,censored_blocks,sustained,audit_controls,write
def settings():
    return [dict(truth=truth,similarity=s,learn=learn,extinguish=ext,steps=384,
                 worlds=200,blocks=8) for s in (0.,.25,.5,.75,1.) for learn in (64,256)
            for ext in (16,64,256) for truth in (0,1)]
def chances(harm,learn,ext,similarity,reference=False):
    base=(1+harm)/(10+learn);erased=(1+harm)/(10+learn+ext)
    sims=np.array([similarity,1.,similarity**2])
    brake=base[:,None]*(1-sims[None,:]*ext/(10+learn+ext))
    signal=np.stack((np.broadcast_to(erased[:,None],brake.shape),brake),axis=-1)
    if reference:return np.clip(1/(1+np.exp(-4*(2*signal-.5))),.02,.98)
    return np.clip(expit(4*(2*signal-.5)),.02,.98)
def generate(s,lane="study",pulse=lambda **kw:None):
    r=rng("E",lane,{k:v for k,v in s.items() if k!="truth"})
    harm=r.binomial(s["learn"],.5,s["worlds"])
    context=np.repeat(np.arange(3),s["steps"]//3)
    p=chances(harm,s["learn"],s["extinguish"],s["similarity"])
    alarm=r.random((s["worlds"],s["steps"]))<p[:,context,s["truth"]]
    return dict(training_harm_count=harm,context=context,alarm=alarm)
def infer(raw,s,reference=False,pulse=lambda **kw:None):
    p=chances(raw["training_harm_count"],s["learn"],s["extinguish"],s["similarity"],reference)
    probs=p[:,raw["context"]]
    ll=np.cumsum(np.where(raw["alarm"][:,:,None],np.log(probs),np.log1p(-probs)),axis=1)
    odds=ll[...,s["truth"]]-ll[...,1-s["truth"]]
    out=dict(separation=sustained(odds),final_odds=odds[:,-1],_likelihood=ll,_chances=p)
    for c in range(3):out["alarm_context_"+str(c)]=raw["alarm"][:,raw["context"]==c].mean(1)
    return out
def summarize(s,m):
    return [dict(measure=k,**(censored_blocks(v,s["steps"],s["blocks"]) if k=="separation" else mean_blocks(v,s["blocks"])))
            for k,v in m.items() if not k.startswith("_")]
def pilot_exclusion(s,m):
    if s["similarity"]==1:return "all-contexts-identical learner alias"
    if np.all(m["separation"]<=10):return "all forty separated within ten steps"
    return None
def controls(dest=None,pulse=lambda **kw:None):
    p=chances(np.arange(9),64,64,.25);same=chances(np.arange(9),64,64,1.)
    checks=[dict(name="extinction context exact matched rates",passed=bool(np.allclose(p[:,1,0],p[:,1,1],atol=1e-14))),
        dict(name="identical contexts exact alias",passed=bool(np.allclose(same[:,:,0],same[:,:,1],atol=1e-14))),
        dict(name="context brake release is stipulated",passed=bool(np.all(p[:,0,1]>p[:,0,0]) and np.all(p[:,2,1]>p[:,2,0]))),
        dict(name="independent logistic arithmetic",passed=bool(np.allclose(p,chances(np.arange(9),64,64,.25,True),atol=1e-14)))]
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
