"""Conditional R performer map: prevalence among non-rule makers."""
from pathlib import Path
import numpy as np
from . import r
from .common import rng,sustained,mean_blocks,censored_blocks,audit_controls,write
def settings():
    return [dict(performer_rate=f,rule_prior=pr,truth=t,slope=1.5,heat=[.8,.1],
        steps=128,worlds=200,blocks=8) for f in (.1,.3) for pr in (.25,.1) for t in range(5)]
def laws(s,reference=False):
    keep=np.r_[r.keep(8,1,1,s["slope"]),r.keep(8,1,1,s["slope"])[0]]
    heat=np.array([.1,.1,.1,.8,.8])
    f=s["performer_rate"];rule=s["rule_prior"]
    prior=np.array([(1-rule)*(1-f)/3]*3+[rule,(1-rule)*f])
    if reference:
        import math
        keep=np.array([1/(1+math.exp(-s["slope"]*(.5-8))),
            1/(1+math.exp(-s["slope"]*(3.5-8))),
            1/(1+math.exp(-s["slope"]*(3.5-8))),.98,
            1/(1+math.exp(-s["slope"]*(.5-8)))])
    return keep,heat,prior
def generate(s,lane="study",pulse=lambda **kw:None):
    rr=rng("R-performer",lane,s);keep,heat,prior=laws(s)
    return dict(refusal=rr.random((s["worlds"],s["steps"]))<keep[s["truth"]],
                heat=rr.random((s["worlds"],s["steps"]))<heat[s["truth"]])
def infer(raw,s,reference=False,pulse=lambda **kw:None):
    keep,heat,prior=laws(s,reference);out={}
    for kind in ("refusal","heat","joint"):
        ll=np.broadcast_to(np.log(prior),(s["worlds"],s["steps"],5)).copy()
        for label,prob in (("refusal",keep),("heat",heat)):
            if kind not in (label,"joint"):continue
            if reference:
                for t in range(s["steps"]):
                    inc=np.stack([np.log(np.where(raw[label][:,t],p,1-p)) for p in prob],axis=1)
                    ll[:,t:]+=inc[:,None,:]
            else:ll+=np.cumsum(np.log(np.where(raw[label][:,:,None],prob,1-prob)),axis=1)
        odds=ll[:,:,s["truth"]]-np.max(np.delete(ll,s["truth"],axis=-1),axis=-1)
        out["separation_"+kind]=sustained(odds);out["_ll_"+kind]=ll
        rule_odds=ll[:,:,3]-np.max(np.delete(ll,3,axis=-1),axis=-1)
        out["rule_confident_"+kind]=(rule_odds[:,-1]>=3).astype(float)
    return out
def summarize(s,m):
    return [dict(measure=k,**(censored_blocks(v,s["steps"],s["blocks"]) if k.startswith("separation") else mean_blocks(v,s["blocks"])))
            for k,v in m.items() if not k.startswith("_")]
def pilot_exclusion(s,m):return None
def controls(dest=None,pulse=lambda **kw:None):
    s=dict(settings()[0],worlds=8,blocks=8,steps=32,truth=4);raw=generate(s,"known-answer")
    a=infer(raw,s);b=infer(raw,s,True);keep,heat,prior=laws(s)
    checks=[dict(name="performer and rule exactly alias on heat alone",passed=heat[3]==heat[4]),
        dict(name="performer and habit exactly alias on refusal alone",passed=keep[0]==keep[4]),
        dict(name="performer prevalence conserves prior",passed=abs(prior.sum()-1)<1e-14),
        dict(name="independent full performer likelihood",passed=all(np.allclose(a[k],b[k],atol=1e-10) for k in a))]
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
