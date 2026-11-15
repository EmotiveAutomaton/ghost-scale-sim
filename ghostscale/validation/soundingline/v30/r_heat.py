"""R repeated staged offers: heat, refusal and their joint evidence."""
import numpy as np
from . import r
from .common import rng,sustained,mean_blocks,censored_blocks,audit_controls,write
from pathlib import Path
def settings():
    return [dict(truth=t,heat=list(heat),rule_prior=prior,slope=1.5,steps=128,worlds=200,blocks=8)
            for heat in ((.8,.1),(.5,.2),(.5,.5)) for prior in (.25,.1) for t in range(4)]
def likelihoods(s):
    keep=r.keep(8,1,1,s["slope"]);heat=np.array([s["heat"][1]]*3+[s["heat"][0]])
    return keep,heat
def generate(s,lane="study",pulse=lambda **kw:None):
    rr=rng("R-heat",lane,s);keep,heat=likelihoods(s);shape=(s["worlds"],s["steps"])
    return dict(refusal=rr.random(shape)<keep[s["truth"]],heat=rr.random(shape)<heat[s["truth"]])
def infer(raw,s,reference=False,pulse=lambda **kw:None):
    keep,heat=likelihoods(s);prior=r.prior(s);out={}
    for kind in ("refusal","heat","joint"):
        ll=np.broadcast_to(np.log(prior),(s["worlds"],s["steps"],4)).copy()
        for name,chance in (("refusal",keep),("heat",heat)):
            if kind not in (name,"joint"):continue
            obs=raw[name]
            if reference:
                increments=np.stack([np.where(obs,np.log(x),np.log(1-x)) for x in chance],axis=-1)
                for k in range(4):ll[:,:,k]+=np.cumsum(increments[:,:,k],axis=1)
            else:ll+=np.cumsum(np.log(np.where(obs[:,:,None],chance,1-chance)),axis=1)
        odds=ll[...,s["truth"]]-np.max(np.delete(ll,s["truth"],axis=-1),axis=-1)
        out["separation_"+kind]=sustained(odds);out["odds_"+kind]=odds[:,-1];out["_ll_"+kind]=ll
    return out
def summarize(s,m):
    return [dict(measure=k,**(censored_blocks(v,s["steps"],s["blocks"]) if k.startswith("separation") else mean_blocks(v,s["blocks"])))
            for k,v in m.items() if not k.startswith("_")]
def pilot_exclusion(s,m):
    # Comparisons include the declared zero-information heat control; never drop it.
    return None
def controls(dest=None,pulse=lambda **kw:None):
    s=dict(settings()[0],heat=[.5,.5],worlds=8,steps=32,blocks=8);raw=generate(s,"known-answer")
    a=infer(raw,s);b=infer(raw,s,True)
    checks=[dict(name="chance heat gives no posterior odds update",passed=bool(np.allclose(a["odds_heat"],0))),
            dict(name="independent full staged likelihood",passed=all(np.allclose(a[k],b[k],atol=1e-10) for k in a))]
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
