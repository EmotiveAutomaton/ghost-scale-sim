"""K attention-only reader: prospectively declared Markov projection plus exact no-gap control."""
from pathlib import Path
from functools import lru_cache
import numpy as np
from . import k
from .common import rng,sustained,mean_blocks,censored_blocks,audit_controls,write

def settings():
    return [dict(arrival=a,opening=w,pair=pair,truth=truth,steps=3000,worlds=200,blocks=8,
                 train_worlds=128,order=2) for a in (.02,.05,.15) for w in (.3,.1)
            for pair in ("plan_strong_vs_noplan","plan_weak_vs_empty") for truth in (0,1)]
def spec(s,truth,worlds=None):
    policy,q,cue=("PLAN",1.,.9) if truth==0 else ("NOPLAN",0.,0.)
    arrival=s["arrival"]
    if s["pair"]=="plan_weak_vs_empty":
        policy,q,cue="PLAN",1.,.3
        if truth==1:arrival=0.
    return dict(arrival=arrival,opening=s["opening"],q=q,cue=cue,policy=policy,deadline=False,
        steps=s["steps"],worlds=worlds or s["worlds"],blocks=8,work_sizes=[3])
def observations(history):
    # Only observable location changes: elsewhere, same named gap, another named gap.
    location=history[:,:,0].astype(int);previous=np.concatenate((np.zeros((len(location),1),int),location[:,:-1]),axis=1)
    return np.where(location==0,0,np.where(location==previous,1,2)).astype(np.int8)
def counts_for(obs,order=2,reference=False):
    out=np.full((3**order,3),.5)
    if reference:
        for row in obs:
            hist=[0]*order
            for value in row:
                index=0
                for item in hist:index=index*3+item
                out[index,value]+=1;hist=hist[1:]+[int(value)]
    else:
        code=np.zeros(len(obs),int)
        for t in range(obs.shape[1]):
            np.add.at(out,(code,obs[:,t]),1);code=(code*3+obs[:,t])%(3**order)
    return out
def fit(s,reference=False):
    models=[]
    for truth in (0,1):
        ss=spec(s,truth,s["train_worlds"]);raw=k.generate(ss,"excluded-attention-training")
        obs=observations(raw["history"]);counts=counts_for(obs,s["order"],reference)
        models.append(counts/counts.sum(1,keepdims=True))
    return np.array(models)
def generate(s,lane="study",pulse=lambda **kw:None):
    ss=spec(s,s["truth"]);raw=k.generate(ss,"K-attention-"+lane,pulse)
    raw["observed_attention"]=observations(raw["history"])
    return raw
def infer(raw,s,reference=False,pulse=lambda **kw:None):
    obs=raw["observed_attention"]
    if reference and not np.array_equal(obs,observations(raw["history"])):raise ArithmeticError("attention extraction differs")
    if s["pair"]=="plan_weak_vs_empty":
        # With q=1 and no deadline, only immediate planning puts attention on a gap.
        # Arrival Bernoulli a versus an exact all-elsewhere path.
        probability=s["arrival"]
        inc=np.where(obs>0,np.log(probability),np.log1p(-probability))
        plan=np.cumsum(inc,axis=1)
        empty=np.where(np.cumsum(obs>0,axis=1)>0,-np.inf,0.)
        odds=plan-empty if s["truth"]==0 else empty-plan
        ll=np.stack((plan,empty),axis=-1)
    else:
        model=fit(s,reference);code=np.zeros(len(obs),int);ll=np.zeros((len(obs),s["steps"],2))
        total=np.zeros((len(obs),2))
        for t in range(s["steps"]):
            for truth in (0,1):total[:,truth]+=np.log(model[truth,code,obs[:,t]])
            ll[:,t]=total;code=(code*3+obs[:,t])%(3**s["order"])
        odds=ll[...,s["truth"]]-ll[...,1-s["truth"]]
    return dict(separation=sustained(odds),final_finite_odds=np.where(np.isfinite(odds[:,-1]),odds[:,-1],0),
        infinite_final_odds=(~np.isfinite(odds[:,-1])).astype(float),_odds=odds,_ll=ll,
        attention_share=(obs>0).mean(1),false_confidence=np.any(odds<=-3,axis=1).astype(float),_observations=obs)
def summarize(s,m):
    rows=[dict(measure="separation",**censored_blocks(m["separation"],s["steps"],s["blocks"]))]
    for name in ("attention_share","infinite_final_odds","false_confidence"):rows.append(dict(measure=name,**mean_blocks(m[name],s["blocks"])))
    rows.append(dict(measure="reader_scope",description="exact arrival/no-gap contrast" if s["pair"]=="plan_weak_vs_empty"
                     else "order-2 Markov projection trained on excluded worlds; not an exact hidden-queue reader"))
    return rows
def pilot_exclusion(s,m):
    if np.all(m["separation"]<=10):return "all forty separated within ten steps"
    return None
def controls(dest=None,pulse=lambda **kw:None):
    rr=rng("K-attention-counts");obs=rr.integers(0,3,size=(8,64))
    a=counts_for(obs);b=counts_for(obs,reference=True)
    empty=np.zeros((8,32,8))
    checks=[dict(name="independent Markov transition sufficient counts",passed=bool(np.array_equal(a,b))),
            dict(name="no gaps give no gap attention",passed=bool(np.all(observations(empty)==0)))]
    ss=dict(settings()[2],worlds=8,steps=128,blocks=8,train_worlds=8)
    raw=generate(ss,"known-answer");a=infer(raw,ss);b=infer(raw,ss,True)
    checks.append(dict(name="exact no-gap likelihood reconstruction",passed=all(np.allclose(a[x],b[x],equal_nan=True) for x in a)))
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
