"""R finite watched and asked readers, written from packet before preflight inspection."""
import itertools
import math
from pathlib import Path
import numpy as np
from scipy.special import expit
from .common import rng,mean_blocks,censored_blocks,sustained,audit_controls,write

REASONS=("HABIT","MEANS","FACE","RULE")
CORE=("DEVALUE","AUDIENCE","PRICE","BLOCK","CLAIM","OFFER")
CASES={"DEVALUE":(0,0,1),"AUDIENCE":(0,1,0),"PRICE":(8,1,1),
       "OFFER":(8,1,1),"SWEETEN":(12,1,1),"SYMBOL":(8,1,1)}

def keep(price,need,watched,slope=1.5):
    price,need,watched=np.broadcast_arrays(price,need,watched)
    return np.stack((expit(slope*(.5-price)),expit(slope*(4*need-.5-price)),
                     expit(slope*(4*watched-.5-price)),np.full(price.shape,.98)),axis=-1)

def prior(s):
    return np.array([(1-s["rule_prior"])/3]*3+[s["rule_prior"]])

def settings():
    watch=[dict(part="watched",truth=t,rate=rate,regime=reg,flags=flags,rule_prior=pr,
                slope=1.5,steps=3000,worlds=200,blocks=8)
           for rate in (.2,.05,.01) for reg in ("LOW","MIXED") for flags in (True,False)
           for pr in (.25,.1) for t in range(4)]
    asked=[dict(part="asked",truth=t,q=q,b=b,heat=list(h),rule_prior=pr,answer=answer,
                slope=1.5,steps=8,worlds=200,blocks=8)
           for q in (1.,.9,.75,.6) for b in (.5,.8) for h in ((.8,.1),(.5,.2),(.5,.5))
           for pr in (.25,.1) for answer in ("binary","graded") for t in range(4)]
    return watch+asked

def emission(s,probe,belief=None,reference=False):
    q=s["q"];b=s["b"]
    if probe=="CLAIM":
        claim=int(np.argmax(np.round(belief if belief is not None else prior(s),12))) # declared auxiliary tie rank only
        a=q*np.eye(4);a[:,claim]+=1-q
        return a
    if probe=="BLOCK":
        a=np.array([[.7,.15,.15],[.1,.8,.1],[.1,.8,.1],[.1,.1,.8]])
        return q*a+(1-q)/3 # untruthful BLOCK: uniform categorical answer
    p,n,w=CASES[probe]
    if reference:
        f=lambda x:1/(1+math.exp(-x))
        chances=np.array([f(s["slope"]*(.5-p)),f(s["slope"]*(4*n-.5-p)),
                          f(s["slope"]*(4*w-.5-p)),.98])
    else: chances=keep(p,n,w,s["slope"])
    if probe in ("OFFER","SWEETEN","SYMBOL"):
        if probe=="SYMBOL":chances[-1]=.5
        hr,fr=s["heat"];heat=np.array([fr,fr,fr,hr])
        if probe=="SWEETEN":heat[-1]=min(1.,hr+.2)
        # staged observations are actual actions, not q-corrupted reports
        a=np.stack(((1-chances)*(1-heat),(1-chances)*heat,
                    chances*(1-heat),chances*heat),axis=1)
    elif s["answer"]=="binary":
        k=q*chances+(1-q)*b;a=np.stack((1-k,k),axis=1)
    else:
        # Truthful grade: nearest fifth plus -1,0,+1 fifth equally, clipped.
        # Untruthful response: the two endpoint grades with keep bias b.
        a=np.zeros((4,6))
        for j,chance in enumerate(chances):
            center=int(np.floor(chance*5+.5))
            for delta in (-1,0,1):a[j,np.clip(center+delta,0,5)]+=q/3
        a[:,0]+=(1-q)*(1-b);a[:,5]+=(1-q)*b
    return a

def watched_probability(raw,s,reference=False):
    price=raw["price"];n=raw["need"];w=raw["watched"]
    if not reference:
        if s["flags"]:return keep(price,n,w,s["slope"])
        p=np.zeros((*price.shape,4))
        for nn,ww in itertools.product((0,1),repeat=2):
            mass=(.7 if nn else .3)*(.6 if ww else .4)
            p+=mass*keep(price,nn,ww,s["slope"])
        return p
    # Independently build a discrete lookup by summing latent contexts.
    values=(0,1,2) if s["regime"]=="LOW" else (0,1,2,4,8)
    p=np.empty((*price.shape,4))
    for amount in values:
        for nn,ww in itertools.product((0,1),repeat=2):
            mask=(price==amount)
            if s["flags"]:mask &= (n==nn)&(w==ww)
            elif nn or ww:continue
            for reason in range(4):
                contexts=[(nn,ww,1.)] if s["flags"] else [
                    (a,b,(.7 if a else .3)*(.6 if b else .4))
                    for a,b in itertools.product((0,1),repeat=2)]
                v=0.
                for a,b,mass in contexts:
                    utility=(.5,4*a-.5,4*b-.5)[reason] if reason<3 else 0.
                    k=1/(1+math.exp(-s["slope"]*(utility-amount))) if reason<3 else .98
                    v+=mass*k
                p[...,reason][mask]=v
    return p

def generate(s,lane="study",pulse=lambda **kw:None):
    r=rng("R",lane,s)
    worlds=s["worlds"]
    if s["part"]=="watched":
        shape=(worlds,s["steps"])
        prices,weights=((0,1,2),(.5,.3,.2)) if s["regime"]=="LOW" else ((0,1,2,4,8),(.25,.25,.2,.15,.15))
        raw=dict(price=r.choice(prices,size=shape,p=weights),need=r.random(shape)<.7,
                 watched=r.random(shape)<.6,occasion=r.random(shape)<s["rate"])
        raw["act"]=r.random(shape)<keep(raw["price"],raw["need"],raw["watched"],s["slope"])[...,s["truth"]]
        return raw
    answers=np.empty((worlds,8),np.int8)
    for world in range(worlds):
        belief=prior(s)
        for j,probe in enumerate((*CORE,"SWEETEN","SYMBOL")):
            a=emission(s,probe,belief);obs=r.choice(a.shape[1],p=a[s["truth"]]);answers[world,j]=obs
            belief*=a[:,obs];belief/=belief.sum()
    return dict(answers=answers)

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    if s["part"]=="watched":
        p=watched_probability(raw,s,reference)
        ll=np.where(raw["act"][...,None],np.log(p),np.log1p(-p))*raw["occasion"][...,None]
        cum=np.cumsum(ll,axis=1)+np.log(prior(s))
        truth=s["truth"];rival=np.max(np.delete(cum,truth,axis=-1),axis=-1)
        odds=cum[...,truth]-rival
        return dict(separation=sustained(odds),final_odds=odds[:,-1],
                    occasion_share=raw["occasion"].mean(1),_odds=odds,
                    _same_probability=np.all(np.isclose(p,p[...,truth,None],rtol=0,atol=1e-14)|
                                             ~raw["occasion"][...,None],axis=1))
    odds=np.empty((s["worlds"],8));ll=np.tile(np.log(prior(s)),(s["worlds"],1))
    probe_names=(*CORE,"SWEETEN","SYMBOL")
    for world,answers in enumerate(raw["answers"]):
        for j,(probe,answer) in enumerate(zip(probe_names,answers)):
            b=np.exp(ll[world]-np.max(ll[world]));b/=b.sum()
            a=emission(s,probe,b,reference)
            ll[world]+=np.log(np.maximum(a[:,answer],1e-300))
            odds[world,j]=ll[world,s["truth"]]-np.max(np.delete(ll[world],s["truth"]))
    return dict(_odds=odds,**{"odds_after_"+name:odds[:,j] for j,name in enumerate(probe_names)},
                **{"settled_after_"+name:(odds[:,j]>=3).astype(float) for j,name in enumerate(probe_names)})

def summarize(s,m):
    rows=[]
    for k,v in m.items():
        if k.startswith("_"):continue
        summary=censored_blocks(v,s["steps"],s["blocks"]) if k=="separation" else mean_blocks(v,s["blocks"])
        rows.append(dict(measure=k,**summary))
    return rows

def pilot_exclusion(s,m):
    if s["part"]=="asked":
        # Reporting eight elicited probes is the estimand, not a long-record discovery.
        return None
    if np.all(m["separation"]<=10):return "all forty separated within ten steps"
    for k in range(4):
        if k!=s["truth"] and np.all(m["_same_probability"][:,k]):return "observed exact action alias"
    return None

def exact_asked(s,probes=("DEVALUE","AUDIENCE","PRICE")):
    states=[(prior(s),np.ones(4),0)] # posterior, likelihood of path under each truth, length
    for name in probes:
        children=[]
        for belief,mass,depth in states:
            a=emission(s,name,belief)
            for outcome in range(a.shape[1]):
                joint=belief*a[:,outcome];z=joint.sum()
                if z>0:children.append((joint/z,mass*a[:,outcome],depth+1))
        states=children
    means=np.zeros(4);settled=np.zeros(4)
    for belief,mass,_ in states:
        log=np.log(np.maximum(belief,1e-300))
        for k in range(4):
            odds=log[k]-max(np.delete(log,k))
            means[k]+=mass[k]*odds;settled[k]+=mass[k]*(odds>=3)
    return dict(expected_odds=means.tolist(),settled_share=settled.tolist())

def controls(dest=None,pulse=lambda **kw:None):
    s=dict(q=.9,b=.5,heat=[.5,.5],rule_prior=.25,answer="binary",slope=1.5)
    checks=[]
    for probe in (*CORE,"SWEETEN","SYMBOL"):
        a=emission(s,probe);b=emission(s,probe,reference=True)
        checks.append(dict(name=probe+" independent normalized emission",passed=bool(
            np.allclose(a,b,atol=1e-13) and np.allclose(a.sum(1),1) and np.all(a>=0))))
    a=emission(s,"OFFER")
    checks.append(dict(name="chance heat cancels conditional likelihood ratios",
                       passed=bool(np.allclose(a[:,0],a[:,1]) and np.allclose(a[:,2],a[:,3]))))
    p=keep(8,1,1)
    checks.append(dict(name="price probe stipulated rule law",passed=bool(p[3]==.98 and max(p[:3])<.002)))
    s.update(q=1.)
    exact=exact_asked(s)
    checks.append(dict(name="exact enumeration conserves finite asked evidence",
                       passed=all(0<=x<=1 for x in exact["settled_share"])))
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
