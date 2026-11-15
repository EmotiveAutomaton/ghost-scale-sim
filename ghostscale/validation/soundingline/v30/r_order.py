"""Exact finite six-probe ordering and prior expected-information exchange rates."""
import itertools,math
from functools import lru_cache
from pathlib import Path
import numpy as np
from . import r
from .common import write,audit_controls
EXACT_MAP=True

def settings():
    return [dict(q=q,b=b,heat=list(h),rule_prior=pr,answer=answer,slope=1.5,steps=6,worlds=200,blocks=8)
            for q in (1.,.9,.75,.6) for b in (.5,.8) for h in ((.8,.1),(.5,.2),(.5,.5))
            for pr in (.25,.1) for answer in ("binary","graded")]
def entropy(p):
    return -np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0),axis=-1)
def information(belief,emission):
    joint=belief[:,None]*emission;mass=joint.sum(0);post=np.divide(joint,mass,out=np.zeros_like(joint),where=mass>0)
    return float(entropy(belief)-np.sum(mass*entropy(post.T)))
def batch_emission(s,name,belief):
    if name!="CLAIM":return np.broadcast_to(r.emission(s,name),(len(belief),*r.emission(s,name).shape))
    claims=np.argmax(belief>=np.max(belief,axis=1,keepdims=True)-1e-12,axis=1)
    result=np.broadcast_to(s["q"]*np.eye(4),(len(belief),4,4)).copy()
    result[np.arange(len(belief)),:,claims]+=1-s["q"]
    return result

def exact_order(s,order):
    belief=r.prior(s)[None,:];mass=np.ones((1,4));first=np.full((1,4),6.)
    false=np.zeros((1,4),bool)
    for depth,name in enumerate(order,1):
        emission=batch_emission(s,name,belief);outcomes=emission.shape[-1]
        joint=belief[:,:,None]*emission;z=joint.sum(1)
        newbelief=np.divide(joint,z[:,None,:],out=np.zeros_like(joint),where=z[:,None,:]>0)
        mass=(mass[:,:,None]*emission).transpose(0,2,1).reshape(-1,4)
        belief=newbelief.transpose(0,2,1).reshape(-1,4)
        first=np.repeat(first,outcomes,axis=0);false=np.repeat(false,outcomes,axis=0)
        alive=mass.sum(1)>0;mass=mass[alive];belief=belief[alive];first=first[alive];false=false[alive]
        log=np.log(np.maximum(belief,1e-300))
        odds=np.stack([log[:,k]-np.max(np.delete(log,k,axis=1),axis=1) for k in range(4)],axis=1)
        first=np.where((odds>=3)&(first==6),depth,first)
        false|=(odds<-3)&(np.max(odds,axis=1)[:,None]>=3)
    settled=odds>=3
    return np.sum(mass*first,axis=0),np.sum(mass*settled,axis=0),np.sum(mass*false,axis=0)

from numba import njit

@njit
def reference_kernel(matrices,cardinality,indices,q,prior):
    expected=np.zeros(4);settled=np.zeros(4);wrong=np.zeros(4)
    total=1
    for index in indices:total*=cardinality[index]
    for path in range(total):
        code=path;answers=np.empty(len(indices),np.int64)
        for depth in range(len(indices)-1,-1,-1):
            card=cardinality[indices[depth]];answers[depth]=code%card;code//=card
        like=np.ones(4);first=np.full(4,6.);false=np.zeros(4,np.bool_);odds=np.zeros(4)
        alive=True
        for depth in range(len(indices)):
            index=indices[depth];answer=answers[depth]
            if index==4:
                belief=like*prior;norm=belief.sum()
                if norm<=0:alive=False;break
                belief/=norm;best=np.max(belief);claim=0
                while belief[claim]<best-1e-12:claim+=1
                for k in range(4):like[k]*=(q if answer==k else 0.)+((1-q) if answer==claim else 0.)
            else:
                for k in range(4):like[k]*=matrices[index,k,answer]
            belief=like*prior;norm=belief.sum()
            if norm<=0:alive=False;break
            log=np.log(np.maximum(belief/norm,1e-300));largest=-math.inf
            for k in range(4):
                rival=-math.inf
                for j in range(4):
                    if j!=k:rival=max(rival,log[j])
                odds[k]=log[k]-rival;largest=max(largest,odds[k])
                if odds[k]>=3 and first[k]==6:first[k]=depth+1
            for k in range(4):
                if odds[k]<-3 and largest>=3:false[k]=True
        if alive:
            for k in range(4):
                expected[k]+=like[k]*first[k]
                if odds[k]>=3:settled[k]+=like[k]
                if false[k]:wrong[k]+=like[k]
    return expected,settled,wrong

def exact_order_reference(s,order):
    matrices=np.zeros((6,4,6));cardinality=np.empty(6,np.int64)
    for i,name in enumerate(r.CORE):
        a=r.emission(s,name,reference=True);matrices[i,:,:a.shape[1]]=a;cardinality[i]=a.shape[1]
    return reference_kernel(matrices,cardinality,np.array([r.CORE.index(n) for n in order]),s["q"],r.prior(s))

def order_emission(s,name,belief):
    if name=="CLAIM":return batch_emission(s,name,belief[None,:])[0]
    return r.emission(s,name,belief)

def greedy(s):
    # Exact adaptive information-gain policy, no stochastic training.
    prior=r.prior(s);cost=np.zeros(4);settled=np.zeros(4);first_probe=None;nodes=0
    def visit(like,remaining,first,depth):
        nonlocal nodes,first_probe
        nodes+=1;belief=like*prior;belief/=belief.sum()
        log=np.log(np.maximum(belief,1e-300))
        odds=np.array([log[k]-np.max(np.delete(log,k)) for k in range(4)])
        if depth:first=np.where((odds>=3)&(first==6),depth,first)
        if not remaining:cost[:]+=like*first;settled[:]+=like*(odds>=3);return
        gains=[information(belief,order_emission(s,n,belief)) for n in remaining]
        index=int(np.argmax(np.round(gains,12)));name=remaining[index]
        if depth==0:first_probe=name
        a=order_emission(s,name,belief);following=tuple(n for n in remaining if n!=name)
        for answer in range(a.shape[1]):
            joint=like*a[:,answer]
            if joint.sum()>0:visit(joint,following,first.copy(),depth+1)
    visit(np.ones(4),r.CORE,np.full(4,6.),0)
    return dict(first_probe=first_probe,cost=cost.tolist(),settled=settled.tolist(),nodes=nodes)

def watched_information(s,regime,flags):
    prices,weights=([0,1,2],[.5,.3,.2]) if regime=="LOW" else ([0,1,2,4,8],[.25,.25,.2,.15,.15])
    out=0.;prior=r.prior(s)
    for price,weight in zip(prices,weights):
        if flags:
            for need,watched in itertools.product((0,1),repeat=2):
                mass=(.7 if need else .3)*(.6 if watched else .4);keep=r.keep(price,need,watched,s["slope"])
                out+=weight*mass*information(prior,np.stack((1-keep,keep),axis=1))
        else:
            keep=sum((.7 if n else .3)*(.6 if w else .4)*r.keep(price,n,w,s["slope"]) for n,w in itertools.product((0,1),repeat=2))
            out+=weight*information(prior,np.stack((1-keep,keep),axis=1))
    return out

def generate(s,lane="study",pulse=lambda **kw:None):
    # Exact map has no sampled worlds. Runner's pilot is a duplicate arithmetic check.
    return dict(identity=np.array([0],np.int8))
def infer(raw,s,reference=False,pulse=lambda **kw:None):
    costs=[];settled=[];wrong=[];orders=list(itertools.permutations(r.CORE))
    for i,order in enumerate(orders):
        values=exact_order_reference(s,order) if reference else exact_order(s,order)
        costs.append(values[0]);settled.append(values[1]);wrong.append(values[2])
        if i%24==0:pulse(order=i+1,orders=720)
    gains=np.array([information(r.prior(s),r.emission(s,n,r.prior(s))) for n in r.CORE])
    watch=np.array([watched_information(s,reg,flags) for reg in ("LOW","MIXED") for flags in (True,False)])
    gr=greedy(s)
    return dict(_cost=np.array(costs),_settled=np.array(settled),_wrong=np.array(wrong),
                _probe_information=gains,_watch_information=watch,
                _greedy_cost=np.array(gr["cost"]),_greedy_settled=np.array(gr["settled"]),
                _greedy_first=np.array([r.CORE.index(gr["first_probe"])],int))
def summarize(s,m):
    weights=r.prior(s);overall=m["_cost"]@weights;best=int(np.argmin(np.round(overall,12)))
    order=list(itertools.permutations(r.CORE))[best]
    rows=[dict(measure="best_fixed_order",order=order,prior_mean_capped_probes=float(overall[best]),
       by_true_reason=m["_cost"][best].tolist(),settled_by_true_reason=m["_settled"][best].tolist(),
       wrong_confidence_by_true_reason=m["_wrong"][best].tolist(),exact_enumeration=True),
       dict(measure="greedy_adaptive",first=r.CORE[int(m["_greedy_first"][0])],
       prior_mean_capped_probes=float(m["_greedy_cost"]@weights),by_true_reason=m["_greedy_cost"].tolist(),
       settled_by_true_reason=m["_greedy_settled"].tolist(),exact_enumeration=True)]
    for j,name in enumerate(r.CORE):
        for i,(reg,flags) in enumerate(itertools.product(("LOW","MIXED"),(True,False))):
            rows.append(dict(measure="prior_expected_information_equivalent_watched_occasions",probe=name,
                regime=reg,flags=flags,estimate=float(m["_probe_information"][j]/m["_watch_information"][i]),
                probe_information=float(m["_probe_information"][j]),watch_information=float(m["_watch_information"][i])))
    return rows
def pilot_exclusion(s,m):return None
def controls(dest=None,pulse=lambda **kw:None):
    s=settings()[0];checks=[]
    for order in (r.CORE,tuple(reversed(r.CORE))):
        a=exact_order(s,order);b=exact_order_reference(s,order)
        checks.append(dict(name="independent exact order "+str(order),passed=all(np.allclose(x,y,atol=1e-10) for x,y in zip(a,b))))
    checks.append(dict(name="identical emissions have zero information",passed=abs(information(np.full(4,.25),np.full((4,2),.5)))<1e-14))
    near=np.array([[.23340704371725565,.3506205384685,.35062053846850005,.0653518793457444]])
    choice=batch_emission(dict(s,q=.6),"CLAIM",near)
    checks.append(dict(name="near-equal leading reasons retain first-index tie",passed=bool(np.allclose(choice[0]-.6*np.eye(4),np.tile([0.,.4,0.,0.],(4,1)),rtol=0,atol=1e-15))))
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
