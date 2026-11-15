"""V hidden-ledger bootstrap filter, with masked observables and checked resolution."""
import math
from pathlib import Path
import numpy as np
from numba import njit
from ..v29 import ledger,x3
from .common import rng,audit_controls,write

@njit
def chance(m,d,price,kind,weight,shutdown,reference=False):
    if kind==3 and shutdown>0:return 0.
    if kind!=0:return .02
    z=1.5*(.5*price+min(price,max(0,d-m))-weight)
    if reference:
        e=math.exp(-abs(z));value=1/(1+e) if z>=0 else e/(1+e)
    else:value=1/(1+math.exp(min(700.,-z)))
    return min(.98,max(.02,value))
@njit
def spending(m,d,kind,weight,reference=False):
    if m<=0 or d<=0:return 0.
    z=1.5*(min(d,3)-(weight if kind==2 else .5))
    return min(.98,max(.02,1/(1+math.exp(-z))))
@njit
def choose(u,prob):
    total=0.
    for j in range(len(prob)):
        total+=prob[j]
        if u<total:return j
    return len(prob)-1
@njit
def normalize_models(ll,prior):
    a=ll+prior;mx=np.max(a);w=np.exp(a-mx);return w/w.sum()
@njit
def kernel(params,prior,prices,price_prob,public_prices,public_prob,qin,qn,loss,K,record_seen,
           truth_kind,truth_weight,draws,observations,seed,generate_mode,reference=False):
    np.random.seed(seed);T=len(draws);J=len(params)
    money=np.full((J,K),5,np.int16);need=np.zeros((J,K),np.int16);stock=np.full((J,K),8,np.int16)
    shutdown=np.zeros((J,K),np.int64);weights=np.full((J,K),1./K);ll=np.zeros(J)
    records=np.zeros((T,10),np.int16);public_after=np.zeros(T,np.int16);private_price=np.zeros(T,np.int16)
    invited=np.zeros(T,np.int8);defects=np.zeros(T,np.int8);prediction=np.zeros(T);delta=np.zeros(T)
    ll_path=np.zeros((T,J));m=5;d=0;x=8
    for t in range(T):
        if generate_mode:
            m=min(30,m+(1 if draws[t,0]<qin else 0))
            d=min(6,d+(1 if draws[t,1]<qn else 0)+(1 if draws[t,2]<qn else 0))
            offered=draws[t,3]<.25 and x>0
            price=public_prices[choose(draws[t,4],public_prob)]
            truechance=chance(m,d,price,0 if truth_kind==7 else truth_kind,truth_weight,0,reference) if offered else 0.
            if truth_kind==7 and price<=4:truechance=0.
            accepted=draws[t,5]<truechance
            records[t,0]=m;records[t,1]=d;records[t,2]=x;records[t,3]=-1 if offered else 0
            records[t,4]=price;records[t,5]=1 if accepted else 0
        else:
            offered=observations[t,0]>0;price=observations[t,1];accepted=observations[t,2]>0
        # Advance all latent candidate ledgers without any evaluator state input.
        old_shutdown=np.maximum(shutdown-1,0);next_shutdown=old_shutdown.copy()
        before_model=np.zeros(J);after_model=np.zeros(J)
        for j in range(J):
            kind=int(params[j,0]);weight=params[j,1];duration=int(params[j,2])
            for z in range(K):
                money[j,z]=min(30,money[j,z]+(1 if np.random.random()<qin else 0))
                need[j,z]=min(6,need[j,z]+(1 if np.random.random()<qn else 0)+(1 if np.random.random()<qn else 0))
                if record_seen:
                    po=.25 if stock[j,z]>0 else 0.
                    weights[j,z]*=po if offered else 1-po
            if record_seen:
                norm=weights[j].sum()
                if norm>0:weights[j]/=norm;ll[j]+=math.log(norm)
                else:weights[j]=1./K;ll[j]=-math.inf
            for z in range(K):
                vm=min(30,money[j,z]+1);pv=0.
                for pr in range(len(prices)):
                    pv+=price_prob[pr]*chance(vm,need[j,z],prices[pr],kind,weight,old_shutdown[j,z],reference)
                before_model[j]+=weights[j,z]*pv
                if record_seen:
                    po=chance(money[j,z],need[j,z],price,kind,weight,old_shutdown[j,z],reference) if offered else 0.
                    weights[j,z]*=po if accepted else 1-po
                    take=accepted;has=offered;pp=price
                else:
                    has=np.random.random()<.25 and stock[j,z]>0
                    pp=public_prices[choose(np.random.random(),public_prob)]
                    po=chance(money[j,z],need[j,z],pp,kind,weight,old_shutdown[j,z],reference) if has else 0.
                    take=np.random.random()<po
                if take:money[j,z]=min(30,money[j,z]+pp);stock[j,z]=max(0,stock[j,z]-1)
                if kind==3 and has and pp>=4:next_shutdown[j,z]=duration+1
        model_before=normalize_models(ll,prior);before=np.sum(model_before*before_model)
        for j in range(J):
            if record_seen:
                norm=weights[j].sum()
                if norm>0:weights[j]/=norm;ll[j]+=math.log(norm)
                else:weights[j]=1./K;ll[j]=-math.inf
            for z in range(K):
                vm=min(30,money[j,z]+1);pv=0.
                for pr in range(len(prices)):
                    pv+=price_prob[pr]*chance(vm,need[j,z],prices[pr],int(params[j,0]),params[j,1],old_shutdown[j,z],reference)
                after_model[j]+=weights[j,z]*pv
        model=normalize_models(ll,prior);pred=np.sum(model*after_model);prediction[t]=pred
        delta[t]=pred-before if record_seen else 0.
        if generate_mode:
            if accepted:m=min(30,m+price);x=max(0,x-1)
            public_after[t]=m;invite=pred<1/(1+loss);m=min(30,m+(1 if invite else 0))
            temptation=prices[choose(draws[t,6],price_prob)];private_price[t]=temptation
            truechance=chance(m,d,temptation,0 if truth_kind==7 else truth_kind,truth_weight,0,reference)
            defect=invite and draws[t,7]<truechance
            if defect:m=min(30,m+temptation)
            spend=draws[t,8]<spending(m,d,truth_kind,truth_weight,reference)
            records[t,6]=m;records[t,7]=1 if spend else 0;m-=1 if spend else 0;d-=1 if spend else 0
        else:invite=observations[t,3]>0;defect=observations[t,4]>0
        invited[t]=1 if invite else 0;defects[t]=1 if defect else 0
        for j in range(J):
            kind=int(params[j,0]);weight=params[j,1]
            if invite:
                for z in range(K):
                    money[j,z]=min(30,money[j,z]+1);law=np.empty(len(prices));pv=0.
                    for pr in range(len(prices)):
                        law[pr]=price_prob[pr]*chance(money[j,z],need[j,z],prices[pr],kind,weight,old_shutdown[j,z],reference);pv+=law[pr]
                    weights[j,z]*=pv if defect else 1-pv
                    if defect:
                        price_index=choose(np.random.random(),law/pv if pv>0 else price_prob)
                        money[j,z]=min(30,money[j,z]+prices[price_index])
                norm=weights[j].sum()
                if norm>0:weights[j]/=norm;ll[j]+=math.log(norm)
                else:weights[j]=1./K;ll[j]=-math.inf
            # Hidden spending and a systematic resample after the observed updates.
            for z in range(K):
                spend=np.random.random()<spending(money[j,z],need[j,z],kind,weight,reference)
                if spend:money[j,z]-=1;need[j,z]-=1
            ess=1/np.sum(weights[j]*weights[j])
            if ess<K*.5:
                cm=np.cumsum(weights[j]);start=np.random.random()/K;indices=np.empty(K,np.int64);index=0
                for z in range(K):
                    u=start+z/K
                    while index<K-1 and cm[index]<u:index+=1
                    indices[z]=index
                money[j]=money[j,indices];need[j]=need[j,indices];stock[j]=stock[j,indices]
                next_shutdown[j]=next_shutdown[j,indices];weights[j]=1./K
        shutdown=next_shutdown;ll_path[t]=ll
    return records,public_after,private_price,invited,defects,prediction,delta,ll_path

def args(s):
    params=ledger.candidates()
    public_prices,prob=([1,2],[.5,.5]) if s["regime"]=="LOW" else ([1,2,4,8],[.4,.3,.2,.1])
    qin,qn=(.8,.15) if s["environment"]=="plenty" else (.3,.3)
    return (params,x3.model_prior(params),x3.PRICES,x3.price_law(s["tail"]),np.array(public_prices),
            np.array(prob),qin,qn,s["loss"],s["particles"],s["arm"]=="RECORD")
def observed(raw,w,arm):
    T=raw["records"].shape[1];out=np.zeros((T,5),np.int16)
    if arm=="RECORD":
        out[:,0]=raw["records"][w,:,3]==-1;out[:,1]=raw["records"][w,:,4]
        out[:,2]=raw["records"][w,:,5]
    out[:,3]=raw["entrusted"][w];out[:,4]=raw["defects"][w]
    return out
def generate(s,lane="study",pulse=lambda **kw:None):
    rr=rng("V-hidden-world",lane,{k:v for k,v in s.items() if k not in ("arm","particles")})
    draws=rr.random((s["worlds"],s["steps"],9))
    seeds=rng("V-hidden-filter",lane,s).integers(1,2**31-1,size=s["worlds"])
    collected=[[] for _ in range(8)]
    for w in range(s["worlds"]):
        out=kernel(*args(s),s["truth"][0],float(s["truth"][1]),draws[w],
                   np.zeros((s["steps"],5),np.int16),int(seeds[w]),True)
        for dest,value in zip(collected,out):dest.append(value)
        pulse(world=w+1)
    names=("records","public_after","private_price","entrusted","defects","prediction","forecast_change","likelihood_path")
    result={key:np.array(value) for key,value in zip(names,collected)}
    result["draws"]=draws;result["filter_seed"]=seeds;result["partner_likelihood"]=result["likelihood_path"][:,-1]
    return result
def reconstruct(raw,s,reference=True,particles=None):
    ss=dict(s,particles=particles or s["particles"]);pred=[];change=[];ll=[]
    for w in range(s["worlds"]):
        # No evaluator ledger, type, private price or physical uniforms are passed.
        out=kernel(*args(ss),-1,0.,np.zeros((s["steps"],9)),observed(raw,w,s["arm"]),
                   int(raw["filter_seed"][w]),False,reference)
        pred.append(out[5]);change.append(out[6]);ll.append(out[7])
    return np.array(pred),np.array(change),np.array(ll)
def infer(raw,s,reference=False,pulse=lambda **kw:None):
    from .v import metrics
    if reference:pred,change,ll=reconstruct(raw,s);last=ll[:,-1]
    else:pred=raw["prediction"];change=raw["forecast_change"];last=raw["partner_likelihood"]
    invite=(pred<1/(1+s["loss"])).astype(float)
    result=metrics(raw,s,last,pred,change,invite)
    result["_decision_matches"]=(invite==raw["entrusted"]).astype(float)
    return result
def controls(dest=None,pulse=lambda **kw:None):
    checks=[]
    for kind in range(4):
        a=[];b=[]
        for m in (0,5,30):
            for d in (0,6):
                for price in (1,8,32):
                    a.append(chance(m,d,price,kind,8.,0,False));b.append(chance(m,d,price,kind,8.,0,True))
        checks.append(dict(name="independent latent choice arithmetic "+str(kind),passed=bool(np.allclose(a,b,atol=1e-14))))
    checks.append(dict(name="zero tail aliases live20 and rule private behavior",passed=all(
        chance(m,d,price,0,20.,0)==.02 for m in (1,6,30) for d in (0,6) for price in (2,4,8,16))))
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
