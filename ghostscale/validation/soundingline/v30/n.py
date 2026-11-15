"""N finite-state attention-only reader; exact marginalization over unseen gaps and ways."""
import itertools
from functools import lru_cache
from pathlib import Path
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from .common import rng,mean_blocks,censored_blocks,sustained,audit_controls,write

STATES=np.array(list(itertools.product(range(7),repeat=3)),np.int16)
BITS=np.array(list(itertools.product((0,1),repeat=3)),np.int8)
POLICIES=("LOUDEST","REACHABLE","WEIGHTED")
def settings():
    return [dict(truth=k,steps=1024,worlds=200,blocks=8,arrival=[.15,.15,.05],
                 opening=opening) for opening in ([.9,.5,.1],[1.,1.,1.]) for k in range(3)]

def choose(gaps,opened,policy,opening):
    if policy==0:return int(np.argmax(gaps))
    if policy==1:return int(np.argmax(np.where(opened,gaps,-1))) if np.any(opened) else int(np.argmax(gaps))
    return int(np.argmax(gaps*np.asarray(opening)))

def generate(s,lane="study",pulse=lambda **kw:None):
    r=rng("N",lane,s);shape=(s["worlds"],s["steps"],3)
    rise=r.random(shape)<s["arrival"];opened=r.random(shape)<s["opening"]
    gaps=np.zeros((s["worlds"],3),np.int16);attention=np.empty(shape[:2],np.int8)
    history=np.empty(shape,np.int8);progress=np.zeros(shape[:2],bool)
    for t in range(s["steps"]):
        gaps=np.minimum(6,gaps+rise[:,t])
        if s["truth"]==0:a=np.argmax(gaps,axis=1)
        elif s["truth"]==1:
            masked=np.where(opened[:,t],gaps,-1)
            a=np.where(opened[:,t].any(1),np.argmax(masked,axis=1),np.argmax(gaps,axis=1))
        else:a=np.argmax(gaps*np.asarray(s["opening"]),axis=1)
        has=(gaps[np.arange(s["worlds"]),a]>0)&opened[np.arange(s["worlds"]),t,a]
        gaps[np.arange(s["worlds"]),a]-=has
        attention[:,t]=a;progress[:,t]=has;history[:,t]=gaps
    return dict(rise=rise,opened=opened,attention=attention,progress=progress,gaps=history)

@lru_cache(None)
def kernels(arrival,opening):
    # T[a][old,new] = joint transition and observed-attention probability.
    outputs=[]
    for policy in range(3):
        rows=[[],[],[]];cols=[[],[],[]];vals=[[],[],[]]
        for i,state in enumerate(STATES):
            for rise in BITS:
                ar=float(np.prod(np.where(rise,arrival,1-np.asarray(arrival))))
                if not ar:continue
                pre=np.minimum(6,state+rise)
                for opened in BITS:
                    op=float(np.prod(np.where(opened,opening,1-np.asarray(opening))))
                    if not op:continue
                    a=choose(pre,opened,policy,opening);post=pre.copy()
                    post[a]-=bool(opened[a] and post[a]>0);j=int(post@[49,7,1])
                    rows[a].append(i);cols[a].append(j);vals[a].append(ar*op)
        mats=[coo_matrix((vals[a],(rows[a],cols[a])),shape=(343,343)).tocsr() for a in range(3)]
        outputs.append(mats)
    return outputs

@njit
def forward_scatter(attention,indptr,indices,data):
    W,T=attention.shape;ll=np.zeros((W,T));N=343
    for w in range(W):
        state=np.zeros(N);state[0]=1.;total=0.
        for t in range(T):
            a=attention[w,t];new=np.zeros(N)
            for old in range(N):
                if state[old]==0:continue
                for k in range(indptr[a,old],indptr[a,old+1]):
                    new[indices[a,k]]+=state[old]*data[a,k]
            z=new.sum()
            if z==0:
                for tt in range(t,T):ll[w,tt]=-np.inf
                break
            state=new/z;total+=np.log(z);ll[w,t]=total
    return ll

def filter_attention(attention,mats,reference=False):
    if reference:
        # Independent column-oriented sparse matrix multiplication across worlds.
        state=np.zeros((len(attention),343));state[:,0]=1
        out=np.zeros(attention.shape);running=np.zeros(len(attention))
        for t in range(attention.shape[1]):
            nxt=np.zeros_like(state)
            for a in range(3):
                mask=attention[:,t]==a
                if mask.any():nxt[mask]=(mats[a].T.tocsc()@state[mask].T).T
            z=nxt.sum(1);valid=z>0
            with np.errstate(divide="ignore"):running+=np.log(z)
            state=np.divide(nxt,z[:,None],out=np.zeros_like(nxt),where=valid[:,None])
            out[:,t]=running
        return out
    length=max(m.nnz for m in mats);indices=np.zeros((3,length),np.int32);data=np.zeros((3,length))
    for a,m in enumerate(mats):indices[a,:m.nnz]=m.indices;data[a,:m.nnz]=m.data
    return forward_scatter(attention,np.array([m.indptr for m in mats]),indices,data)

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    LL=[]
    for k,mats in enumerate(kernels(tuple(s["arrival"]),tuple(s["opening"]))):
        LL.append(filter_attention(raw["attention"],mats,reference));pulse(candidate=k+1)
    ll=np.stack(LL,axis=-1);odds=ll[...,s["truth"]]-np.max(np.delete(ll,s["truth"],axis=-1),axis=-1)
    if not np.isfinite(ll[...,s["truth"]]).all():raise ArithmeticError("truth has impossible observation")
    result=dict(separation=sustained(odds),_ll=ll,
        third_attention_share=(raw["attention"]==2).mean(1),
        no_progress_share=(~raw["progress"]).mean(1),
        returning_without_progress=((raw["attention"][:,1:]==raw["attention"][:,:-1])&~raw["progress"][:,1:]).mean(1),
        total_unmet=raw["gaps"].sum(2).mean(1),infinite_final_odds=np.isposinf(odds[:,-1]).astype(float))
    for k in range(3):result["need_"+str(k)+"_unmet"]=raw["gaps"][:,:,k].mean(1)
    return result

def summarize(s,m):
    return [dict(measure=k,**(censored_blocks(v,s["steps"],s["blocks"]) if k=="separation" else mean_blocks(v,s["blocks"])))
            for k,v in m.items() if not k.startswith("_")]

def pilot_exclusion(s,m):
    if s["opening"]==[1.,1.,1.]:return "all-long-run-ways-open exact policy alias"
    if np.all(m["separation"]<=10):return "all forty separated within ten steps"
    return None

def controls(dest=None,pulse=lambda **kw:None):
    checks=[];a=(.15,.15,.05)
    for op in ((.9,.5,.1),(1.,1.,1.)):
        ks=kernels(a,op)
        checks.append(dict(name="transition mass "+str(op),passed=all(
            np.max(abs(sum(m.sum(axis=1).A.ravel() for m in mats)-1))<1e-12 for mats in ks)))
    checks.append(dict(name="all-open exact alias",passed=all(
        choose(g,np.ones(3,bool),0,(1,1,1))==choose(g,np.ones(3,bool),1,(1,1,1))==
        choose(g,np.ones(3,bool),2,(1,1,1)) for g in STATES)))
    s=dict(truth=0,arrival=list(a),opening=[.9,.5,.1],worlds=8,blocks=8,steps=20)
    raw=generate(s,"known-answer");x=infer(raw,s);y=infer(raw,s,True)
    checks.append(dict(name="independent sparse HMM",passed=bool(np.allclose(x["_ll"],y["_ll"],atol=1e-10))))
    for policy in range(3):
        s2=dict(s,truth=policy);z=generate(s2,"trajectory-control");gap=np.zeros((8,3),int);valid=True
        for t in range(20):
            gap=np.minimum(6,gap+z["rise"][:,t])
            for w in range(8):
                choice=choose(gap[w],z["opened"][w,t],policy,s["opening"])
                valid &= choice==z["attention"][w,t]
                gap[w,choice]-=bool(gap[w,choice] and z["opened"][w,t,choice])
            valid &= np.array_equal(gap,z["gaps"][:,t])
        checks.append(dict(name="independent state recursion "+str(policy),passed=bool(valid)))
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
