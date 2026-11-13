"""Chosen rises under a latent external-shock alternative; finite exact readers."""
from pathlib import Path
import math
import numpy as np
from scipy.special import logsumexp
from numba import njit
from .common import rng,logistic,class_evidence,sustained,mean_blocks,censored_blocks,conditional_blocks,write,digest,audit_controls
from .ledger_reference import lse


def candidates():
    return np.array([(0,0,0)]+[(1,l,r) for l in (1,2,3,4,5) for r in (.05,.1,.2,.3)]+[(2,0,r) for r in (.05,.1,.2,.3)],float)


def settings():
    truths=[(0,0,0)]+[(1,l,r) for l in (2,4) for r in (.1,.3)]+[(2,0,r) for r in (.1,.3)]
    return [dict(part='Q1',truth=list(truth),shock=q,pressure=p,steps=1024,worlds=200,blocks=8) for truth in truths for q in (.02,.05,.10) for p in (.5,1.,1.5)]+[dict(part='Q1b',truth=truth,pressure=p,steps=1024,rate=.2,worlds=200,blocks=8) for truth in (0,1) for p in (.5,1.,1.5)]


def ride_prob(level,params,s):
    base=params[None,:,2]*logistic(s.get('slope',1.5)*(params[None,:,1]-level[:,None]),0.)
    base=np.where(params[None,:,0]==2,params[None,:,2]/2,base)
    base=np.where(params[None,:,0]==0,0.,base)
    lapse=s.get('tremble',0.)
    return np.clip(base,lapse,1-lapse)


def generate(s,lane='study',worlds=None):
    n=worlds or s['worlds'];r=rng('Q1',lane,s);steps=s['steps'];d=np.zeros((n,3),int);y=np.zeros(n,int);remaining=np.zeros(n,int)
    deficits=np.empty((n,steps,3),np.int8);before=np.empty((n,steps),np.int16);after=np.empty_like(before);rides=np.zeros_like(before,np.int8);shocks=np.zeros_like(rides);offers=np.full_like(rides,-1)
    for t in range(steps):
        d=np.minimum(5,d+(r.random((n,3))<np.array([.2,.15,.1])*s['pressure']));deficits[:,t]=d;level=d.sum(1);before[:,t]=y
        if s['part']=='Q1':
            y=np.maximum(0,y-1);shock=(r.random(n)<s['shock'])*r.choice([1,2,3],n,p=[.3,.4,.3]);shocks[:,t]=shock;y+=shock
            prob=ride_prob(level,np.array([s['truth']],float),s)[:,0];act=(y==0)&(r.random(n)<prob);y+=2*act
        else:
            remaining=np.maximum(0,remaining-1);eligible=remaining==0;kind=r.integers(4,size=n);offers[:,t]=np.where(eligible,kind,-1)
            threshold=np.full(n,3.) if s['truth']==0 else np.array([2.,3.,4.,5.])[kind]
            prob=np.clip(s.get('rate',.2)*logistic(s.get('slope',1.5)*(threshold-level),0.),s.get('tremble',0.),1-s.get('tremble',0.))
            act=eligible&(r.random(n)<prob);remaining=np.where(act,np.array([1,2,4,8])[kind],remaining);y=np.where(act,kind+1,np.maximum(0,y-1))
        rides[:,t]=act;after[:,t]=y
        choose=np.argmax(d*np.array([1.,1.5,3.]),axis=1);serve=(d.sum(1)>0)&(r.random(n)<.9);d[np.arange(n),choose]-=serve
    return dict(deficits=deficits,arousal_before=before,arousal_after=after,rides=rides,shocks=shocks,ride_offers=offers)


def observation_prob(level,before,after,params,s):
    p=ride_prob(level,params,s);z=np.maximum(0,before-1);delta=after-z;q=s['shock']
    base=np.where(delta==0,1-q,np.where(delta==1,.3*q,np.where(delta==2,.4*q,np.where(delta==3,.3*q,0.))))[:,None]
    eligible=z==0
    return np.where(eligible[:,None],np.where(after[:,None]==0,(1-q)*(1-p),np.where(after[:,None]==2,.4*q+(1-q)*p,base)),base)


@njit(cache=True)
def reference_main(level,before,after,params,truth,shock,slope,lapse):
    n,steps=level.shape;m=len(params);classes=params[:,0].astype(np.int64);ll=np.zeros((n,m));pair=np.full((n,3),-1);odds=np.zeros((n,3));same=np.ones((n,3),np.bool_);false=np.zeros(n)
    for w in range(n):
        for t in range(steps):
            z=max(0,before[w,t]-1);out=after[w,t];change=out-z
            for j in range(m):
                kind,threshold,rate=params[j]
                p=0. if kind==0 else (rate/2 if kind==2 else rate/(1+math.exp(-slope*(threshold-level[w,t]))))
                p=min(1-lapse,max(lapse,p));prob=0.
                if change==0:prob=1-shock
                elif change==1:prob=.3*shock
                elif change==2:prob=.4*shock
                elif change==3:prob=.3*shock
                if z==0:
                    if out==0:prob=(1-shock)*(1-p)
                    elif out==2:prob=.4*shock+(1-shock)*p
                ll[w,j]+=math.log(prob) if prob else -math.inf
            e=np.empty(3)
            for k in range(3):e[k]=lse(ll[w,classes==k])-math.log(np.sum(classes==k))
            for k in range(3):
                odds[w,k]=e[truth]-e[k]
                if odds[w,k]<3:pair[w,k]=t
                if abs(odds[w,k])>=1e-12:same[w,k]=False
            if max(e[1],e[2])-e[0]>=3:false[w]=1.
        same[w,truth]=False
    return pair+2,odds,ll,same.astype(np.float64),false


def infer_main(raw,s,reference=False):
    level=raw['deficits'].sum(2);n,steps=level.shape;params=candidates();classes=params[:,0].astype(int);truth=int(s['truth'][0])
    if reference:pair,odds,ll,same,false=reference_main(level,raw['arousal_before'],raw['arousal_after'],params,truth,s['shock'],s.get('slope',1.5),s.get('tremble',0.))
    else:
        ll=np.zeros((n,len(params)));last=np.full((n,3),-1);same=np.ones((n,3),bool);false=np.zeros(n)
        for t in range(steps):
            p=observation_prob(level[:,t],raw['arousal_before'][:,t],raw['arousal_after'][:,t],params,s)
            with np.errstate(divide='ignore'):ll+=np.log(p)
            e=class_evidence(ll,classes);odds=e[:,truth,None]-e;last=np.where(odds<3,t,last);same &= abs(odds)<1e-12;false=np.maximum(false,(e[:,1:].max(1)-e[:,0]>=3))
        pair=last+2;same[:,truth]=False;same=same.astype(float)
    gated=classes==1;posterior=np.exp(ll[:,gated]-logsumexp(ll[:,gated],axis=1,keepdims=True));threshold=posterior@params[gated,1]
    return dict(separation=np.max(np.delete(pair,truth,axis=1),axis=1),pair_separation=pair,final_likelihood=ll,final_class_odds=odds,same_probability=same,false_riding_class=false,recovered_threshold=threshold,ride_rate=raw['rides'].mean(1),ambiguous_two_unit_rises=np.mean((np.maximum(0,raw['arousal_before']-1)==0)&(raw['arousal_after']==2),axis=1))


def infer_shared(raw,s,reference=False):
    n,steps=raw['rides'].shape;ll=np.zeros((n,4,5));last=np.full(n,-1);same=np.ones(n,bool);thresholds=np.arange(1,6);final=np.zeros(n)
    for t in range(steps):
        offer=raw['ride_offers'][:,t];eligible=offer>=0;level=raw['deficits'][:,t].sum(1);act=raw['rides'][:,t]
        if reference:
            for w in range(n):
                if not eligible[w]:continue
                for j in range(5):
                    p=s.get('rate',.2)/(1+math.exp(-s.get('slope',1.5)*(j+1-level[w])));p=min(1-s.get('tremble',0.),max(s.get('tremble',0.),p));ll[w,offer[w],j]+=math.log(p if act[w] else 1-p)
            shared=np.empty(n);separate=np.empty(n)
            for w in range(n):
                shared[w]=float(lse(ll[w].sum(0)))-math.log(5);separate[w]=sum(float(lse(ll[w,k]))-math.log(5) for k in range(4))
        else:
            p=np.clip(s.get('rate',.2)*logistic(s.get('slope',1.5)*(thresholds[None,:]-level[:,None]),0.),s.get('tremble',0.),1-s.get('tremble',0.));inc=np.where(act[:,None],np.log(p),np.log1p(-p))
            ll[np.flatnonzero(eligible),offer[eligible]]+=inc[eligible]
            shared=logsumexp(ll.sum(1),axis=1)-np.log(5);separate=(logsumexp(ll,axis=2)-np.log(5)).sum(1)
        final=separate-shared;odds=final if s['truth']==1 else -final;last=np.where(odds<3,t,last);same &= abs(final)<1e-12
    offers=(raw['ride_offers']>=0).sum(1)
    return dict(separation=last+2,final_likelihood=ll.reshape(n,-1),same_probability=same[:,None].astype(float),separate_minus_shared=final,nats_per_ride_opportunity=final/np.maximum(1,offers),ride_opportunities=offers,rides=raw['rides'].sum(1))


def infer(raw,s,reference=False):return infer_main(raw,s,reference) if s['part']=='Q1' else infer_shared(raw,s,reference)


def run(destination,mode='original',original=None,pulse=lambda **kw:None,selected=None,lane='study'):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);roster=settings() if selected is None else selected;rows=[];files={}
    for i,s in enumerate(roster):
        name=f'cell-{i:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:raw=generate(s,lane);np.savez_compressed(dest/name,**raw);files[name]=digest(dest/name)
        out=infer(raw,s,mode=='reconstruct');np.savez_compressed(dest/f'cell-{i:03d}_measures.npz',**out)
        rows.append(dict(setting=s,measure='separation',**censored_blocks(out['separation'],s['steps'],s['blocks'])))
        if s['part']=='Q1':
            for j,label in enumerate(('NON','GATED','UNGATED')):
                if j!=s['truth'][0]:rows.append(dict(setting=s,measure='pair_separation',rival=label,**censored_blocks(out['pair_separation'][:,j],s['steps'],s['blocks'])))
            for key in ('false_riding_class','recovered_threshold','ride_rate','ambiguous_two_unit_rises'):rows.append(dict(setting=s,measure=key,**mean_blocks(out[key],s['blocks'])))
            if s['truth'][0]==1:rows.append(dict(setting=s,measure='threshold_bias_given_gated_class',**mean_blocks(out['recovered_threshold']-s['truth'][1],s['blocks'])))
        else:
            for key in ('separate_minus_shared','nats_per_ride_opportunity','ride_opportunities','rides'):rows.append(dict(setting=s,measure=key,**mean_blocks(out[key],s['blocks'])))
        pulse(family='Q1',cell=i+1,cells=len(roster))
    result=dict(family='Q1',settings=roster,rows=rows,no_registered_claim=True,fixed='GATED declines as the queue rises by definition; this map tests proposed rules, not their truth about people.')
    write(dest/'SUMMARY.json',result)
    if mode!='reconstruct':write(dest/'RAW_MANIFEST.json',files)
    return result


def controls(destination,pulse=lambda **kw:None):
    checks=[]
    for base in (settings()[0],settings()[13],settings()[40],settings()[-1]):
        s=base|dict(steps=64,worlds=8,blocks=8);raw=generate(s,'known-answer');a=infer(raw,s);b=infer(raw,s,True)
        checks.append(dict(name=str(s)+' independent observation marginal',passed=all(np.allclose(a[k],b[k],rtol=0,atol=1e-9,equal_nan=True) for k in a)))
    params=candidates();s=dict(shock=.05)
    total=sum(observation_prob(np.array([3]),np.array([0]),np.array([j]),params,s) for j in range(4))
    checks.append(dict(name='latent shock and chosen-rise law normalized',passed=bool(np.allclose(total,1,atol=1e-14))))
    checks.append(dict(name='NON rides zero; gated monotonically falls',passed=bool(ride_prob(np.arange(16),params,s)[:,0].sum()==0) and bool(np.all(np.diff(ride_prob(np.arange(16),params,s)[:,params[:,0]==1],axis=0)<=0))))
    s=settings()[0]|dict(steps=32,worlds=8,blocks=8);raw=generate(s,'hidden-label-invariance');changed={k:v.copy() for k,v in raw.items()};changed['rides'][:]=1-changed['rides'];changed['shocks'][:]=3-changed['shocks']
    checks.append(dict(name='reader never uses hidden chosen-rise or shock labels',passed=bool(np.array_equal(infer(raw,s)['final_likelihood'],infer(changed,s)['final_likelihood']))))
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result