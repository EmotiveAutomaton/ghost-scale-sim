"""Held-out prediction from one, two or three declared standing readouts."""
from pathlib import Path
import math
import numpy as np
from .common import rng,logistic,mean_blocks,conditional_blocks,write,digest,audit_controls
LEVELS=(8,16,32,64,128,256)


def settings():
    return [dict(part='B3',mu=mu,chi=chi,population=pop,steps=512,worlds=200,blocks=8) for mu in (1.,.5,0.) for chi in (0.,.5,1.) for pop in ('independent','opposed')]+[dict(part='B3b',mu=mu,chi=.5,population=pop,steps=512,worlds=200,blocks=8) for mu in (1.,.5,0.) for pop in ('independent','opposed')]


def generate(s,lane='study',worlds=None):
    n=worlds or s['worlds'];steps=s['steps'];r=rng('B3',lane,s);a=r.choice(np.arange(5)/4,n);g=1-a if s['population']=='opposed' else r.choice(np.arange(5)/4,n);warm=r.choice(np.arange(5)/4,n)
    traits=np.column_stack([a,g,warm]);acts=(r.random((n,steps,3))<traits[:,None,:]).astype(np.int8)
    means=(1+acts.cumsum(1))/(2+np.arange(1,steps+1)[None,:,None]);respect=means[:,:,0];trust=means[:,:,1]-s['chi']*respect;liking=means[:,:,2]
    predictors=np.stack([respect+s['mu']*trust-.75,s['mu']*respect+trust-.75],axis=2)
    if s['part']=='B3b':predictors=np.concatenate([predictors,(liking-.5)[:,:,None]],axis=2)
    probabilities=logistic(s.get('slope',2.)*predictors,s.get('tremble',.02));decisions=(r.random(probabilities.shape)<probabilities).astype(np.int8)
    return dict(traits=traits,acts=acts,gauges=np.stack([respect,trust,liking],axis=2),probabilities=probabilities,decisions=decisions)


def projection(logits,rank,reference=False):
    center=logits.mean(0);x=logits-center
    if reference:
        covariance=np.zeros((x.shape[1],x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):covariance[i,j]=math.fsum(float(row[i])*float(row[j]) for row in x)
        values,vectors=np.linalg.eigh(covariance);basis=vectors[:,-rank:];return center+(x@basis)@basis.T
    u,values,v=np.linalg.svd(x,full_matrices=False)
    return center+(u[:,:rank]*values[:rank])@v[:rank]


def predictions(data,s,reference=False):
    n,length,decisions=data.shape
    if reference:
        counts=np.array([[sum(int(v) for v in data[w,:,j]) for j in range(decisions)] for w in range(n)],float)
    else:counts=data.sum(1).astype(float)
    separate=(counts+.5)/(length+1);pooled=(counts[:,:2].sum(1)+.5)/(2*length+1)
    if s['part']=='B3':
        low=np.repeat(pooled[:,None],2,axis=1);high=low.copy() if s['mu']==1 else separate
    else:
        if s['mu']==1:
            low=separate.copy();low[:,:2]=pooled[:,None];high=low.copy()
        else:
            logits=np.log(separate/(1-separate));fitted=projection(logits,2,reference);low=logistic(fitted,0.);high=separate
    return low,high


def infer(raw,s,reference=False):
    data=raw['decisions'];levels=[k for k in LEVELS if 2*k<=s['steps']];n=len(data);gains=[];low_scores=[];high_scores=[]
    for k in levels:
        low,high=predictions(data[:,:k],s,reference);test=data[:,k:2*k]
        if reference:
            ls=np.empty(n);hs=np.empty(n)
            for w in range(n):
                ls[w]=math.fsum(math.log(float(low[w,j]) if act[j] else 1-float(low[w,j])) for act in test[w] for j in range(test.shape[2]))/(k*test.shape[2])
                hs[w]=math.fsum(math.log(float(high[w,j]) if act[j] else 1-float(high[w,j])) for act in test[w] for j in range(test.shape[2]))/(k*test.shape[2])
        else:
            ls=np.where(test,np.log(low[:,None,:]),np.log1p(-low[:,None,:])).mean((1,2));hs=np.where(test,np.log(high[:,None,:]),np.log1p(-high[:,None,:])).mean((1,2))
        gains.append(hs-ls);low_scores.append(ls);high_scores.append(hs)
    # A nested learning curve, not extra independent observations.
    low,high=predictions(data[:,:s['steps']//2],s,reference);one=np.log(low[:,0]/(1-low[:,0]));two=np.log(high[:,:2]/(1-high[:,:2]))
    if s['mu']==1:respect=np.full(n,np.nan);trust=np.full(n,np.nan)
    else:
        total=two/s.get('slope',2.)+.75;respect=(total[:,0]-s['mu']*total[:,1])/(1-s['mu']**2);trust=(total[:,1]-s['mu']*total[:,0])/(1-s['mu']**2)
    return dict(gains=np.stack(gains,axis=1),low_log_score=np.stack(low_scores,axis=1),high_log_score=np.stack(high_scores,axis=1),same_probability=np.full((n,1),float(s['mu']==1)),one_fitted_logit=one,respect_estimate=respect,trust_estimate=trust,levels=np.array(levels))


def run(destination,mode='original',original=None,pulse=lambda **kw:None,selected=None,lane='study'):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);roster=settings() if selected is None else selected;rows=[];files={}
    for i,s in enumerate(roster):
        name=f'cell-{i:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:raw=generate(s,lane);np.savez_compressed(dest/name,**raw);files[name]=digest(dest/name)
        out=infer(raw,s,mode=='reconstruct');np.savez_compressed(dest/f'cell-{i:03d}_measures.npz',**out);first=None
        for j,k in enumerate(out['levels']):
            value=mean_blocks(out['gains'][:,j],s['blocks']);rows.append(dict(setting=s,measure='heldout_gain_per_decision',training_records=int(k),test_records=int(k),**value))
            if first is None and value['interval95'][0]>0:first=int(k)
        rows.append(dict(setting=s,measure='first_pointwise_positive_interval',training_records=first,meaning='First of six nested learning-curve intervals entirely above zero; descriptive, no multiplicity correction.'))
        if s['part']=='B3' and s['mu']!=1:
            for trait in np.arange(5)/4:
                mask=raw['traits'][:,0]==trait
                for key in ('one_fitted_logit','respect_estimate','trust_estimate'):rows.append(dict(setting=s,measure=key,assertiveness=float(trait),**conditional_blocks(out[key],mask,s['blocks'])))
        pulse(family='B3',cell=i+1,cells=len(roster))
    result=dict(family='B3',settings=roster,rows=rows,no_registered_claim=True,reader_limit='Declared finite count fits; B3b compares rank-two and saturated three-readout fits. Not a general architecture test.')
    write(dest/'SUMMARY.json',result)
    if mode!='reconstruct':write(dest/'RAW_MANIFEST.json',files)
    return result


def controls(destination,pulse=lambda **kw:None):
    checks=[]
    for base in (settings()[0],settings()[8],settings()[-1]):
        s=base|dict(steps=64,worlds=16,blocks=8);raw=generate(s,'known-answer');a=infer(raw,s);b=infer(raw,s,True)
        checks.append(dict(name=str(s)+' independent fits and held-out score',passed=all(np.allclose(a[k],b[k],atol=1e-9,rtol=0,equal_nan=True) for k in a)))
    for part in ('B3','B3b'):
        s=dict(part=part,mu=1.,chi=.5,population='independent',steps=64,worlds=16,blocks=8);raw=generate(s,'known-alias');out=infer(raw,s)
        checks.append(dict(name=part+' mu-one equality and no extra gain',passed=bool(np.array_equal(raw['probabilities'][:,:,0],raw['probabilities'][:,:,1])) and bool(np.all(out['gains']==0))))
    s=settings()[-1]|dict(steps=64,worlds=16,blocks=8);raw=generate(s,'heldout');low,high=predictions(raw['decisions'][:,:32],s);changed={k:v.copy() for k,v in raw.items()};changed['decisions'][:,32:]=1-changed['decisions'][:,32:]
    low2,high2=predictions(changed['decisions'][:,:32],s);checks.append(dict(name='held-out outcomes cannot change fitted predictions',passed=bool(np.array_equal(low,low2)) and bool(np.array_equal(high,high2))))
    fixture=np.column_stack([np.arange(20),np.arange(20)**2,np.arange(20)*2+3.]);a=projection(fixture,2);b=projection(fixture,2,True)
    checks.append(dict(name='rank-two known matrix reproduced',passed=bool(np.allclose(a,fixture,atol=1e-10)) and bool(np.allclose(b,fixture,atol=1e-10))))
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result