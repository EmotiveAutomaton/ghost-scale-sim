"""Exact enumeration of two trigger tables and remedy-label uncertainty."""
from itertools import product
from pathlib import Path
import math
import numpy as np
from .common import rng,mean_blocks,censored_blocks,sustained,write,digest,audit_controls

POINTS=('body','Knowledge','Tending','Belonging','seen as competent','seen as kind')
MOVES=('sudden drop','drop coming fast','drop already happened','slow drift','closable gap')
REMEDIES=('own push restores','can get clear','only another can undo','cannot be contested','none needed')
CHANNELS=('none','RAGE','FEAR','PANIC/GRIEF','SHAME','SEEKING')
EVENTS=np.array(list(product(range(6),range(5),range(5))),int)


def named(point,move,remedy,table):
    if table==0:
        if move==0 and remedy==0:return 1
        if move==1 and remedy==1:return 2
        if move==2 and remedy==2:return 3
        if point in (4,5) and move in (0,2) and remedy==3:return 4
        if move==4 and remedy==4:return 5
    elif table==1:
        if point==0 and move==1:return 2
        if point==2 and move==0:return 1
        if point==3 and move==0:return 3
        if point in (4,5) and move in (0,2):return 4
        if point==1 and move==4:return 5
    return 0


def table_probabilities(tremble=.1):
    out=np.full((3,len(EVENTS),6),tremble/5)
    for table in range(2):
        for i,(p,m,r) in enumerate(EVENTS):out[table,i,named(p,m,r,table)]=1-tremble
    out[2]=1/6
    return out


def mix_weights(name):
    weight=np.ones(len(EVENTS))
    if name=='social-heavy':weight*=np.where(EVENTS[:,0]>=3,3.,1.)
    if name=='loss-heavy':weight*=np.where(EVENTS[:,1]<=2,3.,1.)
    return weight/weight.sum()


def observed_model(s,reference=False):
    probs=table_probabilities(s.get('tremble',.1));weights=mix_weights(s['mix']);noise=s['noise']
    out=np.zeros_like(probs)
    for o,(point,move,remedy) in enumerate(EVENTS):
        candidates=np.flatnonzero((EVENTS[:,0]==point)&(EVENTS[:,1]==move))
        w=weights[candidates]*np.where(EVENTS[candidates,2]==remedy,1-noise,noise/4)
        if reference:
            for model in range(3):
                for ch in range(6):out[model,o,ch]=math.fsum(float(w[j])*float(probs[model,e,ch]) for j,e in enumerate(candidates))/math.fsum(w)
        else:out[:,o]=np.einsum('e,mec->mc',w,probs[:,candidates])/w.sum()
    return out


def settings():
    return [dict(truth=truth,mix=mix,noise=noise,steps=1024,worlds=200,blocks=8)
            for truth in range(3) for mix in ('even','social-heavy','loss-heavy') for noise in (0.,.2,.4)]


def generate(s,lane='study',worlds=None):
    n=worlds or s['worlds'];r=rng('C1',lane,s);shape=(n,s['steps'])
    events=r.choice(len(EVENTS),size=shape,p=mix_weights(s['mix']))
    table=table_probabilities(s.get('tremble',.1))[s['truth']]
    actions=(r.random(shape)[...,None]>np.cumsum(table[events],axis=-1)).sum(-1).astype(np.int8)
    remedy=EVENTS[events,2];error=r.random(shape)<s['noise']
    alternate=r.integers(4,size=shape);alternate+=alternate>=remedy
    observed=events-remedy+np.where(error,alternate,remedy)
    return dict(true_events=events.astype(np.int16),observed_events=observed.astype(np.int16),actions=actions)


def infer(raw,s,reference=False):
    model=observed_model(s,reference);events=raw['observed_events'];acts=raw['actions'];n,steps=acts.shape
    if reference:
        ll=np.zeros((n,3));last=np.full(n,-1);pairs=np.full((n,3),-1)
        for t in range(steps):
            for j in range(3):ll[:,j]+=np.log(model[j,events[:,t],acts[:,t]])
            odds=ll[:,s['truth'],None]-ll
            last=np.where(np.min(np.delete(odds,s['truth'],axis=1),axis=1)<3,t,last)
            pairs=np.where(odds<3,t,pairs)
    else:
        increments=np.stack([np.log(model[j,events,acts]) for j in range(3)],axis=-1)
        path=np.cumsum(increments,axis=1);ll=path[:,-1]
        delta=path[:,:,s['truth'],None]-path
        last=sustained(np.min(np.delete(delta,s['truth'],axis=2),axis=2))-2
        pairs=np.stack([sustained(delta[:,:,j])-2 for j in range(3)],axis=1)
    return dict(separation=last+2,pair_separation=pairs+2,final_likelihood=ll)


def enumerate_events():
    p=table_probabilities();rows=[]
    for i,(point,move,remedy) in enumerate(EVENTS):
        divergence=float(np.sum(p[0,i]*np.log(p[0,i]/p[1,i])))
        rows.append(dict(id=i,set_point=POINTS[point],movement=MOVES[move],remedy=REMEDIES[remedy],
                         probabilities={name:p[j,i].tolist() for j,name in enumerate(('R15','R14','T-SIZE'))},
                         divergence_nats=divergence,by_construction=True))
    return dict(events=rows,dropped=[],exclusion_basis='No combination is logically impossible at this coarse resolution; unsupported cells remain none.')


def controls(destination,pulse=lambda **kw:None):
    p=table_probabilities();s=dict(truth=0,mix='social-heavy',noise=.4,steps=32,worlds=40,blocks=8)
    raw=generate(s,'known-answer');a=infer(raw,s);b=infer(raw,s,True)
    error=max(float(np.max(abs(a[k]-b[k]))) for k in a)
    checks=[dict(name='all categorical laws normalized',passed=bool(np.allclose(p.sum(2),1))),
            dict(name='slow drift no R15 channel',passed=all(named(int(v[0]),3,int(v[2]),0)==0 for v in EVENTS)),
            dict(name='identical table rows zero information',passed=all(row['divergence_nats']==0 for row in enumerate_events()['events'] if row['probabilities']['R15']==row['probabilities']['R14'])),
            dict(name='independent mixture and path reconstruction',passed=error<1e-9,error=error)]
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result


def run(destination,mode='original',original=None,pulse=lambda **kw:None,selected=None,lane='study'):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);rows=[];files={}
    roster=settings() if selected is None else selected
    for i,s in enumerate(roster):
        name=f'cell-{i:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:
            raw=generate(s,lane);np.savez_compressed(dest/name,**raw);files[name]=digest(dest/name)
        result=infer(raw,s,mode=='reconstruct');np.savez_compressed(dest/f'cell-{i:03d}_measures.npz',**result)
        rows.append(dict(setting=s,measure='separation',**censored_blocks(result['separation'],s['steps'],s['blocks'])))
        for j in range(3):
            if j!=s['truth']:rows.append(dict(setting=s,measure='pair_separation',rival=j,**censored_blocks(result['pair_separation'][:,j],s['steps'],s['blocks'])))
        pulse(family='C1',cell=i+1,cells=len(roster))
    result=dict(family='C1',rows=rows,enumeration=enumerate_events(),settings=roster,no_registered_claim=True)
    write(dest/'SUMMARY.json',result)
    if mode!='reconstruct':write(dest/'RAW_MANIFEST.json',files)
    return result
