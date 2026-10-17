"""N9: explicit offers distinguish consideration from preference in a finite world."""
import math
import numpy as np
from .common import rng,normalize,canonical
from . import ledger

FAMILY='N9';CLASSES=4
MECHANISMS=('baseline','fear-pruned','dislike','expert-narrow','attended-local')
RELIABILITY=np.linspace(.60,.95,16)

def variants():
    return [dict(id=f'N9-{w}-{n}-{timing}',family=FAMILY,settings=dict(width=w,history=n,timing=timing))
            for w in (1,2) for n in (0,8,32) for timing in (False,True)]

def intended(m,width,scene,offer):
    # Response = bold*2 + slow. Local omission is integrated as a fair attention draw.
    if m==0:return [3]
    if m==2:return [1]
    if m==4:return [3] if offer else [0,3]
    pruned=scene<(width if m==1 else 1)
    return [0] if pruned and not offer else [3]

def laws(width,contexts):
    out=np.broadcast_to((1-RELIABILITY)[None,:,None,None]/4,(5,16,len(contexts),4)).copy()
    for m in range(5):
        for j,(scene,offer) in enumerate(contexts):
            selected=intended(m,width,scene,offer)
            for y in selected:out[m,:,j,y]+=RELIABILITY/len(selected)
    return out

def sample(v,phase,world,case):
    s=v['settings'];w,n,timing=s['width'],s['history'],s['timing'];m=(case+world)%5
    r=rng(FAMILY,'paired',w,n,timing,phase,world,case//5)
    # Ordinary behavior uses the conditioning scene: fear, expertise and dislike alias in action.
    hist= laws(w,[(0,0)])[m,world%16,0]
    history=[int(np.searchsorted(np.cumsum(hist),u)) for u in r.random(n)]
    if not timing:history=[y//2*2 for y in history]
    contexts=[[s,o] for s in range(3) for o in (0,1)];order=r.permutation(6).tolist()
    contexts=[contexts[i] for i in order];prob=laws(w,contexts)[m,world%16]
    y=[int(np.searchsorted(np.cumsum(p),u)) for p,u in zip(prob,r.random(6),strict=True)]
    return dict(width=w,timing=timing,history=history,query=contexts,probe_order=order),dict(y=y,mechanism=m,law=world%16)

def posterior(p,y=None,mode='all',*,product=False):
    mass=np.ones((5,16))/80;logmass=np.full((5,16),-math.log(80))
    history=laws(p['width'],[(0,0)])[:,:,0]
    if not p['timing']:history=history.reshape(5,16,2,2).sum(-1)
    for observed in p['history']:
        factor=history[:,:,observed if p['timing'] else observed//2]
        if product:mass*=factor;mass/=mass.sum()
        else:logmass+=np.log(factor)
    if y is not None:
        probs=laws(p['width'],p['query'])
        for j,observed in enumerate(y):
            scene,offer=p['query'][j]
            if mode=='offers' and not offer:continue
            if mode in ('timing','actions') and offer:continue
            if mode=='timing' and not p['timing']:continue
            if mode=='timing':factor=probs[:,:,j].reshape(5,16,2,2).sum(-2)[:,:,observed%2]
            elif mode in ('offers','actions') or not p['timing']:factor=probs[:,:,j].reshape(5,16,2,2).sum(-1)[:,:,observed//2]
            else:factor=probs[:,:,j,observed]
            if product:mass*=factor;mass/=mass.sum()
            else:logmass+=np.log(factor)
    return mass/mass.sum() if product else normalize(logmass.ravel()).reshape(5,16)

def features(v,p):
    x=np.zeros((32,5))
    for j,y in enumerate(p['history']):x[j,y]=1;x[j,4]=1
    return np.r_[x.ravel(),p['width'],int(p['timing']),np.array(p['query']).ravel()]

def evaluate(v,p,t):
    before=posterior(p);pred={'exact':np.einsum('ml,mlqc->qc',before,laws(p['width'],p['query']))}
    q=before.sum(1);loss=-math.log(q[t['mechanism']]);detail=dict(before=q.tolist(),after={});metrics={}
    for name in ('all','offers','timing','actions'):
        mass=posterior(p,t['y'],name)
        if not np.allclose(mass,posterior(p,t['y'],name,product=True),atol=1e-12,rtol=0):raise ValueError('N9 probability products differ')
        q=mass.sum(1);detail['after'][name]=q.tolist();metrics['primary' if name=='all' else name+'_gain']=loss+math.log(q[t['mechanism']])
    return pred,detail,metrics

def emit(p,schema,identity):
    e=ledger.Emission(schema,identity)
    for j,y in enumerate(p['history']):e.decision('history',j,list(range(4)),y)
    for j,q in enumerate(p['query']):e.decision('query',j,list(range(4)),None,observed=dict(query=q,probe=p['probe_order'][j]))
    e.decision('design',0,[0],0,observed=dict(width=p['width'],timing=p['timing']))
    return e.finish()

def decode(bundle,schema):
    ledger.validate(bundle,schema);rows=bundle['rows']['Decision']
    p=dict(next(r for r in rows if r['context']['channel']=='design')['context']['observed']);p.update(history=[],query=[],probe_order=[])
    for channel in ('history','query'):
        for r in sorted((r for r in rows if r['context']['channel']==channel),key=lambda r:r['context']['index']):
            if channel=='history':p['history'].append(next(o['position'] for o in r['options'] if o['option_id']==r['taken']))
            else:
                if r['taken'] is not None:raise ValueError('query target leaked')
                p['query'].append(r['context']['observed']['query']);p['probe_order'].append(r['context']['observed']['probe'])
    return p

def controls():
    # At narrow width fear and expertise are genuinely identical: never require their separation.
    if not np.array_equal(laws(1,[[s,o] for s in range(3) for o in (0,1)])[1],laws(1,[[s,o] for s in range(3) for o in (0,1)])[3]):raise ValueError('narrow fear/expertise alias')
    baseline=laws(2,[(0,0)]).reshape(5,16,1,2,2).sum(-1)
    if not np.allclose(baseline[1],baseline[2],atol=1e-15,rtol=0):raise ValueError('baseline action alias')
    recover=[]
    for m in range(5):
        p=dict(width=2,timing=True,history=[],query=[[s,o] for s in range(3) for o in (0,1)],probe_order=list(range(6)))
        y=[intended(m,2,s,o)[0] for s,o in p['query']]
        mass=posterior(p,y);q=mass.sum(1)
        if q[m]<=.2:raise ValueError('changed-probe positive control')
        recover.append(float(q[m]))
        evaluate({},p,dict(y=y,mechanism=m))
    p,t=sample(variants()[-1],'controls',15,1)
    if not np.array_equal(features(variants()[0],p),features(variants()[-1],p)):raise ValueError('hidden setting leak')
    return dict(passed=True,narrow_fear_expertise_aliased=True,baseline_action_alias=True,
        changed_probe_information=True,minimum_planted_posterior=min(recover),independent_mass_agrees=True,
        evaluator_labels_excluded=True,miniature_architecture_untested=True)
