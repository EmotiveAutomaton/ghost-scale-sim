"""N10: soft correction likelihoods and context-aligned unexplained directions."""
from functools import lru_cache
import math
import numpy as np
from scipy.special import softmax
from .common import rng
from . import finite_common as c
FAMILY='N10';CLASSES=6
VECTORS=np.r_[np.eye(3),-np.eye(3)]
HYPOTHESES=[(hidden,goal,direction) for hidden in (0,1) for goal in (0,1) for direction in (-1,1)]

def variants():
    return [dict(id=f'N10-{s}-{rho}-{noise}',family=FAMILY,settings=dict(share=s,rho=rho,noise=noise)) for s in (0.,.1,.3) for rho in (0.,.5) for noise in (0,1)]

@lru_cache(None)
def laws(share,rho,noise,context):
    beta=(5.,2.)[noise];rows=[]
    for hidden,goal,direction in HYPOTHESES:
        catalog=np.eye(3)[goal];hidden_goal=rho*catalog+math.sqrt(1-rho*rho)*direction*context*np.eye(3)[2]
        base=softmax(beta*(VECTORS@catalog));other=softmax(beta*(VECTORS@hidden_goal))
        rows.append((1-share*hidden)*base+share*hidden*other)
    return np.stack([c.noisy(row) for row in rows])

def posterior(p,mode='residual',product=False):
    d=p['design'];share=d['alternative_share'];logmass=np.full((8,16),-math.log(128));mass=np.ones((8,16))/128
    contexts=[r['context'][0] for r in p['observations']]
    if mode=='shuffled':contexts=contexts[1:]+contexts[:1]
    for row,context in zip(p['observations'],contexts,strict=True):
        law=laws(share,d['rho'],d['noise'],context);y=row['y']
        # Soft-only keeps the catalog projection; +/- third axis becomes one category.
        factor=law[:,:,2]+law[:,:,5] if mode=='soft' and y in (2,5) else law[:,:,y]
        if product:mass*=factor;mass/=mass.sum()
        else:
            term=np.log(factor);logmass+=term-term.max()
    return mass if product else c.normalize(logmass)

def sample(v,phase,world,case):
    s=v['settings'];hidden=int(s['share']>0 and case%2==1);goal=(case//2)%2;direction=(-1,1)[(case//4)%2]
    index=HYPOTHESES.index((hidden,goal,direction));r=rng(FAMILY,phase,s['share'],s['rho'],s['noise'],world,case//2)
    d=dict(alternative_share=s['share'] or .1,rho=s['rho'],noise=s['noise']);observations=[]
    for context,u in zip(r.choice([-1,1],32),r.random(32),strict=True):
        law=laws(d['alternative_share'],d['rho'],d['noise'],int(context))[index,world%16]
        observations.append(dict(context=[int(context)],y=c.draw(law,u)))
    query=[[int(x)] for x in r.choice([-1,1],6)]
    y=[c.draw(laws(d['alternative_share'],d['rho'],d['noise'],q[0])[index,world%16],u) for q,u in zip(query,r.random(6),strict=True)]
    return dict(design=d,observations=observations,query=query),dict(y=y,mechanism=hidden,hypothesis=index,law=world%16)

def evaluate(v,p,t):
    d=p['design'];soft=posterior(p,'soft');full=posterior(p);shuffled=posterior(p,'shuffled')
    if not np.allclose(full,posterior(p,product=True),atol=1e-12,rtol=0):raise ValueError('N10 likelihood disagreement')
    marginal=lambda q:np.array([q[:4].sum(),q[4:].sum()])
    qs,qf,qsh=map(marginal,(soft,full,shuffled));query=np.stack([laws(d['alternative_share'],d['rho'],d['noise'],q[0]) for q in p['query']],axis=2)
    pred=dict(soft=np.einsum('hl,hlqc->qc',soft,query),exact=np.einsum('hl,hlqc->qc',full,query),
        oracle=query[t['hypothesis'],t['law']])
    support=[all(VECTORS[row['y']]@np.eye(3)[g]>0 for row in p['observations']) for g in (0,1)]
    # Direction estimate is evaluated only where a hidden component is planted.
    direction_vector=lambda goal,direction:d['rho']*np.eye(3)[goal]+math.sqrt(1-d['rho']**2)*direction*np.eye(3)[2]
    estimate=sum((full[h].sum()*direction_vector(goal,direction) for h,(hidden,goal,direction) in enumerate(HYPOTHESES) if hidden),np.zeros(3))
    _,goal,direction=HYPOTHESES[t['hypothesis']];true_direction=direction_vector(goal,direction)
    detail=dict(soft=qs.tolist(),residual=qf.tolist(),shuffled=qsh.tolist())
    return pred,detail,dict(primary=c.mechanism_gain(qs,qf,t['mechanism']),soft_empty=0.,hard_empty=float(not any(support)),
        hidden_probability=float(qf[1]),hidden_truth=float(t['mechanism']),shuffled_hidden_probability=float(qsh[1]),
        direction_cosine=float(estimate@true_direction/np.linalg.norm(estimate)) if t['mechanism'] and np.linalg.norm(estimate)>0 else 0.,direction_defined=float(t['mechanism']))

def features(v,p):return c.features(p,32,1,CLASSES,('alternative_share','rho','noise'))
def emit(p,schema,identity):return c.emit(p,schema,identity,CLASSES)
decode=c.decode

def controls():
    minimum=1.
    for law in range(16):
        for hidden in (0,1):
            # A deterministic, balanced strongest-setting planted likelihood fixture.
            d=dict(alternative_share=.3,rho=0.,noise=0);obs=[]
            for context in (-1,1):
                probability=laws(.3,0.,0,context)[HYPOTHESES.index((hidden,0,1)),law]
                for u in (np.arange(192)+.5)/192:obs.append(dict(context=[context],y=c.draw(probability,u)))
            p=dict(design=d,observations=obs,query=[[1]]*6);q=posterior(p);m=np.array([q[:4].sum(),q[4:].sum()])
            minimum=min(minimum,float(m[hidden]));assert m[hidden]>.5
            assert np.allclose(q,posterior(p,product=True),atol=1e-12,rtol=0)
    p,t=sample(variants()[-1],'controls',15,1);assert np.isfinite(evaluate({},p,t)[2]['primary'])
    return dict(passed=True,minimum_planted_posterior=minimum,soft_support_positive=True,
        probability_products_agree=True,evaluator_labels_excluded=True,miniature_architecture_untested=True)

def block_readout(v,world,metrics):
    truth=np.asarray(metrics['hidden_truth']);positive=truth==1;negative=truth==0
    def auc(key):
        if not positive.any() or not negative.any():return None
        p=np.asarray(metrics[key]);difference=p[positive,None]-p[negative][None,:]
        return float(np.mean((difference>0)+.5*(difference==0)))
    return dict(hidden_detection_auc=auc('hidden_probability'),shuffled_detection_auc=auc('shuffled_hidden_probability'),
        direction_cosine_given_hidden=float(np.mean(np.asarray(metrics['direction_cosine'])[positive])) if positive.any() else None,
        hidden_cases=int(positive.sum()),null_cases=int(negative.sum()))
