"""N12: concurrent goals, replacement, and final authority."""
from functools import lru_cache
import math
import numpy as np
from .common import rng
from . import finite_common as c
FAMILY='N12';CLASSES=2;TIMES=np.arange(3,9)

def variants():return [dict(id=f'N12-{m}-{rate}',family=FAMILY,settings=dict(mechanism=m,conflict=rate)) for m in range(4) for rate in (0.,.2,.5)]

@lru_cache(None)
def laws(context,mode='overlay'):
    a,b,editor,index=context;post=np.array([.5*(a+b),b,a,editor],float)
    if mode in ('single','change'):post[0]=b
    p=np.broadcast_to(post[:,None],(4,6)).copy()
    if mode!='single':p=np.where(index<TIMES[None,:],a,p)
    return np.transpose(c.noisy(c.categorical(p)),(1,2,0,3))

def posterior(p,mode='overlay',product=False):
    mass=np.ones((4,6,16))/384;logmass=np.zeros((4,6,16))
    for row in p['observations']:
        factor=laws(tuple(row['context']),mode)[...,row['y']]
        if product:mass*=factor;mass/=mass.sum()
        else:
            term=np.log(factor);logmass+=term-term.max()
    return mass if product else c.normalize(logmass)

def sample(v,phase,world,case):
    m=v['settings']['mechanism'];rate=v['settings']['conflict'];r=rng(FAMILY,'paired',rate,phase,world,case)
    adoption=int(r.integers(6));a=r.integers(2,size=18);b=a^(r.random(18)<rate);editor=a^(r.random(18)<rate)
    contexts=[[int(a[i]),int(b[i]),int(editor[i]),i] for i in range(18)]
    y=[c.draw(laws(tuple(q))[m,adoption,world%16],u) for q,u in zip(contexts,r.random(18),strict=True)]
    p=dict(design=dict(conflict=rate),observations=[dict(context=q,y=y[i]) for i,q in enumerate(contexts[:12])],query=contexts[12:])
    return p,dict(y=y[12:],mechanism=m,law=world%16,adoption=int(TIMES[adoption]))

def evaluate(v,p,t):
    masses={name:posterior(p,name) for name in ('single','overlay','change')}
    if not np.allclose(masses['overlay'],posterior(p,product=True),atol=1e-12,rtol=0):raise ValueError('N12 inference disagreement')
    qm={name:q.sum(axis=(1,2)) for name,q in masses.items()}
    forecast={name:np.stack([np.einsum('mtl,mtlc->c',q,laws(tuple(context),name)) for context in p['query']]) for name,q in masses.items()}
    forecast['exact']=forecast['overlay'].copy()
    estimated=float(masses['overlay'].sum(axis=(0,2))@TIMES)
    authority=qm['overlay'][2:]/qm['overlay'][2:].sum();defined=t['mechanism']>=2
    detail=dict(posteriors={name:q.tolist() for name,q in qm.items()},adoption_posterior=masses['overlay'].sum(axis=(0,2)).tolist(),final_authority=authority.tolist())
    metrics=dict(primary=c.mechanism_gain(qm['single'],qm['overlay'],t['mechanism']),adoption_absolute_error=abs(estimated-t['adoption']),
        final_authority_error=float(np.argmax(authority)!=t['mechanism']-2) if defined else 0.,authority_defined=float(defined))
    for name,q in qm.items():metrics[name+'_mechanism_log_loss']=-math.log(q[t['mechanism']])
    return forecast,detail,metrics

def features(v,p):return c.features(p,12,4,2,('conflict',))
def emit(p,schema,identity):return c.emit(p,schema,identity,2)
decode=c.decode

def controls():
    p,t=sample(variants()[0],'controls',15,0);q=posterior(p).sum(axis=(1,2));assert np.allclose(q,.25,atol=1e-13,rtol=0)
    minimum=1.
    # Expected-count fixtures under every actual law, with every goal contrast.
    for m in range(4):
        for law in range(16):
            logmass=np.zeros((4,6,16))
            for index in range(12):
                for bits in range(8):
                    context=(bits%2,(bits//2)%2,bits//4,index);candidate=laws(context)
                    true=candidate[m,3,law]
                    logmass+=64*np.sum(true*np.log(candidate),axis=-1)
            value=float(c.normalize(logmass)[m].sum());minimum=min(minimum,value);assert value>.25
    p,t=sample(variants()[-1],'controls',15,4);assert np.allclose(posterior(p),posterior(p,product=True),atol=1e-12,rtol=0)
    return dict(passed=True,zero_conflict_exact_alias=True,minimum_planted_posterior=minimum,probability_products_agree=True,
        evaluator_labels_excluded=True,miniature_architecture_untested=True)
