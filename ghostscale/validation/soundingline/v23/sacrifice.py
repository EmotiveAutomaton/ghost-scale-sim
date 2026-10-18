"""N4d: goal ordering under forced conflicts and varying stakes."""
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp
from .common import rng
from . import finite_common as c
FAMILY='N4d';CLASSES=2

def variants():return [dict(id=f'N4d-{m}-{n}',family=FAMILY,settings=dict(mechanism=m,probes=n)) for m in range(4) for n in (0,2,6)]

@lru_cache(None)
def laws(context,hidden=False):
    a,b=context;p=np.array([0.,.5 if hidden or a==b else float(b>a),1.,.5])
    return np.transpose(c.noisy(c.categorical(p)),(1,0,2))

def posterior(p,y,hidden=False,product=False):
    mass=np.ones((4,16))/64;logmass=np.zeros((4,16))
    for context,outcome in zip(p['query'][:p['design']['probes']],y[:p['design']['probes']],strict=True):
        factor=laws(tuple(context),hidden)[:,:,outcome]
        if product:mass*=factor;mass/=mass.sum()
        else:
            logmass+=np.log(factor);logmass-=logsumexp(logmass)
    return mass if product else c.normalize(logmass)

def sample(v,phase,world,case):
    m=v['settings']['mechanism'];n=v['settings']['probes'];r=rng(FAMILY,'paired',phase,world,case)
    # Independent higher/lower stakes; hidden-stakes inference integrates this law.
    query=[[[1.,3.],[3.,1.]][int(i)] for i in r.integers(2,size=6)]
    y=[c.draw(laws(tuple(context))[m,world%16],u) for context,u in zip(query,r.random(6),strict=True)]
    return dict(design=dict(probes=n),observations=[],query=query),dict(y=y,mechanism=m,law=world%16)

def evaluate(v,p,t):
    q=posterior(p,t['y']);hidden=posterior(p,t['y'],True)
    if not np.allclose(q,posterior(p,t['y'],product=True),atol=1e-12,rtol=0):raise ValueError('N4d inference disagreement')
    qm=q.sum(1);qh=hidden.sum(1);prior=np.full(4,.25)
    forecast=np.stack([laws(tuple(context)).mean(axis=(0,1)) for context in p['query']])
    return dict(exact=forecast),dict(after=qm.tolist(),stakes_hidden=qh.tolist()),dict(primary=c.mechanism_gain(prior,qm,t['mechanism']),stakes_hidden_gain=c.mechanism_gain(prior,qh,t['mechanism']))

def features(v,p):return np.r_[np.asarray(p['query']).ravel(),p['design']['probes']]
def emit(p,schema,identity):return c.emit(p,schema,identity,2)
decode=c.decode

def controls():
    p,t=sample(variants()[0],'controls',0,0);baseline=posterior(p,t['y']);assert np.all(baseline==baseline[0,0]) and np.isclose(baseline.sum(),1)
    assert np.array_equal(laws((3.,1.))[0],laws((3.,1.))[1])
    assert not np.array_equal(laws((1.,3.))[0],laws((1.,3.))[1])
    minimum=1.
    for m in range(4):
        for law in range(16):
            contexts=[];ys=[]
            for context in ((1.,3.),(3.,1.)):
                for u in (np.arange(128)+.5)/128:contexts.append(list(context));ys.append(c.draw(laws(context)[m,law],u))
            p=dict(design=dict(probes=len(ys)),observations=[],query=contexts);q=posterior(p,ys)
            assert np.allclose(q,posterior(p,ys,product=True),atol=1e-12,rtol=0)
            value=float(q[m].sum());minimum=min(minimum,value);assert value>.25
    return dict(passed=True,baseline_exact_alias=True,fixed_stakes_alias=True,varying_stakes_change_law=True,minimum_planted_posterior=minimum,
        probability_products_agree=True,evaluator_labels_excluded=True,miniature_architecture_untested=True)
