"""N11: precision, learned habit priors, and reset interventions."""
import math
import numpy as np
from scipy.special import logsumexp
from scipy.special import softmax
from .common import rng
from . import finite_common as c
FAMILY='N11';CLASSES=2
INITIAL=np.array([[1.,1.],[12.,1.],[1.,1.],[12.,1.]])
GAMMA=np.array([4.,.1,4.,2.])
PROBES=[[0,.1,0],[1,1.,1],[1,.5,2]]*2

def variants():return [dict(id=f'N11-{m}-{n}',family=FAMILY,settings=dict(mechanism=m,history=n)) for m in range(4) for n in (0,8,32)]

def rule(e,context):
    best,scale,_=context;value=np.zeros((4,2));value[:,int(best)]=GAMMA*scale
    return softmax(np.log(e)+value,axis=-1)

def laws(e,context):return np.transpose(c.noisy(rule(e,context)),(1,0,2))

def history_state(p,product=False):
    e=INITIAL.copy();logmass=np.zeros((4,16));mass=np.ones((4,16))/64
    for row in p['observations']:
        factor=laws(e,row['context'])[:,:,row['y']]
        if product:mass*=factor;mass/=mass.sum()
        else:
            logmass+=np.log(factor);logmass-=logsumexp(logmass)
        e[2,row['y']]+=1
    return (mass if product else c.normalize(logmass)),e

def infer_probes(prior,e,p,y,kind=None,product=False):
    mass=prior.copy();logmass=np.log(prior)
    for context,outcome in zip(p['query'],y,strict=True):
        if kind is not None and int(context[2])!=kind:continue
        factor=laws(e,context)[:,:,outcome]
        if product:mass*=factor;mass/=mass.sum()
        else:
            logmass+=np.log(factor);logmass-=logsumexp(logmass)
    return mass if product else c.normalize(logmass)

def sample(v,phase,world,case):
    m=v['settings']['mechanism'];n=v['settings']['history'];r=rng(FAMILY,'paired',n,phase,world,case)
    e=INITIAL.copy();obs=[];effort=[]
    for u in r.random(n):
        context=[0,1.,-1];policy=rule(e,context)[m];base=e[m]/e[m].sum()
        effort.append(float(np.sum(policy*np.log(policy/base))));y=c.draw(laws(e,context)[m,world%16],u)
        obs.append(dict(context=context,y=y));e[2,y]+=1
    query=[PROBES[int(i)] for i in r.permutation(6)];y=[c.draw(laws(e,q)[m,world%16],u) for q,u in zip(query,r.random(6),strict=True)]
    return dict(design=dict(history=n),observations=obs,query=query),dict(y=y,mechanism=m,law=world%16,effort=effort)

def evaluate(v,p,t):
    before,e=history_state(p);after=infer_probes(before,e,p,t['y']);qm=before.sum(1);qa=after.sum(1)
    if not np.allclose(after,infer_probes(history_state(p,True)[0],e,p,t['y'],product=True),atol=1e-12,rtol=0):raise ValueError('N11 inference disagreement')
    forecast=np.stack([np.einsum('ml,mlc->c',before,laws(e,q)) for q in p['query']])
    diagnostic=dict(primary=c.mechanism_gain(qm,qa,t['mechanism']))
    for kind,name in enumerate(('pressure','shift','devaluation')):diagnostic[name+'_gain']=c.mechanism_gain(qm,infer_probes(before,e,p,t['y'],kind).sum(1),t['mechanism'])
    effort=t['effort'];diagnostic.update(effort_first=effort[0] if effort else 0.,effort_last=effort[-1] if effort else 0.,effort_observed=float(bool(effort)))
    return dict(exact=forecast),dict(before=qm.tolist(),after=qa.tolist(),effort_trajectory=effort),diagnostic

def features(v,p):return c.features(p,32,3,2,('history',))
def emit(p,schema,identity):return c.emit(p,schema,identity,2)
decode=c.decode

def controls():
    p=dict(observations=[],query=PROBES,design=dict(history=0));q,e=history_state(p)
    assert np.array_equal(e[0],e[2]) and GAMMA[0]==GAMMA[2]
    assert np.allclose(laws(e,PROBES[0])[0],laws(e,PROBES[0])[2])
    assert not np.allclose(laws(e,[1,1.,1]),laws(e,[1,.1,1]))
    minimum=1.
    for m in range(4):
        v=dict(settings=dict(mechanism=m,history=32));p,t=sample(v,'controls',15,0);prior,e=history_state(p)
        for law in range(16):
            contexts=[];outcomes=[]
            for context in PROBES[:3]:
                prob=laws(e,context)[m,law]
                for u in (np.arange(256)+.5)/256:contexts.append(context);outcomes.append(c.draw(prob,u))
            control=dict(p,query=contexts);a=infer_probes(prior,e,control,outcomes);b=infer_probes(prior,e,control,outcomes,product=True)
            assert np.allclose(a,b,atol=1e-12,rtol=0)
            value=float(a[m].sum());minimum=min(minimum,value);assert value>prior[m].sum()
    return dict(passed=True,zero_history_goal_automatization_alias=True,pressure_changes_law=True,minimum_planted_posterior=minimum,
        probability_products_agree=True,evaluator_labels_excluded=True,miniature_architecture_untested=True)
