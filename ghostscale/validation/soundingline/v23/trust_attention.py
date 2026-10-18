"""N15: source-specific precision tracking versus global vigilance."""
import numpy as np
from scipy.special import logsumexp
from .common import rng
from . import finite_common as c
FAMILY='N15';CLASSES=2

def variants():return [dict(id=f'N15-{m}-{n}',family=FAMILY,settings=dict(mechanism=m,history=n)) for m in range(4) for n in (0,8,32)]

def laws(seen,failed,source,stakes):
    checking=np.array([.2,.1+.8*failed[source]/(2+seen[source]),.1+.8*sum(failed)/(2+sum(seen)),.1+.8*stakes])
    return np.transpose(c.noisy(c.categorical(checking)),(1,0,2))

def history_state(p,product=False):
    seen=np.zeros(2);failed=np.zeros(2);mass=np.ones((4,16))/64;logmass=np.zeros((4,16))
    for row in p['observations']:
        source,failure,stakes=row['context'];source=int(source);seen[source]+=1;failed[source]+=failure
        factor=laws(seen,failed,source,stakes)[:,:,row['y']]
        if product:mass*=factor;mass/=mass.sum()
        else:
            logmass+=np.log(factor);logmass-=logsumexp(logmass)
    return (mass if product else c.normalize(logmass)),seen,failed

def probe_law(seen,failed,context):
    source,stakes=context;source=int(source);seen=seen.copy();failed=failed.copy();seen[source]+=1;failed[source]+=1
    return laws(seen,failed,0,stakes)

def posterior(p,y,same_only=False,product=False):
    mass,seen,failed=history_state(p,product);logmass=np.log(mass)
    for context,outcome in zip(p['query'],y,strict=True):
        if same_only and context[0]!=0:continue
        factor=probe_law(seen,failed,context)[:,:,outcome]
        if product:mass*=factor;mass/=mass.sum()
        else:
            logmass+=np.log(factor);logmass-=logsumexp(logmass)
    return mass if product else c.normalize(logmass)

def sample(v,phase,world,case):
    m=v['settings']['mechanism'];n=v['settings']['history'];r=rng(FAMILY,'paired',n,phase,world,case)
    seen=np.zeros(2);failed=np.zeros(2);obs=[]
    failures=r.random(n)<.25;stakes=r.integers(2,size=n)
    for f,s,u in zip(failures,stakes,r.random(n),strict=True):
        seen[0]+=1;failed[0]+=f;context=[0,int(f),int(s)]
        obs.append(dict(context=context,y=c.draw(laws(seen,failed,0,s)[m,world%16],u)))
    contexts=[[0,0],[0,1],[0,0],[1,0],[1,1],[1,0]];query=[contexts[int(i)] for i in r.permutation(6)]
    y=[c.draw(probe_law(seen,failed,q)[m,world%16],u) for q,u in zip(query,r.random(6),strict=True)]
    return dict(design=dict(history=n),observations=obs,query=query),dict(y=y,mechanism=m,law=world%16)

def evaluate(v,p,t):
    before,seen,failed=history_state(p);after=posterior(p,t['y']);same=posterior(p,t['y'],True)
    if not np.allclose(after,posterior(p,t['y'],product=True),atol=1e-12,rtol=0):raise ValueError('N15 inference disagreement')
    qb,qa,qs=[x.sum(1) for x in (before,after,same)]
    forecast=np.stack([np.einsum('ml,mlc->c',before,probe_law(seen,failed,q)) for q in p['query']])
    return dict(exact=forecast),dict(before=qb.tolist(),after=qa.tolist(),same_source=qs.tolist()),dict(primary=c.mechanism_gain(qb,qa,t['mechanism']),same_source_gain=c.mechanism_gain(qb,qs,t['mechanism']))

def features(v,p):return c.features(p,32,3,2,('history',))
def emit(p,schema,identity):return c.emit(p,schema,identity,2)
decode=c.decode

def controls():
    minimum=1.
    for n in (0,8,32):
        p,t=sample(dict(settings=dict(mechanism=1,history=n)),'controls',15,3)
        a=posterior(p,t['y'],True);assert np.array_equal(a[1],a[2])
    p=dict(design=dict(history=0),observations=[],query=[]);prior,seen,failed=history_state(p)
    assert np.array_equal(probe_law(seen,failed,[0,0])[1],probe_law(seen,failed,[0,0])[2])
    assert not np.array_equal(probe_law(seen,failed,[1,0])[1],probe_law(seen,failed,[1,0])[2])
    for m in range(4):
        for law in range(16):
            query=[];ys=[]
            for context in ([0,0],[0,1],[1,0],[1,1]):
                probability=probe_law(seen,failed,context)[m,law]
                for u in (np.arange(192)+.5)/192:query.append(context);ys.append(c.draw(probability,u))
            fixture=dict(p,query=query);q=posterior(fixture,ys)
            assert np.allclose(q,posterior(fixture,ys,product=True),atol=1e-12,rtol=0)
            value=float(q[m].sum());minimum=min(minimum,value);assert value>.25
    return dict(passed=True,same_source_exact_alias=True,second_source_changes_law=True,minimum_planted_posterior=minimum,
        probability_products_agree=True,evaluator_labels_excluded=True,miniature_architecture_untested=True)
