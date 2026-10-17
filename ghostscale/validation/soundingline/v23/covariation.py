"""N14: declared finite covariation and process observations, not human attribution."""
from functools import lru_cache
import math
import numpy as np
from scipy.special import logsumexp
from .common import rng, normalize, canonical
from . import ledger

FAMILY='N14'
CLASSES=8
MECHANISMS=('attended','scene-habit','one-shot-fear','deliberate-policy','genre')
RELIABILITY=np.linspace(.60,.95,16)
READOUTS=('pervasiveness','covariation','process')


def variants():
    return [dict(id=f'N14-{g}-{b}-{n}',family=FAMILY,settings=dict(genre=g,budget=b,works=n))
            for g in (0,1) for b in (8,24) for n in (1,2,3)]


def intended(m,g,b,trait,char,work,scene,threat):
    """Eight observed responses: fear*4 + slow*2 + revision."""
    if m==0:
        attended=(2*scene+threat)*4+char < b
        return (int(not trait or threat)*4+3) if attended else 4*g
    if m==1:return 4*int(scene==0 or (work>0 and scene==1))
    if m==2:return 4*int(scene==0 or (scene==1 and threat))
    if m==3:return 7
    return 4*g


@lru_cache(maxsize=6)
def tables(b,n):
    # genre x mechanism x law x trait x character x observation x response
    base=np.broadcast_to((1-RELIABILITY)[None,None,:,None,None,None,None]/8,
        (2,5,16,2,4,6*n,8)).copy()
    for g in range(2):
        for m in range(5):
            for trait in range(2):
                for char in range(4):
                    for j in range(6*n):
                        w,rest=divmod(j,6);s,h=divmod(rest,2)
                        base[g,m,:,trait,char,j,intended(m,g,b,trait,char,w,s,h)]+=RELIABILITY
    # Analytic marginal avoids position-dependent summation roundoff in exact aliases.
    action=np.broadcast_to((1-RELIABILITY)[None,None,:,None,None,None,None]/2,
        (2,5,16,2,4,6*n,2)).copy()
    modal=base.argmax(-1)//4
    action+=RELIABILITY[None,None,:,None,None,None,None]*(np.arange(2)==modal[...,None])
    # Distribution of the exact pooled fear count; contexts/order are absent.
    counts=np.zeros((2,5,16,2,4,6*n+1));counts[...,0]=1
    for j in range(6*n):
        p=action[...,j,1]
        old=counts.copy();counts=old*(1-p)[...,None]
        counts[...,1:]+=old[...,:-1]*p[...,None]
    base.setflags(write=False);action.setflags(write=False);counts.setflags(write=False)
    return base,action,counts


def sample(v,phase,world,case):
    s=v['settings'];g,b,n=s['genre'],s['budget'],s['works'];m=(case+world)%5
    r=rng(FAMILY,'paired',g,b,n,phase,world,case//5)
    traits=r.integers(2,size=4).tolist()
    uniforms=r.random((4,6*n));rel=RELIABILITY[world%16]
    law=tables(b,n)[0][g,m,world%16]
    probabilities=law[np.array(traits),np.arange(4)]
    values=(uniforms[:,:,None]>np.cumsum(probabilities,axis=-1)).sum(-1).tolist()
    consensus=(r.random(32)<(.20 if g==0 else .80)).astype(int).tolist()
    query=[[c,n,scene,threat] for c in (case%4,) for scene in range(3) for threat in range(2)]
    y=[]
    for c,w,scene,threat in query:
        p=np.full(8,(1-rel)/8);p[intended(m,g,b,traits[c],c,w,scene,threat)]+=rel
        y.append(int(r.choice(8,p=p)))
    return dict(budget=b,works=n,traits=traits,history=values,consensus=consensus,query=query),dict(y=y,mechanism=m,law=world%16)


def posterior(p,mode,*,product=False):
    b,n=p['budget'],p['works'];full,action,counts=tables(b,n)
    logmass=np.zeros((2,5,16));mass=np.ones((2,5,16))
    for c,history in enumerate(p['history']):
        if mode=='pervasiveness':
            observed=sum(y//4 for y in history)
            factor=counts[:,:,:,:,c,observed].mean(-1)
            logs=np.log(factor)
        else:
            probs=full if mode=='process' else action
            y=np.array(history) if mode=='process' else np.array(history)//4
            likelihood=probs[:,:,:,:,c,np.arange(6*n),y]
            if mode=='process':likelihood=likelihood[:,:,:,p['traits'][c],:]
            if product:
                factor=likelihood.prod(-1)
                if mode=='covariation':factor=factor.mean(-1)
            else:
                logs=np.log(likelihood).sum(-1)
                if mode=='covariation':logs=logsumexp(logs,axis=-1)-math.log(2)
        if product:
            mass*=factor;mass/=mass.sum()
        else:logmass+=logs
    if mode!='pervasiveness':
        k=sum(p['consensus']);N=len(p['consensus'])
        rates=np.array([.20,.80])
        if product:mass*=(rates**k*(1-rates)**(N-k))[:,None,None]
        else:logmass+=(k*np.log(rates)+(N-k)*np.log1p(-rates))[:,None,None]
    return mass/mass.sum() if product else normalize(logmass.ravel()).reshape(2,5,16)


def features(v,p):
    h=np.zeros((4,18,9))
    for c,history in enumerate(p['history']):
        for j,y in enumerate(history):h[c,j,y]=1;h[c,j,8]=1
    return np.r_[h.ravel(),p['traits'],sum(p['consensus'])/32,p['budget']/24,p['works']/3,np.array(p['query']).ravel()/4]


def forecast(p,posterior_mass):
    result=np.zeros((6,8))
    for j,(c,w,s,h) in enumerate(p['query']):
        result[j,:]=np.sum(posterior_mass*(1-RELIABILITY)[None,None,:])/8
        for g in range(2):
            for m in range(5):
                y=intended(m,g,p['budget'],p['traits'][c],c,w,s,h)
                result[j,y]+=np.sum(posterior_mass[g,m]*RELIABILITY)
    return result


def evaluate(v,p,t):
    masses={name:posterior(p,name) for name in READOUTS}
    for name,mass in masses.items():
        if not np.allclose(mass,posterior(p,name,product=True),rtol=0,atol=2e-12):raise ValueError('N14 independent probability products differ')
    post={name:mass.sum((0,2)).tolist() for name,mass in masses.items()}
    losses={name:-math.log(q[t['mechanism']]) for name,q in post.items()}
    metrics=dict(primary=losses['pervasiveness']-losses['process'],
        covariation_gain=losses['pervasiveness']-losses['covariation'],
        process_gain=losses['covariation']-losses['process'])
    # Fractional tie allocation: no arbitrary first-class promotion.
    confusion={}
    for name,q in post.items():
        winners=np.isclose(q,max(q),atol=1e-12,rtol=0)
        row=np.zeros((5,5));row[t['mechanism']]=winners/winners.sum()
        confusion[name]=row.tolist()
    return {'exact':forecast(p,masses['process'])},dict(posteriors=post,confusion=confusion),metrics


def emit(p,schema,identity):
    e=ledger.Emission(schema,identity)
    for c,values in enumerate(p['history']):
        for j,y in enumerate(values):e.decision('history',c*18+j,list(range(8)),y,observed=dict(character=c,record=j))
    for j,q in enumerate(p['query']):e.decision('query',j,list(range(8)),None,observed=dict(query=q))
    e.decision('design',0,[0],0,observed={k:p[k] for k in ('budget','works','traits','consensus')})
    return e.finish()


def decode(bundle,schema):
    ledger.validate(bundle,schema);rows=bundle['rows']['Decision']
    p=dict(next(r for r in rows if r['context']['channel']=='design')['context']['observed'])
    p['history']=[[] for _ in range(4)];p['query']=[]
    for r in sorted(rows,key=lambda r:r['context']['index']):
        channel=r['context']['channel'];o=r['context']['observed']
        if channel=='history':p['history'][o['character']].append(next(x['position'] for x in r['options'] if x['option_id']==r['taken']))
        elif channel=='query':
            if r['taken'] is not None:raise ValueError('query target leaked')
            p['query'].append(o['query'])
    return p


def controls():
    # Independent noisy planted fixtures, strongest grid, every law and mechanism.
    recovered=[]
    for law in range(16):
        for m in range(5):
            v=dict(settings=dict(genre=0,budget=24,works=3))
            p,t=sample(v,'planted-control',law,(m-law)%5)
            assert t['mechanism']==m
            q=posterior(p,'process').sum((0,2))
            if q[m]<=.2:raise ValueError('N14 planted recovery failed')
            recovered.append(float(q[m]))
    p,t=sample(variants()[-1],'controls',15,0)
    evaluate({},p,t)
    if not np.array_equal(features(variants()[0],p),features(variants()[-1],p)):raise ValueError('evaluator setting leaked')
    if any(not np.allclose(tables(b,n)[0].sum(-1),1) for b in (8,24) for n in (1,2,3)):raise ValueError('N14 response mass')
    # Deliberate-policy and genre share action at strong convention, differ in process.
    full,action,_=tables(24,3)
    if not np.array_equal(action[1,3],action[1,4]) or np.array_equal(full[1,3],full[1,4]):raise ValueError('policy/genre alias control failed')
    return dict(passed=True,planted_cases=80,minimum_planted_posterior=min(recovered),
        action_policy_genre_alias=True,process_breaks_alias=True,probability_products_agree=True,
        evaluator_labels_excluded=True,miniature_architecture_untested=True)
