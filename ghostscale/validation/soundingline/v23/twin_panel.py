"""T4 crossed constructed raters; nominal and ordinal scales are distinct.

No ordering is inferred for Yes/No/Unsure. Shape-only engineering studies cannot
be cited as real-shape results without the separately authenticated realism gate.
"""
from itertools import combinations,product
import numpy as np
from .common import rng,normalize

RECIPES=tuple(combinations(range(4),2))
PROFILES=tuple((r,) for r in RECIPES)+tuple(combinations(RECIPES,2))
TYPES=tuple(product(range(len(PROFILES)),(.6,.9)))


def items(identity,count):return rng('T4-shared-items',identity).uniform(0,2,(count,4))


def ratings(features,semantics):
    if semantics not in ('nominal_yes_no_unsure','ordinal_3'):raise ValueError('explicit supported rating semantics required')
    values=[]
    for profile,reliability in TYPES:
        utility=np.mean([features[:,a]*features[:,b] for a,b in PROFILES[profile]],axis=0)-1
        if semantics=='nominal_yes_no_unsure':
            target=normalize(np.column_stack((-utility,utility,.5-np.abs(utility))))
        else:
            left=1/(1+np.exp(utility+.5));right=1/(1+np.exp(-utility+.5))
            target=np.column_stack((left,1-left-right,right))
        values.append(reliability*target+(1-reliability)/3)
    return np.asarray(values)


def prior(group):
    weights=np.array([2. if any(group in r for r in PROFILES[p]) else 1. for p,_ in TYPES])
    return weights/weights.sum()


def rater(v,phase,world,index,*,semantics,repeat=None,records=None):
    s=v['settings'];r=rng('T4-rater',v['id'],phase,world,index);group=index%2
    x=items((v['id'],phase,world),s['items']+1);hidden=int(r.choice(len(TYPES),p=prior(group)))
    probs=ratings(x,semantics)[hidden]
    flooded=s['flood']=='individual' and index==0 or s['flood']=='group' and group==0
    repetitions=(100 if flooded else 1) if repeat is None else repeat
    if repetitions not in (1,100):raise ValueError('flood uses exactly one or one hundred contributions per item')
    n=s['items'] if records is None else records
    if not 1<=n<=s['items']:raise ValueError('invalid panel prefix')
    history=[];marks=[]
    for repetition in range(repetitions):
        for item in range(n):
            y=int(r.choice(3,p=probs[item]));missing=s['dirty'] and r.random()<.05
            noisy=s['dirty'] and r.random()<.02
            observed=int(r.integers(3)) if noisy else y
            record=dict(item=item,features=x[item].tolist(),rating=observed,record_id=[index,repetition,item],duplicate_of=None)
            marks.append(dict(record_id=record['record_id'],rating=y,missing=bool(missing),label_noise=bool(noisy)))
            if not missing:
                history.append(record)
                if s['dirty'] and r.random()<.02:history.append(dict(record,record_id=record['record_id']+['dup'],duplicate_of=record['record_id']))
    # A finite stated-reason vector names recipe presence with a shared shift.
    presence=np.array([float(recipe in PROFILES[TYPES[hidden][0]]) for recipe in RECIPES])
    reason=presence+.5*np.array([1,0,0,0,0,-1])+r.normal(0,1,len(RECIPES))
    query=x[-1];y=int(r.choice(3,p=probs[-1]))
    return dict(maker=index,group=group,semantics=semantics,history=history,query=query.tolist(),reason=reason.tolist()),dict(
        type=hidden,profile=TYPES[hidden][0],reliability=TYPES[hidden][1],recipe_presence=presence.tolist(),
        y=[y],marks=marks,flood_repetitions=repetitions)


def evaluate(v,public,*,use_reason=True):
    lp=np.log(prior(public['group']));seen=set()
    if use_reason:
        for i,(profile,_) in enumerate(TYPES):
            presence=np.array([float(recipe in PROFILES[profile]) for recipe in RECIPES])
            residual=np.asarray(public['reason'])-.5*np.array([1,0,0,0,0,-1])-presence
            lp[i]-=.5*float(residual@residual)
    for row in public['history']:
        key=tuple(row['duplicate_of'] or row['record_id'])
        if key in seen:continue
        seen.add(key);p=ratings(np.array([row['features']]),public['semantics'])[:,0,row['rating']]
        if v['settings']['dirty']:p=.98*p+.02/3
        lp+=np.log(p)
    post=normalize(lp);q=ratings(np.array([public['query']]),public['semantics'])[:,0]
    return {'individual':(post@q)[None,:],'group-population':(prior(public['group'])@q)[None,:]},dict(
        type_posterior=post.tolist(),unique_ratings=len(seen),semantics=public['semantics'])


def verify(v,public,detail):
    weights=[]
    for state,(profile,reliability) in enumerate(TYPES):
        presence=np.array([float(recipe in PROFILES[profile]) for recipe in RECIPES])
        residual=np.array(public['reason'])-.5*np.array([1,0,0,0,0,-1])-presence
        value=np.log(prior(public['group'])[state])-.5*float(residual@residual);seen=set()
        for row in public['history']:
            key=tuple(row['duplicate_of'] or row['record_id'])
            if key in seen:continue
            seen.add(key);x=row['features'];u=sum(x[a]*x[b] for a,b in PROFILES[profile])/len(PROFILES[profile])-1
            if public['semantics']=='nominal_yes_no_unsure':
                p=np.exp([-u,u,.5-abs(u)]);p/=p.sum()
            else:
                left=1/(1+np.exp(u+.5));right=1/(1+np.exp(-u+.5));p=np.array([left,1-left-right,right])
            p=reliability*p+(1-reliability)/3;q=p[row['rating']]
            if v['settings']['dirty']:q=.98*q+.02/3
            value+=np.log(q)
        weights.append(value)
    post=np.exp(np.array(weights)-max(weights));post/=post.sum()
    if not np.allclose(post,detail['type_posterior'],atol=1e-11,rtol=0):raise ValueError('panel posterior mismatch')
    return dict(independent_scalar_posterior=True)


def population(posteriors,counts):
    """Keep person-balanced and contribution-weighted estimates separate."""
    p=np.asarray(posteriors,float);c=np.asarray(counts,float)
    if p.ndim!=2 or c.shape!=(len(p),) or np.any(c<=0) or not np.allclose(p.sum(1),1):raise ValueError('invalid population evidence')
    return dict(person_balanced=p.mean(0).tolist(),contribution_weighted=np.average(p,axis=0,weights=c).tolist())
