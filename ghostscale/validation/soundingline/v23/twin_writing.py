"""T1 finite writing decisions with planted preferences and isolated truth.

Shape/timing must be supplied explicitly. Engineering fixtures are not empirical
CoAuthor shape, and this model does not claim human preference recovery.
"""
from itertools import product
import numpy as np
from .common import rng,normalize
from .twin_contract import empirical_quantiles

TYPES=np.array(list(product((-1.,1.),repeat=4)))
HANDLING=('accept','edit','dismiss','other')


def probabilities(row,effect,base):
    x=np.asarray(row['options']);utility=effect*(TYPES@x.T)
    choice=normalize(utility+.1*row['momentum']*x[:,0])
    selected=utility[:,row['selected']]
    fatigue=row['phase']
    handling=normalize(np.log(base)+np.column_stack((selected,-.3*selected+.2*fatigue,-selected,np.zeros(len(TYPES)))))
    gap=np.max(utility,axis=1)-np.partition(utility,-2,axis=1)[:,-2]
    fast=1/(1+np.exp(-gap))
    latency=np.column_stack((fast,1-fast))
    edit=1/(1+np.exp(-effect*(TYPES@np.asarray(row['edit_delta']))))
    correction=.7*edit+.3*.5
    return choice,handling,latency,correction


def maker(v,phase,world,index,shape,*,records=None):
    required=('records_per_maker','sessions_per_maker','options_per_decision','latency_quantiles_s','keystroke_quantiles','handling_marginals')
    if any(k not in shape for k in required):raise ValueError('explicit writing shape required')
    count=shape['records_per_maker'][index] if records is None else records
    if not 1<=count<=shape['records_per_maker'][index]:raise ValueError('invalid engineering prefix')
    sessions=shape['sessions_per_maker'][index];option_shape=shape['options_per_decision']
    if isinstance(option_shape,dict):
        sizes=np.array([int(x) for x in sorted(option_shape,key=int)]);masses=np.array([option_shape[str(x)] if str(x) in option_shape else option_shape[int(x)] for x in sizes],float)
        if np.any(sizes<2) or not np.isfinite(masses).all() or np.any(masses<=0):raise ValueError('invalid empirical option histogram')
        masses/=masses.sum();options=None
    else:options=int(option_shape)
    if sessions<1 or (options is not None and options<2):raise ValueError('invalid sessions/options')
    base=np.array(shape['handling_marginals'],float)
    if base.shape!=(4,) or np.any(base<=0) or not np.isclose(base.sum(),1):raise ValueError('positive handling marginals required')
    r=rng('T1-maker',v['id'],phase,world,index);hidden=int(r.integers(len(TYPES)));effect=v['settings']['personal_effect']
    timing=shape['latency_quantiles_s'];keys=shape['keystroke_quantiles']
    clocks=empirical_quantiles(timing['probabilities'],timing['values'],r,count)
    keystrokes=empirical_quantiles(keys['probabilities'],keys['values'],r,count)
    statement=effect*TYPES[hidden]+.5*np.array([1,0,0,-1])+r.normal(size=4)
    history=[];marks=[]
    def context(i):
        session=min(sessions-1,i*sessions//count)
        offered=options if options is not None else int(r.choice(sizes,p=masses))
        return dict(options=r.integers(-1,2,(offered,4)).tolist(),selected=0,session=session,
            phase=(i*sessions/count)%1,momentum=int(r.choice((-1,1))),edit_delta=r.integers(-1,2,4).tolist())
    for i in range(count):
        row=context(i);choices,*_=probabilities(row,effect,base);row['selected']=int(r.choice(len(row['options']),p=choices[hidden]))
        _,handling,latency,correction=probabilities(row,effect,base)
        action=int(r.choice(4,p=handling[hidden]));bin_=int(r.choice(2,p=latency[hidden]));local=bool(r.random()<.3)
        q=.5 if local else float(1/(1+np.exp(-effect*(TYPES[hidden]@row['edit_delta']))))
        row.update(action=action,latency_bin=bin_,latency_s=float(clocks[i]*(.5 if bin_==0 else 1.5)),
            keystrokes=int(round(keystrokes[i])),corrected=int(r.random()<q),record_id=i,duplicate_of=None)
        noise=v['settings']['dirty'] and r.random()<.02
        if noise:row['action']=int(r.integers(4))
        missing=v['settings']['dirty'] and r.random()<.05
        marks.append(dict(record_id=i,local_fix=local,hidden_action=action,label_noise=bool(noise),missing=bool(missing)))
        if not missing:
            history.append(row)
            if v['settings']['dirty'] and r.random()<.02:history.append(dict(row,record_id=str(i)+'-duplicate',duplicate_of=i))
    query=context(count-1);choices,*_=probabilities(query,effect,base);query['selected']=int(r.choice(len(query['options']),p=choices[hidden]))
    target=probabilities(query,effect,base)[1][hidden];y=int(r.choice(4,p=target))
    return dict(maker=index,history=history,query=query,statement=statement.tolist(),handling_base=base.tolist()),dict(
        type=hidden,preference=TYPES[hidden].tolist(),marks=marks,y=[y],records_assigned=count,sessions=sessions)


def evaluate(v,public,*,statement=True):
    effect=v['settings']['personal_effect'];base=public['handling_base'];lp=np.zeros(len(TYPES));seen=set()
    if statement:
        residual=np.asarray(public['statement'])-.5*np.array([1,0,0,-1])-effect*TYPES
        lp-=.5*np.sum(residual**2,axis=1)
    for row in public['history']:
        key=row['record_id'] if row['duplicate_of'] is None else row['duplicate_of']
        if key in seen:continue
        seen.add(key);choice,handling,latency,correction=probabilities(row,effect,base)
        q=handling[:,row['action']]
        if v['settings']['dirty']:q=.98*q+.02/4
        lp+=np.log(choice[:,row['selected']])+np.log(q)+np.log(latency[:,row['latency_bin']])
        lp+=np.log(correction if row['corrected'] else 1-correction)
    choice,query,*_=probabilities(public['query'],effect,base)
    query_evidence=choice[:,public['query']['selected']];post=normalize(lp+np.log(query_evidence))
    population=query_evidence/query_evidence.sum()
    return {'individual':(post@query)[None,:],'population':(population@query)[None,:]},dict(type_posterior=post.tolist(),
        unique_records=len(seen),exact_scope='finite observed-decision model; timing within bins ancillary; missingness independent')


def verify(v,public,detail):
    """Separate scalar state enumeration and probability multiplication."""
    effect=v['settings']['personal_effect'];base=np.array(public['handling_base']);mass=[]
    soft=lambda x:np.exp(x-np.max(x))/np.exp(x-np.max(x)).sum()
    for pref in TYPES:
        residual=np.array(public['statement'])-.5*np.array([1,0,0,-1])-effect*pref
        logweight=-.5*float(residual@residual);seen=set()
        for row in public['history']:
            key=row['record_id'] if row['duplicate_of'] is None else row['duplicate_of']
            if key in seen:continue
            seen.add(key);options=np.array(row['options']);u=effect*(options@pref);selected=u[row['selected']]
            pc=soft(u+.1*row['momentum']*options[:,0])[row['selected']]
            ph=soft(np.log(base)+np.array([selected,-.3*selected+.2*row['phase'],-selected,0]))[row['action']]
            if v['settings']['dirty']:ph=.98*ph+.005
            ordered=sorted(u);fast=1/(1+np.exp(-(ordered[-1]-ordered[-2])))
            pl=fast if row['latency_bin']==0 else 1-fast
            correction=.7/(1+np.exp(-effect*(pref@row['edit_delta'])))+.15
            logweight+=np.log(pc*ph*pl*(correction if row['corrected'] else 1-correction))
        query=public['query'];options=np.array(query['options']);u=effect*(options@pref)
        logweight+=np.log(soft(u+.1*query['momentum']*options[:,0])[query['selected']])
        mass.append(logweight)
    post=soft(np.array(mass))
    if not np.allclose(post,detail['type_posterior'],atol=1e-11,rtol=0):raise ValueError('writing posterior mismatch')
    return dict(independent_scalar_posterior=True)
