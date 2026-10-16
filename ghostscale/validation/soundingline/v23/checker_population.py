"""Independent N1 reconstruction using scaled joint culture/precision/type mass.

No production likelihood, filtering, posterior, or prediction function is used.
The frozen culture menu is an input, like a declared prior. Exact zero prior
mass stays zero. Full population ownership and leave-one-out evidence are
reconstructed before forecasting either context and recipe presence.
"""
from itertools import combinations, product
from functools import lru_cache
import numpy as np


def catalog():
    recipes=[(i,) for i in range(4)]+list(combinations(range(4),2))
    types=[]
    for count in (1,2):
        for indices in combinations(range(10),count):
            for weights in product((1.,2.),repeat=count):
                row=np.zeros(10)
                for index,weight in zip(indices,weights):row[index]=weight
                types.append(row)
    return recipes,np.array(types)


def normalize(logmass):
    x=np.exp(logmass-np.max(logmass,axis=-1,keepdims=True))
    return x/x.sum(axis=-1,keepdims=True)


@lru_cache(4096)
def _choice_likelihood(key):
    recipes,types=catalog()
    menu=np.asarray(key).reshape(3,4)
    satisfaction=np.array([[np.prod(option[list(recipe)]) for recipe in recipes] for option in menu])
    utility=types@satisfaction.T
    result=np.array([normalize(beta*utility) for beta in (1.,2.,4.)])
    result.flags.writeable=False
    return result


def choice_likelihood(menu,recipes,types):
    # Independently constructed dictionary and probability-space likelihood;
    # no production cache or likelihood is shared with the checked reader.
    menu=np.asarray(menu)
    if menu.shape!=(3,4) or not np.isin(menu,(0,1)).all():
        raise ValueError('invalid finite population menu')
    return _choice_likelihood(tuple(map(int,menu.ravel())))


def filter_joint(menus,choices,priors,drift,recipes,types):
    priors=np.atleast_2d(priors)
    mass=np.broadcast_to(priors[:,None,:]/3,(len(priors),3,len(types))).copy()
    evidence=np.zeros(len(priors))
    for step,(menu,chosen) in enumerate(zip(menus,choices,strict=True)):
        if step:
            mass=(1-drift)*mass+drift*priors[:,None,:]*mass.sum(-1)[:,:,None]
        mass*=choice_likelihood(menu,recipes,types)[None,:,:,int(chosen)]
        scale=mass.sum((1,2))
        if np.any(scale<=0):raise ValueError('independent population support empty')
        evidence+=np.log(scale);mass/=scale[:,None,None]
    return mass,evidence


def reconstruct(v,p,culture_menu,pulse=lambda **kw:None):
    recipes,types=catalog();priors=np.asarray(culture_menu,dtype=float)
    if priors.shape!=(32,200) or np.any(priors<0) or not np.allclose(priors.sum(1),1,atol=1e-12,rtol=0):
        raise ValueError('invalid frozen population menu')
    structure=v['settings']['structure']
    drift=float(structure.split('-')[1]) if structure.startswith('drift-') else 0.
    people=len(p['choices']);conditional=[];evidence=[];individual=[]
    combined=np.vstack([priors,np.full(200,1/200)])
    for i in range(people):
        post,logz=filter_joint(p['menus'][i],p['choices'][i],combined,drift,recipes,types)
        conditional.append(post[:32]);evidence.append(logz[:32]);individual.append(post[32])
        pulse(phase='independent-population',maker=i+1,makers=people)
    conditional=np.array(conditional);evidence=np.array(evidence)
    culture=normalize(evidence.sum(0));personal=np.einsum('c,ncbt->nbt',culture,conditional)
    leave_out=normalize(evidence.sum(0)[None,:]-evidence)
    population_prior=leave_out@priors
    pooled=[]
    for i in range(people):
        post,_=filter_joint(p['menus'][i],p['choices'][i],population_prior[i],drift,recipes,types)
        pooled.append(post[0])
    pooled=np.array(pooled)
    distributions={'law-marginal':personal,'individual-only':np.array(individual),
        'population-only':np.repeat((population_prior/3)[:,None,:],3,axis=1),'partial-pooling':pooled}
    for count in (4,16):
        centers=sorted(range(200),key=lambda t:(-population_prior[:,t].mean(),t))[:count]
        assignments=[min(range(count),key=lambda j:sum((types[t]-types[centers[j]])**2)) for t in range(200)]
        quantized=np.zeros_like(pooled)
        for t,j in enumerate(assignments):quantized[:,:,centers[j]]+=pooled[:,:,t]
        distributions['prototypes-'+str(count)]=quantized
    forecasts={};details={}
    for name,mass in distributions.items():
        outside=[];inside=[]
        for i in range(people):
            outside.append(np.einsum('bt,bta->a',mass[i],choice_likelihood(p['other'][i],recipes,types)))
            inside.append(np.einsum('bt,bta->a',mass[i],choice_likelihood(p['own'][i],recipes,types)))
        forecasts[name]=np.array(outside)
        details[name]=dict(in_context=np.array(inside),recipe_presence=mass.sum(1)@(types>0),
                           top_type=mass.sum(1).max(1))
    return forecasts,dict(readers=details,culture_posterior=culture,population_types=population_prior)


def verify_n1(v,p,truth,prediction,detail,culture_menu,pulse=lambda **kw:None):
    expected,diagnostic=reconstruct(v,p,culture_menu,pulse)
    if set(expected)!=set(prediction):raise ValueError('population reader coverage differs')
    error=0.
    def compare(a,b):
        nonlocal error
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('invalid population reconstruction shape/support')
        error=max(error,float(np.max(np.abs(a-b))))
    for name in expected:
        compare(expected[name],prediction[name])
        for key in diagnostic['readers'][name]:compare(diagnostic['readers'][name][key],detail['readers'][name][key])
    for key in ('culture_posterior','population_types'):compare(diagnostic[key],detail[key])
    if error>1e-10:raise ValueError('independent population posterior mismatch')
    return error
