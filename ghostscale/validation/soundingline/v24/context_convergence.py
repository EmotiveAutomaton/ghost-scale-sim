"""Exact finite-prior convergence construction for the V24 design card."""
import math

import numpy as np

from ..v23.common import normalize, rng

MECHANISMS=("a","b","c")


def emission(mechanism,index):
    if mechanism not in MECHANISMS or index<0:raise ValueError("unknown mechanism or context")
    if index<2:return .5  # declared finite-context observational alias
    return {"a":.8,"b":.2,"c":.65}[mechanism]


def generate(mechanism,identity,*,length=64):
    if mechanism not in MECHANISMS or length<1:raise ValueError("finite mechanism and positive context required")
    random=rng("v24-context-convergence",identity,mechanism,length)
    observations=[int(random.random()<emission(mechanism,index)) for index in range(length)]
    return dict(identity=identity,observations=observations),dict(mechanism=mechanism)


def read(public,prior,support=MECHANISMS):
    support=tuple(support);prior=np.asarray(prior,float)
    if (not support or len(set(support))!=len(support) or any(x not in MECHANISMS for x in support)
            or prior.shape!=(len(support),) or np.any(prior<=0) or not np.isclose(prior.sum(),1)):
        raise ValueError("positive normalized prior on a unique finite support required")
    log_mass=np.log(prior)
    for index,value in enumerate(public["observations"]):
        for j,mechanism in enumerate(support):
            p=emission(mechanism,index);log_mass[j]+=math.log(p if value else 1-p)
    posterior=normalize(log_mass)
    next_index=len(public["observations"])
    predictive=float(sum(weight*emission(mechanism,next_index)
                         for weight,mechanism in zip(posterior,support,strict=True)))
    return dict(support=list(support),posterior=posterior.tolist(),predictive_one=predictive,
                normalized=bool(np.isclose(posterior.sum(),1)))


def controls():
    aliased,_=generate("a","v24-engineering-excluded-context-alias",length=2)
    first=read(aliased,[.4,.4,.2]);second=read(aliased,[.4,.4,.2])
    planted,_=generate("c","v24-engineering-excluded-context-omitted",length=128)
    full=read(planted,[.8,.1,.1]);different=read(planted,[.1,.1,.8]);omitted=read(planted,[.5,.5],support=("a","b"))
    return dict(
        identical_prior_agreement=first==second,
        exact_observational_alias=bool(np.isclose(first["posterior"][0]/.4,first["posterior"][1]/.4)),
        common_support_predictive_convergence=abs(full["predictive_one"]-different["predictive_one"])<.01,
        known_omitted_support="c" not in omitted["support"] and abs(omitted["predictive_one"]-.5)<.31,
        normalization=all(row["normalized"] for row in (first,full,different,omitted)),
    )
