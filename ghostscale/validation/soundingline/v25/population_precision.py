"""Resolve eight-decimal ranking boundaries from the exact finite choice model.

Double precision screens only: a near-boundary score is recomputed from integer
counts, rational weights and the original effects at both 50 and 80 digits.
The registered eight-decimal, half-even score and five-percent rule are unchanged.
"""
from decimal import Decimal,localcontext,ROUND_HALF_EVEN
from functools import lru_cache
import numpy as np
from . import population,pieces


@lru_cache(maxsize=128)
def log_laws(effect_key,precision):
    # Cache only deterministic model constants, separately by precision. No
    # observations, scores, random state or setting choices enter this cache.
    with localcontext() as context:
        context.prec=precision;d=Decimal
        weights=[[d(int(v))/10 for v in row] for row in population.TENTHS]
        effects=[[d(v) for v in row] for row in effect_key];logs=[]
        for available in pieces.TRAIN:
            laws=[]
            for w in weights:
                utilities=[3*sum((w[j]*effects[k][j] for j in range(3)),d(0)) for k in available]
                top=max(utilities);exp=[(u-top).exp() for u in utilities];total=sum(exp,d(0))
                laws.append(tuple((x/total).ln() for x in exp))
            logs.append(tuple(laws))
        return tuple(logs)


def rounded_flags(post,e,counts,main,excluded,alpha,placebo,reader):
    raw=population.flags(post)
    scaled=raw*1e8
    candidates=np.flatnonzero(abs(scaled-np.floor(scaled)-.5)<.001)
    rounded=np.round(raw,8)
    if not len(candidates):return rounded
    answers=[]
    for precision in (50,80):
        with localcontext() as context:
            context.prec=precision
            d=Decimal;weights=[[d(int(v))/10 for v in row] for row in population.TENTHS]
            logs=log_laws(tuple(tuple(str(float(v)) for v in row) for row in e),precision)
            cache={}
            def likelihood(row):
                if row not in cache:
                    ll=[sum((d(int(counts[row,si,k]))*logs[si][h][k] for si,a in enumerate(pieces.TRAIN) for k in range(len(a))),d(0)) for h in range(36)]
                    top=max(ll);values=[(v-top).exp() for v in ll];total=sum(values,d(0));cache[row]=[v/total for v in values]
                return cache[row]
            size=len(main)
            if reader=='estimated':members=[likelihood(i) for i in range(size)]
            if reader=='oracle':
                logprior=[sum(((d(str(alpha))*mu-1)*w[j].ln() for j,mu in enumerate((d('.5'),d('.3'),d('.2')))),d(0)) for w in weights]
                top=max(logprior);prior=[(v-top).exp() if not placebo or population.DEVIATION[h]<=1 else d(0) for h,v in enumerate(logprior)]
            resolved=[]
            for row in candidates.tolist():
                probabilities=likelihood(row)
                if reader=='estimated':
                    omitted=row if row<size else int(excluded[row-size]) if row<size+512 else row-size-512
                    prior=[sum((members[i][h] for i in range(size) if i!=omitted),d(0))/(size-1) for h in range(36)]
                if reader!='individual':
                    probabilities=[v*p for v,p in zip(probabilities,prior)];total=sum(probabilities,d(0));probabilities=[v/total for v in probabilities]
                flag=max(sum((probabilities[h] for h in range(36) if population.TAILS[h,j]),d(0)) for j in range(6))
                resolved.append(flag.quantize(d('.00000001'),rounding=ROUND_HALF_EVEN))
            answers.append(resolved)
    if answers[0]!=answers[1]:raise ValueError('population ranking boundary unresolved at 50/80 digits')
    rounded[candidates]=[float(v) for v in answers[1]]
    return rounded
