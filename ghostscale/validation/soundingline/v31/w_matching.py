"""Excluded prospective matching, with an independent validation panel."""
import itertools
import numpy as np
from .common import rng,write
from .w_settings import BASE,simulate

BOUNDS=((.05,.95),(.05,2.),(.05,8.),(.05,4.))

def statistics(path):
    return np.array([np.mean(path[1:,4]!=path[:-1,4])*100,path[:,9].mean()*1000,np.mean(path[:,5]*(path[:,4]==path[:,0]))])

def panel(s,theta,lane,n=4,steps=4096):
    scores=[]
    for j in range(n):
        g=rng('W-matching',s['volatility'],s['reliability'],s['actionable'],lane,j)
        draws=g.random((steps,11));draws[:,10]=g.normal(size=steps)
        scores.append(statistics(simulate(draws,s['volatility'],s['reliability'],s['actionable'],theta)))
    return np.array(scores)

def match(s,left,right,fraction):
    metric=1 if 3 in (left,right) else 0 if (left,right)==(0,1) else 2
    baseline=panel(s,BASE,'fit');base=baseline[:,metric].mean();target=base*(1+fraction if metric==0 else 1-fraction)
    fitted=[];fit_records=[]
    for idx in (left,right):
        cache={}
        def evaluate(value):
            if value not in cache:
                theta=BASE.copy();theta[idx]=value;cache[value]=panel(s,theta,'fit')[:,metric].mean()
            return cache[value]
        grid=sorted(set([float(BASE[idx])]+list(np.linspace(*BOUNDS[idx],17))))
        vals=np.array([evaluate(x) for x in grid]);brackets=[(grid[i],grid[i+1]) for i in range(len(grid)-1) if (vals[i]-target)*(vals[i+1]-target)<=0]
        if not brackets:return dict(passed=False,reason='target outside measured parameter response',metric=int(metric),target=float(target),parameter=idx,curve=[[float(x),float(y)] for x,y in zip(grid,vals)])
        lo,hi=min(brackets,key=lambda ab:min(abs(ab[0]-BASE[idx]),abs(ab[1]-BASE[idx])))
        for _ in range(12):
            mid=(lo+hi)/2
            if (evaluate(lo)-target)*(evaluate(mid)-target)<=0:hi=mid
            else:lo=mid
        value=min(cache,key=lambda x:abs(cache[x]-target));theta=BASE.copy();theta[idx]=value
        fitted.append(theta);fit_records.append(dict(parameter=idx,value=float(value),target=float(target),estimate=float(cache[value])))
    freshbase=panel(s,BASE,'validation',n=16,steps=5000)[:,metric]
    fresh=[panel(s,theta,'validation',n=16,steps=5000)[:,metric] for theta in fitted]
    difference=fresh[0]-fresh[1];se=float(difference.std(ddof=1)/4)
    # The match is numerical, not a scientific acceptance gate: retain any
    # genuine remaining difference and refuse the comparison above this band.
    tolerance=max(.01*abs(float(freshbase.mean())),3*se,1e-8)
    target_errors=[float(np.mean(x-freshbase*(1+fraction if metric==0 else 1-fraction))) for x in fresh]
    target_tolerances=[max(.02*abs(float(freshbase.mean())),3*float((x-freshbase).std(ddof=1)/4),1e-8) for x in fresh]
    passed=abs(float(difference.mean()))<=tolerance and all(abs(e)<=t for e,t in zip(target_errors,target_tolerances))
    return dict(passed=passed,reason=None if passed else 'independent matching panel failed',metric=int(metric),fraction=fraction,models=[x.tolist() for x in fitted],fit=fit_records,
        independent_means=[float(x.mean()) for x in fresh],independent_pair_difference=float(difference.mean()),pair_tolerance=tolerance,target_errors=target_errors,target_tolerances=target_tolerances,
        fit_worlds=4,fit_steps=4096,validation_worlds=16,validation_steps=5000)

def validate(s,dest,pulse=lambda **kw:None):
    result=match(s,s['left'],s['right'],s['fraction']);write(dest/'MATCH.json',result);pulse(phase='matching_complete')
    # Infeasibility is a recorded design disposition, not an instrument failure.
    return dict(passed=True,matched=result['passed'],matching=result)
