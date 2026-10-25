"""W8: finite drift filters and an exact finite library of trend trajectories.

Histories are prefixes of one trajectory. U is computed from mean error curves
after whole-world aggregation, never by averaging positive within-case minima.
"""
import math,itertools
from functools import lru_cache
import numpy as np
from . import pieces
from .common import rng,simplex,softmax,posterior,scores,record,close

GRID=simplex(10);DIRECTIONS=list(itertools.permutations(range(3),2));MULTIPLIERS=(0.,.1,.25,.5,1.,2.);WINDOWS=(4,8,16,32,64,128);HISTORIES=(16,32,64,128,512)
NEIGHBORS=np.zeros((66,66))
for i,w in enumerate(GRID):
    js=np.flatnonzero(np.sum(abs(GRID-w),axis=1)<.20000000001);js=js[js!=i];NEIGHBORS[i,js]=1/len(js)
MODELS=[('walk',s) for s in (0.,.002,.005,.01,.02,.05,.1)]+[('jump',h) for h in (0.,1/640,1/256,1/128,1/64,1/32,1/16,1/8)]

def transition(p,kind,rate,independent=False):
    if kind=='jump':return (1-rate)*p+rate*np.sum(p)/66
    moved=np.einsum('i,ij->j',p,NEIGHBORS) if independent else p@NEIGHBORS
    return (1-rate)*p+rate*moved

@lru_cache(None)
def trend_positions(speed):
    # Continuous-valued slope, finite declared initial/slope library; reflect
    # transfer direction at the simplex boundary, then project onto the .1 grid.
    level=np.repeat(GRID,6,axis=0).copy();direction=np.tile(np.array(DIRECTIONS),(66,1));result=[]
    indices=np.arange(len(level));grid_norm=np.sum(GRID**2,axis=1)
    for t in range(512):
        distance=np.sum(level**2,axis=1)[:,None]+grid_norm[None,:]-2*level@GRID.T
        result.append(np.argmax(distance<=distance.min(1)[:,None]+1e-12,axis=1))
        source=direction[:,0];target=direction[:,1];available=level[indices,source];total=available+level[indices,target]
        move=np.minimum(speed,available);remaining=np.where(total>1e-12,speed-move,0.)
        level[indices,source]-=move;level[indices,target]+=move
        level[indices,target]-=remaining;level[indices,source]+=remaining
        reflected=(available<speed)&(total>1e-12);direction[reflected]=direction[reflected,::-1]
        level=np.clip(level,0,1)
    return np.array(result)

def readout(p,current,lifetime):
    mixture=.99*p+.01/66;estimate=p@GRID;metrics=scores(mixture,current)
    metrics.update(log_loss=-math.log(float(mixture[current])),current_squared_error=float(np.sum((estimate-GRID[current])**2)),lifetime_squared_error=float(np.sum((estimate-lifetime)**2)),posterior_variance=float(p@np.sum((GRID-estimate)**2,axis=1)))
    return metrics

def block(setting,identity,independent=False):
    r=rng(*identity);e=pieces.library('partial',r);laws=[softmax(3*(GRID@e[list(a)].T),independent) for a in pieces.TRAIN];out=[];raw=[]
    adaptive_trends=np.concatenate([trend_positions(s) for s in (.005,.01,.02)],axis=1);components=len(MODELS)+3
    for maker in range(8):
        state=int(r.integers(66));initial=state;direction=int(r.integers(6));true_trend=trend_positions(setting['rate'])[:,state*6+direction] if setting['process']=='trend' else None
        adaptive=np.full((len(MODELS),66),1/(components*66));trend_mass=np.full(adaptive_trends.shape[1],1/(components*396));fixed={mult:np.full(66,1/66) for mult in MULTIPLIERS}
        fixed_trends={mult:trend_positions(setting['rate']*mult) for mult in MULTIPLIERS} if true_trend is not None else {}
        trend_fixed_mass={mult:np.full(396,1/396) for mult in MULTIPLIERS};logs=[];states=[];history=[]
        for t in range(512):
            if true_trend is not None:state=int(true_trend[t])
            elif t:
                p=transition(np.eye(66)[state],setting['process'],setting['rate']);state=int(r.choice(66,p=p))
            if t:
                for i,(kind,rate) in enumerate(MODELS):adaptive[i]=transition(adaptive[i],kind,rate,independent)
                if true_trend is None:
                    for mult in MULTIPLIERS:fixed[mult]=transition(fixed[mult],setting['process'],setting['rate']*mult,independent)
            si=t%4;choice=int(r.choice(len(pieces.TRAIN[si]),p=laws[si][state]));likelihood=laws[si][:,choice];logs.append(np.log(likelihood));states.append(state);history.append(dict(situation=si,choice=choice))
            adaptive*=likelihood;trend_mass*=likelihood[adaptive_trends[t]];normal=float(adaptive.sum()+trend_mass.sum());adaptive/=normal;trend_mass/=normal
            for mult in MULTIPLIERS:
                if true_trend is None:fixed[mult]*=likelihood;fixed[mult]/=fixed[mult].sum()
                else:
                    trend_fixed_mass[mult]*=likelihood[fixed_trends[mult][t]];trend_fixed_mass[mult]/=trend_fixed_mass[mult].sum()
                    fixed[mult]=np.bincount(fixed_trends[mult][t],weights=trend_fixed_mass[mult],minlength=66)
            n=t+1
            if n not in HISTORIES:continue
            lifetime=np.mean(GRID[states],axis=0);logarray=np.array(logs);static=posterior(logarray.sum(0),independent=independent)
            ps={'static':static,'adaptive':adaptive.sum(0)+np.bincount(adaptive_trends[t],weights=trend_mass,minlength=66)}
            ps.update({f'fixed-{mult:g}':p for mult,p in fixed.items()})
            if not (n==512 and setting.get('omit_windowed_512')):
                for window in WINDOWS:
                    data=logarray[-min(n,window):];p=posterior(data.sum(0),independent=independent);ps[f'window-{window}']=p
                    quarter=max(1,len(data)//4);first=posterior(data[:quarter].sum(0),independent=independent)@GRID;last=posterior(data[-quarter:].sum(0),independent=independent)@GRID
                    extrapolated=last+(last-first)/6;ps[f'slope-{window}']=softmax(-np.sum((GRID-extrapolated)**2,axis=1)/(2*.1**2),independent)
            for name,p in ps.items():out.append(record(readout(p,state,lifetime),maker=maker,reader=name,history=n))
        raw.append(dict(initial=initial,trend_direction=direction,states=states,observations=history))
    return dict(public=dict(effects=e.tolist(),grid=GRID.tolist(),choice_rule='beta3, no habit',trajectory='One512-record trajectory; histories are prefixes and share the initial prior. Fixed windows restart a uniform prior on their suffix.',adaptive_models=MODELS,adaptive_trend_speeds=[.005,.01,.02],trend_prior='Uniform66 initial weights and6 transfer directions for each declared speed; exact finite-trajectory likelihood. Boundary reflection, nearest-grid projection.',slope_baseline='Approximate two-quarter extrapolation with Gaussian width.1; not an exact Bayesian reader.',primary_aggregation='U is final mean log loss minus minimum mean log loss across histories, after averaging makers within world and worlds within setting.'),truth_and_observations=raw,readouts=out)

def controls():
    close(NEIGHBORS.sum(1).tolist(),[1.]*66,tol=1e-12)
    law=softmax(np.stack([GRID[:,0],GRID[:,1]],axis=1));p=np.full(66,1/66);ll=np.zeros(66)
    for choice in (0,1,0,0,1):
        p=transition(p,'walk',0);p*=law[:,choice];p/=p.sum();ll+=np.log(law[:,choice])
    close(p.tolist(),posterior(ll,independent=True).tolist(),tol=1e-12)
    for kind in ('walk','jump'):close(transition(p,kind,.1).tolist(),transition(p,kind,.1,True).tolist(),tol=1e-12)
    assert np.array_equal(trend_positions(0),np.tile(np.repeat(np.arange(66),6),(512,1)))
    return dict(passed=True,zero_change_exact_static_posterior=True,normalized_transition=True,independent_filter_arithmetic=True)
