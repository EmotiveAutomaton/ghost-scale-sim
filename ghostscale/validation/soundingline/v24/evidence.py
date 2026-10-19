"""Prospective probability mixtures and past-only bets on independent blocks.

For q>0, ((1-epsilon)*p + epsilon*q)/q >= epsilon. This bounds
the unmodified mixture score, not a clipped observation. For a block mean X
bounded below by L under E[X|past]<=m, 1+lambda*(X-m) is nonnegative
and has expectation <=1 whenever predictable lambda is in [0,1/(m-L)].
The stake grid below is our explicit strategy, not a reproduction of a paper's
optimal betting recipe. Sample sizing is a pilot forecast, never a power proof.
"""
import math
import numpy as np


def mixture(probabilities, comparator, epsilon):
    p,q=np.asarray(probabilities,float),np.asarray(comparator,float)
    if p.shape!=q.shape or p.ndim<1 or not 0<epsilon<=1:raise ValueError('invalid mixture')
    for a in (p,q):
        if not np.isfinite(a).all() or np.any(a<0) or not np.allclose(a.sum(-1),1,atol=1e-12,rtol=0):raise ValueError('invalid probabilities')
    return (1-epsilon)*p+epsilon*q


def mixture_gain(raw_log_ratio,epsilon):
    """Equivalent true-label score without reconstructing the whole vector."""
    x=np.asarray(raw_log_ratio,float)
    if not 0<epsilon<=1 or np.isnan(x).any():raise ValueError('invalid mixture score')
    if epsilon==1:return np.zeros_like(x)
    return np.logaddexp(math.log1p(-epsilon)+x,math.log(epsilon))


def choose_epsilon(raw_gains,candidates=(.01,.02,.05,.1),max_relative_loss=.02):
    values=np.asarray(raw_gains,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or not 0<max_relative_loss<1:raise ValueError('complete finite discovery scores required')
    if not candidates or len(set(candidates))!=len(candidates):raise ValueError('distinct epsilon candidates required')
    mean=float(values.mean());rows=[]
    for eps in sorted(candidates):
        gain=float(mixture_gain(values,eps).mean())
        rows.append(dict(epsilon=float(eps),mixture_mean=gain,relative_loss=(mean-gain)/mean if mean>0 else None,eligible=mean>0 and gain>(1-max_relative_loss)*mean))
    eligible=[row['epsilon'] for row in rows if row['eligible']]
    return dict(state='DESIGN_CALIBRATION_ONLY',raw_mean=mean,candidates=rows,epsilon=max(eligible) if eligible else None,confirmation=False)


def adaptive_bets(blocks,lower_bound,margin=.02):
    x=np.asarray(blocks,float)
    if x.ndim!=1 or not np.isfinite(x).all() or not math.isfinite(margin) or not math.isfinite(lower_bound) or lower_bound>=margin:raise ValueError('finite complete independent blocks and certified bound required')
    if np.any(x<lower_bound):raise ValueError('support violation; do not clip')
    grid=np.array([0,.1,.25,.5,.9])/(margin-lower_bound)
    history_log=np.zeros(len(grid));factors=[];stakes=[]
    for i,value in enumerate(x):
        # The current observation cannot select its own stake.
        chosen=2 if i==0 else int(np.argmax(history_log))
        stake=float(grid[chosen]);all_factors=1+grid*(value-margin)
        if np.any(all_factors<=0):raise ValueError('nonpositive betting factor')
        stakes.append(stake);factors.append(float(all_factors[chosen]))
        history_log+=np.log(all_factors)
    return dict(log_e=float(np.log(factors).sum()),stakes=stakes,factors=factors,independent_blocks=len(x),lower_bound=lower_bound,margin=margin,clipped_observations=0)


def expected_sample_plan(pilot_blocks,lower_bound,log_threshold,cpu_per_block,max_cpu_hours=64,margin=.02):
    x=np.asarray(pilot_blocks,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or np.any(x<lower_bound):raise ValueError('invalid pilot support')
    if not all(math.isfinite(z) and z>0 for z in (log_threshold,cpu_per_block,max_cpu_hours)) or not math.isfinite(margin) or not math.isfinite(lower_bound) or lower_bound>=margin:raise ValueError('invalid sizing inputs')
    grid=np.array([.1,.25,.5,.9])/(margin-lower_bound)
    rates=np.log1p(grid[:,None]*(x-margin)).mean(1);best=int(np.argmax(rates));rate=float(rates[best])
    blocks=math.ceil(1.5*log_threshold/rate) if rate>0 else None
    cpu=blocks*cpu_per_block if blocks else None
    return dict(state='EXPECTED_EVIDENCE_FORECAST' if cpu is not None and cpu<=max_cpu_hours*3600 else 'FIXED_SAMPLE_DESIGN_REQUIRED',independent_blocks=blocks,expected_log_e_per_block=rate,fixed_forecast_stake=float(grid[best]),cpu_hours=cpu/3600 if cpu else None,log_threshold=log_threshold,target_multiplier=1.5,power_guarantee=False,adaptive_strategy_guaranteed_to_match_forecast=False)
