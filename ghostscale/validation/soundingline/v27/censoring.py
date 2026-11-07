"""Retained-data bounds for ratios of finite-horizon learning medians.

A non-crossing is later than the horizon, not an observed infinity. Bootstrap
both endpoints of the identified ratio set using the same whole-block draws.
No discarded draws and no extrapolation beyond the recorded horizon.
"""
import gzip,json,math,statistics
import numpy as np
from .common import read,digest,write,rng,close


def ratio_bounds(blocks,horizon,identity,independent=False):
    def endpoints(values):
        median=statistics.median if independent else np.median
        lower_a=float(median([horizon+1 if x is None else x for x,y in values]))
        upper_a=float(median([math.inf if x is None else x for x,y in values]))
        lower_b=float(median([horizon+1 if y is None else y for x,y in values]))
        upper_b=float(median([math.inf if y is None else y for x,y in values]))
        return lower_a/upper_b,upper_a/lower_b
    low,high=endpoints([v for b in blocks for v in b]);n=len(blocks)
    if n<2:ci=[None,None]
    else:
        draws=[endpoints([v for i in pick for v in blocks[i]]) for pick in rng('censor-bound-bootstrap',identity).integers(0,n,(2000,n))]
        ci=[sorted(v[0] for v in draws)[49],sorted(v[1] for v in draws)[1949]]
    finite=lambda x:None if x is None or not math.isfinite(x) else float(x)
    return dict(estimate=None if low!=high else finite(low),identified_estimate_bounds=[finite(low),finite(high)],
        upper_bound_unbounded=math.isinf(high),interval95=list(map(finite,ci)),upper_interval_unbounded=ci[1] is not None and math.isinf(ci[1]),
        interval_unavailable=n<2,estimate_censored=low!=high,lower_censored=False,upper_censored=ci[1] is not None and math.isinf(ci[1]),
        method='Pointwise whole-block bootstrap of finite-horizon ratio bounds, 2000 resamples; non-crossings lie in [horizon+1, infinity).')


def retained(packet):
    summary=read(packet/'original/SUMMARY.json')
    affected=[r for r in summary['cells'] if r['statistic']=='median_ratio' and (r['estimate_censored'] or r['interval_unavailable'] or r['upper_censored'] or r['lower_censored'])]
    if not affected:return []
    result=[];index=read(packet/'original/RAW_INDEX.json')
    for row in affected:
        setting=row['setting'];blocks=[];s=summary['settings'][setting]
        horizon=s.get('records',s.get('structures'))
        if horizon is None:raise ValueError('ratio censor horizon missing')
        for b in range(summary['blocks']):
            path=packet/'original'/f'world-{setting:04d}-{b:04d}_points.json.gz'
            if digest(path)!=index[path.name]['sha256']:raise ValueError('censor bound input changed')
            with gzip.open(path,'rt',encoding='utf-8') as f:raw=json.load(f)
            m=next(m for m in raw['data']['measures'] if m['metric']==row['metric'] and m['labels']==row['labels'])
            blocks.append(m['values'])
        identity=[summary['card'],setting,row['metric'],row['labels']]
        bound=ratio_bounds(blocks,horizon,identity);close(bound,ratio_bounds(blocks,horizon,identity,True),tol=1e-12)
        result.append(dict(setting=setting,metric=row['metric'],labels=row['labels'],**bound))
    return result
