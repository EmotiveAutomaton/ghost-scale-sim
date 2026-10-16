"""Raw reconstruction for explicit unavailable N-family observations."""
import math
from pathlib import Path
import numpy as np
from . import checker,registry
from .common import digest,interval
from .family import rows,verify_packet


def same(actual,expected):
    if actual is None or expected is None:return actual is expected
    return math.isclose(actual,expected,abs_tol=1e-11,rel_tol=0)


def estimate(values,identity,stable=False):
    if any(x is None for x in values):return dict(mean=None,lower=None,upper=None,se=None,finite=False)
    if stable:
        from .adaptive_numerics import interval as stable_interval
        return stable_interval(values,identity)
    return interval(values,identity)


def reconstruct(root,assembly_pending=False):
    root=Path(root);plan,summary=verify_packet(root,assembly_pending);family=plan['family'];out={}
    rival,mechanism=registry.PRIMARY[family][:2]
    for variant in summary['variants']:
        identity=variant['variant']['id'];folder=root/'original'/identity
        groups={w:[] for w in range(plan['worlds'])};scores={};seen=set();unavailable={}
        for pred,truth in zip(rows(folder/'prediction_points.jsonl.gz'),rows(folder/'truth_points.jsonl.gz'),strict=True):
            world,case=pred['world'],pred['case'];key=(world,case)
            if key in seen or world not in groups or case not in range(plan['cases']):raise ValueError('duplicate or invalid raw case')
            seen.add(key)
            if truth['identity']!=[family,identity,plan['batch'],world,case]:raise ValueError('raw identity mismatch')
            label=truth['truth'];rebuilt={}
            for name,prob in pred['probabilities'].items():
                if prob is None:metrics={m:None for m in ('log_loss','half_brier','accuracy')}
                else:
                    measured=checker.scores(prob,label['y'])
                    if plan.get('numerical_validation_packet'):
                        from .adaptive_numerics import independent_log_loss
                        measured['log_loss']=independent_log_loss(prob,label['y'])
                    metrics={m:float(np.mean(x)) if np.isfinite(x).all() else None for m,x in measured.items()}
                available=all(x is not None for x in metrics.values())
                if available!=pred['scores'][name]['available']:raise ValueError('availability was changed')
                unavailable[name]=unavailable.get(name,0)+int(not available)
                for m,x in metrics.items():
                    if not same(x,pred['scores'][name][m]):raise ValueError('saved proper score mismatch')
                    scores.setdefault((name,m),{}).setdefault(world,[]).append(x)
                rebuilt[name]=metrics
            eligible=family!='N6' or label['warmth']==label['competence']==0
            if pred['primary_eligible']!=eligible:raise ValueError('primary stratum mismatch')
            if eligible:
                a,b=rebuilt[rival]['log_loss'],rebuilt[mechanism]['log_loss']
                groups[world].append(a-b if a is not None and b is not None else None)
        if len(seen)!=plan['worlds']*plan['cases']:raise ValueError('incomplete packet')
        mean=lambda xs:math.fsum(xs)/len(xs) if xs and all(x is not None for x in xs) else None
        gains=[mean(groups[w]) for w in groups]
        if any(not same(a,b) for a,b in zip(gains,variant['world_gains'],strict=True)):raise ValueError('world gains changed')
        if [len(groups[w]) for w in groups]!=variant['primary_counts']:raise ValueError('primary denominator changed')
        if unavailable!=variant['unavailable_predictions']:raise ValueError('unavailable cases were deleted')
        primary=estimate(gains,[identity,plan['batch'],'primary'],stable=bool(plan.get('numerical_validation_packet')))
        for key in ('mean','se','lower','upper'):
            if not same(primary[key],variant['primary'][key]):raise ValueError('primary interval mismatch')
        for (name,metric),worlds in scores.items():
            rebuilt=estimate([mean(worlds[w]) for w in groups],[identity,plan['batch'],name,metric],stable=bool(plan.get('numerical_validation_packet')))
            for key in ('mean','se','lower','upper'):
                if not same(rebuilt[key],variant['readers'][name][metric][key]):raise ValueError('reader interval mismatch')
        out[identity]=dict(cases=len(seen),worlds=len(groups),estimate=primary,
            unavailable_predictions=unavailable,all_proper_scores_rebuilt=True)
    return dict(state='passed',plan_sha256=digest(root/'PLAN.json'),variants=out)
