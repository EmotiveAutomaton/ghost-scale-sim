"""Hash-bound batch review and the calibrated finite family succession rule."""
import gzip,json,math
from pathlib import Path
import numpy as np
from . import adaptive,registry,checker
from .common import digest,read,write,interval


def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as source:
        for line in source:yield json.loads(line)


def verify_packet(root,assembly_pending=False):
    root=Path(root);complete=read(root/('ASSEMBLY_VERIFICATION.json' if assembly_pending else 'COMPLETE.json'));plan=read(root/'PLAN.json')
    if complete['state']!='verified' or complete['plan_sha256']!=digest(root/'PLAN.json'):
        raise ValueError('unverified packet')
    if complete['summary_sha256']!=digest(root/'original/SUMMARY.json') or complete['replay_sha256']!=digest(root/'REPLAY.json'):
        raise ValueError('packet completion binding changed')
    replay=read(root/'REPLAY.json')
    for name,expected in replay['files'].items():
        if digest(root/'original'/name)!=expected or digest(root/'replay'/name)!=expected:
            raise ValueError('accepted replay file changed')
    summary=read(root/'original/SUMMARY.json')
    return plan,summary


def reconstruct(root,assembly_pending=False):
    """Rebuild aggregate gains from saved probabilities and evaluator labels."""
    root=Path(root);plan,summary=verify_packet(root,assembly_pending);family=plan['family']
    if family.startswith('N'):
        from .family_new import reconstruct as reconstruct_new
        return reconstruct_new(root,assembly_pending)
    rival,mechanism,_,_=registry.PRIMARY[family];out={}
    for variant in summary['variants']:
        identity=variant['variant']['id'];folder=root/'original'/identity
        by_world={};count=0
        for pred,truth in zip(rows(folder/'prediction_points.jsonl.gz'),rows(folder/'truth_points.jsonl.gz'),strict=True):
            world,case=pred['world'],pred['case']
            if truth['identity']!=[family,identity,plan['batch'],world,case]:raise ValueError('raw row identity mismatch')
            rebuilt={name:checker.scores(prob,truth['truth']['y']) for name,prob in pred['probabilities'].items()}
            for name,metrics in rebuilt.items():
                for metric,value in metrics.items():
                    if not np.isclose(np.mean(value),pred['scores'][name][metric],atol=1e-12,rtol=0):
                        raise ValueError('saved score mismatch')
            value=float(np.mean(rebuilt[rival]['log_loss']-rebuilt[mechanism]['log_loss']))
            by_world.setdefault(world,[]).append(value);count+=1
        if count!=plan['worlds']*plan['cases'] or set(by_world)!=set(range(plan['worlds'])):
            raise ValueError('incomplete whole-world packet')
        gains=[math.fsum(by_world[w])/len(by_world[w]) for w in range(plan['worlds'])]
        if not np.allclose(gains,variant['world_gains'],atol=1e-12,rtol=0):raise ValueError('world gain aggregation differs')
        estimate=interval(gains,[identity,plan['batch'],'primary'])
        for key in ('mean','se','lower','upper'):
            if not math.isclose(estimate[key],variant['primary'][key],abs_tol=1e-12,rel_tol=0):
                raise ValueError('bootstrap aggregate differs')
        out[identity]=dict(cases=count,worlds=len(gains),estimate=estimate)
    return dict(state='passed',plan_sha256=digest(root/'PLAN.json'),variants=out)


def decide(family,packets,calibration):
    """Structural zeros stay on the curve; no invented SE enters shrinkage."""
    variants=[v for v in registry.variants(family) if family!='N4' or not v['settings']['extension']];identities=[v['id'] for v in variants]
    estimates={identity:[] for identity in identities}; structural=set();previous=None
    for batch,root in enumerate(packets,1):
        plan,summary=verify_packet(root)
        if plan.get('world_indices') is not None:raise ValueError('partial worlds cannot enter a family decision')
        if plan['family']!=family or plan['batch']!=batch or plan['worlds']!=adaptive.RUNGS[batch-1]:
            raise ValueError('family batch sequence/rung changed')
        expected=set(identities if batch<=2 else previous['active_ids'])
        if set(plan['variants'])!=expected:raise ValueError('burn-in or survivor membership changed')
        unavailable=[item['variant']['id'] for item in summary['variants'] if not item['primary'].get('finite',True)]
        if unavailable:
            return dict(state='PRIMARY_UNAVAILABLE',family=family,active_ids=[],best_id=None,
                unavailable_ids=unavailable,curve=[dict(variant=item['variant'],estimate=item['primary'],
                    primary_status=item.get('primary_status','UNAVAILABLE')) for item in summary['variants']],
                reason='At least one declared primary has missing support, infinite loss or an empty target stratum; no case deletion or finite family selection.')
        for item in summary['variants']:
            identity=item['variant']['id'];estimate=item['primary']
            if item['structural_zero']:
                if max(map(abs,item['world_gains']))>1e-12:raise ValueError('structural null violated')
                structural.add(identity)
            else:
                if not np.isfinite([estimate['mean'],estimate['se']]).all() or estimate['se']<=0:
                    raise ValueError('nonfinite or unexplained zero standard error')
                estimates[identity].append(estimate)
        modeled=[v for v in variants if v['id'] not in structural]
        model_ids=[v['id'] for v in modeled]
        if not modeled:raise ValueError('no estimable primary contrast')
        active=[i for i,v in enumerate(modeled) if v['id'] in expected]
        late=[i for i,v in enumerate(modeled) if v['late_effect']]
        eligible=[i for i,v in enumerate(modeled) if registry.selection_eligible(v)]
        decision=adaptive.decide(batch,[estimates[v['id']] for v in modeled],active,late,
            kill=calibration['kill_threshold'],mu_scale=calibration['prior'][0],tau_scale=calibration['prior'][1],eligible=eligible)
        active_ids=identities if batch==1 else [model_ids[i] for i in decision['active']]
        if batch==2 and decision['state']=='CONTINUE':
            # Controls remain visible on their existing curve after burn-in;
            # deterministic zeros need no new sampling just for occupancy.
            active_ids=[model_ids[i] for i in decision['active']]
        curve=[]
        for v in variants:
            identity=v['id']
            if identity in structural:
                curve.append(dict(variant=v,raw_mean=0.,lower=0.,upper=0.,shrunk_mean=0.,probability_above_margin=0.,structural_zero=True))
            else:
                means,ses=adaptive.combine([estimates[identity]])
                index=model_ids.index(identity)
                curve.append(dict(variant=v,raw_mean=means[0],lower=means[0]-1.96*ses[0],upper=means[0]+1.96*ses[0],
                    interval_kind='normal approximation of inverse-variance pooled whole-world bootstrap estimates',
                    shrunk_mean=decision['posterior']['mean'][index],
                    probability_above_margin=decision['posterior']['probability_above_margin'][index],structural_zero=False))
        previous=dict(decision,family=family,active_ids=active_ids,curve=sorted(curve,key=lambda x:(x['raw_mean'],x['variant']['id'])),
            best_id=model_ids[decision['best']] if 'best' in decision else None)
        if decision['state'] not in ('BURN_IN','CONTINUE') and batch<len(packets):
            raise ValueError('batch executed after a family terminal decision')
    return previous


def successor(last_plan,decision,first_packet):
    if decision['state'] not in ('BURN_IN','CONTINUE'):raise ValueError('family is at a review boundary')
    plan=dict(last_plan);batch=last_plan['batch']+1
    if batch>7:raise ValueError('finite ladder exhausted')
    fit_root=Path(last_plan['fit_directory']) if last_plan.get('fit_directory') else Path(first_packet)/'original/fits'
    plan.update(batch=batch,worlds=adaptive.RUNGS[batch-1],variants=decision['active_ids'],
        fit_directory=str(fit_root),fit_hashes={p.name:digest(p) for p in fit_root.iterdir() if p.is_file()},
        predecessor_decision='must bind the finalized machine-card hash at admission')
    return plan
