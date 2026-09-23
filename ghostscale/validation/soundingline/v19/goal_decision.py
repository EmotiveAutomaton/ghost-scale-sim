"""Compare joint and coordinatewise goal decisions with each forecast fixed.

Full-label numeric tie order is explicit. Native targets only score decisions;
they never constrain a learned choice or enter reader inputs.
"""
from itertools import product
import gzip
import json
import time
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import joint_support as S
from .witnessed_goal_factorization import factorize, GOALS, ARMS, FIELDS
from .joint_uncertainty import distribution

METRICS = ('goal_accuracy', 'path_accuracy', 'top_incompatible',
           'forecast_hamming_loss', 'forecast_path_loss', 'loss')
FORECASTS = ('restricted', 'goal-product')
HAMMING = (GOALS[:, None, :] != GOALS[None, :, :]).mean(2)


def decisions(q):
    q = np.asarray(q, float)
    if q.ndim != 2 or q.shape[1] != 27 or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('goal forecast')
    if not np.allclose(q.sum(1), 1, rtol=0, atol=4e-12): raise ValueError('normalization')
    marg = np.stack([q @ (GOALS[:, t, None] == np.arange(3)) for t in range(3)], axis=1)
    joint = np.argmax(q, axis=1)
    coord = np.argmax(marg, axis=2) @ np.array([9, 3, 1])
    ordered = np.sort(q, axis=1); sorted_marg = np.sort(marg, axis=2)
    result = dict(joint=joint, coordinate=coord, marginals=marg,
                  joint_ties=(q == q.max(1)[:, None]).sum(1),
                  coordinate_ties=(marg == marg.max(2)[:, :, None]).sum(2),
                  joint_margin=ordered[:, -1]-ordered[:, -2],
                  coordinate_margins=sorted_marg[:, :, -1]-sorted_marg[:, :, -2])
    # These are objectives under the forecast, not criteria about unknown truth.
    risks = q @ HAMMING
    if np.any(risks[np.arange(len(q)), coord] > risks.min(1)+1e-12): raise ValueError('Hamming optimum')
    if np.any(q[np.arange(len(q)), joint] < q.max(1)-1e-12): raise ValueError('path optimum')
    result['hamming_risks'] = risks
    return result


def score(q, selected, targets, mass):
    """Targets: lineage x frame x goal; mass: lineage x frame."""
    rows = np.arange(len(q)); targets = np.asarray(targets); mass = np.asarray(mass)
    if targets.ndim != 3 or targets.shape[1:] != q.shape or mass.shape != targets.shape[:2]:
        raise ValueError('reference shape')
    d = decisions(q); chosen = d[selected]
    hits = (GOALS[chosen, None, :] == GOALS[None, :, :]).mean(2)
    weighted = targets * mass[:, :, None]
    support = targets > 0
    if np.any((q[None] <= 0) & support): raise ValueError('zero true probability')
    logq = np.zeros_like(q); positive = q > 0; logq[positive] = np.log(q[positive])
    chosen_mass = targets[:, rows, chosen]
    return dict(loss=-(weighted*logq).sum((1, 2)),
        goal_accuracy=(weighted*hits).sum((1, 2)),
        path_accuracy=(mass*chosen_mass).sum(1),
        top_incompatible=(mass*(chosen_mass == 0)).sum(1),
        forecast_hamming_loss=mass @ d['hamming_risks'][rows, chosen],
        forecast_path_loss=mass @ (1-q[rows, chosen]),
        decision_disagreement=mass @ (d['joint'] != d['coordinate']),
        joint_tie=mass @ (d['joint_ties'] > 1),
        coordinate_tie=mass @ (d['coordinate_ties'] > 1).any(1),
        joint_margin=mass @ d['joint_margin'],
        coordinate_margin=mass @ d['coordinate_margins'].min(1))


def controls():
    q = np.zeros((1, 27)); q[0, [0, 12, 10, 4]] = [.30, .26, .26, .18]
    d = decisions(q); s = np.zeros((1, 27)); s[0, 26] = 1
    uniform = decisions(np.ones((1, 27))/27)
    return {'live:opposed_decisions': bool(d['joint'][0] == 0 and d['coordinate'][0] == 9),
            'positive:distinct_objectives': bool(d['hamming_risks'][0, 9] < d['hamming_risks'][0, 0] and q[0, 0] > q[0, 9]),
            'placebo:point_identity': bool(decisions(s)['joint'][0] == decisions(s)['coordinate'][0] == 26),
            'positive:numeric_ties': bool(uniform['joint'][0] == uniform['coordinate'][0] == 0 and uniform['joint_ties'][0] == 27)}


def aggregate(rows, cfg):
    arms = tuple(a+'-'+f+'-'+d for a in ARMS for f in FORECASTS for d in ('joint', 'coordinate'))
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages','training_draws','fit_seeds','budgets'))
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    if len(idx) != len(rows) or set(idx) != set(product(cfg['tiers'], bs, arms, ls, ds, ss)): raise ValueError('stratum roster')
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls), size=(cfg['bootstrap_resamples'], len(ls)))
    def estimate(v):
        line=v.mean((1,2)); boot=line[samples].mean(1)
        return dict(mean=float(line.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),
            lineage_values=line.tolist(),draw_means=v.mean((0,2)).tolist(),feature_seed_means=v.mean((0,1)).tolist())
    means=[]; contrasts=[]; areas=[]; fields=sorted(set(rows[0])-set(FIELDS)-{'frames'})
    for tier in cfg['tiers']:
        for arm,budget in product(arms,bs):
            rr=[idx[tier,budget,arm,l,d,s] for l,d,s in product(ls,ds,ss)]
            means.append(dict(tier=tier,arm=arm,budget=budget,**{k:float(np.mean([r[k] for r in rr])) for k in fields}))
        for arm,forecast,metric in product(ARMS,FORECASTS,METRICS):
            base=arm+'-'+forecast; curves=[]
            for budget in bs:
                v=np.array([[[idx[tier,budget,base+'-coordinate',l,d,s][metric]-idx[tier,budget,base+'-joint',l,d,s][metric] for s in ss] for d in ds] for l in ls])
                curves.append(v);contrasts.append(dict(tier=tier,budget=budget,arm=base+'-coordinate',baseline=base+'-joint',metric=metric,**estimate(v)))
            area=np.trapezoid(curves,x=np.log(bs),axis=0)/np.log(bs[-1]/bs[0])
            areas.append(dict(tier=tier,arm=base+'-coordinate',baseline=base+'-joint',metric=metric,**estimate(area)))
    return dict(contrasts=contrasts,normalized_log_budget_area=areas,means=means)


def run(root, plan, pulse):
    cfg=plan['design']; base=root/'inputs'; checks=controls(); write(root/'CONTROLS.json',checks)
    if not all(checks.values()): raise ValueError('known-answer controls')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h: raise ValueError('input binding')
    parent=read(base/'parent/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if cfg[k]!=parent[k]: raise ValueError('population changed')
    if cfg['tiers']!=['E2-full']: raise ValueError('complete witnesses')
    packets=read(base/'reader/PACKETS.json'); keys=sorted(packets['packets'])
    operations=[]
    for k in keys:
        p=packets['packets'][k];S.validate(p)
        if digest(p)!=k or p['tier']!='E2-full': raise ValueError('packet identity')
        operations.append(sum(S.L.OPERATIONS.index(e['operation'])*6**(2-t) for t,e in enumerate(p['inputs']['observations'])))
    operations=np.array(operations); labels=216*np.arange(27)[None,:]+operations[:,None]
    refs=read(base/'evaluator/REFERENCES.json'); ls=cfg['development_lineages']; pos={k:i for i,k in enumerate(keys)}
    if len(refs)!=len(ls) or {r['lineage'] for r in refs}!=set(ls): raise ValueError('native lineage roster')
    targets=np.zeros((len(ls),len(keys),27)); mass=np.zeros((len(ls),len(keys)))
    for r in refs:
        li=ls.index(r['lineage'])
        if r['tier']!='E2-full' or len(r['frames'])!=len(keys) or {f['frame'] for f in r['frames']}!=set(keys): raise ValueError('native frames')
        for f in r['frames']:
            i=pos[f['frame']];mass[li,i]=f['mass']; pairs=f['target']
            if len(dict(pairs))!=len(pairs): raise ValueError('duplicate target')
            for k,w in pairs:
                if k%216!=operations[i] or not 0<=k<5832 or w<=0: raise ValueError('native support')
                targets[li,i,k//216]=w
    if not np.isfinite(targets).all() or not np.isfinite(mass).all() or np.any(mass<=0) or not np.allclose(targets.sum(2),1,rtol=0,atol=1e-10) or not np.allclose(mass.sum(1),1,rtol=0,atol=1e-10): raise ValueError('native normalization')
    write(root/'reader/PACKETS.json',packets);write(root/'evaluator/REFERENCES.json',refs)
    raw=json.loads(gzip.decompress((base/'parent/goal_mass_points.json.gz').read_bytes()))
    idx={tuple(r[k] for k in FIELDS):r for r in raw}
    expected=set(product(cfg['tiers'],cfg['budgets'],tuple(a+s for a in ARMS for s in ('-restricted','-mass-matched','-goal-product')),ls,cfg['training_draws'],cfg['fit_seeds']))
    if len(idx)!=len(raw) or set(idx)!=expected: raise ValueError('parent roster')
    native=[]
    for li,l in enumerate(ls):
        q=targets[li];d=decisions(q)
        for decision in ('joint','coordinate'):
            v=score(q,decision,q[None],mass[li:li+1])
            native.append(dict(tier='E2-full',lineage=l,decision=decision,**{k:float(a[0]) for k,a in v.items()}))
        np.savez_compressed(root/'evaluator'/f'native-{l}.npz',probabilities=q,**d)
    write(root/'evaluator/NATIVE_SCORES.json',native)
    rows=[];timing=[];reproduced=0;error=0.;frames=0;(root/'decisions').mkdir()
    for draw,seed,budget,arm in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],ARMS):
        pulse(phase='goal-decisions',draw=draw,seed=seed,budget=budget,arm=arm);tick=time.process_time()
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}'
        if read(base/'forecasts'/(stem+'-frames.json'))!=keys: raise ValueError('forecast frame roster')
        with np.load(base/'forecasts'/(stem+'.npz'),allow_pickle=False) as z:
            if set(z.files)!={'probabilities','alphabet'}: raise ValueError('forecast fields')
            q=distribution(z['probabilities'],z['alphabet'],budget)
        restricted,fact,marg,norm=factorize(q,operations)
        with np.load(base/'masses'/(stem+'.npz'),allow_pickle=False) as z:
            if not np.array_equal(z['operations'],operations) or not np.array_equal(z['normalizer'],norm) or not np.array_equal(z['goals'],marg): raise ValueError('accepted forecast changed')
        for forecast,full in zip(FORECASTS,(restricted,fact)):
            prob=full[np.arange(len(keys))[:,None],labels];d=decisions(prob)
            np.savez_compressed(root/'decisions'/(stem+'-'+forecast+'.npz'),probabilities=prob,operations=operations,**d)
            for decision in ('joint','coordinate'):
                v=score(prob,decision,targets,mass)
                for li,l in enumerate(ls):
                    old=idx['E2-full',budget,arm+'-'+forecast,l,draw,seed]['loss'];e=abs(float(v['loss'][li])-old);error=max(error,e)
                    if e>1e-10: raise ValueError('prediction identity')
                    if decision=='joint':reproduced+=1
                    rows.append(dict(tier='E2-full',budget=budget,arm=arm+'-'+forecast+'-'+decision,lineage=l,draw=draw,seed=seed,frames=len(keys),**{k:float(a[li]) for k,a in v.items()}))
            frames+=len(keys)
        timing.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,cpu_seconds=time.process_time()-tick))
    (root/'goal_decision_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'TIMING.jsonl',dict(measurements=timing));write(root/'PARENT_REPRODUCTION.json',dict(passed=True,loss_cells=reproduced,max_error=error))
    return dict(controls=checks,rows=len(rows),frame_forecasts=frames,packets=len(keys),native_rows=len(native),parent_loss_cells=reproduced,parent_max_error=error,
        **aggregate(rows,cfg),fits=0,scope='constructed-method decision comparison;forecast fixed;numeric-label ties;native targets evaluator-only;historical correspondence unestablished')
