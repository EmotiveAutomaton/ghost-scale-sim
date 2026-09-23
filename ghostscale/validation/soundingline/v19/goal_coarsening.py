"""Fixed exhaustive goal partitions; reader evidence and fitted forecasts stay fixed."""
from itertools import product
import gzip
import json
import time
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import goal_decision as D

ARMS = D.ARMS
FORECASTS = D.FORECASTS
COSTS = (.1, .25, .5)
PARTITIONS = {'fine': (0, 1, 2), 'meaning': (0, 1, 1),
              'dependency': (1, 0, 1), 'presentation': (1, 1, 0), 'one-class': (0, 0, 0)}
CHOICES = ('mapped-fine', 'coarse-mode')
POLICIES = ('always', 'abstain')
FIELDS = ('tier', 'budget', 'arm', 'forecast', 'partition', 'choice', 'policy', 'cost', 'lineage', 'draw', 'seed')
METRICS = ('coverage', 'correct', 'incorrect', 'cost_value', 'forecast_cost', 'incompatible',
           'full_incompatible', 'coarse_loss', 'fine_loss', 'alternatives')
PAIRED = ('coverage', 'incorrect', 'cost_value', 'coarse_loss', 'alternatives')


def geometry(partition):
    if partition not in PARTITIONS: raise ValueError('partition')
    mapping = np.asarray(PARTITIONS[partition]); n = int(mapping.max())+1
    goals = np.asarray(list(product(range(n), repeat=3)))
    labels = (mapping[D.GOALS]*np.array([n*n, n, 1])).sum(1)
    projection = np.eye(n**3)[labels]
    reports = np.asarray(list(product(range(-1, n), repeat=3)))
    active = reports >= 0
    hits = (reports[:, None, :] == goals[None, :, :]) & active[:, None, :]
    correctness = hits.mean(2)
    compatible = (hits | ~active[:, None, :]).all(2)
    sizes = np.bincount(mapping)
    alternatives = np.where(active, sizes[np.maximum(reports, 0)], 0).mean(1)
    return dict(n=n, mapping=mapping, goals=goals, projection=projection, reports=reports,
                correct=correctness, compatible=compatible, coverage=active.mean(1),
                alternatives=alternatives, full=active.all(1), any=active.any(1))


def forecasts(q, partition):
    d = D.decisions(q)  # validates finite normalized 27-path probabilities
    geo = geometry(partition); p = q @ geo['projection']; n = geo['n']
    marginals = np.stack([p @ (geo['goals'][:, t, None] == np.arange(n)).astype(float) for t in range(3)], axis=1)
    selected = {'mapped-fine': geo['mapping'][D.GOALS[d['coordinate']]],
                'coarse-mode': marginals.argmax(2)}
    records = dict(probabilities=p, marginals=marginals)
    for choice, cost, policy in product(CHOICES, COSTS, POLICIES):
        r = selected[choice]
        if policy == 'abstain':
            risk = 1-marginals[np.arange(len(q))[:, None], np.arange(3)[None, :], r]
            r = np.where(risk < cost, r, -1)
        records[f'{choice}-{cost}-{policy}'] = r
    return records


def native_tables(targets, mass, partition):
    g = geometry(partition); targets = np.asarray(targets); mass = np.asarray(mass)
    if (targets.ndim != 3 or targets.shape[2] != 27 or mass.shape != targets.shape[:2]
        or not np.isfinite(targets).all() or not np.isfinite(mass).all()
        or np.any(targets < 0) or np.any(mass < 0)
        or not np.allclose(targets.sum(2), 1, rtol=0, atol=1e-10)
        or not np.allclose(mass.sum(1), 1, rtol=0, atol=1e-10)): raise ValueError('native law')
    coarse = targets @ g['projection']
    return dict(geometry=g, targets=coarse, mass=mass,
                correct=coarse @ g['correct'].T,
                incompatible=((coarse > 0).astype(int) @ g['compatible'].T.astype(int)) == 0)


def score(p, reported, tables, cost, fine_loss):
    g = tables['geometry']; n = g['n']; mass = tables['mass']; target = tables['targets']
    if cost not in COSTS or reported.shape != (len(p), 3) or not np.isin(reported, range(-1, n)).all(): raise ValueError('report')
    if (p.shape != target.shape[1:] or np.any(p < 0) or not np.isfinite(p).all()
        or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-10)
        or np.any((p[None] <= 0) & (target > 0))): raise ValueError('forecast')
    codes = ((reported+1)*np.array([(n+1)**2, n+1, 1])).sum(1).astype(int)
    frame = np.arange(len(p)); cov = g['coverage'][codes]
    coverage = mass @ cov
    correct = (mass*tables['correct'][:, frame, codes]).sum(1)
    incorrect = coverage-correct
    forecast_correct = (p*g['correct'][codes]).sum(1)
    incompat = tables['incompatible'][:, frame, codes]
    logp = np.zeros_like(p); np.log(p, out=logp, where=p > 0)
    return dict(coverage=coverage, correct=correct, incorrect=incorrect,
        cost_value=incorrect+cost*(1-coverage),
        forecast_cost=mass @ (cov-forecast_correct+cost*(1-cov)),
        incompatible=(mass*(incompat & g['any'][codes])).sum(1),
        full_incompatible=(mass*(incompat & g['full'][codes])).sum(1),
        coarse_loss=-(target*mass[:, :, None]*logp).sum((1, 2)),
        fine_loss=np.asarray(fine_loss), alternatives=mass @ g['alternatives'][codes])


def controls():
    q = np.full((1, 27), 1/27)
    one = forecasts(q, 'one-class')
    boundary = np.zeros((1, 27)); boundary[0, [0, 13]] = [.75, .25]
    return {'live:broader_report': bool(np.all(forecasts(q, 'meaning')['coarse-mode-0.5-always'] == 1)),
        'placebo:fine_uniform_abstains': bool(np.all(forecasts(q, 'fine')['coarse-mode-0.5-abstain'] == -1)),
        'positive:one_class_certain': bool(np.all(one['coarse-mode-0.1-abstain'] == 0) and np.allclose(one['probabilities'], 1)),
        'positive:equality_abstains': bool(np.all(forecasts(boundary, 'fine')['coarse-mode-0.25-abstain'] == -1))}


def aggregate(rows, cfg):
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages', 'training_draws', 'fit_seeds', 'budgets'))
    expected = set(product(cfg['tiers'], bs, ARMS, FORECASTS, PARTITIONS, CHOICES, POLICIES, COSTS, ls, ds, ss))
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    if len(idx) != len(rows) or set(idx) != expected: raise ValueError('stratum roster')
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls), size=(cfg['bootstrap_resamples'], len(ls)))
    def estimate(v):
        line = v.mean((1, 2)); boot = line[samples].mean(1)
        return dict(mean=float(line.mean()), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)),
            lineage_values=line.tolist(), draw_means=v.mean((0, 2)).tolist(), feature_seed_means=v.mean((0, 1)).tolist())
    means = []; contrasts = []; areas = []
    for tier, arm, forecast, partition, cost in product(cfg['tiers'], ARMS, FORECASTS, PARTITIONS, COSTS):
        def values(choice, policy, budget, metric):
            return np.asarray([[[idx[tier,budget,arm,forecast,partition,choice,policy,cost,l,d,s][metric] for s in ss] for d in ds] for l in ls])
        for choice, policy, budget in product(CHOICES, POLICIES, bs):
            m = {k:float(values(choice,policy,budget,k).mean()) for k in METRICS}
            means.append(dict(tier=tier,arm=arm,forecast=forecast,partition=partition,choice=choice,policy=policy,cost=cost,budget=budget,
                conditional_error=m['incorrect']/m['coverage'] if m['coverage']>0 else None,
                alternatives_per_claim=m['alternatives']/m['coverage'] if m['coverage']>0 else None,**m))
        comparisons = [(choice,'abstain',choice,'always') for choice in CHOICES]
        if partition not in ('fine','one-class'): comparisons += [('coarse-mode',p,'mapped-fine',p) for p in POLICIES]
        for (choice,policy,base_choice,base_policy),metric in product(comparisons,PAIRED):
            identity=dict(tier=tier,arm=arm,forecast=forecast,partition=partition,cost=cost,choice=choice,policy=policy,base_choice=base_choice,base_policy=base_policy,metric=metric)
            curves=[]
            for budget in bs:
                v=values(choice,policy,budget,metric)-values(base_choice,base_policy,budget,metric)
                curves.append(v);contrasts.append(dict(identity,budget=budget,**estimate(v)))
            area=np.trapezoid(curves,x=np.log(bs),axis=0)/np.log(bs[-1]/bs[0])
            areas.append(dict(identity,**estimate(area)))
    return dict(means=means,contrasts=contrasts,normalized_log_budget_area=areas)


def run(root, plan, pulse):
    cfg=plan['design'];base=root/'inputs';checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    parent=read(base/'parent/PLAN.json')['design']
    for k in ('tiers','budgets','training_draws','fit_seeds','development_lineages'):
        if cfg[k]!=parent[k]:raise ValueError('population')
    if cfg['tiers']!=['E2-full'] or cfg['costs']!=list(COSTS) or cfg['partitions']!={k:list(v) for k,v in PARTITIONS.items()}:raise ValueError('design')
    packets=read(base/'reader/PACKETS.json');keys=sorted(packets['packets']);ops=[]
    for key in keys:
        p=packets['packets'][key];D.S.validate(p)
        if digest(p)!=key or p['tier']!='E2-full':raise ValueError('packet')
        ops.append(sum(D.S.L.OPERATIONS.index(e['operation'])*6**(2-t) for t,e in enumerate(p['inputs']['observations'])))
    refs=read(base/'evaluator/REFERENCES.json');ls=cfg['development_lineages'];pos={k:i for i,k in enumerate(keys)}
    if len(refs)!=len(ls) or {r['lineage'] for r in refs}!=set(ls):raise ValueError('reference roster')
    targets=np.zeros((len(ls),len(keys),27));mass=np.zeros((len(ls),len(keys)))
    for r in refs:
        li=ls.index(r['lineage'])
        if r['tier']!='E2-full' or len(r['frames'])!=len(keys) or {f['frame'] for f in r['frames']}!=set(keys):raise ValueError('native frames')
        for f in r['frames']:
            i=pos[f['frame']];mass[li,i]=f['mass']
            if len(dict(f['target']))!=len(f['target']):raise ValueError('duplicate target')
            for k,w in f['target']:
                if k%216!=ops[i] or not 0<=k<5832 or w<=0:raise ValueError('native support')
                targets[li,i,k//216]=w
    tables={partition:native_tables(targets,mass,partition) for partition in PARTITIONS}
    write(root/'reader/PACKETS.json',packets);write(root/'evaluator/REFERENCES.json',refs)
    old=json.loads(gzip.decompress((base/'parent/goal_abstention_points.json.gz').read_bytes()))
    oldfields=('tier','budget','arm','forecast','policy','cost','lineage','draw','seed')
    idx={tuple(r[k] for k in oldfields):r for r in old}
    expected=set(product(cfg['tiers'],cfg['budgets'],ARMS,FORECASTS,('always-coordinate','joint-all-or-none','step-abstention'),COSTS,ls,cfg['training_draws'],cfg['fit_seeds']))
    if len(idx)!=len(old) or set(idx)!=expected:raise ValueError('parent roster')
    rows=[];native=[];timings=[];reproduced=0;error=0.;frames=0;(root/'reports').mkdir()
    def process(q, truth, weights, prepared, identity, dest, native_run=False):
        nonlocal reproduced,error
        D.decisions(q)
        logq=np.zeros_like(q);np.log(q,out=logq,where=q>0)
        if np.any((q[None]<=0)&(truth>0)):raise ValueError('zero true probability')
        fine_loss=-(truth*weights[:,:,None]*logq).sum((1,2));saved={}
        for partition in PARTITIONS:
            record=forecasts(q,partition);saved.update({partition+'-'+k:v for k,v in record.items()})
            for choice,cost,policy in product(CHOICES,COSTS,POLICIES):
                v=score(record['probabilities'],record[f'{choice}-{cost}-{policy}'],prepared[partition],cost,fine_loss)
                for li,lineage in enumerate([identity['lineage']] if native_run else ls):
                    cell={k:float(x[li]) for k,x in v.items()}
                    if partition=='fine' and choice=='coarse-mode' and not native_run:
                        prior=idx['E2-full',identity['budget'],identity['arm'],identity['forecast'],'always-coordinate' if policy=='always' else 'step-abstention',cost,lineage,identity['draw'],identity['seed']]
                        e=max([abs(cell[k]-prior[k]) for k in METRICS if k in prior]+[abs(cell['fine_loss']-prior['loss']),abs(cell['coarse_loss']-prior['loss'])]);error=max(error,e);reproduced+=1
                        if e>1e-10:raise ValueError('parent score identity')
                    row=dict(identity,lineage=lineage,partition=partition,choice=choice,cost=cost,policy=policy,
                        conditional_error=cell['incorrect']/cell['coverage'] if cell['coverage']>0 else None,
                        alternatives_per_claim=cell['alternatives']/cell['coverage'] if cell['coverage']>0 else None,**cell)
                    (native if native_run else rows).append(row)
        np.savez_compressed(dest,**saved)
    for li,lineage in enumerate(ls):
        pulse(phase='native-coarsening',lineage=lineage)
        prepared={k:native_tables(targets[li:li+1],mass[li:li+1],k) for k in PARTITIONS}
        process(targets[li],targets[li:li+1],mass[li:li+1],prepared,dict(tier='E2-full',lineage=lineage),root/'evaluator'/f'native-{lineage}.npz',True)
    for draw,seed,budget,arm,forecast in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],ARMS,FORECASTS):
        pulse(phase='goal-coarsening',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast);tick=time.process_time()
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(base/'reports'/(stem+'.npz'),allow_pickle=False) as z:q=z['probabilities']
        if len(q)!=len(keys):raise ValueError('forecast roster')
        process(q,targets,mass,tables,dict(tier='E2-full',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,frames=len(keys)),root/'reports'/(stem+'.npz'))
        frames+=len(keys);timings.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,cpu_seconds=time.process_time()-tick))
    write(root/'evaluator/NATIVE_SCORES.json',native)
    (root/'goal_coarsening_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'TIMING.jsonl',dict(measurements=timings));write(root/'PARENT_REPRODUCTION.json',dict(passed=True,cells=reproduced,max_error=error))
    pulse(phase='coarsening-aggregate')
    return dict(controls=checks,rows=len(rows),native_rows=len(native),packets=len(keys),frame_forecasts=frames,parent_cells=reproduced,parent_max_error=error,
                **aggregate(rows,cfg),fits=0,scope='constructed-method specificity/coverage tradeoff; all partitions; truth only scores; broad reports do not recover fine goals or human intent')
