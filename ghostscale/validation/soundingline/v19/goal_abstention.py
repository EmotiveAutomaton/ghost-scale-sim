"""Fixed-cost goal reports from accepted forecasts; truth only scores reports."""
from itertools import product
import gzip
import json
import time
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import goal_decision as D

GOALS = D.GOALS
ARMS = D.ARMS
FORECASTS = D.FORECASTS
POLICIES = ('always-coordinate', 'joint-all-or-none', 'step-abstention')
COSTS = (.1, .25, .5)
FIELDS = ('tier', 'budget', 'arm', 'forecast', 'policy', 'cost', 'lineage', 'draw', 'seed')
METRICS = ('coverage', 'correct', 'incorrect', 'cost_value', 'forecast_cost',
           'incompatible', 'full_incompatible', 'loss')


def reports(q, cost):
    if cost not in COSTS: raise ValueError('undeclared cost')
    d = D.decisions(q)
    coordinate = GOALS[d['coordinate']]
    joint = GOALS[d['joint']]
    rows = np.arange(len(q))[:, None]
    steps = np.arange(3)[None, :]
    coordinate_risk = 1-d['marginals'][rows, steps, coordinate]
    joint_risk = 1-d['marginals'][rows, steps, joint]
    # Binary64 strict comparison: equality abstains; no tolerance creates a tie.
    joint_report = np.where((joint_risk.mean(1) < cost)[:, None], joint, -1)
    step_report = np.where(coordinate_risk < cost, coordinate, -1)
    return dict(zip(POLICIES, (coordinate, joint_report, step_report)))


def score(q, reported, targets, mass, cost):
    targets = np.asarray(targets); mass = np.asarray(mass)
    if (targets.ndim != 3 or targets.shape[1:] != q.shape or mass.shape != targets.shape[:2]
            or reported.shape != (len(q), 3) or not np.isin(reported, [-1, 0, 1, 2]).all()):
        raise ValueError('score shapes or reports')
    if (not np.isfinite(targets).all() or not np.isfinite(mass).all()
            or np.any(targets < 0) or np.any(mass < 0)
            or not np.allclose(targets.sum(2), 1, rtol=0, atol=1e-10)
            or not np.allclose(mass.sum(1), 1, rtol=0, atol=1e-10)):
        raise ValueError('reference normalization')
    active = reported >= 0
    hits = (reported[:, None, :] == GOALS[None, :, :]) & active[:, None, :]
    correct_per_goal = hits.sum(2)/3
    coverage_per_frame = active.mean(1)
    compatible = (hits | ~active[:, None, :]).all(2)
    native_support = targets > 0
    if np.any((q[None] <= 0) & native_support): raise ValueError('zero true probability')
    logq = np.zeros_like(q); np.log(q, out=logq, where=q > 0)
    correct = ((targets*mass[:, :, None])*correct_per_goal).sum((1, 2))
    coverage = mass @ coverage_per_frame
    incorrect = coverage-correct
    has_extension = (native_support & compatible[None]).any(2)
    forecast_correct = (q*correct_per_goal).sum(1)
    value = dict(coverage=coverage, correct=correct, incorrect=incorrect,
        cost_value=incorrect+cost*(1-coverage),
        forecast_cost=mass @ (coverage_per_frame-forecast_correct+cost*(1-coverage_per_frame)),
        incompatible=(mass*(~has_extension & active.any(1)[None])).sum(1),
        full_incompatible=(mass*(~has_extension & active.all(1)[None])).sum(1),
        loss=-(targets*mass[:, :, None]*logq).sum((1, 2)))
    return value


def controls():
    point = np.zeros((1, 27)); point[0, 0] = 1
    uniform = np.ones((1, 27))/27
    mixed = np.zeros((1, 27)); mixed[0, [0, 1, 2]] = [1/3]*3
    boundary = np.zeros((1, 27)); boundary[0, [0, 13]] = [.75, .25]
    r = reports(mixed, .25)['step-abstention']
    v = score(uniform, reports(uniform, .1)['step-abstention'], uniform[None], np.ones((1, 1)), .1)
    return {'live:partial_report': bool(np.array_equal(r, [[0, 0, -1]])),
        'positive:point_all': bool(np.array_equal(reports(point, .1)['step-abstention'], [[0, 0, 0]])),
        'placebo:uniform_none': bool(v['coverage'][0] == 0 and v['incorrect'][0] == 0 and v['cost_value'][0] == .1),
        'positive:equality_abstains': bool(np.all(reports(boundary, .25)['step-abstention'] == -1))}


def aggregate(rows, cfg):
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages', 'training_draws', 'fit_seeds', 'budgets'))
    expected = set(product(cfg['tiers'], bs, ARMS, FORECASTS, POLICIES, COSTS, ls, ds, ss))
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    if len(idx) != len(rows) or set(idx) != expected: raise ValueError('stratum roster')
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls), size=(cfg['bootstrap_resamples'], len(ls)))
    def estimate(v):
        line = v.mean((1, 2)); boot = line[samples].mean(1)
        return dict(mean=float(line.mean()), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)),
            lineage_values=line.tolist(), draw_means=v.mean((0, 2)).tolist(), feature_seed_means=v.mean((0, 1)).tolist())
    means = []; contrasts = []; areas = []
    for tier, arm, forecast, cost in product(cfg['tiers'], ARMS, FORECASTS, COSTS):
        for policy, budget in product(POLICIES, bs):
            rr = [idx[tier, budget, arm, forecast, policy, cost, l, d, s] for l, d, s in product(ls, ds, ss)]
            m = {k: float(np.mean([r[k] for r in rr])) for k in METRICS}
            means.append(dict(tier=tier, arm=arm, forecast=forecast, policy=policy, cost=cost, budget=budget,
                conditional_error=m['incorrect']/m['coverage'] if m['coverage'] > 0 else None, **m))
        for policy, metric in product(POLICIES[1:], METRICS):
            curves = []
            identity = dict(tier=tier, arm=arm, forecast=forecast, policy=policy, cost=cost, metric=metric, baseline=POLICIES[0])
            for budget in bs:
                v = np.array([[[idx[tier,budget,arm,forecast,policy,cost,l,d,s][metric]-idx[tier,budget,arm,forecast,POLICIES[0],cost,l,d,s][metric]
                                for s in ss] for d in ds] for l in ls])
                curves.append(v); contrasts.append(dict(identity, budget=budget, **estimate(v)))
            area = np.trapezoid(curves, x=np.log(bs), axis=0)/np.log(bs[-1]/bs[0])
            areas.append(dict(identity, **estimate(area)))
    return dict(means=means, contrasts=contrasts, normalized_log_budget_area=areas)


def run(root, plan, pulse):
    cfg = plan['design']; base = root/'inputs'; checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('known-answer controls')
    for n, h in cfg['input_files'].items():
        if file_digest(base/n) != h: raise ValueError('input binding')
    parent = read(base/'parent/PLAN.json')['design']
    for k in ('tiers', 'budgets', 'training_draws', 'fit_seeds', 'development_lineages'):
        if cfg[k] != parent[k]: raise ValueError('population changed')
    if cfg['tiers'] != ['E2-full'] or cfg['costs'] != list(COSTS): raise ValueError('evidence or cost design')
    packets = read(base/'reader/PACKETS.json'); keys = sorted(packets['packets']); ops = []
    for key in keys:
        p = packets['packets'][key]; D.S.validate(p)
        if digest(p) != key or p['tier'] != 'E2-full': raise ValueError('packet identity')
        ops.append(sum(D.S.L.OPERATIONS.index(e['operation'])*6**(2-t) for t,e in enumerate(p['inputs']['observations'])))
    refs = read(base/'evaluator/REFERENCES.json'); ls = cfg['development_lineages']; pos = {k:i for i,k in enumerate(keys)}
    if len(refs) != len(ls) or {r['lineage'] for r in refs} != set(ls): raise ValueError('reference roster')
    targets = np.zeros((len(ls),len(keys),27)); mass = np.zeros((len(ls),len(keys)))
    for r in refs:
        li = ls.index(r['lineage'])
        if r['tier'] != 'E2-full' or len(r['frames']) != len(keys) or {f['frame'] for f in r['frames']} != set(keys): raise ValueError('native frames')
        for f in r['frames']:
            i = pos[f['frame']]; mass[li,i] = f['mass']
            if len(dict(f['target'])) != len(f['target']): raise ValueError('duplicate target')
            for k,w in f['target']:
                if k%216 != ops[i] or not 0 <= k < 5832 or w <= 0: raise ValueError('native support')
                targets[li,i,k//216] = w
    write(root/'reader/PACKETS.json', packets); write(root/'evaluator/REFERENCES.json', refs)
    old = json.loads(gzip.decompress((base/'parent/goal_decision_points.json.gz').read_bytes()))
    idx = {tuple(r[k] for k in D.FIELDS):r for r in old}
    expected = set(product(cfg['tiers'], cfg['budgets'], tuple(a+'-'+f+'-'+d for a,f,d in product(ARMS,FORECASTS,('joint','coordinate'))), ls, cfg['training_draws'], cfg['fit_seeds']))
    if len(idx) != len(old) or set(idx) != expected: raise ValueError('parent roster')
    rows = []; native = []; timings = []; reproduced = 0; error = 0.; frames = 0
    (root/'reports').mkdir()
    def score_policies(q, truth, weights, identity, destination, native_run=False):
        nonlocal reproduced, error
        records = {}
        for cost in COSTS:
            selected = reports(q, cost)
            for policy in POLICIES:
                reported = selected[policy]; records[f'{cost}-{policy}'] = reported
                values = score(q, reported, truth, weights, cost)
                for li, lineage in enumerate([identity['lineage']] if native_run else ls):
                    v = {k:float(x[li]) for k,x in values.items()}
                    row = dict(identity, lineage=lineage, cost=cost, policy=policy,
                        conditional_error=v['incorrect']/v['coverage'] if v['coverage'] > 0 else None, **v)
                    if not native_run and cost == COSTS[0] and policy == POLICIES[0]:
                        prior = idx['E2-full',identity['budget'],identity['arm']+'-'+identity['forecast']+'-coordinate',lineage,identity['draw'],identity['seed']]
                        e = max(abs(v['loss']-prior['loss']), abs(v['correct']-prior['goal_accuracy']))
                        error = max(error,e); reproduced += 1
                        if e > 1e-10: raise ValueError('parent prediction or accuracy identity')
                    (native if native_run else rows).append(row)
        np.savez_compressed(destination, probabilities=q, **records)
    for li, lineage in enumerate(ls):
        pulse(phase='native-goal-abstention', lineage=lineage)
        score_policies(targets[li], targets[li:li+1], mass[li:li+1], dict(tier='E2-full',lineage=lineage), root/'evaluator'/f'native-{lineage}.npz', True)
    for draw,seed,budget,arm,forecast in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],ARMS,FORECASTS):
        pulse(phase='goal-abstention', draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast); tick=time.process_time()
        stem=f'{draw}-E2-full-{seed}-{budget}-{arm}-{forecast}'
        with np.load(base/'decisions'/(stem+'.npz'), allow_pickle=False) as z:
            if set(z.files) != {'probabilities','operations','joint','coordinate','marginals','joint_ties','coordinate_ties','joint_margin','coordinate_margins','hamming_risks'}: raise ValueError('decision fields')
            q=z['probabilities']; d=D.decisions(q)
            if len(q) != len(keys) or not np.array_equal(z['operations'],ops): raise ValueError('forecast roster')
            if any(not np.array_equal(z[k],v) for k,v in d.items()): raise ValueError('accepted decisions changed')
        score_policies(q,targets,mass,dict(tier='E2-full',draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,frames=len(keys)),root/'reports'/(stem+'.npz'))
        frames += len(keys);timings.append(dict(draw=draw,seed=seed,budget=budget,arm=arm,forecast=forecast,cpu_seconds=time.process_time()-tick))
    write(root/'evaluator/NATIVE_SCORES.json', native)
    (root/'goal_abstention_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'TIMING.jsonl',dict(measurements=timings))
    write(root/'PARENT_REPRODUCTION.json',dict(passed=True,cells=reproduced,max_error=error))
    return dict(controls=checks,rows=len(rows),native_rows=len(native),packets=len(keys),frame_forecasts=frames,
        parent_cells=reproduced,parent_max_error=error,**aggregate(rows,cfg),fits=0,
        scope='constructed-method reporting tradeoff;stipulated costs;native targets only score;no historical or human-intent claim')
