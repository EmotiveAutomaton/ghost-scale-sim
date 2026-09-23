"""Frozen joint forecasts versus the products of their own sequence marginals.

No fitting, evidence restriction or latent policy enters a learned forecast.
Native posterior products are separate evaluator-only references.
"""
from itertools import product
import gzip
import time
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import joint_support as S
from .joint_uncertainty import distribution

N = 5832
ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
FIELDS = ('tier', 'budget', 'arm', 'lineage', 'draw', 'seed')


def factorize(q):
    q = np.asarray(q, float)
    if q.ndim != 2 or q.shape[1] != N or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('joint distribution')
    if not np.allclose(q.sum(1), 1, rtol=0, atol=1e-12): raise ValueError('joint normalization')
    cube = q.reshape(-1, 27, 216)
    goals, operations = cube.sum(2), cube.sum(1)
    independent = (goals[:, :, None] * operations[:, None, :]).reshape(-1, N)
    if not np.allclose(independent.sum(1), 1, rtol=0, atol=2e-12): raise ValueError('product normalization')
    return independent, goals, operations


def controls():
    q = np.zeros((1, N)); q[0, [0, 217]] = .5
    p, _, _ = factorize(q)
    independent = np.zeros((1, N)); independent[0, [0, 1, 216, 217]] = .25
    return {'live:correlated_joint_changed': bool(np.allclose(p, independent)),
            'placebo:independent_joint_identity': bool(np.array_equal(factorize(independent)[0], independent)),
            'positive:incompatible_cross_pairs_created': bool(p[0, 1] == .25 and p[0, 216] == .25),
            'positive:zeros_preserved_outside_marginals': bool(np.count_nonzero(p) == 4)}


def aggregate(rows, cfg):
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages', 'training_draws', 'fit_seeds', 'budgets'))
    arms = tuple(a+s for a in ARMS for s in ('', '-product'))
    expected = set(product(cfg['tiers'], bs, arms, ls, ds, ss))
    if len(idx) != len(rows) or set(idx) != expected: raise ValueError('complete stratum roster')
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls), size=(cfg['bootstrap_resamples'], len(ls)))
    def estimate(v):
        line = v.mean((1, 2)); boot = line[samples].mean(1)
        return dict(mean=float(line.mean()), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)),
                    lineage_values=line.tolist(), draw_means=v.mean((0, 2)).tolist(), feature_seed_means=v.mean((0, 1)).tolist())
    contrasts = []; areas = []; means = []
    metrics = sorted(set(rows[0]) - set(FIELDS) - {'frames'})
    for tier in cfg['tiers']:
        for arm, budget in product(arms, bs):
            rr = [idx[tier, budget, arm, l, d, s] for l, d, s in product(ls, ds, ss)]
            means.append(dict(tier=tier, budget=budget, arm=arm, **{k:float(np.mean([r[k] for r in rr])) for k in metrics}))
        for arm in ARMS:
            curve = []
            for budget in bs:
                v = np.array([[[idx[tier, budget, arm+'-product', l, d, s]['loss']-idx[tier, budget, arm, l, d, s]['loss']
                               for s in ss] for d in ds] for l in ls])
                contrasts.append(dict(tier=tier, budget=budget, arm=arm+'-product', baseline=arm, **estimate(v)))
                curve.append(v)
            area = np.trapezoid(curve, x=np.log(bs), axis=0)/np.log(bs[-1]/bs[0])
            areas.append(dict(tier=tier, arm=arm+'-product', baseline=arm, **estimate(area)))
    return dict(contrasts=contrasts, normalized_log_budget_area=areas, means=means)


def run(root, plan, pulse):
    cfg = plan['design']; base = root/'inputs'; checks = controls()
    write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('known-answer controls')
    for n, h in cfg['input_files'].items():
        if file_digest(base/n) != h: raise ValueError('input binding')
    parent = read(base/'PLAN.json')['design']
    for k in ('tiers', 'budgets', 'training_draws', 'fit_seeds', 'development_lineages'):
        if parent[k] != cfg[k]: raise ValueError('parent population changed')
    packets = read(base/'reader/PACKETS.json'); refs = read(base/'evaluator/REFERENCES.json')
    for key, p in packets['packets'].items():
        S.validate(p)
        if key != digest(p): raise ValueError('packet identity')
    write(root/'reader/PACKETS.json', packets); write(root/'evaluator/REFERENCES.json', refs)
    old = read(base/'SUMMARY.json')['cells']; idx = {tuple(r[k] for k in FIELDS):r for r in old}
    expected = set(product(cfg['tiers'], cfg['budgets'], ARMS, cfg['development_lineages'], cfg['training_draws'], cfg['fit_seeds']))
    if len(idx) != len(old) or set(idx) != expected: raise ValueError('parent complete roster')
    rows = []; oracle_rows = []; timing = []; reproduced = 0; max_error = 0.; frames_scored = 0
    (root/'marginals').mkdir()
    for tier in cfg['tiers']:
        keys = sorted(k for k,p in packets['packets'].items() if p['tier']==tier)
        # All labels allowed: this comparison makes no public support restriction.
        ref = S.reference_arrays(refs, tier, keys, cfg['development_lineages'], np.ones((len(keys), N), bool))
        if set(r['lineage'] for r in refs if r['tier']==tier) != set(cfg['development_lineages']): raise ValueError('native lineage roster')
        # Native laws are evaluator-only rulers, one record per tier/lineage.
        for li, lineage in enumerate(cfg['development_lineages']):
            pulse(phase='native-joint-factorization', tier=tier, lineage=lineage)
            r = next(r for r in refs if r['tier']==tier and r['lineage']==lineage)
            frames = {f['frame']:f for f in r['frames']}
            if len(frames)!=len(r['frames']) or set(frames)!=set(keys): raise ValueError('native frame roster')
            q = np.zeros((len(keys), N))
            for i, key in enumerate(keys):
                target = frames[key]['target']
                if len(dict(target))!=len(target) or any(w<=0 for _,w in target): raise ValueError('native target')
                for k,w in target:q[i,k]=w
            fact, goals, ops = factorize(q)
            np.savez_compressed(root/'evaluator'/f'{tier}-{lineage}-marginals.npz', goals=goals, operations=ops)
            one = dict(ref, weighted=ref['weighted'][li:li+1], mass=ref['mass'][li:li+1], squared=ref['squared'][li:li+1])
            for arm, prob in (('native-joint', q), ('native-product', fact)):
                scores = S.scores(prob, S.LABELS, one)
                oracle_rows.append(dict(tier=tier,lineage=lineage,arm=arm,frames=len(keys),**{k:float(v[0]) for k,v in scores.items()}))
        native_loss = {r['lineage']:r['loss'] for r in oracle_rows if r['tier']==tier and r['arm']=='native-joint'}
        for draw, seed, budget, arm in product(cfg['training_draws'], cfg['fit_seeds'], cfg['budgets'], ARMS):
            pulse(phase='learned-joint-factorization', tier=tier, draw=draw, seed=seed, budget=budget, arm=arm)
            tick = time.process_time(); stem = f'{draw}-{tier}-{seed}-{budget}-{arm}'
            if read(base/'forecasts'/(stem+'-frames.json'))!=keys: raise ValueError('forecast frame roster')
            with np.load(base/'forecasts'/(stem+'.npz'), allow_pickle=False) as data:
                if set(data.files)!={'probabilities','alphabet'}: raise ValueError('forecast fields')
                alphabet = data['alphabet']; q = distribution(data['probabilities'], alphabet, budget)
            if len(q)!=len(keys): raise ValueError('forecast length')
            fact, goals, ops = factorize(q)
            np.savez_compressed(root/'marginals'/(stem+'.npz'), goals=goals, operations=ops)
            scores = {'':S.scores(q, alphabet, ref), '-product':S.scores(fact, alphabet, ref)}
            for li, lineage in enumerate(cfg['development_lineages']):
                prior = idx[tier,budget,arm,lineage,draw,seed]
                for k,v in scores[''].items():
                    error = abs(float(v[li])-prior[k]); max_error = max(max_error,error)
                    if error>1e-10: raise ValueError(f'original number reproduction: {k}')
                reproduced += 1
                for suffix, values in scores.items():
                    excess = float(values['loss'][li])-native_loss[lineage]
                    if excess < -1e-10: raise ValueError('negative excess')
                    rows.append(dict(tier=tier,budget=budget,arm=arm+suffix,lineage=lineage,draw=draw,seed=seed,frames=len(keys),
                                     **{k:float(v[li]) for k,v in values.items()},excess_loss=excess))
            frames_scored += len(keys)
            timing.append(dict(tier=tier,draw=draw,seed=seed,budget=budget,arm=arm,cpu_seconds=time.process_time()-tick))
    if reproduced!=len(old): raise ValueError('original coverage')
    (root/'factorization_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'evaluator/NATIVE_SCORES.json', oracle_rows)
    write(root/'ORIGINAL_REPRODUCTION.json',dict(passed=True,cells=reproduced,max_error=max_error,metrics='all original metrics'))
    write(root/'TIMING.jsonl',dict(measurements=timing))
    checks['positive:complete_original_reproduction']=True
    return dict(controls=checks,rows=len(rows),original_cells_reproduced=reproduced,original_max_error=max_error,
                frame_forecasts=frames_scored,packets=len(packets['packets']),native_rows=len(oracle_rows),**aggregate(rows,cfg),
                fits=0,scope='constructed-method learned goal/operation dependence; native posterior products are separate evaluator rulers; no support restriction or historical correspondence')
