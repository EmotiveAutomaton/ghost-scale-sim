"""Frozen joint, independent step pairs, and independent goal/operation coordinates.

No fitting, evidence restriction or latent policy enters a learned forecast.
Native posterior products are separate evaluator-only references.
"""
from itertools import product
import gzip
import json
import time
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from . import joint_support as S
from .joint_uncertainty import distribution
from .temporal_factorization import factorize as step_factorize

N = 5832
ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
FIELDS = ('tier', 'budget', 'arm', 'lineage', 'draw', 'seed')


# Integer label: goal sequence in base 3, then operation sequence in base 6.
PAIRS = np.array([[(k//216//(3**(2-t)))%3*6+(k%216//(6**(2-t)))%6
                   for t in range(3)] for k in range(N)], dtype=np.int64)

def factorize(q):
    pair, steps = step_factorize(q)
    cube = steps.reshape(-1, 3, 3, 6)
    goals = cube.sum(axis=3)
    operations = cube.sum(axis=2)
    single = np.ones_like(pair)
    for t in range(3):
        single *= goals[:, t, PAIRS[:, t]//6] * operations[:, t, PAIRS[:, t]%6]
    if not np.allclose(single.sum(1), 1, rtol=0, atol=6e-12):
        raise ValueError('single product normalization')
    return pair, single, steps, goals, operations


def controls():
    q = np.zeros((1, N)); q[0, [0, 1980]] = .5
    pair, single, *_ = factorize(q)
    expected = np.zeros_like(q); expected[0, [0, 36, 1944, 1980]] = .25
    return {'live:within_step_dependence_removed': bool(np.array_equal(single, expected)),
            'placebo:fully_independent_identity': bool(np.array_equal(factorize(expected)[1], expected)),
            'positive:independent_steps_pair_identity': bool(np.array_equal(pair, q)),
            'positive:incompatible_pairs_created': bool(single[0, 36] == .25 and single[0, 1944] == .25)}


def aggregate(rows, cfg):
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages', 'training_draws', 'fit_seeds', 'budgets'))
    arms = tuple(a+s for a in ARMS for s in ('', '-step-product', '-single-product'))
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
                v = np.array([[[idx[tier, budget, arm+'-single-product', l, d, s]['loss']-idx[tier, budget, arm+'-step-product', l, d, s]['loss']
                               for s in ss] for d in ds] for l in ls])
                contrasts.append(dict(tier=tier, budget=budget, arm=arm+'-single-product', baseline=arm+'-step-product', **estimate(v)))
                curve.append(v)
            area = np.trapezoid(curve, x=np.log(bs), axis=0)/np.log(bs[-1]/bs[0])
            areas.append(dict(tier=tier, arm=arm+'-single-product', baseline=arm+'-step-product', **estimate(area)))
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
    temporal_rows = json.loads(gzip.decompress((base/'temporal/temporal_points.json.gz').read_bytes()))
    temporal_idx = {tuple(r[k] for k in FIELDS): r for r in temporal_rows}
    temporal_expected = set(product(cfg['tiers'], cfg['budgets'], tuple(a+x for a in ARMS for x in ('', '-product')), cfg['development_lineages'], cfg['training_draws'], cfg['fit_seeds']))
    if len(temporal_idx) != len(temporal_rows) or set(temporal_idx) != temporal_expected: raise ValueError('temporal roster')
    native_temporal = read(base/'temporal/NATIVE_SCORES.json')
    nt_idx = {(r['tier'],r['lineage'],r['arm']):r for r in native_temporal}
    if len(nt_idx) != len(native_temporal) or set(nt_idx) != set(product(cfg['tiers'],cfg['development_lineages'],('native-joint','native-product'))): raise ValueError('native temporal roster')
    temporal_error = 0.
    rows = []; oracle_rows = []; timing = []; reproduced = 0; max_error = 0.; frames_scored = 0
    (root/'marginals').mkdir()
    for tier in cfg['tiers']:
        keys = sorted(k for k,p in packets['packets'].items() if p['tier']==tier)
        # All labels allowed: this comparison makes no public support restriction.
        ref = S.reference_arrays(refs, tier, keys, cfg['development_lineages'], np.ones((len(keys), N), bool))
        if set(r['lineage'] for r in refs if r['tier']==tier) != set(cfg['development_lineages']): raise ValueError('native lineage roster')
        # Native laws are evaluator-only rulers, one record per tier/lineage.
        for li, lineage in enumerate(cfg['development_lineages']):
            pulse(phase='native-within-step-factorization', tier=tier, lineage=lineage)
            r = next(r for r in refs if r['tier']==tier and r['lineage']==lineage)
            frames = {f['frame']:f for f in r['frames']}
            if len(frames)!=len(r['frames']) or set(frames)!=set(keys): raise ValueError('native frame roster')
            q = np.zeros((len(keys), N))
            for i, key in enumerate(keys):
                target = frames[key]['target']
                if len(dict(target))!=len(target) or any(w<=0 for _,w in target): raise ValueError('native target')
                for k,w in target:q[i,k]=w
            pair, single, steps, goals, operations = factorize(q)
            np.savez_compressed(root/'evaluator'/f'{tier}-{lineage}-marginals.npz', steps=steps, goals=goals, operations=operations)
            one = dict(ref, weighted=ref['weighted'][li:li+1], mass=ref['mass'][li:li+1], squared=ref['squared'][li:li+1])
            for arm, prob in (('native-joint', q), ('native-step-product', pair), ('native-single-product', single)):
                scores = S.scores(prob, S.LABELS, one)
                row = dict(tier=tier,lineage=lineage,arm=arm,frames=len(keys),**{k:float(v[0]) for k,v in scores.items()})
                if arm != 'native-single-product':
                    prior = nt_idx[tier,lineage,'native-product' if arm == 'native-step-product' else arm]
                    for k in scores:
                        e = abs(row[k]-prior[k]); temporal_error = max(temporal_error,e)
                        if e > 1e-10: raise ValueError('native temporal reproduction')
                oracle_rows.append(row)
        native_loss = {r['lineage']:r['loss'] for r in oracle_rows if r['tier']==tier and r['arm']=='native-joint'}
        for draw, seed, budget, arm in product(cfg['training_draws'], cfg['fit_seeds'], cfg['budgets'], ARMS):
            pulse(phase='learned-within-step-factorization', tier=tier, draw=draw, seed=seed, budget=budget, arm=arm)
            tick = time.process_time(); stem = f'{draw}-{tier}-{seed}-{budget}-{arm}'
            if read(base/'forecasts'/(stem+'-frames.json'))!=keys: raise ValueError('forecast frame roster')
            with np.load(base/'forecasts'/(stem+'.npz'), allow_pickle=False) as data:
                if set(data.files)!={'probabilities','alphabet'}: raise ValueError('forecast fields')
                alphabet = data['alphabet']; q = distribution(data['probabilities'], alphabet, budget)
            if len(q)!=len(keys): raise ValueError('forecast length')
            pair, single, steps, goals, operations = factorize(q)
            np.savez_compressed(root/'marginals'/(stem+'.npz'), steps=steps, goals=goals, operations=operations)
            scores = {'':S.scores(q, alphabet, ref), '-step-product':S.scores(pair, alphabet, ref), '-single-product':S.scores(single, alphabet, ref)}
            for li, lineage in enumerate(cfg['development_lineages']):
                prior = idx[tier,budget,arm,lineage,draw,seed]
                for k,v in scores[''].items():
                    error = abs(float(v[li])-prior[k]); max_error = max(max_error,error)
                    if error>1e-10: raise ValueError(f'original number reproduction: {k}')
                for k,v in scores['-step-product'].items():
                    e = abs(float(v[li])-temporal_idx[tier,budget,arm+'-product',lineage,draw,seed][k]); temporal_error=max(temporal_error,e)
                    if e > 1e-10: raise ValueError('learned temporal reproduction')
                reproduced += 1
                for suffix, values in scores.items():
                    excess = float(values['loss'][li])-native_loss[lineage]
                    if excess < -1e-10: raise ValueError('negative excess')
                    rows.append(dict(tier=tier,budget=budget,arm=arm+suffix,lineage=lineage,draw=draw,seed=seed,frames=len(keys),
                                     **{k:float(v[li]) for k,v in values.items()},excess_loss=excess))
            frames_scored += len(keys)
            timing.append(dict(tier=tier,draw=draw,seed=seed,budget=budget,arm=arm,cpu_seconds=time.process_time()-tick))
    if reproduced!=len(old): raise ValueError('original coverage')
    (root/'within_step_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'evaluator/NATIVE_SCORES.json', oracle_rows)
    write(root/'ORIGINAL_REPRODUCTION.json',dict(passed=True,cells=reproduced,max_error=max_error,temporal_max_error=temporal_error,metrics='all original and temporal product metrics'))
    write(root/'TIMING.jsonl',dict(measurements=timing))
    checks['positive:complete_original_reproduction']=True
    return dict(controls=checks,rows=len(rows),original_cells_reproduced=reproduced,original_max_error=max_error,temporal_max_error=temporal_error,
                frame_forecasts=frames_scored,packets=len(packets['packets']),native_rows=len(oracle_rows),**aggregate(rows,cfg),
                fits=0,scope='constructed-method within-step dependence after temporal factorization; native posterior products are separate evaluator rulers; no support restriction or historical correspondence')
