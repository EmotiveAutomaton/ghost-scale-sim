"""Frozen goal dependence conditional on complete witnessed operations.

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


N = 5832
ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
FIELDS = ('tier', 'budget', 'arm', 'lineage', 'draw', 'seed')


GOALS = np.array([(g//9,g//3%3,g%3) for g in range(27)])


def factorize(q, operations):
    q = np.asarray(q, float); operations = np.asarray(operations)
    if q.ndim != 2 or q.shape[1] != N or not np.isfinite(q).all() or np.any(q < 0): raise ValueError('distribution')
    if not np.allclose(q.sum(1),1,rtol=0,atol=1e-12): raise ValueError('normalization')
    if operations.shape != (len(q),) or not np.issubdtype(operations.dtype,np.integer) or np.any(operations<0) or np.any(operations>=216): raise ValueError('operations')
    labels=216*np.arange(27)[None,:]+operations[:,None]
    cond=q[np.arange(len(q))[:,None],labels].copy()
    # Preserve the accepted public-support decoder's binary64 normalizer.
    z=(q*(np.arange(N)[None,:]%216==operations[:,None])).sum(1)
    if np.any(z<=0): raise ValueError('zero witnessed-operation mass')
    cond/=z[:,None]
    cube=cond.reshape(-1,3,3,3)
    marg=np.stack([cube.sum(axis=tuple(a for a in (1,2,3) if a!=t+1)) for t in range(3)],axis=1)
    fact=np.ones_like(cond)
    for t in range(3):fact*=marg[:,t,GOALS[:,t]]
    restricted=np.zeros_like(q);result=np.zeros_like(q)
    restricted[np.arange(len(q))[:,None],labels]=cond
    result[np.arange(len(q))[:,None],labels]=fact
    if not np.allclose(result.sum(1),1,rtol=0,atol=3e-12): raise ValueError('product normalization')
    return restricted,result,marg,z


def controls():
    q=np.zeros((1,N));q[0,[0,1728]]=.5
    r,f,_,_=factorize(q,np.array([0]));expected=np.zeros_like(q);expected[0,[0,432,1296,1728]]=.25
    return {'live:goal_dependence_removed':bool(np.array_equal(f,expected)),
            'placebo:independent_goal_identity':bool(np.array_equal(factorize(expected,np.array([0]))[1],expected)),
            'positive:fixed_operations_preserved':bool(np.array_equal(r,q)),
            'positive:incompatible_goals_created':bool(f[0,432]==.25)}


def aggregate(rows, cfg):
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages', 'training_draws', 'fit_seeds', 'budgets'))
    arms = tuple(a+s for a in ARMS for s in ('', '-restricted', '-goal-product'))
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
                v = np.array([[[idx[tier, budget, arm+'-goal-product', l, d, s]['loss']-idx[tier, budget, arm+'-restricted', l, d, s]['loss']
                               for s in ss] for d in ds] for l in ls])
                contrasts.append(dict(tier=tier, budget=budget, arm=arm+'-goal-product', baseline=arm+'-restricted', **estimate(v)))
                curve.append(v)
            area = np.trapezoid(curve, x=np.log(bs), axis=0)/np.log(bs[-1]/bs[0])
            areas.append(dict(tier=tier, arm=arm+'-goal-product', baseline=arm+'-restricted', **estimate(area)))
    return dict(contrasts=contrasts, normalized_log_budget_area=areas, means=means)


def run(root, plan, pulse):
    cfg = plan['design']; base = root/'inputs'; checks = controls()
    write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('known-answer controls')
    for n, h in cfg['input_files'].items():
        if file_digest(base/n) != h: raise ValueError('input binding')
    parent = read(base/'PLAN.json')['design']
    for k in ('tiers', 'budgets', 'training_draws', 'fit_seeds', 'development_lineages'):
        if k == 'tiers':
            if cfg[k] != ['E2-full'] or 'E2-full' not in parent[k]: raise ValueError('witness tier')
        elif parent[k] != cfg[k]: raise ValueError('parent population changed')
    packets = read(base/'reader/PACKETS.json'); refs = read(base/'evaluator/REFERENCES.json')
    packets = dict(packets,packets={k:p for k,p in packets['packets'].items() if p['tier']=='E2-full'})
    refs = [r for r in refs if r['tier']=='E2-full']
    for key, p in packets['packets'].items():
        S.validate(p)
        if key != digest(p): raise ValueError('packet identity')
    write(root/'reader/PACKETS.json', packets); write(root/'evaluator/REFERENCES.json', refs)
    old = [r for r in read(base/'SUMMARY.json')['cells'] if r['tier']=='E2-full']; idx = {tuple(r[k] for k in FIELDS):r for r in old}
    expected = set(product(cfg['tiers'], cfg['budgets'], ARMS, cfg['development_lineages'], cfg['training_draws'], cfg['fit_seeds']))
    if len(idx) != len(old) or set(idx) != expected: raise ValueError('parent complete roster')
    support_rows = json.loads(gzip.decompress((base/'support/support_points.json.gz').read_bytes()))
    support_rows = [r for r in support_rows if r['tier']=='E2-full']
    support_idx = {tuple(r[k] for k in FIELDS):r for r in support_rows}
    support_arms = tuple(ARMS)+tuple(a+'-restricted' for a in ARMS)+('uniform-support',)
    expected_support=set(product(cfg['tiers'],cfg['budgets'],support_arms,cfg['development_lineages'],cfg['training_draws'],cfg['fit_seeds']))
    if len(support_idx)!=len(support_rows) or set(support_idx)!=expected_support:raise ValueError('support roster')
    support_error=0.
    rows = []; oracle_rows = []; timing = []; reproduced = 0; max_error = 0.; frames_scored = 0
    (root/'marginals').mkdir()
    for tier in cfg['tiers']:
        keys = sorted(k for k,p in packets['packets'].items() if p['tier']==tier)
        operations=np.array([sum(S.L.OPERATIONS.index(e['operation'])*6**(2-t) for t,e in enumerate(packets['packets'][k]['inputs']['observations'])) for k in keys])
        if read(base/'support/E2-full-frames.json')!=keys:raise ValueError('support frames')
        with np.load(base/'support/E2-full-support.npz',allow_pickle=False) as data:masks=data['masks']
        expected_masks=np.arange(N)[None,:]%216==operations[:,None]
        if not np.array_equal(masks,expected_masks):raise ValueError('witness support differs')
        ref = S.reference_arrays(refs, tier, keys, cfg['development_lineages'], np.ones((len(keys), N), bool))
        if set(r['lineage'] for r in refs if r['tier']==tier) != set(cfg['development_lineages']): raise ValueError('native lineage roster')
        # Native laws are evaluator-only rulers, one record per tier/lineage.
        for li, lineage in enumerate(cfg['development_lineages']):
            pulse(phase='native-witnessed-goals', tier=tier, lineage=lineage)
            r = next(r for r in refs if r['tier']==tier and r['lineage']==lineage)
            frames = {f['frame']:f for f in r['frames']}
            if len(frames)!=len(r['frames']) or set(frames)!=set(keys): raise ValueError('native frame roster')
            q = np.zeros((len(keys), N))
            for i, key in enumerate(keys):
                target = frames[key]['target']
                if len(dict(target))!=len(target) or any(w<=0 for _,w in target): raise ValueError('native target')
                for k,w in target:q[i,k]=w
            restricted, factored, goals, normalizers = factorize(q, operations)
            np.savez_compressed(root/'evaluator'/f'{tier}-{lineage}-marginals.npz', goals=goals, normalizers=normalizers, operations=operations)
            one = dict(ref, weighted=ref['weighted'][li:li+1], mass=ref['mass'][li:li+1], squared=ref['squared'][li:li+1])
            for arm, prob in (('native-joint', q), ('native-restricted', restricted), ('native-goal-product', factored)):
                scores = S.scores(prob, S.LABELS, one)
                row = dict(tier=tier,lineage=lineage,arm=arm,frames=len(keys),**{k:float(v[0]) for k,v in scores.items()})
                oracle_rows.append(row)
        native_loss = {r['lineage']:r['loss'] for r in oracle_rows if r['tier']==tier and r['arm']=='native-joint'}
        for draw, seed, budget, arm in product(cfg['training_draws'], cfg['fit_seeds'], cfg['budgets'], ARMS):
            pulse(phase='learned-witnessed-goals', tier=tier, draw=draw, seed=seed, budget=budget, arm=arm)
            tick = time.process_time(); stem = f'{draw}-{tier}-{seed}-{budget}-{arm}'
            if read(base/'forecasts'/(stem+'-frames.json'))!=keys: raise ValueError('forecast frame roster')
            with np.load(base/'forecasts'/(stem+'.npz'), allow_pickle=False) as data:
                if set(data.files)!={'probabilities','alphabet'}: raise ValueError('forecast fields')
                alphabet = data['alphabet']; q = distribution(data['probabilities'], alphabet, budget)
            if len(q)!=len(keys): raise ValueError('forecast length')
            restricted, factored, goals, normalizers = factorize(q, operations)
            np.savez_compressed(root/'marginals'/(stem+'.npz'), goals=goals, normalizers=normalizers, operations=operations)
            scores = {'':S.scores(q, alphabet, ref), '-restricted':S.scores(restricted, alphabet, ref), '-goal-product':S.scores(factored, alphabet, ref)}
            for li, lineage in enumerate(cfg['development_lineages']):
                prior = idx[tier,budget,arm,lineage,draw,seed]
                for k,v in scores[''].items():
                    error = abs(float(v[li])-prior[k]); max_error = max(max_error,error)
                    if error>1e-10: raise ValueError(f'original number reproduction: {k}')
                for k,v in scores['-restricted'].items():
                    e = abs(float(v[li])-support_idx[tier,budget,arm+'-restricted',lineage,draw,seed][k]); support_error=max(support_error,e)
                    if e > 1e-10: raise ValueError('public support reproduction')
                reproduced += 1
                for suffix, values in scores.items():
                    excess = float(values['loss'][li])-native_loss[lineage]
                    if excess < -1e-10: raise ValueError('negative excess')
                    rows.append(dict(tier=tier,budget=budget,arm=arm+suffix,lineage=lineage,draw=draw,seed=seed,frames=len(keys),
                                     **{k:float(v[li]) for k,v in values.items()},excess_loss=excess))
            frames_scored += len(keys)
            timing.append(dict(tier=tier,draw=draw,seed=seed,budget=budget,arm=arm,cpu_seconds=time.process_time()-tick))
    if reproduced!=len(old): raise ValueError('original coverage')
    (root/'witnessed_goal_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'evaluator/NATIVE_SCORES.json', oracle_rows)
    write(root/'ORIGINAL_REPRODUCTION.json',dict(passed=True,cells=reproduced,max_error=max_error,support_max_error=support_error,metrics='all original and public support metrics'))
    write(root/'TIMING.jsonl',dict(measurements=timing))
    checks['positive:complete_original_reproduction']=True
    return dict(controls=checks,rows=len(rows),original_cells_reproduced=reproduced,original_max_error=max_error,support_max_error=support_error,
                frame_forecasts=frames_scored,packets=len(packets['packets']),native_rows=len(oracle_rows),**aggregate(rows,cfg),
                fits=0,scope='constructed-method goal dependence conditional on public witnessed operations; native rulers separate; no historical correspondence or new fit')
