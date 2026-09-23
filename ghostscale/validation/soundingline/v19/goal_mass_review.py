"""Independent conditional-goal group masses, scores and paired estimates.

Scalar sums validate all saved binary64 quantities before tie-sensitive scoring.
No producer probability, score, mass matching or aggregation code is imported.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from .joint_factorization_review import (
    N, ARMS, FIELDS, close, expand, references, priority_for, score,
    compare_groups, validate,
)
from .witnessed_goal_review import witness, conditional, reconstruct

SUFFIXES = ('-restricted', '-mass-matched', '-goal-product')


def group_match(q, fact, alphabet, saved_before=None, saved_after=None):
    """Rebuild group-conditional forecasts using independent scalar sums."""
    q, fact = np.asarray(q), np.asarray(fact)
    if q.shape != (N,) or fact.shape != (N,): raise ValueError('distribution shape')
    active = np.flatnonzero((q != 0) | (fact != 0)).tolist()
    for v in (q, fact):
        if not np.isfinite(v).all() or np.any(v < 0): raise ValueError('distribution')
        close(math.fsum(float(v[k]) for k in active), 1., 3e-12)
    alphabet = np.asarray(alphabet)
    if alphabet.ndim != 1 or not np.issubdtype(alphabet.dtype, np.integer): raise ValueError('alphabet')
    known = set(map(int, alphabet))
    if len(known) != len(alphabet) or any(k < 0 or k >= N for k in known): raise ValueError('alphabet')
    # Both omitted vectors are exactly zero; this is lossless sparse summation.
    groups = ([k for k in active if k in known], [k for k in active if k not in known])
    before = np.array([math.fsum(float(q[k]) for k in ids) for ids in groups])
    after = np.array([math.fsum(float(fact[k]) for k in ids) for ids in groups])
    error = 0.
    a, b = before, after
    if (saved_before is None) != (saved_after is None): raise ValueError('paired saved masses')
    if saved_before is not None:
        a, b = np.asarray(saved_before), np.asarray(saved_after)
        error = max(close(a, before, 1e-12), close(b, after, 1e-12))
        if np.any(a < 0) or np.any(b < 0): raise ValueError('negative saved mass')
        if not np.array_equal(a == 0, before == 0) or not np.array_equal(b == 0, after == 0):
            raise ValueError('saved support')
    result = np.zeros(N)
    for j, ids in enumerate(groups):
        if before[j] == 0:
            if after[j] > 0: raise ValueError('undefined original conditional')
            continue
        scalar = np.array([float(q[k])/before[j]*after[j] for k in ids])
        executed = np.array([float(q[k])*(b[j]/a[j]) for k in ids])
        error = max(error, close(scalar, executed, 1e-12))
        result[ids] = executed
    close(math.fsum(float(result[k]) for k in active), 1., 4e-12)
    return result, before, after, error


def loss_shift(reference, alphabet, before, after):
    """Exact group-mass contribution to logarithmic loss; target weights separate."""
    labels, target, mass = reference
    known = set(map(int, alphabet))
    deltas = []
    for label in labels:
        j = 0 if int(label) in known else 1
        if before[j] <= 0 or after[j] <= 0: raise ValueError('zero true group mass')
        deltas.append(-math.log(after[j]/before[j]))
    return np.array([math.fsum(float(w)*d for w, d in zip(t, deltas))*m
                     for t, m in zip(target, mass)])


def controls():
    q = np.zeros(N); q[:4] = [.1, .2, .3, .4]
    f = np.zeros(N); f[:4] = [.3, .3, .2, .2]
    a = np.array([0, 1]); r, before, after, _ = group_match(q, f, a)
    ref = (np.array([0, 3]), np.array([[.6, .4]]), np.array([1.]))
    old, new = score(q, priority_for(a), ref), score(r, priority_for(a), ref)
    return dict({
        'live:mass_change': bool(np.allclose(r[:4], [.2, .4, 6/35, 8/35], rtol=0, atol=1e-15)),
        'placebo:identity': bool(np.array_equal(group_match(q, q, a)[0], q)),
        'positive:conditional_odds': bool(abs(r[0]/r[1]-.5) < 1e-15),
        'positive:loss_decomposition': bool(close(new['loss']-old['loss'], loss_shift(ref, a, before, after)) < 1e-12),
    })


def regroup(rows, cfg):
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages', 'training_draws', 'fit_seeds', 'budgets'))
    arms = tuple(a+s for a in ARMS for s in SUFFIXES)
    expected = set(product(cfg['tiers'], bs, arms, ls, ds, ss))
    if len(idx) != len(rows) or set(idx) != expected: raise ValueError('complete stratum roster')
    if len(bs) < 2 or any(b >= c for b, c in zip(bs, bs[1:])): raise ValueError('ordered budgets')
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls), size=(cfg['bootstrap_resamples'], len(ls)))
    counts = np.array([np.bincount(s, minlength=len(ls)) for s in samples])
    mean = lambda v: math.fsum(v)/len(v)
    def estimate(values):
        v = [mean([values[l, d, s] for d, s in product(ds, ss)]) for l in ls]
        boot = counts @ np.array(v)/len(ls)
        return dict(mean=mean(v), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)),
                    lineage_values=v, draw_means=[mean([values[l, d, s] for l, s in product(ls, ss)]) for d in ds],
                    feature_seed_means=[mean([values[l, d, s] for l, d in product(ls, ds)]) for s in ss])
    contrasts = []; areas = []; means = []; span = math.log(bs[-1]/bs[0])
    metrics = sorted(set(rows[0])-set(FIELDS)-{'frames'})
    for tier in cfg['tiers']:
        for arm, budget in product(arms, bs):
            subset = [idx[tier, budget, arm, l, d, s] for l, d, s in product(ls, ds, ss)]
            means.append(dict(tier=tier, budget=budget, arm=arm, **{k: mean([r[k] for r in subset]) for k in metrics}))
        for arm in ARMS:
            for suffix, baseline in (('-mass-matched', '-restricted'), ('-goal-product', '-mass-matched')):
                curves = {}
                for budget in bs:
                    values = {(l, d, s): idx[tier, budget, arm+suffix, l, d, s]['loss']-idx[tier, budget, arm+baseline, l, d, s]['loss']
                              for l, d, s in product(ls, ds, ss)}
                    curves[budget] = values
                    contrasts.append(dict(tier=tier, budget=budget, arm=arm+suffix, baseline=arm+baseline, **estimate(values)))
                area = {key: math.fsum((curves[b][key]+curves[c][key])*.5*(math.log(c)-math.log(b)) for b, c in zip(bs, bs[1:]))/span
                        for key in product(ls, ds, ss)}
                areas.append(dict(tier=tier, arm=arm+suffix, baseline=arm+baseline, **estimate(area)))
    return dict(contrasts=contrasts, normalized_log_budget_area=areas, means=means)


def run(root, plan, pulse):
    checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('known-answer controls')
    cfg = plan['design']; inputs = root/'inputs'; original = inputs/'original'; parent = inputs/'parent'
    for n, h in cfg['input_files'].items():
        if file_digest(inputs/n) != h: raise ValueError('input hash')
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan')
    design = read(original/'PLAN.json')['design']; prior_design = read(parent/'witnessed/PLAN.json')['design']
    for k in ('tiers', 'budgets', 'training_draws', 'fit_seeds', 'development_lineages'):
        if design[k] != prior_design[k]: raise ValueError('parent population')
    if design['tiers'] != ['E2-full']: raise ValueError('complete witnesses required')
    packets = read(original/'reader/PACKETS.json'); pp = read(parent/'reader/PACKETS.json')
    if packets != dict(pp, packets={k:p for k,p in pp['packets'].items() if p['tier']=='E2-full'}): raise ValueError('reader changed')
    for key, p in packets['packets'].items():
        validate(p)
        if key != digest(p): raise ValueError('packet identity')
    refs = [r for r in read(parent/'evaluator/REFERENCES.json') if r['tier']=='E2-full']
    if refs != read(original/'evaluator/REFERENCES.json'): raise ValueError('native targets changed')
    native = read(parent/'witnessed/NATIVE_SCORES.json')
    if native != read(original/'evaluator/INHERITED_NATIVE_SCORES.json'): raise ValueError('native scores changed')
    native_idx = {(r['tier'],r['lineage'],r['arm']):r for r in native}
    if len(native_idx)!=len(native) or set(native_idx)!=set(product(design['tiers'],design['development_lineages'],('native-joint','native-restricted','native-goal-product'))): raise ValueError('native roster')
    raw = json.loads(gzip.decompress((original/'goal_mass_points.json.gz').read_bytes()))
    old = json.loads(gzip.decompress((parent/'witnessed/witnessed_goal_points.json.gz').read_bytes()))
    idx = {tuple(r[k] for k in FIELDS):r for r in raw}; prior = {tuple(r[k] for k in FIELDS):r for r in old}
    def roster(arms):return set(product(design['tiers'],design['budgets'],arms,design['development_lineages'],design['training_draws'],design['fit_seeds']))
    if len(idx)!=len(raw) or set(idx)!=roster(tuple(a+s for a in ARMS for s in SUFFIXES)): raise ValueError('raw roster')
    if len(prior)!=len(old) or set(prior)!=roster(tuple(a+s for a in ARMS for s in ('','-restricted','-goal-product'))): raise ValueError('parent roster')
    tier = 'E2-full'; keys = sorted(packets['packets']); ops = np.array([witness(packets['packets'][k]) for k in keys])
    ls = design['development_lineages']; reference = references(refs, tier, keys, ls)
    entropies = np.zeros(len(ls))
    for labels, target, masses in reference:
        for i, (t, mass) in enumerate(zip(target, masses)):entropies[i] -= mass*math.fsum(float(w)*math.log(float(w)) for w in t)
    for i,l in enumerate(ls):close(entropies[i], native_idx[tier,l,'native-joint']['loss'])
    rows = []; frames = reproduced = 0; score_error = parent_error = probability_error = decomposition_error = 0.
    def compare_row(computed, saved, identity):
        if set(saved)!=set(identity)|set(computed) or any(saved[k]!=v for k,v in identity.items()): raise ValueError('metric identity')
        return close([computed[k] for k in computed],[saved[k] for k in computed])
    for draw, seed, budget, arm in product(design['training_draws'],design['fit_seeds'],design['budgets'],ARMS):
        pulse(phase='independent-conditional-goal-mass',draw=draw,seed=seed,budget=budget,arm=arm)
        stem = f'{draw}-{tier}-{seed}-{budget}-{arm}'
        if read(parent/'forecasts'/(stem+'-frames.json'))!=keys: raise ValueError('forecast frames')
        with np.load(parent/'forecasts'/(stem+'.npz'),allow_pickle=False) as z:
            if set(z.files)!={'probabilities','alphabet'}: raise ValueError('forecast fields')
            p, alphabet = z['probabilities'], z['alphabet']
        with np.load(original/'masses'/(stem+'.npz'),allow_pickle=False) as z:
            if set(z.files)!={'goals','normalizer','original','product','operations'}: raise ValueError('mass fields')
            goals, norm, before, after, operations = (z[k] for k in ('goals','normalizer','original','product','operations'))
        if p.ndim!=2 or len(p)!=len(keys) or goals.shape!=(len(keys),3,3) or norm.shape!=(len(keys),) or before.shape!=(len(keys),2) or after.shape!=(len(keys),2) or not np.array_equal(operations,ops): raise ValueError('forecast/mass shape')
        ranking=priority_for(alphabet); totals={s:defaultdict(list) for s in SUFFIXES}
        for i,ref in enumerate(reference):
            q=expand(p[i],alphabet,budget)
            restricted,fact,error=reconstruct(q,int(ops[i]),goals[i],norm[i]);probability_error=max(probability_error,error)
            matched,a,b,error=group_match(restricted,fact,alphabet,before[i],after[i]);probability_error=max(probability_error,error)
            values={s:score(prob,ranking,ref) for s,prob in zip(SUFFIXES,(restricted,matched,fact))}
            decomposition_error=max(decomposition_error,close(values['-mass-matched']['loss']-values['-restricted']['loss'],loss_shift(ref,alphabet,a,b)))
            for s,v in values.items():
                for k,value in v.items():totals[s][k].append(value)
            frames+=1
        for suffix in SUFFIXES:
            values={k:np.sum(v,axis=0) for k,v in totals[suffix].items()}
            for li,lineage in enumerate(ls):
                computed={k:float(v[li]) for k,v in values.items()};computed['excess_loss']=computed['loss']-entropies[li]
                if computed['excess_loss'] < -1e-10: raise ValueError('negative excess')
                identity=dict(tier=tier,budget=budget,arm=arm+suffix,lineage=lineage,draw=draw,seed=seed,frames=len(keys))
                key=tuple(identity[k] for k in FIELDS)
                score_error=max(score_error,compare_row(computed,idx[key],identity))
                if suffix!='-mass-matched':parent_error=max(parent_error,compare_row(computed,prior[key],identity));reproduced+=1
                rows.append(dict(identity,**computed))
    summary=read(original/'SUMMARY.json')
    if len(rows)!=summary['rows'] or frames!=summary['frame_forecasts'] or len(keys)!=summary['packets'] or reproduced!=summary['witnessed_cells_reproduced'] or len(native)!=summary['inherited_native_rows']: raise ValueError('whole coverage')
    groups=regroup(rows,design);original_groups=regroup(raw,design);expected={k:summary[k] for k in groups}
    regroup_error=compare_groups(groups,expected);original_regroup_error=compare_groups(original_groups,expected)
    write(root/'INDEPENDENT_REGROUP.json',groups);write(root/'ORIGINAL_ROW_REGROUP.json',original_groups)
    (root/'reconstructed_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    result=dict(passed=True,rows=len(rows),frame_forecasts=frames,packets=len(keys),inherited_native_rows=len(native),parent_cells_reproduced=reproduced,
                max_probability_error=probability_error,max_score_error=score_error,max_parent_error=parent_error,max_decomposition_error=decomposition_error,
                max_regroup_error=regroup_error,max_original_regroup_error=original_regroup_error,controls=checks,
                scope='independent scalar conditional-goal masses and full scores; accepted fitted forecasts and native references inherited; binary64 masses verified before tie decisions')
    write(root/'INDEPENDENT_REVIEW.json',result)
    return result
