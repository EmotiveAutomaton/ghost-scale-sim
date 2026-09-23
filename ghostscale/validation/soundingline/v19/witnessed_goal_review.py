"""Independent witnessed-goal normalization, goal products and complete scores.

Integer labels and accurate scalar sums are independent of producer tensor axes.
Verified saved binary64 normalizers/marginals preserve the executed modal ties.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from .joint_factorization_review import (
    N, ARMS, FIELDS, LABELS, close, expand, references, priority_for, score,
    compare_groups, validate,
)

OPERATIONS=('edit-claim','repair-evidence','replace-presentation','accept-tool','inspect','undo')


def witness(packet):
    validate(packet)
    if packet['tier']!='E2-full': raise ValueError('complete witness required')
    events=packet['inputs']['observations']
    if len(events)!=3 or [e['step'] for e in events]!=[0,1,2]: raise ValueError('witness order')
    code=0
    for event in events: code=6*code+OPERATIONS.index(event['operation'])
    return code


def conditional(q, operation):
    q=np.asarray(q)
    if q.shape!=(N,) or not np.isfinite(q).all() or np.any(q<0): raise ValueError('distribution')
    if type(operation) is not int or not 0<=operation<216: raise ValueError('operation code')
    close(math.fsum(q.tolist()),1.,1e-12)
    labels=[216*g+operation for g in range(27)]
    z=math.fsum(float(q[k]) for k in labels)
    if z<=0: raise ValueError('zero witnessed-operation mass')
    cond=np.array([float(q[k])/z for k in labels])
    marg=np.array([[math.fsum(float(cond[g]) for g in range(27) if g//3**(2-t)%3==v)
                    for v in range(3)] for t in range(3)])
    return labels,cond,marg,z


def reconstruct(q, operation, saved_goals, saved_normalizer):
    labels,cond,marg,z=conditional(q,operation)
    saved=np.asarray(saved_goals); saved_z=float(saved_normalizer)
    if saved_z<=0: raise ValueError('saved normalizer')
    error=max(close(z,saved_z,1e-12),close(marg,saved,1e-12))
    restricted=np.zeros(N);fact=np.zeros(N)
    for g,k in enumerate(labels):
        restricted[k]=q[k]/saved_z
        fact[k]=math.prod(float(saved[t,g//3**(2-t)%3]) for t in range(3))
    scalar=np.array([math.prod(float(marg[t,g//3**(2-t)%3]) for t in range(3)) for g in range(27)])
    error=max(error,close(cond,restricted[labels],1e-12),close(scalar,fact[labels],1e-12))
    close(math.fsum(restricted.tolist()),1.,2e-12)
    close(math.fsum(fact.tolist()),1.,3e-12)
    return restricted,fact,error


def controls():
    q=np.zeros(N);q[[0,1728]]=.5
    _,_,m,z=conditional(q,0);r,f,_=reconstruct(q,0,m,z)
    independent=np.zeros(N);independent[[0,432,1296,1728]]=.25
    _,_,im,iz=conditional(independent,0)
    ref=(np.array([0,1728]),np.array([[.5,.5]]),np.array([1.]))
    a=score(q,priority_for(LABELS),ref);b=score(f,priority_for(LABELS),ref)
    return {'live:dependent_goal_loss':abs(float(b['loss'][0]-a['loss'][0])-math.log(2))<1e-12,
        'placebo:independent_goal_identity':bool(np.array_equal(reconstruct(independent,0,im,iz)[1],independent)),
        'positive:fixed_operations_preserved':bool(np.array_equal(r,q)),
        'positive:incompatible_goal_paths':bool(b['compatible_mass'][0]==.5),
        'positive:candidate_coverage':bool(b['candidate_coverage'][0]==1.)}


def regroup(rows, cfg):
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages', 'training_draws', 'fit_seeds', 'budgets'))
    arms = tuple(a+s for a in ARMS for s in ('', '-restricted', '-goal-product'))
    expected = set(product(cfg['tiers'], bs, arms, ls, ds, ss))
    if len(idx) != len(rows) or set(idx) != expected: raise ValueError('complete stratum roster')
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls), size=(cfg['bootstrap_resamples'], len(ls)))
    counts = np.array([np.bincount(s, minlength=len(ls)) for s in samples])
    mean = lambda values: math.fsum(values)/len(values)
    def estimate(values):
        v = [mean([values[l, d, s] for d, s in product(ds, ss)]) for l in ls]
        boot = counts @ np.array(v)/len(ls)
        return dict(mean=mean(v), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)),
            lineage_values=v, draw_means=[mean([values[l, d, s] for l, s in product(ls, ss)]) for d in ds],
            feature_seed_means=[mean([values[l, d, s] for l, d in product(ls, ds)]) for s in ss])
    contrasts = []; areas = []; means = []; span = math.log(bs[-1]/bs[0])
    if span <= 0 or any(b >= c for b, c in zip(bs, bs[1:])): raise ValueError('ordered budgets')
    metrics = sorted(set(rows[0])-set(FIELDS)-{'frames'})
    for tier in cfg['tiers']:
        for arm, budget in product(arms, bs):
            subset = [idx[tier, budget, arm, l, d, s] for l, d, s in product(ls, ds, ss)]
            means.append(dict(tier=tier, budget=budget, arm=arm, **{k: mean([r[k] for r in subset]) for k in metrics}))
        for arm in ARMS:
            curves = {}
            for budget in bs:
                v = {(l, d, s): idx[tier, budget, arm+'-goal-product', l, d, s]['loss']-idx[tier, budget, arm+'-restricted', l, d, s]['loss']
                     for l, d, s in product(ls, ds, ss)}
                curves[budget] = v
                contrasts.append(dict(tier=tier, budget=budget, arm=arm+'-goal-product', baseline=arm+'-restricted', **estimate(v)))
            area = {k: math.fsum((curves[b][k]+curves[c][k])*.5*(math.log(c)-math.log(b)) for b, c in zip(bs, bs[1:]))/span
                    for k in product(ls, ds, ss)}
            areas.append(dict(tier=tier, arm=arm+'-goal-product', baseline=arm+'-restricted', **estimate(area)))
    return dict(contrasts=contrasts, normalized_log_budget_area=areas, means=means)


def run(root, plan, pulse):
    checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('known-answer controls')
    cfg = plan['design']; inputs = root/'inputs'; original = inputs/'original'; parent = inputs/'parent'
    for n, h in cfg['input_files'].items():
        if file_digest(inputs/n) != h: raise ValueError('input hash')
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan')
    design = read(original/'PLAN.json')['design']; prior_design = read(parent/'PLAN.json')['design']
    for k in ('tiers', 'budgets', 'training_draws', 'fit_seeds', 'development_lineages'):
        if k == 'tiers':
            if design[k] != ['E2-full'] or 'E2-full' not in prior_design[k]: raise ValueError('witness tier')
        elif design[k] != prior_design[k]: raise ValueError('parent roster')
    packets = read(original/'reader/PACKETS.json')
    parent_packets=read(parent/'reader/PACKETS.json')
    selected=dict(parent_packets,packets={k:p for k,p in parent_packets['packets'].items() if p['tier']=='E2-full'})
    if packets != selected: raise ValueError('reader changed')
    for key, p in packets['packets'].items():
        validate(p)
        if key != digest(p): raise ValueError('packet identity')
    refs = [r for r in read(parent/'evaluator/REFERENCES.json') if r['tier']=='E2-full']
    if refs != read(original/'evaluator/REFERENCES.json'): raise ValueError('native targets changed')
    raw = json.loads(gzip.decompress((original/'witnessed_goal_points.json.gz').read_bytes()))
    old = [r for r in read(parent/'SUMMARY.json')['cells'] if r['tier']=='E2-full']; summary = read(original/'SUMMARY.json')
    idx = {tuple(r[k] for k in FIELDS): r for r in raw}; prior = {tuple(r[k] for k in FIELDS): r for r in old}
    expected = set(product(design['tiers'], design['budgets'], ARMS, design['development_lineages'], design['training_draws'], design['fit_seeds']))
    all_expected = set(product(design['tiers'], design['budgets'], tuple(a+s for a in ARMS for s in ('', '-restricted', '-goal-product')),
                               design['development_lineages'], design['training_draws'], design['fit_seeds']))
    if len(idx) != len(raw) or set(idx) != all_expected or len(prior) != len(old) or set(prior) != expected:
        raise ValueError('complete raw roster')
    native_raw = read(original/'evaluator/NATIVE_SCORES.json')
    native_idx = {(r['tier'], r['lineage'], r['arm']): r for r in native_raw}
    native_expected = set(product(design['tiers'], design['development_lineages'], ('native-joint', 'native-restricted', 'native-goal-product')))
    if len(native_raw) != len(native_idx) or set(native_idx) != native_expected: raise ValueError('native score roster')
    support_rows=[r for r in json.loads(gzip.decompress((parent/'support/support_points.json.gz').read_bytes())) if r['tier']=='E2-full']
    support_idx={tuple(r[k] for k in FIELDS):r for r in support_rows}
    support_arms=tuple(ARMS)+tuple(a+'-restricted' for a in ARMS)+('uniform-support',)
    support_expected=set(product(design['tiers'],design['budgets'],support_arms,design['development_lineages'],design['training_draws'],design['fit_seeds']))
    if len(support_rows)!=len(support_idx) or set(support_idx)!=support_expected: raise ValueError('support roster')
    rows = []; native_rows = []; score_error = original_error = marginal_error = support_error = 0.; frames_scored = 0
    def finish(totals): return {k: np.sum(v, axis=0) for k, v in totals.items()}
    def compare_row(computed, saved, identity):
        if set(saved) != set(identity) | set(computed): raise ValueError('metric fields')
        if any(saved[k] != v for k, v in identity.items()): raise ValueError('score identity')
        return close([computed[k] for k in computed], [saved[k] for k in computed])
    for tier in design['tiers']:
        keys = sorted(k for k, p in packets['packets'].items() if p['tier'] == tier)
        operations=np.array([witness(packets['packets'][k]) for k in keys])
        if read(parent/'support/E2-full-frames.json')!=keys: raise ValueError('support frames')
        with np.load(parent/'support/E2-full-support.npz',allow_pickle=False) as data:
            if set(data.files)!={'masks'}: raise ValueError('support fields')
            masks=data['masks']
        if not np.array_equal(masks, np.array([[k%216==o for k in range(N)] for o in operations])): raise ValueError('support mask')
        ls = design['development_lineages']; reference = references(refs, tier, keys, ls)
        entropies = {}
        for li, lineage in enumerate(ls):
            pulse(phase='independent-native-witnessed-goals', tier=tier, lineage=lineage)
            totals = {a: defaultdict(list) for a in ('native-joint', 'native-restricted', 'native-goal-product')}
            with np.load(original/'evaluator'/f'{tier}-{lineage}-marginals.npz', allow_pickle=False) as data:
                if set(data.files) != {'goals', 'normalizers', 'operations'}: raise ValueError('native marginal fields')
                g, z, o = data['goals'], data['normalizers'], data['operations']
            if g.shape != (len(keys), 3, 3) or z.shape != (len(keys),) or not np.array_equal(o, operations): raise ValueError('native marginal shape')
            for i, (labels, target, masses) in enumerate(reference):
                q = np.zeros(N); q[labels] = target[li]
                restricted, fact, error = reconstruct(q, int(o[i]), g[i], z[i]); marginal_error = max(marginal_error, error)
                ref = (labels, target[li:li+1], masses[li:li+1])
                for arm, prob in (('native-joint', q), ('native-restricted', restricted), ('native-goal-product', fact)):
                    for k, v in score(prob, priority_for(LABELS), ref).items(): totals[arm][k].append(v)
            for arm in totals:
                computed = {k: float(v[0]) for k, v in finish(totals[arm]).items()}
                identity = dict(tier=tier, lineage=lineage, arm=arm, frames=len(keys))
                score_error = max(score_error, compare_row(computed, native_idx[tier, lineage, arm], identity))
                native_rows.append(dict(identity, **computed))
                if arm == 'native-joint': entropies[lineage] = computed['loss']
        for draw, seed, budget, arm in product(design['training_draws'], design['fit_seeds'], design['budgets'], ARMS):
            pulse(phase='independent-learned-witnessed-goals', tier=tier, draw=draw, seed=seed, budget=budget, arm=arm)
            stem = f'{draw}-{tier}-{seed}-{budget}-{arm}'
            if read(parent/'forecasts'/(stem+'-frames.json')) != keys: raise ValueError('forecast frames')
            with np.load(parent/'forecasts'/(stem+'.npz'), allow_pickle=False) as data:
                if set(data.files) != {'probabilities', 'alphabet'}: raise ValueError('forecast fields')
                probabilities, alphabet = data['probabilities'], data['alphabet']
            with np.load(original/'marginals'/(stem+'.npz'), allow_pickle=False) as data:
                if set(data.files) != {'goals', 'normalizers', 'operations'}: raise ValueError('learned marginal fields')
                goals, normalizers, ops = data['goals'], data['normalizers'], data['operations']
            if probabilities.ndim != 2 or len(probabilities) != len(keys) or goals.shape != (len(keys), 3, 3) or normalizers.shape != (len(keys),) or not np.array_equal(ops, operations):
                raise ValueError('forecast/marginal shape')
            ranking = priority_for(alphabet); totals = {s: defaultdict(list) for s in ('', '-restricted', '-goal-product')}
            for i, ref in enumerate(reference):
                q = expand(probabilities[i], alphabet, budget)
                restricted, fact, error = reconstruct(q, int(ops[i]), goals[i], normalizers[i]); marginal_error = max(marginal_error, error)
                for suffix, prob in (('', q), ('-restricted', restricted), ('-goal-product', fact)):
                    for k, v in score(prob, ranking, ref).items(): totals[suffix][k].append(v)
                frames_scored += 1
            for suffix in totals:
                values = finish(totals[suffix])
                for li, lineage in enumerate(ls):
                    computed = {k: float(v[li]) for k, v in values.items()}
                    identity = dict(tier=tier, budget=budget, arm=arm+suffix, lineage=lineage, draw=draw, seed=seed, frames=len(keys))
                    key = (tier, budget, arm+suffix, lineage, draw, seed)
                    if not suffix: original_error = max(original_error, compare_row(computed, prior[key], identity))
                    if suffix=='-restricted': support_error=max(support_error,compare_row(computed,support_idx[key],identity))
                    computed['excess_loss'] = computed['loss']-entropies[lineage]
                    if computed['excess_loss'] < -1e-10: raise ValueError('negative excess')
                    score_error = max(score_error, compare_row(computed, idx[key], identity))
                    rows.append(dict(identity, **computed))
    if (len(rows) != summary['rows'] or frames_scored != summary['frame_forecasts'] or len(native_rows) != summary['native_rows']
            or len(packets['packets']) != summary['packets']): raise ValueError('whole coverage')
    groups = regroup(rows, design); original_groups = regroup(raw, design)
    expected_groups = {k: summary[k] for k in groups}
    error = compare_groups(groups, expected_groups); original_regroup_error = compare_groups(original_groups, expected_groups)
    write(root/'INDEPENDENT_REGROUP.json', groups); write(root/'ORIGINAL_ROW_REGROUP.json', original_groups)
    write(root/'NATIVE_SCORES.json', native_rows)
    (root/'review_points.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
    checks['positive:complete_reconstruction'] = True
    return dict(controls=checks, rows=len(rows), native_rows=len(native_rows), original_cells_reproduced=len(old),
        frame_forecasts=frames_scored, native_frame_laws=sum(len(r['frames']) for r in refs), packets=len(packets['packets']),
        contrasts=len(groups['contrasts']), learning_areas=len(groups['normalized_log_budget_area']), means=len(groups['means']),
        max_support_error=support_error, max_score_error=score_error, max_original_error=original_error, max_marginal_product_error=marginal_error,
        max_regroup_error=error, max_original_row_regroup_error=original_regroup_error,
        target_plan_sha256=cfg['target_plan_sha256'],
        scope='independent scalar witnessed-operation normalization and goal marginals/products; all native/learned scores and paired regroup; saved binary64 marginal products retain executed ties; inherited frozen fits/native targets; numerical adjudication pending')
