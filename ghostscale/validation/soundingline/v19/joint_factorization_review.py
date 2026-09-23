"""Independent sequence marginals, sparse native scores and paired grouping.

No producer distribution, factorization, score or regroup routine is imported.
Saved marginals are verified by scalar sums before their binary64 products are
scored, preserving the executed tie decisions rather than changing the forecast.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from .joint_uncertainty_review import validate

N = 5832
ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
FIELDS = ('tier', 'budget', 'arm', 'lineage', 'draw', 'seed')
LABELS = np.arange(N)
GOALS = np.array([k // 216 for k in range(N)])
OPERATIONS = np.array([k % 216 for k in range(N)])
DIGITS = np.array([[k//1944, k//648%3, k//216%3,
                    k//36%6, k//6%6, k%6] for k in range(N)])


def close(a, b, tolerance=1e-10):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('shape or nonfinite value')
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    if error > tolerance: raise ValueError(f'independent factorization mismatch: {error}')
    return error


def expand(p, alphabet, budget):
    p, a = np.asarray(p), np.asarray(alphabet)
    if (a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or len(set(a)) != len(a)
            or len(a) >= N or np.any(a < 0) or np.any(a >= N) or budget <= 0):
        raise ValueError('alphabet or budget')
    if p.shape != (len(a),) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('forecast')
    close(math.fsum(p.tolist()), budget/(budget+1), 1e-12)
    q = np.full(N, 1/((budget+1)*(N-len(a))))
    for k, value in zip(a, p): q[int(k)] = value
    return q


def marginals(q):
    q = np.asarray(q)
    if q.shape != (N,) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('distribution')
    values = q.tolist()
    close(math.fsum(values), 1., 1e-12)
    # Integer quotient/remainder defines membership; no shared tensor axes.
    goals = np.array([math.fsum(values[g*216:(g+1)*216]) for g in range(27)])
    operations = np.array([math.fsum(values[o::216]) for o in range(216)])
    return goals, operations


def reconstruct(q, saved_goals, saved_operations):
    goals, operations = marginals(q)
    error = max(close(goals, saved_goals, 1e-12), close(operations, saved_operations, 1e-12))
    scalar = goals[GOALS] * operations[OPERATIONS]
    # These independently checked arrays define the actual scored forecast.
    scored = np.asarray(saved_goals)[GOALS] * np.asarray(saved_operations)[OPERATIONS]
    error = max(error, close(scalar, scored, 1e-12))
    close(math.fsum(scored.tolist()), 1., 2e-12)
    return scored, error


def references(records, tier, keys, lineages):
    selected = [r for r in records if r['tier'] == tier]
    bylin = {r['lineage']: {f['frame']: f for f in r['frames']} for r in selected}
    if len(selected) != len(lineages) or set(bylin) != set(lineages): raise ValueError('native lineages')
    for r in selected:
        frames = bylin[r['lineage']]
        if len(frames) != len(r['frames']) or set(frames) != set(keys): raise ValueError('native frames')
        masses = [f['mass'] for f in frames.values()]
        if any(not math.isfinite(m) or m <= 0 for m in masses): raise ValueError('native mass')
        close(math.fsum(masses), 1.)
    result = []
    for key in keys:
        common = None; targets = []; masses = []
        for lineage in lineages:
            f = bylin[lineage][key]; pairs = f['target']; t = dict(pairs)
            if (not t or len(t) != len(pairs) or any(type(k) is not int or k < 0 or k >= N for k in t)
                    or any(not math.isfinite(w) or w <= 0 for w in t.values())): raise ValueError('native target')
            labels = sorted(t)
            if common is not None and common != labels: raise ValueError('native support roster')
            common = labels; weights = [t[k] for k in labels]
            close(math.fsum(weights), 1.)
            targets.append(weights); masses.append(f['mass'])
        result.append((np.array(common), np.array(targets), np.array(masses)))
    return result


def priority_for(alphabet):
    known = set(map(int, alphabet))
    if len(known) != len(alphabet) or any(k < 0 or k >= N for k in known): raise ValueError('ranking alphabet')
    unseen = np.array([k for k in range(N) if k not in known], dtype=int)
    priority = list(map(int, alphabet)) + unseen.tolist()
    ranks = np.empty(N, dtype=int)
    for i, k in enumerate(priority): ranks[k] = i
    return ranks, unseen


def score(q, ranking, reference):
    """Independently rank one frame and sum sparse native target contributions."""
    q = np.asarray(q); ranks, unseen = ranking; labels, target, mass = reference
    if q.shape != (N,) or not np.isfinite(q).all() or np.any(q < 0): raise ValueError('score distribution')
    close(q.sum(), 1., 2e-12)
    order = np.lexsort((ranks, -q))
    cumulative = np.cumsum(q[order])
    size = int(np.searchsorted(cumulative, .9, side='left'))+1
    if size > N: raise ValueError('candidate normalization')
    mode = int(order[0]); truth_q = q[labels]
    if np.any(truth_q <= 0): raise ValueError('zero true probability')
    included = np.zeros(N, dtype=bool); included[order[:size]] = True
    equality = DIGITS[labels] == DIGITS[mode]
    accuracy = target @ equality
    values = dict(loss=-(target @ np.log(truth_q)),
        squared_error=float(np.dot(q, q))-2*(target @ truth_q)+np.sum(target*target, axis=1),
        compatible_mass=math.fsum(truth_q.tolist()), candidate_coverage=target @ included[labels],
        candidate_size=size, top_incompatible=mode not in labels, abstain=q[mode] < .5,
        top_probability=q[mode], unknown_mass=math.fsum(q[unseen].tolist()),
        truth_outside_alphabet=target @ np.isin(labels, unseen),
        goal_accuracy=accuracy[:, :3].mean(1), operation_accuracy=accuracy[:, 3:].mean(1))
    values.update({f'goal_{i}': accuracy[:, i] for i in range(3)})
    values.update({f'operation_{i}': accuracy[:, i+3] for i in range(3)})
    return {k: mass*v for k, v in values.items()}


def regroup(rows, cfg):
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages', 'training_draws', 'fit_seeds', 'budgets'))
    arms = tuple(a+s for a in ARMS for s in ('', '-product'))
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
                v = {(l, d, s): idx[tier, budget, arm+'-product', l, d, s]['loss']-idx[tier, budget, arm, l, d, s]['loss']
                     for l, d, s in product(ls, ds, ss)}
                curves[budget] = v
                contrasts.append(dict(tier=tier, budget=budget, arm=arm+'-product', baseline=arm, **estimate(v)))
            area = {k: math.fsum((curves[b][k]+curves[c][k])*.5*(math.log(c)-math.log(b)) for b, c in zip(bs, bs[1:]))/span
                    for k in product(ls, ds, ss)}
            areas.append(dict(tier=tier, arm=arm+'-product', baseline=arm, **estimate(area)))
    return dict(contrasts=contrasts, normalized_log_budget_area=areas, means=means)


def compare_groups(actual, expected):
    error = 0.
    if set(actual) != set(expected): raise ValueError('summary groups')
    for name in actual:
        if len(actual[name]) != len(expected[name]): raise ValueError('summary roster')
        for a, b in zip(actual[name], expected[name]):
            if set(a) != set(b): raise ValueError('summary fields')
            for k in a:
                if k in ('tier', 'budget', 'arm', 'baseline'):
                    if a[k] != b[k]: raise ValueError('summary identity')
                else: error = max(error, close(a[k], b[k]))
    return error


def controls():
    q = np.zeros(N); q[[0, 217]] = .5
    g, o = marginals(q); fact, _ = reconstruct(q, g, o)
    independent = np.zeros(N); independent[[0, 1, 216, 217]] = .25
    ig, io = marginals(independent)
    ref = (np.array([0, 217]), np.array([[.5, .5]]), np.array([1.]))
    original = score(q, priority_for(LABELS), ref); factored = score(fact, priority_for(LABELS), ref)
    return {'live:correlated_joint_loss': abs(float(factored['loss'][0]-original['loss'][0])-math.log(2)) < 1e-12,
            'placebo:independent_joint_identity': bool(np.array_equal(reconstruct(independent, ig, io)[0], independent)),
            'positive:incompatible_cross_pairs': bool(factored['compatible_mass'][0] == .5),
            'positive:full_native_coverage': bool(factored['candidate_coverage'][0] == 1.)}


def run(root, plan, pulse):
    checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('known-answer controls')
    cfg = plan['design']; inputs = root/'inputs'; original = inputs/'original'; parent = inputs/'parent'
    for n, h in cfg['input_files'].items():
        if file_digest(inputs/n) != h: raise ValueError('input hash')
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan')
    design = read(original/'PLAN.json')['design']; prior_design = read(parent/'PLAN.json')['design']
    for k in ('tiers', 'budgets', 'training_draws', 'fit_seeds', 'development_lineages'):
        if design[k] != prior_design[k]: raise ValueError('parent roster')
    packets = read(original/'reader/PACKETS.json')
    if packets != read(parent/'reader/PACKETS.json'): raise ValueError('reader changed')
    for key, p in packets['packets'].items():
        validate(p)
        if key != digest(p): raise ValueError('packet identity')
    refs = read(parent/'evaluator/REFERENCES.json')
    if refs != read(original/'evaluator/REFERENCES.json'): raise ValueError('native targets changed')
    raw = json.loads(gzip.decompress((original/'factorization_points.json.gz').read_bytes()))
    old = read(parent/'SUMMARY.json')['cells']; summary = read(original/'SUMMARY.json')
    idx = {tuple(r[k] for k in FIELDS): r for r in raw}; prior = {tuple(r[k] for k in FIELDS): r for r in old}
    expected = set(product(design['tiers'], design['budgets'], ARMS, design['development_lineages'], design['training_draws'], design['fit_seeds']))
    all_expected = set(product(design['tiers'], design['budgets'], tuple(a+s for a in ARMS for s in ('', '-product')),
                               design['development_lineages'], design['training_draws'], design['fit_seeds']))
    if len(idx) != len(raw) or set(idx) != all_expected or len(prior) != len(old) or set(prior) != expected:
        raise ValueError('complete raw roster')
    native_raw = read(original/'evaluator/NATIVE_SCORES.json')
    native_idx = {(r['tier'], r['lineage'], r['arm']): r for r in native_raw}
    native_expected = set(product(design['tiers'], design['development_lineages'], ('native-joint', 'native-product')))
    if len(native_raw) != len(native_idx) or set(native_idx) != native_expected: raise ValueError('native score roster')
    rows = []; native_rows = []; score_error = original_error = marginal_error = 0.; frames_scored = 0
    def finish(totals): return {k: np.sum(v, axis=0) for k, v in totals.items()}
    def compare_row(computed, saved, identity):
        if set(saved) != set(identity) | set(computed): raise ValueError('metric fields')
        if any(saved[k] != v for k, v in identity.items()): raise ValueError('score identity')
        return close([computed[k] for k in computed], [saved[k] for k in computed])
    for tier in design['tiers']:
        keys = sorted(k for k, p in packets['packets'].items() if p['tier'] == tier)
        ls = design['development_lineages']; reference = references(refs, tier, keys, ls)
        entropies = {}
        for li, lineage in enumerate(ls):
            pulse(phase='independent-native-factorization', tier=tier, lineage=lineage)
            totals = {a: defaultdict(list) for a in ('native-joint', 'native-product')}
            with np.load(original/'evaluator'/f'{tier}-{lineage}-marginals.npz', allow_pickle=False) as data:
                if set(data.files) != {'goals', 'operations'}: raise ValueError('native marginal fields')
                g, o = data['goals'], data['operations']
            if g.shape != (len(keys), 27) or o.shape != (len(keys), 216): raise ValueError('native marginal shape')
            for i, (labels, target, masses) in enumerate(reference):
                q = np.zeros(N); q[labels] = target[li]
                fact, error = reconstruct(q, g[i], o[i]); marginal_error = max(marginal_error, error)
                ref = (labels, target[li:li+1], masses[li:li+1])
                for arm, prob in (('native-joint', q), ('native-product', fact)):
                    for k, v in score(prob, priority_for(LABELS), ref).items(): totals[arm][k].append(v)
            for arm in totals:
                computed = {k: float(v[0]) for k, v in finish(totals[arm]).items()}
                identity = dict(tier=tier, lineage=lineage, arm=arm, frames=len(keys))
                score_error = max(score_error, compare_row(computed, native_idx[tier, lineage, arm], identity))
                native_rows.append(dict(identity, **computed))
                if arm == 'native-joint': entropies[lineage] = computed['loss']
        for draw, seed, budget, arm in product(design['training_draws'], design['fit_seeds'], design['budgets'], ARMS):
            pulse(phase='independent-learned-factorization', tier=tier, draw=draw, seed=seed, budget=budget, arm=arm)
            stem = f'{draw}-{tier}-{seed}-{budget}-{arm}'
            if read(parent/'forecasts'/(stem+'-frames.json')) != keys: raise ValueError('forecast frames')
            with np.load(parent/'forecasts'/(stem+'.npz'), allow_pickle=False) as data:
                if set(data.files) != {'probabilities', 'alphabet'}: raise ValueError('forecast fields')
                probabilities, alphabet = data['probabilities'], data['alphabet']
            with np.load(original/'marginals'/(stem+'.npz'), allow_pickle=False) as data:
                if set(data.files) != {'goals', 'operations'}: raise ValueError('learned marginal fields')
                goals, ops = data['goals'], data['operations']
            if probabilities.ndim != 2 or len(probabilities) != len(keys) or goals.shape != (len(keys), 27) or ops.shape != (len(keys), 216):
                raise ValueError('forecast/marginal shape')
            ranking = priority_for(alphabet); totals = {s: defaultdict(list) for s in ('', '-product')}
            for i, ref in enumerate(reference):
                q = expand(probabilities[i], alphabet, budget)
                fact, error = reconstruct(q, goals[i], ops[i]); marginal_error = max(marginal_error, error)
                for suffix, prob in (('', q), ('-product', fact)):
                    for k, v in score(prob, ranking, ref).items(): totals[suffix][k].append(v)
                frames_scored += 1
            for suffix in totals:
                values = finish(totals[suffix])
                for li, lineage in enumerate(ls):
                    computed = {k: float(v[li]) for k, v in values.items()}
                    identity = dict(tier=tier, budget=budget, arm=arm+suffix, lineage=lineage, draw=draw, seed=seed, frames=len(keys))
                    key = (tier, budget, arm+suffix, lineage, draw, seed)
                    if not suffix: original_error = max(original_error, compare_row(computed, prior[key], identity))
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
        max_score_error=score_error, max_original_error=original_error, max_marginal_product_error=marginal_error,
        max_regroup_error=error, max_original_row_regroup_error=original_regroup_error,
        target_plan_sha256=cfg['target_plan_sha256'],
        scope='independent scalar sequence marginals and products; all native/learned scores and paired regroup; saved binary64 marginal products retain executed ties; inherited frozen fits/native targets; numerical adjudication pending')
