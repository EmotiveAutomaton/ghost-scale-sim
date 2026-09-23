"""Independent scalar reconstruction of frozen joint forecast chain rules.

No producer distribution, component, score or regroup implementation is imported.
Previously accepted fitted forecasts and native posterior masses are inherited.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest

N = 5832
ARMS = ('raw-history', 'frozen-latent', 'learned-bank', 'matched-frequency')
LOSSES = ('loss', 'operation_loss', 'conditional_goal_loss')
ENTROPIES = ('joint_entropy', 'operation_entropy', 'conditional_goal_entropy')
EXCESSES = ('joint_excess', 'operation_excess', 'conditional_goal_excess')
METRICS = LOSSES + EXCESSES
FIELDS = ('tier', 'budget', 'arm', 'lineage', 'draw', 'seed')


def close(a, b, tolerance=1e-10):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('shape or nonfinite value')
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    if error > tolerance:
        raise ValueError(f'independent uncertainty mismatch: {error}')
    return error


def validate(packet):
    tiers = ('E0', 'E1', 'E2-sparse', 'E2-full')
    if (set(packet) != {'schema', 'tier', 'inputs'} or
            packet['schema'] != 'v19.local.public.1' or packet['tier'] not in tiers):
        raise ValueError('reader schema')
    tier = packet['tier']; p = packet['inputs']; allowed = {'artifact'}
    if tier != 'E0': allowed.update(('initial', 'requested_purpose'))
    if tier.startswith('E2'): allowed.add('observations')
    if set(p) != allowed: raise ValueError('reader fields')
    for key in ('artifact', 'initial'):
        if key in p and (len(p[key]) != 3 or any(v not in (0, 1) for v in p[key])):
            raise ValueError('reader artifact')
    if 'requested_purpose' in p and p['requested_purpose'] not in (0, 1):
        raise ValueError('request')
    events = p.get('observations', [])
    if len(events) != {'E0': 0, 'E1': 0, 'E2-sparse': 1, 'E2-full': 3}[tier]:
        raise ValueError('witness length')
    for i, e in enumerate(events):
        if (set(e) != {'step', 'operation', 'before', 'after', 'tool_proposal'} or e['step'] != i
                or e['operation'] not in ('edit-claim', 'repair-evidence', 'replace-presentation', 'accept-tool', 'inspect', 'undo')):
            raise ValueError('witness fields')
        for key in ('before', 'after'):
            if len(e[key]) != 3 or any(v not in (0, 1) for v in e[key]): raise ValueError('witness artifact')
        if e['tool_proposal'] != (e['after'] if e['operation'] == 'accept-tool' else None):
            raise ValueError('tool proposal')


def expand(probabilities, alphabet, budget):
    a = np.asarray(alphabet); p = np.asarray(probabilities)
    if (a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or len(set(a)) != len(a)
            or len(a) >= N or np.any(a < 0) or np.any(a >= N) or budget <= 0):
        raise ValueError('alphabet or budget')
    if p.shape != (len(a),) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('forecast')
    close(math.fsum(map(float, p)), budget/(budget+1), 1e-12)
    q = [1/((budget+1)*(N-len(a)))] * N
    for label, value in zip(a, p): q[int(label)] = float(value)
    close(math.fsum(q), 1., 1e-12)
    return q


def native(target):
    """Sparse scalar target moments, cached once per frame and native law."""
    labels = [pair[0] for pair in target]; weights = [float(pair[1]) for pair in target]
    if (not labels or len(set(labels)) != len(labels) or any(type(k) is not int or k < 0 or k >= N for k in labels)
            or any(not math.isfinite(v) or v <= 0 for v in weights)):
        raise ValueError('native target')
    close(math.fsum(weights), 1.)
    by_op = defaultdict(list)
    for k, w in zip(labels, weights): by_op[k % 216].append(w)
    op = {k: math.fsum(v) for k, v in by_op.items()}
    entropy = -math.fsum(w*math.log(w) for w in weights)
    op_entropy = -math.fsum(w*math.log(w) for w in op.values())
    goal_entropy = -math.fsum(w*math.log(w/op[k % 216]) for k, w in zip(labels, weights))
    close(entropy, op_entropy+goal_entropy)
    return dict(labels=labels, weights=weights, operations=op, entropies=(entropy, op_entropy, goal_entropy))


def components(q, targets):
    if len(q) != N or any(not math.isfinite(x) or x < 0 for x in q): raise ValueError('distribution')
    close(math.fsum(q), 1., 1e-12)
    # Label modulo 216, with no reshape/axis convention shared with the producer.
    marginal = [math.fsum(q[k] for k in range(op, N, 216)) for op in range(216)]
    labels = set(k for t in targets for k in t['labels'])
    if any(q[k] <= 0 for k in labels): raise ValueError('zero probability true label')
    log_joint = {k: math.log(q[k]) for k in labels}
    log_goal = {k: math.log(q[k]/marginal[k % 216]) for k in labels}
    log_op = {k: math.log(v) for k, v in enumerate(marginal) if v > 0}
    rows = []; maximum = 0.
    for t in targets:
        pairs = list(zip(t['labels'], t['weights']))
        joint = -math.fsum(w*log_joint[k] for k, w in pairs)
        operation = -math.fsum(w*log_op[k] for k, w in t['operations'].items())
        conditional = -math.fsum(w*log_goal[k] for k, w in pairs)
        entropies = t['entropies']; losses = (joint, operation, conditional)
        excesses = tuple(x-y for x, y in zip(losses, entropies))
        maximum = max(maximum, close(joint, operation+conditional),
                      close(entropies[0], entropies[1]+entropies[2]),
                      close(excesses[0], excesses[1]+excesses[2]))
        if min(excesses) < -1e-10: raise ValueError('negative excess')
        rows.append(dict(zip(LOSSES+ENTROPIES+EXCESSES, losses+entropies+excesses)))
    return rows, maximum


def regroup(rows, cfg):
    """Scalar means and bootstrap multiplicities independently group all strata."""
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages', 'training_draws', 'fit_seeds', 'budgets'))
    expected = set(product(cfg['tiers'], bs, ARMS, ls, ds, ss))
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
    for tier in cfg['tiers']:
        for arm, budget in product(ARMS, bs):
            subset = [idx[tier, budget, arm, l, d, s] for l, d, s in product(ls, ds, ss)]
            means.append(dict(tier=tier, budget=budget, arm=arm,
                              **{k: mean([r[k] for r in subset]) for k in METRICS+ENTROPIES}))
        for base, metric in product((a for a in ARMS if a != 'learned-bank'), METRICS):
            curves = {}
            for budget in bs:
                values = {(l, d, s): idx[tier, budget, 'learned-bank', l, d, s][metric]-idx[tier, budget, base, l, d, s][metric]
                          for l, d, s in product(ls, ds, ss)}
                curves[budget] = values
                contrasts.append(dict(tier=tier, budget=budget, arm='learned-bank', baseline=base, metric=metric, **estimate(values)))
            area = {k: math.fsum((curves[b][k]+curves[c][k])*.5*(math.log(c)-math.log(b)) for b, c in zip(bs, bs[1:]))/span
                    for k in product(ls, ds, ss)}
            areas.append(dict(tier=tier, arm='learned-bank', baseline=base, metric=metric, **estimate(area)))
    return dict(contrasts=contrasts, normalized_log_budget_area=areas, means=means)


def compare_groups(actual, expected):
    error = 0.
    if set(actual) != set(expected): raise ValueError('summary groups')
    for name in actual:
        if len(actual[name]) != len(expected[name]): raise ValueError('summary roster')
        for a, b in zip(actual[name], expected[name]):
            if set(a) != set(b): raise ValueError('summary fields')
            for k in a:
                if k in ('tier', 'budget', 'arm', 'baseline', 'metric'):
                    if a[k] != b[k]: raise ValueError('summary identity')
                else: error = max(error, close(a[k], b[k]))
    return error


def controls():
    q = [1/N]*N
    same, _ = components(q, [native([[0, .5], [216, .5]])])
    q = [0.]*N; q[0] = .25; q[1] = .75
    exact, _ = components(q, [native([[0, .25], [1, .75]])])
    return {'live:uniform_operation_loss': abs(same[0]['operation_loss']-math.log(216)) < 1e-12,
            'positive:same_operation_goal_entropy': abs(same[0]['conditional_goal_entropy']-math.log(2)) < 1e-12,
            'placebo:perfect_forecast_zero_excess': all(abs(exact[0][k]) < 1e-12 for k in EXCESSES),
            'positive:distinct_operations_one_goal': exact[0]['conditional_goal_entropy'] == 0}


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
    raw = json.loads(gzip.decompress((original/'uncertainty_points.json.gz').read_bytes()))
    old = read(parent/'SUMMARY.json')['cells']; summary = read(original/'SUMMARY.json')
    idx = {tuple(r[k] for k in FIELDS): r for r in raw}
    prior = {tuple(r[k] for k in FIELDS): r for r in old}
    expected = set(product(design['tiers'], design['budgets'], ARMS, design['development_lineages'], design['training_draws'], design['fit_seeds']))
    if len(idx) != len(raw) or len(prior) != len(old) or set(idx) != expected or set(prior) != expected:
        raise ValueError('complete raw roster')
    rows = []; score_error = original_error = chain_error = 0.; frame_count = 0
    for tier in design['tiers']:
        keys = sorted(k for k, p in packets['packets'].items() if p['tier'] == tier)
        records = [r for r in refs if r['tier'] == tier]
        bylin = {r['lineage']: {f['frame']: f for f in r['frames']} for r in records}
        ls = design['development_lineages']
        if len(records) != len(ls) or set(bylin) != set(ls): raise ValueError('native lineage roster')
        for r in records:
            frames = bylin[r['lineage']]
            if len(frames) != len(r['frames']) or set(frames) != set(keys): raise ValueError('native frame roster')
            if any(not math.isfinite(f['mass']) or f['mass'] <= 0 for f in frames.values()): raise ValueError('native mass')
            close(math.fsum(f['mass'] for f in frames.values()), 1.)
        targets = {k: [native(bylin[l][k]['target']) for l in ls] for k in keys}
        for draw, seed, budget, arm in product(design['training_draws'], design['fit_seeds'], design['budgets'], ARMS):
            pulse(phase='independent-uncertainty', tier=tier, draw=draw, seed=seed, budget=budget, arm=arm)
            stem = f'{draw}-{tier}-{seed}-{budget}-{arm}'
            if read(parent/'forecasts'/(stem+'-frames.json')) != keys: raise ValueError('forecast frame roster')
            with np.load(parent/'forecasts'/(stem+'.npz'), allow_pickle=False) as data:
                if set(data.files) != {'probabilities', 'alphabet'}: raise ValueError('forecast fields')
                p, alphabet = data['probabilities'], data['alphabet']
            if p.ndim != 2 or len(p) != len(keys): raise ValueError('forecast shape')
            totals = {l: defaultdict(list) for l in ls}
            for i, key in enumerate(keys):
                values, error = components(expand(p[i], alphabet, budget), targets[key]); chain_error = max(chain_error, error)
                for l, v in zip(ls, values):
                    for metric, value in v.items(): totals[l][metric].append(bylin[l][key]['mass']*value)
                frame_count += 1
            for lineage in ls:
                key = (tier, budget, arm, lineage, draw, seed)
                computed = {k: math.fsum(v) for k, v in totals[lineage].items()}
                saved = idx[key]; inherited = prior[key]
                if set(saved) != set(inherited) | set(computed): raise ValueError('raw field coverage')
                for k in inherited:
                    if k not in computed and inherited[k] != saved[k]: raise ValueError('inherited metric changed')
                score_error = max(score_error, close([computed[k] for k in computed], [saved[k] for k in computed]))
                original_error = max(original_error, close(computed['loss'], inherited['loss']))
                rows.append(dict(inherited, **computed))
    if (len(rows) != summary['rows'] or frame_count != summary['frame_forecasts_scored']
            or len(packets['packets']) != summary['packets']): raise ValueError('whole coverage')
    groups = regroup(rows, design)
    regroup_error = compare_groups(groups, {k: summary[k] for k in groups})
    # A separate grouping of original produced rows isolates aggregation errors
    # from reconstructed-score roundoff; all draw/seed summaries are retained.
    original_groups = regroup(raw, design)
    original_regroup_error = compare_groups(original_groups, {k: summary[k] for k in groups})
    write(root/'INDEPENDENT_REGROUP.json', groups)
    write(root/'ORIGINAL_ROW_REGROUP.json', original_groups)
    (root/'review_points.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
    checks['positive:complete_reconstruction'] = True
    return dict(controls=checks, rows=len(rows), original_cells_reproduced=len(old), frame_forecasts_scored=frame_count,
                native_frame_laws=sum(len(r['frames']) for r in refs), packets=len(packets['packets']),
                contrasts=len(groups['contrasts']), learning_areas=len(groups['normalized_log_budget_area']), means=len(groups['means']),
                max_score_error=score_error, max_original_error=original_error, max_chain_error=chain_error,
                max_regroup_error=regroup_error, max_original_row_regroup_error=original_regroup_error,
                target_plan_sha256=cfg['target_plan_sha256'],
                scope='independent scalar chain rules, native weighted components and paired lineage regroup; frozen fits/native targets and unchanged localization metrics inherited; numerical adjudication pending')
