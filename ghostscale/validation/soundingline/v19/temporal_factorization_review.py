"""Independent between-step dependence reconstruction and full score review.

Label memberships use integer division and remainder, without producer tensor
axes. The common independent checker supplies scores and paired grouping.
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
    regroup, compare_groups, validate,
)

# Decode each label independently of the producer's reshape/axis reductions.
MEMBERS = tuple(tuple((k//(216*3**(2-t))%3)*6 + k//(6**(2-t))%6
                     for t in range(3)) for k in range(N))
BUCKETS = tuple(tuple(tuple(k for k in range(N) if MEMBERS[k][t] == v)
                     for v in range(18)) for t in range(3))
INDICES = np.array(MEMBERS, dtype=int)


def marginals(q):
    q = np.asarray(q)
    if q.shape != (N,) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('distribution')
    values = q.tolist()
    close(math.fsum(values), 1., 1e-12)
    return np.array([[math.fsum(values[k] for k in bucket) for bucket in step]
                     for step in BUCKETS])


def reconstruct(q, saved_steps):
    steps = marginals(q)
    error = close(steps, saved_steps, 1e-12)
    saved = np.asarray(saved_steps)
    scalar = steps[0, INDICES[:, 0]] * steps[1, INDICES[:, 1]] * steps[2, INDICES[:, 2]]
    # Verify first; then use the executed binary64 values for tie-sensitive scores.
    scored = saved[0, INDICES[:, 0]] * saved[1, INDICES[:, 1]] * saved[2, INDICES[:, 2]]
    error = max(error, close(scalar, scored, 1e-12))
    close(math.fsum(scored.tolist()), 1., 3e-12)
    return scored, error


def controls():
    q = np.zeros(N); q[[0, 2634]] = .5
    fact, _ = reconstruct(q, marginals(q))
    independent = np.zeros(N); independent[[0, 1980, 654, 2634]] = .25
    ref = (np.array([0, 2634]), np.array([[.5, .5]]), np.array([1.]))
    joint = score(q, priority_for(LABELS), ref)
    temporal = score(fact, priority_for(LABELS), ref)
    return {'live:temporal_dependence_loss': abs(float(temporal['loss'][0]-joint['loss'][0])-math.log(2)) < 1e-12,
            'placebo:independent_steps_identity': bool(np.array_equal(reconstruct(independent, marginals(independent))[0], independent)),
            'positive:within_step_dependence_retained': bool(np.array_equal(fact, independent)),
            'positive:incompatible_cross_paths': bool(temporal['compatible_mass'][0] == .5),
            'positive:full_native_coverage': bool(temporal['candidate_coverage'][0] == 1.)}


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
    raw = json.loads(gzip.decompress((original/'temporal_points.json.gz').read_bytes()))
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
            pulse(phase='independent-native-temporal', tier=tier, lineage=lineage)
            totals = {a: defaultdict(list) for a in ('native-joint', 'native-product')}
            with np.load(original/'evaluator'/f'{tier}-{lineage}-marginals.npz', allow_pickle=False) as data:
                if set(data.files) != {'steps'}: raise ValueError('native marginal fields')
                steps = data['steps']
            if steps.shape != (len(keys), 3, 18): raise ValueError('native marginal shape')
            for i, (labels, target, masses) in enumerate(reference):
                q = np.zeros(N); q[labels] = target[li]
                fact, error = reconstruct(q, steps[i]); marginal_error = max(marginal_error, error)
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
            pulse(phase='independent-learned-temporal', tier=tier, draw=draw, seed=seed, budget=budget, arm=arm)
            stem = f'{draw}-{tier}-{seed}-{budget}-{arm}'
            if read(parent/'forecasts'/(stem+'-frames.json')) != keys: raise ValueError('forecast frames')
            with np.load(parent/'forecasts'/(stem+'.npz'), allow_pickle=False) as data:
                if set(data.files) != {'probabilities', 'alphabet'}: raise ValueError('forecast fields')
                probabilities, alphabet = data['probabilities'], data['alphabet']
            with np.load(original/'marginals'/(stem+'.npz'), allow_pickle=False) as data:
                if set(data.files) != {'steps'}: raise ValueError('learned marginal fields')
                steps = data['steps']
            if probabilities.ndim != 2 or len(probabilities) != len(keys) or steps.shape != (len(keys), 3, 18):
                raise ValueError('forecast/marginal shape')
            ranking = priority_for(alphabet); totals = {s: defaultdict(list) for s in ('', '-product')}
            for i, ref in enumerate(reference):
                q = expand(probabilities[i], alphabet, budget)
                fact, error = reconstruct(q, steps[i]); marginal_error = max(marginal_error, error)
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
        scope='independent scalar temporal marginals and products; all native/learned scores and paired regroup; saved binary64 marginal products retain executed ties; inherited frozen fits/native targets; numerical adjudication pending')
