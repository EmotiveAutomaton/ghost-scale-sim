"""Fixed-mass reweighting of saved practice statistics; no new episodes."""
from itertools import product
import gzip
import time
import numpy as np
from ..v18_3.io import canonical, read, write, file_digest
from . import practice as P
from .readout_model import save_arrays

WEIGHTS = (0., .25, .5, .75, 1.)


def mix_counts(restricted, matched, weight):
    if restricted.shape != P.SHAPE or matched.shape != P.SHAPE or weight not in WEIGHTS:
        raise ValueError('mixture shape or frozen weight differs')
    if np.any(restricted < .5) or np.any(matched < .5): raise ValueError('prior missing')
    if not np.isclose(restricted.sum(), matched.sum(), rtol=0, atol=1e-10):
        raise ValueError('unequal feedback mass')
    return .5 + (1-weight)*(restricted-.5) + weight*(matched-.5)


def reconstruct(log, episodes):
    if log.ndim != 2 or log.shape[1] != 6 or len(log) < 3*episodes:
        raise ValueError('incomplete observed transitions')
    counts = np.full(P.SHAPE, .5)
    for row in log[:3*episodes]:
        if any(int(x) != x or not 0 <= int(x) < size for x, size in zip(row, P.SHAPE)):
            raise ValueError('transition outside alphabet')
        counts[tuple(map(int, row))] += 1
    return counts


def native_kernel(rate):
    """Independent conditional law for the retained high-skill actor."""
    kernel = np.zeros(P.SHAPE)
    for step, context, a, previous, goal in product(range(3), range(2), range(8), range(8), range(3)):
        x, y, z = P.ARTIFACTS[a]
        after = (1-x, 1-x, z) if goal == 0 else (x, x, z) if goal == 1 else (x, y, 1-z)
        other = previous if goal == 1 and step == 2 else a
        kernel[step, context, a, previous, goal, P.INDEX[after]] += rate
        kernel[step, context, a, previous, goal, other] += 1-rate
    return kernel


def exhaustive_success(kernel, policy):
    """Sum all 8^3 endpoint paths per context, independently of backward DP."""
    result = []
    for ctx, initial in enumerate(P.INITIAL):
        total = 0.
        for a, b, last in product(range(8), repeat=3):
            g0 = policy[0, ctx, initial, initial]; g1 = policy[1, ctx, a, initial]; g2 = policy[2, ctx, b, a]
            total += (kernel[0, ctx, initial, initial, g0, a] * kernel[1, ctx, a, initial, g1, b]
                      * kernel[2, ctx, b, a, g2, last] * P.SUCCESS[last])
        result.append(total)
    return np.asarray(result)


def controls():
    r = np.full(P.SHAPE, .5); m = r.copy()
    r[0, 0, 2, 2, 0, 6] += 3; m[0, 1, 5, 5, 0, 6] += 3
    mixed = mix_counts(r, m, .25)
    known = np.zeros(P.SHAPE); known[..., 6] = 1
    inert = np.broadcast_to(np.eye(8)[None, None, :, None, None, :], P.SHAPE).copy()
    pi, _ = P.policy(known); ipi, _ = P.policy(inert)
    return {
        'live:context_support_changes': bool(np.any(mixed[:, 1] > .5) and np.all(r[:, 1] == .5)),
        'positive:one_prior': bool(mixed.sum() == .5*np.prod(P.SHAPE)+3),
        'positive:controlled_success': bool(np.array_equal(exhaustive_success(known, pi), [1, 1])),
        'placebo:action_inert': bool(np.array_equal(exhaustive_success(inert, ipi), [0, 0])),
        'placebo:identical_tables': all(np.array_equal(mix_counts(r, r, w), r) for w in WEIGHTS),
        'positive:endpoint_identity': bool(np.array_equal(mix_counts(r, m, 0), r) and np.array_equal(mix_counts(r, m, 1), m)),
        'positive:no_cross_context_write': bool(mixed[0, 0, 2, 2, 0, 6] == 2.75 and mixed[0, 1, 5, 5, 0, 6] == 1.25),
    }


def arrays(path):
    with np.load(path, allow_pickle=False) as z: return {n: z[n] for n in z.files}


def run(root, plan, pulse):
    cfg = plan['design']; checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('support-mixture controls failed')
    if cfg['weights'] != list(WEIGHTS): raise ValueError('frozen weights changed')
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('parent input changed')
    parent = root/'inputs/parent'; old_rows = read(parent/'POINTS.json')
    key = lambda r: tuple(r[k] for k in ('lineage', 'draw', 'policy_seed', 'condition', 'episodes', 'arm'))
    old = {key(r): r for r in old_rows}
    if len(old) != len(old_rows): raise ValueError('duplicate parent score')
    worlds = {r['lineage']: r['world'] for r in read(root/'inputs/ENUMERATION.json')['lineages']}
    rows = []; timings = []; reproduced = 0; max_error = 0.
    for lineage in cfg['lineages']:
        pulse(phase='parent-reconstruction', lineage=lineage)
        kernel = native_kernel(worlds[lineage]['action_rate']); law = arrays(parent/'evaluator'/f'law-{lineage}.npz')
        if not np.allclose(kernel, law['kernel'], rtol=0, atol=1e-12): raise ValueError('native law mismatch')
        for draw, seed in product(cfg['training_draws'], cfg['policy_seeds']):
            parents = {}
            for condition in ('restricted-start', 'matched-start'):
                base = f'{lineage}-{draw}-{seed}-{condition}'
                logs = {arm: arrays(parent/'observed'/f'{base}-{arm}.npz')['transitions'] for arm in ('active', 'replay', 'demonstration')}
                if not np.array_equal(logs['active'], logs['replay']): raise ValueError('ordered replay differs')
                for arm, log in logs.items():
                    for budget in cfg['budgets']:
                        counts = reconstruct(log, budget)
                        if counts.sum() != .5*np.prod(P.SHAPE)+3*budget: raise ValueError('count mass differs')
                        if condition == 'restricted-start' and not np.all(counts[:, 1] == .5): raise ValueError('cross-context update')
                        model = arrays(parent/'models'/f'{base}-{budget}-{arm}.npz'); pi, values = P.policy(counts)
                        if not np.array_equal(counts, model['counts']) or not np.array_equal(pi, model['policy']):
                            raise ValueError('parent count or fixed-arithmetic policy differs')
                        if not np.allclose(values, model['learned_values'], rtol=0, atol=1e-12): raise ValueError('parent values differ')
                        scores = exhaustive_success(kernel, pi); saved = old[lineage, draw, seed, condition, budget, arm]
                        error = max(float(np.max(abs(scores-saved['context_success']))), abs(float(scores.mean())-saved['success']))
                        if error > 1e-12: raise ValueError('parent endpoint reproduction failed')
                        max_error = max(max_error, error); reproduced += 1; parents[condition, budget, arm] = (counts, pi, scores)
            for budget, arm, weight in product(cfg['budgets'], ('active', 'demonstration'), WEIGHTS):
                pulse(phase='fixed-support-mixture', lineage=lineage, draw=draw, seed=seed, budget=budget, arm=arm, weight=weight)
                start = time.process_time(); restricted, rpi, _ = parents['restricted-start', budget, arm]; matched, _, _ = parents['matched-start', budget, arm]
                counts = mix_counts(restricted, matched, weight); pi, values = P.policy(counts)
                success, _, _ = P.evaluate(kernel, pi); independent = exhaustive_success(kernel, pi)
                error = float(np.max(abs(success-independent))); max_error = max(max_error, error)
                if error > 1e-12: raise ValueError('independent exhaustive policy evaluation differs')
                if weight in (0, 1):
                    endpoint = parents['restricted-start' if weight == 0 else 'matched-start', budget, arm]
                    if not np.array_equal(counts, endpoint[0]) or not np.array_equal(pi, endpoint[1]) or not np.allclose(success, endpoint[2], rtol=0, atol=1e-12):
                        raise ValueError('mixture endpoint identity differs')
                visit = counts.sum(-1) > 4; rvisit = restricted.sum(-1) > 4
                stem = f'{lineage}-{draw}-{seed}-{budget}-{arm}-{int(weight*4)}'
                save_arrays(root/'models'/f'{stem}.npz', counts=counts, policy=pi, learned_values=values, context_success=success)
                rows.append(dict(lineage=lineage, draw=draw, policy_seed=seed, episodes=budget, arm=arm, matched_weight=weight,
                    feedback_mass=float((counts-.5).sum()), context_success=success.tolist(), success=float(success.mean()), restricted_success=float(success[0]),
                    visited_by_context=visit.sum((0, 2, 3, 4)).tolist(), newly_visited_by_context=(visit & ~rvisit).sum((0, 2, 3, 4)).tolist(),
                    policy_changes_by_context=(pi != rpi).sum((0, 2, 3)).tolist(),
                    changed_seen_probabilities_by_context=(rvisit & np.any(abs(counts/counts.sum(-1,keepdims=True)-restricted/restricted.sum(-1,keepdims=True)) > 1e-14,axis=-1)).sum((0,2,3,4)).tolist()))
                timings.append(dict(stem=stem, cpu_seconds=time.process_time()-start))
    if reproduced != len(old): raise ValueError('unconsumed parent score')
    (root/'raw').mkdir(exist_ok=True); (root/'raw/support_mix_points.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
    write(root/'PARENT_REPRODUCTION.json', dict(passed=True, tables=reproduced, maximum_error=max_error, scope='ordered counts, saved fixed-arithmetic policy, learned values and independent context scores; endpoints identity controls'))
    write(root/'TIMING.jsonl', dict(measurements=timings, note='included in native charge'))
    write(root/'INPUT_SCHEMA.json', dict(reader='no new reader task; unchanged parent observed-transition export remains available', evaluator='table pairings, counts, policies, native kernels and production scores; never reader inputs'))
    checks['positive:parent_reconstruction'] = True; checks['positive:independent_path_sum'] = True
    cells = []
    for budget, arm, weight in product(cfg['budgets'], ('active', 'demonstration'), WEIGHTS):
        selected = [r for r in rows if (r['episodes'], r['arm'], r['matched_weight']) == (budget, arm, weight)]
        cells.append(dict(episodes=budget, arm=arm, matched_weight=weight, records=len(selected), success=float(np.mean([r['success'] for r in selected])), context_success=np.mean([r['context_success'] for r in selected], axis=0).tolist()))
    return dict(controls=checks, cells=cells, rows=len(rows), parent_tables=reproduced, max_error=max_error, new_episodes=0,
        scope='fractional sufficient-statistic reweighting with one prior; fixed feedback mass; not on-policy training or inverse process inference', warrant='exploratory constructed method; miniature — architecture untested')
