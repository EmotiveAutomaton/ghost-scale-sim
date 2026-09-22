"""Fixed primitive feedback on retained count tables; no new episode sampling."""
from itertools import product
import gzip
import time
import numpy as np
from ..v18_3.io import canonical, digest, read, write, file_digest
from . import rollout as R, forward_support as S, support_transfer as T
from . import missing_tool as M, local_world as L

ARMS = ('unchanged', 'add-one', 'replace-row')
BUDGETS = (0, 4, 16, 64)
ORDERS = ('forward', 'reverse')
FEEDBACK = ('true', 'wrong')


def roster(order):
    # Tool use requires skill=1. Round-robin beliefs inside lexicographic
    # (current, undo) blocks balances both legal belief states in every budget.
    rows = [(1, belief, a, b, 3) for a, b, belief in product(range(8), range(8), range(2))]
    if order == 'reverse': rows.reverse()
    elif order != 'forward': raise ValueError('unknown ordering')
    return rows


def observation(key, rule, wrong=False):
    skill, belief, a, b, op = key
    if key not in roster('forward'): raise ValueError('illegal feedback input')
    target = R.code(M.execute(R.ARTIFACTS[a], R.ARTIFACTS[b], L.OPERATIONS[op],
                              (0, skill, belief, 0), rule))
    return target ^ 1 if wrong else target


def update(counts, observations, arm):
    if counts.shape != (2, 2, 8, 8, 6, 8) or not np.isfinite(counts).all() or (counts < 1).any():
        raise ValueError('saved unit-prior counts required')
    if arm not in ARMS: raise ValueError('unknown update')
    unique = {}
    for key, target in observations:
        key = tuple(key)
        if key not in roster('forward') or target not in range(8): raise ValueError('invalid feedback')
        if key in unique and unique[key] != target: raise ValueError('conflicting duplicate input')
        unique[key] = target
    table = R.normalized(counts)
    for key, target in unique.items():
        if arm == 'add-one':
            row = counts[key].copy(); row[target] += 1
            table[key] = row / row.sum()
        elif arm == 'replace-row': table[key] = np.eye(8)[target]
    return table


def path_sum(table, query):
    """Independent explicit sum over three intermediate artifact choices."""
    skill, belief, initial, o1, o2, o3 = query
    answer = np.zeros(8)
    for a, b, final in product(range(8), repeat=3):
        answer[final] += (table[skill, belief, initial, initial, o1, a] *
                          table[skill, belief, a, initial, o2, b] *
                          table[skill, belief, b, a, o3, final])
    return answer


def controls():
    counts = np.ones((2, 2, 8, 8, 6, 8)); key = (1, 0, 2, 2, 3)
    target = observation(key, 'presentation-tool'); obs = [(key, target)]
    added = update(counts, obs, 'add-one'); replaced = update(counts, obs, 'replace-row')
    baseline = R.normalized(counts); untouched = np.ones(counts.shape[:-1], dtype=bool); untouched[key] = False
    q = (1, 0, 2, 3, 5, 4)
    return {
        'live:changed_tool': observation(key, 'original') != target,
        'placebo:zero_feedback': all(np.array_equal(update(counts, [], a), baseline) for a in ARMS),
        'positive:unit_prior_once': bool(added[key][target] == 2/9 and added[key].sum() == 1),
        'positive:deduplicate': np.array_equal(added, update(counts, obs * 2, 'add-one')),
        'positive:only_observed_rows': np.array_equal(replaced[untouched], baseline[untouched]),
        'positive:known_row': bool(replaced[key][target] == 1),
        'placebo:unchanged_counts': np.array_equal(update(counts, obs, 'unchanged'), baseline),
        'positive:independent_path_sum': np.allclose(path_sum(replaced, q), R.propagate(replaced, q), atol=1e-12, rtol=0),
        'positive:wrong_feedback_distinct': observation(key, 'presentation-tool', True) != target,
        'positive:balanced_unique_roster': len(set(roster('forward'))) == 128 and all(sum(k[1] == 0 for k in roster(o)[:n]) == n//2 for o in ORDERS for n in BUDGETS),
    }


def cells_for(loss, squared, probability, mass, diagnostics, changed, identity):
    cells = []
    for change in ('all', 'changed', 'stay'):
        for subset in T.SUPPORT:
            mask = np.array([T.select(d, subset) for d in diagnostics])
            if change != 'all': mask &= changed == (change == 'changed')
            for weighting in ('native', 'equal-query'):
                selected = mask & (mass > 0) if weighting == 'native' else mask
                weights = mass if weighting == 'native' else np.ones(len(mass))/len(mass)
                denom = float(weights[selected].sum())
                values = {k: float(np.dot(weights[selected], v[selected])/denom) if denom else None
                          for k, v in (('loss', loss), ('squared_error', squared), ('true_probability', probability))}
                cells.append(dict(identity, change=change, subset=subset, weighting=weighting,
                                  queries=int(selected.sum()), population_mass=denom, **values))
    return cells


def run(root, plan, pulse):
    cfg = plan['design']; inputs = root/'inputs'; checks = controls()
    if (not all(checks.values()) or cfg['arms'] != list(ARMS) or cfg['budgets'] != list(BUDGETS)
            or cfg['orders'] != list(ORDERS) or cfg['feedback'] != list(FEEDBACK)
            or cfg['epsilon'] != S.EPSILON or cfg['saved_prior'] != 1):
        raise ValueError('primitive feedback admission failed')
    for n, h in cfg['input_files'].items():
        if file_digest(inputs/n) != h: raise ValueError('frozen feedback input changed')
    for folder in ('raw', 'models', 'forecasts', 'reader', 'evaluator'): (root/folder).mkdir()
    write(root/'CONTROLS.json', checks)
    qs = [tuple(r['query']) for r in read(inputs/'QUERY_TRUTH.json')]
    if len(qs) != cfg['queries'] or len(set(qs)) != len(qs): raise ValueError('query roster')
    write(root/'reader/QUERIES.json', [R.visible(q) for q in qs])
    truth = read(inputs/'QUERY_TRUTH.json')
    targets = {rule: np.array([r['targets'][rule] for r in truth]) for rule in M.RULES}
    changed = targets['original'] != targets['presentation-tool']
    populations = {}
    for lineage in cfg['lineages']:
        for rule in M.RULES:
            # Reuse complete retained native trajectories, including their weights.
            rows = read_gzip(inputs/'raw'/f'{lineage}-{rule}_points.json.gz')
            mass = np.zeros(len(qs)); index = {q: i for i, q in enumerate(qs)}
            for r in rows:
                i = index[R.query(r)]
                if R.code(r['final']) != targets[rule][i]: raise ValueError('retained target mismatch')
                mass[i] += r['probability']
            if not np.isclose(mass.sum(), 1, atol=1e-12, rtol=0): raise ValueError('retained mass')
            populations[lineage, rule] = mass
    write(root/'evaluator/POPULATIONS.json', [dict(lineage=l, rule=r, masses=m.tolist()) for (l,r),m in populations.items()])
    write(root/'evaluator/FEEDBACK_ROSTER.json', {o: [list(k) for k in roster(o)] for o in ORDERS})
    all_cells = []; timing = []; row_count = 0; baseline_count = 0; feedback_map = []
    for draw in cfg['training_draws']:
        for mode in ('original', 'composition-holdout'):
            start = time.process_time(); pulse(phase='feedback-parent', draw=draw, mode=mode)
            with np.load(inputs/'models'/f'{draw}-{mode}.npz') as saved: counts = saved['transition_counts']
            with np.load(inputs/'forecasts'/f'{draw}-{mode}.npz') as saved:
                baseline = saved['learned-exact']; saved_qs = saved['queries']
            if not np.array_equal(saved_qs, qs): raise ValueError('parent query mismatch')
            check = np.array([S.smooth(R.propagate(R.normalized(counts), q)) for q in qs])
            if not np.allclose(check, baseline, atol=1e-12, rtol=0): raise ValueError('parent forecast mismatch')
            baseline_count += len(qs)
            diagnostics = read(inputs/'forecasts'/f'{draw}-{mode}-support.json')
            for rule in M.RULES:
                for order in ORDERS:
                    for feedback in FEEDBACK:
                        full = [(k, observation(k, rule, feedback == 'wrong')) for k in roster(order)]
                        # No rule labels, targets for queries, scores or population weights in reader input.
                        public = [dict(skill=k[0], belief_error=k[1], current=list(R.ARTIFACTS[k[2]]),
                                       undo_buffer=list(R.ARTIFACTS[k[3]]), operation='accept-tool',
                                       next_artifact=list(R.ARTIFACTS[t])) for k,t in full]
                        file_key = f'{rule}-{order}-{feedback}'
                        for budget in BUDGETS:
                            visible = public[:budget]; public_id = digest(visible)
                            public_path = root/'reader'/f'{public_id}.json'
                            if not public_path.exists(): write(public_path, visible)
                            feedback_map.append(dict(draw=draw, mode=mode, rule=rule, order=order,
                                                     feedback=feedback, budget=budget, reader_id=public_id))
                            for arm in ARMS:
                                pulse(phase='feedback-query', draw=draw, mode=mode, rule=rule, order=order, feedback=feedback, budget=budget, arm=arm)
                                table = update(counts, full[:budget], arm)
                                pred = np.array([S.smooth(R.propagate(table, q)) for q in qs])
                                if budget == 0 or arm == 'unchanged':
                                    if not np.array_equal(pred, check): raise ValueError('zero feedback identity')
                                # Independent full path sums for all 640 queries in each changed table.
                                independent = np.array([S.smooth(path_sum(table, q)) for q in qs])
                                discrepancy = float(np.max(abs(pred-independent)))
                                if discrepancy > 1e-12: raise ValueError('independent path sum failure')
                                t = targets[rule]; prob = pred[np.arange(len(qs)), t]
                                loss = -np.log(prob); squared = np.sum((pred-np.eye(8)[t])**2, axis=1)
                                identity = dict(draw=draw, mode=mode, rule=rule, order=order, feedback=feedback, budget=budget, arm=arm)
                                key = f'{draw}-{mode}-{file_key}-{budget}-{arm}'
                                np.savez_compressed(root/'models'/f'{key}.npz', table=table)
                                np.savez_compressed(root/'forecasts'/f'{key}_points.npz', queries=saved_qs, predictions=pred, loss=loss,
                                                    squared_error=squared, true_probability=prob, path_sum_max_abs=discrepancy)
                                for lineage in cfg['lineages']:
                                    all_cells.extend(cells_for(loss, squared, prob, populations[lineage,rule], diagnostics, changed, dict(identity,lineage=lineage)))
                                row_count += len(qs)*len(cfg['lineages'])
            timing.append(dict(draw=draw, mode=mode, cpu_seconds=time.process_time()-start, new_fits=0))
    write(root/'PARENT_REPRODUCTION.json', dict(passed=True, forecast_vectors=baseline_count, saved_unit_prior=True))
    write(root/'evaluator/FEEDBACK_MAP.json', feedback_map)
    write(root/'TIMING.jsonl',dict(measurements=timing,accounting='all new feedback, propagation, reconstruction and scoring included; no refit'))
    return dict(controls=checks, cells=all_cells, queries=len(qs), score_rows=row_count, fits=0,
                feedback_roster_inputs=128, independent_feedback_trials=False,
                scope='constructed-method forward feedback comparison; supplied skill/belief/operation metadata; no historical process, human intent or generalization to unobserved rows')


def read_gzip(path):
    import json
    return json.loads(gzip.decompress(path.read_bytes()))
