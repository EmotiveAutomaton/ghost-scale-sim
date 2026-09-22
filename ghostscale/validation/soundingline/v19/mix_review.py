"""Independent saved-count and policy audit for the practice mixture.

No producer planner, mixture, kernel or score function is imported. Independent
matrix-vector reductions may resolve numerical ties differently; saved policies
are checked for optimality and then evaluated unchanged by explicit path sums.
"""
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, file_digest

SHAPE = (3, 2, 8, 8, 3, 8)
ARTS = list(product(range(2), repeat=3))
INITIAL = (2, 5)
SUCCESS = np.array([int(a == b == 1) for a, b, c in ARTS])
WEIGHTS = (0., .25, .5, .75, 1.)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def near(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('nonfinite value or shape mismatch')
    delta = float(np.max(abs(a - b)))
    if delta > 1e-12:
        raise ValueError(f'numerical reconstruction differs: {delta}')
    return delta


def kernel(rate):
    out = np.zeros(SHAPE)
    for t, ctx, a, undo, goal in product(range(3), range(2), range(8), range(8), range(3)):
        x, y, z = ARTS[a]
        target = (1-x, 1-x, z) if goal == 0 else (x, x, z) if goal == 1 else (x, y, 1-z)
        out[t, ctx, a, undo, goal, ARTS.index(target)] += rate
        out[t, ctx, a, undo, goal, undo if goal == 1 and t == 2 else a] += 1-rate
    return out


def counts_from_log(log, episodes, condition, law):
    if log.shape[1:] != (6,) or len(log) < episodes*3:
        raise ValueError('incomplete log')
    counts = np.full(SHAPE, .5)
    for j, row in enumerate(log[:episodes*3]):
        if any(x != int(x) or not 0 <= x < size for x, size in zip(row, SHAPE, strict=True)):
            raise ValueError('invalid transition')
        t, ctx, a, undo, goal, b = map(int, row)
        if t != j % 3 or ctx != ((j//3) % 2 if condition == 'matched-start' else 0):
            raise ValueError('episode ordering differs')
        if t == 0:
            if a != INITIAL[ctx] or undo != a: raise ValueError('initial state differs')
        elif a != log[j-1, 5] or undo != log[j-1, 2]:
            raise ValueError('transient state differs')
        if law[t, ctx, a, undo, goal, b] <= 0: raise ValueError('impossible transition')
        counts[t, ctx, a, undo, goal, b] += 1
    return counts


def policy_check(counts, saved_policy, saved_values):
    probability = counts/counts.sum(-1, keepdims=True)
    value = np.zeros((4, 2, 8, 8)); value[3] = SUCCESS[None, :, None]
    ties = 0; worst_gap = 0.
    for t in (2, 1, 0):
        for ctx, a, undo in product(range(2), range(8), range(8)):
            q = probability[t, ctx, a, undo] @ value[t+1, ctx, :, a]
            chosen = saved_policy[t, ctx, a, undo]
            if chosen != int(chosen) or not 0 <= chosen < 3: raise ValueError('invalid policy')
            chosen = int(chosen); gap = float(q.max()-q[chosen])
            if gap > 1e-12: raise ValueError('saved policy not optimal')
            ties += int(chosen != int(np.argmax(q))); worst_gap = max(worst_gap, gap)
            value[t, ctx, a, undo] = q.max()
    return near(value, saved_values), ties, worst_gap


def score(law, policy):
    result = []
    for ctx, start in enumerate(INITIAL):
        terms = []
        for a, b, end in product(range(8), repeat=3):
            terms.append(float(law[0, ctx, start, start, policy[0, ctx, start, start], a]
                * law[1, ctx, a, start, policy[1, ctx, a, start], b]
                * law[2, ctx, b, a, policy[2, ctx, b, a], end] * SUCCESS[end]))
        result.append(math.fsum(terms))
    return np.asarray(result)


def metrics(row):
    result = {k: row[k] for k in ('success', 'restricted_success')}
    for k in ('context_success', 'visited_by_context', 'newly_visited_by_context',
              'policy_changes_by_context', 'changed_seen_probabilities_by_context'):
        for i, value in enumerate(row[k]): result[f'{k}_{i}'] = value
    return result


def regroup(rows, design, cfg):
    lookup = {tuple(r[k] for k in ('lineage', 'draw', 'policy_seed', 'episodes', 'arm', 'matched_weight')): metrics(r) for r in rows}
    expected = len(design['lineages'])*len(design['training_draws'])*len(design['policy_seeds'])*len(design['budgets'])*2*5
    if len(lookup) != len(rows) or len(rows) != expected: raise ValueError('mixture denominator differs')
    fits = list(product(design['training_draws'], design['policy_seeds']))
    lineages = design['lineages']; names = tuple(metrics(rows[0]))
    sample = np.random.default_rng(cfg['bootstrap_seed']).integers(len(lineages), size=(cfg['bootstrap_resamples'], len(lineages)))
    def estimate(left, right=None):
        result = {}
        for metric in names:
            values = np.array([[lookup[l, d, s, *left][metric] - (lookup[l, d, s, *right][metric] if right else 0.) for d, s in fits] for l in lineages])
            paired = values.mean(1); boot = paired[sample].mean(1)
            result[metric] = dict(mean=float(paired.mean()), low=float(np.quantile(boot,.025)), high=float(np.quantile(boot,.975)),
                lineage_values=paired.tolist(), fit_means=values.mean(0).tolist())
        return result
    means = []; contrasts = []
    for budget, arm, weight in product(design['budgets'], ('active','demonstration'), WEIGHTS):
        left = (budget, arm, weight); identity = dict(episodes=budget, arm=arm, matched_weight=weight)
        means.append(dict(identity, metrics=estimate(left)))
        if weight > 0:
            contrasts.append(dict(identity, contrast='adjacent-weight', baseline_weight=weight-.25, metrics=estimate(left,(budget,arm,weight-.25))))
        if weight == 1:
            contrasts.append(dict(identity, contrast='endpoint-span', baseline_weight=0., metrics=estimate(left,(budget,arm,0.))))
        if arm == 'active':
            contrasts.append(dict(identity, contrast='active-minus-demonstration', metrics=estimate(left,(budget,'demonstration',weight))))
    return dict(means=means, contrasts=contrasts, lineages=lineages, fit_pairs=fits,
        population='equal paired lineages; all draw/acquisition-seed pairs averaged within lineage; fits and repeated queries are not independent worlds')


def controls():
    pi = np.zeros((3,2,8,8),int); known = np.zeros(SHAPE); known[...,6] = 1
    inert = np.broadcast_to(np.eye(8)[None,None,:,None,None,:],SHAPE)
    return {'live:controlled_success': bool(np.array_equal(score(known,pi),[1,1])),
        'placebo:inert_actions': bool(np.array_equal(score(inert,pi),[0,0])),
        'positive:normalized_native_kernel': bool(np.allclose(kernel(.8).sum(-1),1))}


def run(root, plan, pulse):
    checks = controls(); write(root/'CONTROLS.json',checks)
    if not all(checks.values()): raise ValueError('independent mixture controls failed')
    cfg = plan['design']; original = root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('review input differs')
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan differs')
    design = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json')
    if design['weights'] != list(WEIGHTS): raise ValueError('weight roster differs')
    parent = original/'inputs/parent'
    old_rows = read(parent/'POINTS.json')
    old = {tuple(r[k] for k in ('lineage','draw','policy_seed','condition','episodes','arm')):r for r in old_rows}
    if len(old) != len(old_rows): raise ValueError('duplicate parent row')
    rows = json.loads(gzip.decompress((original/'raw/support_mix_points.json.gz').read_bytes()))
    worlds = {r['lineage']:r['world'] for r in read(original/'inputs/ENUMERATION.json')['lineages']}
    expected = {}; parent_count = 0; worst = 0.; ties = 0; gap = 0.
    for lineage in design['lineages']:
        pulse(phase='independent-mixture-reconstruction',lineage=lineage)
        law = kernel(worlds[lineage]['action_rate'])
        worst = max(worst,near(law,arrays(parent/'evaluator'/f'law-{lineage}.npz')['kernel']))
        for draw, seed in product(design['training_draws'],design['policy_seeds']):
            tables = {}
            for condition in ('restricted-start','matched-start'):
                stem = f'{lineage}-{draw}-{seed}-{condition}'
                logs = {arm:arrays(parent/'observed'/f'{stem}-{arm}.npz')['transitions'] for arm in ('active','replay','demonstration')}
                if not np.array_equal(logs['active'],logs['replay']): raise ValueError('ordered replay differs')
                for budget, arm in product(design['budgets'],logs):
                    counts = counts_from_log(logs[arm],budget,condition,law)
                    model = arrays(parent/'models'/f'{stem}-{budget}-{arm}.npz')
                    if not np.array_equal(counts,model['counts']): raise ValueError('parent counts differ')
                    e,t,g = policy_check(counts,model['policy'],model['learned_values']); ties += t; gap=max(gap,g);worst=max(worst,e)
                    success = score(law,model['policy']); saved=old[lineage,draw,seed,condition,budget,arm]
                    worst=max(worst,near(success,saved['context_success']),near(success.mean(),saved['success']))
                    tables[condition,budget,arm]=(counts,model['policy'],success);parent_count+=1
            for budget,arm,weight in product(design['budgets'],('active','demonstration'),WEIGHTS):
                restricted,rpi,_ = tables['restricted-start',budget,arm];matched,_,_=tables['matched-start',budget,arm]
                # One prior plus reweighted empirical evidence; exactly representable quarter weights.
                counts = ((4-int(4*weight))*(restricted-.5)+int(4*weight)*(matched-.5))/4+.5
                if counts.sum() != 4608+3*budget: raise ValueError('count mass differs')
                stem=f'{lineage}-{draw}-{seed}-{budget}-{arm}-{int(4*weight)}';model=arrays(original/'models'/f'{stem}.npz')
                if not np.array_equal(counts,model['counts']):raise ValueError('mixture counts differ')
                e,t,g=policy_check(counts,model['policy'],model['learned_values']);ties+=t;gap=max(gap,g);worst=max(worst,e)
                success=score(law,model['policy']);worst=max(worst,near(success,model['context_success']))
                if weight in (0,1):
                    endpoint=tables['restricted-start' if weight==0 else 'matched-start',budget,arm]
                    if not np.array_equal(model['policy'],endpoint[1]):raise ValueError('endpoint policy differs')
                    worst=max(worst,near(success,endpoint[2]))
                visit=np.any(counts>.5,axis=-1);rvisit=np.any(restricted>.5,axis=-1)
                prob=counts/counts.sum(-1,keepdims=True);rprob=restricted/restricted.sum(-1,keepdims=True)
                expected[lineage,draw,seed,budget,arm,weight]=dict(feedback_mass=float((counts-.5).sum()),context_success=success.tolist(),success=float(success.mean()),restricted_success=float(success[0]),
                    visited_by_context=[int(visit[:,i].sum()) for i in range(2)],newly_visited_by_context=[int((visit & ~rvisit)[:,i].sum()) for i in range(2)],
                    policy_changes_by_context=[int((model['policy']!=rpi)[:,i].sum()) for i in range(2)],
                    changed_seen_probabilities_by_context=[int((rvisit & np.any(abs(prob-rprob)>1e-14,axis=-1))[:,i].sum()) for i in range(2)])
    if parent_count != len(old):raise ValueError('parent denominator differs')
    if len(rows)!=len(expected):raise ValueError('raw denominator differs')
    for row in rows:
        key=tuple(row[k] for k in ('lineage','draw','policy_seed','episodes','arm','matched_weight'))
        for k,v in expected[key].items():worst=max(worst,near(v,row[k]))
    for cell in summary['cells']:
        selected=[r for r in rows if all(r[k]==cell[k] for k in ('episodes','arm','matched_weight'))]
        if len(selected)!=cell['records']:raise ValueError('summary denominator differs')
        for k in ('success','context_success'):worst=max(worst,near(np.mean([r[k] for r in selected],axis=0),cell[k]))
    pulse(phase='paired-mixture-regroup')
    grouped=regroup(rows,design,cfg);write(root/'REGROUP.json',grouped)
    checks.update({'positive:all_counts_and_scores':True,'positive:endpoint_identity':True,'positive:paired_regroup':True})
    return dict(controls=checks,passed=True,rows=len(rows),parent_tables=parent_count,maximum_error=worst,
        independent_policy_tie_differences=ties,maximum_saved_action_gap=gap,
        means=len(grouped['means']),contrasts=len(grouped['contrasts']),metrics=len(metrics(rows[0])),
        scope='independent native kernel, observed-count reconstruction, optimality and saved-policy path sums; original numerical tie rule retained; no re-execution of acquisition randomness',
        warrant='exploratory constructed method; conditional on saved acquisition; miniature — architecture untested')
