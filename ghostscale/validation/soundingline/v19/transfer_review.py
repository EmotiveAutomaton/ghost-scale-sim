"""Independent frozen-table transfer reconstruction, separate from its producer.

Reuses the validated support checker's explicit path-sum and count reconstruction,
and the crossed-law checker's independently implemented native executor.
"""
from itertools import product
import numpy as np
from ..v18_3.io import read, write, file_digest
from ..v18_3.world import rng
from . import support_review as V

ARMS = V.ARMS + ('true-law-oracle',)
RULES = ('original', 'presentation-tool')
CHANGES = ('all', 'changed', 'stay')
METRICS = ('loss', 'squared_error', 'true_probability')
AXES = ('rule', 'mode', 'change', 'subset')


def endpoint(q, rule):
    before = undo = V.ARTS[q[2]]
    for op in q[3:]:
        after = V.execute(before, undo, V.OPS[op], q[0], q[1], rule)
        undo, before = before, after
    return V.code(before)


def strata(entries, identity):
    answer = []
    for change in CHANGES:
        rr = [r for r in entries if change == 'all' or r['changed'] == (change == 'changed')]
        subsets = [rr, [r for r in rr if r['query_seen']], [r for r in rr if not r['query_seen']],
                   [r for r in rr if r['withheld_composition']],
                   [r for r in rr if r['withheld_composition'] and r['all_primitives_seen']]]
        for subset, selected in zip(V.SUBSETS, subsets, strict=True):
            mass = sum(r['probability_mass'] for r in selected)
            answer.append(dict(identity, change=change, subset=subset, queries=len(selected), population_mass=mass,
                **{k: sum(r['probability_mass'] * r[k] for r in selected) / mass if mass else None for k in METRICS}))
    return answer


def regroup(cells, lineages, draws, cfg):
    key = lambda r: tuple(r[k] for k in ('lineage', 'draw', *AXES, 'arm'))
    lookup = {key(r): r for r in cells}
    if len(lookup) != len(cells): raise ValueError('duplicate aggregate')
    means = []; contrasts = []
    def collect(left, right, metric):
        vals = []; ids = []; draw_vals = {d: [] for d in draws}
        for lineage in lineages:
            paired = []
            for draw in draws:
                a = lookup[(lineage, draw, *left)][metric]
                b = lookup[(lineage, draw, *right)][metric] if right else 0.
                if a is not None and b is not None:
                    paired.append(a - b); draw_vals[draw].append(a - b)
            if len(paired) == len(draws):
                vals.append(float(np.mean(paired))); ids.append(lineage)
        return dict(complete_lineages=len(ids), lineage_ids=ids,
                    draw_means=[float(np.mean(draw_vals[d])) if draw_vals[d] else None for d in draws],
                    **(V.interval(vals, cfg) if vals else dict(mean=None, low=None, high=None, lineage_values=[])))
    for rule, mode, change, subset, arm in product(RULES, ('original', 'composition-holdout'), CHANGES, V.SUBSETS, ARMS):
        left = (rule, mode, change, subset, arm)
        identity = dict(zip((*AXES, 'arm'), left, strict=True))
        means.append(dict(identity, metrics={m: collect(left, None, m) for m in METRICS}))
        if arm != 'direct':
            right = (rule, mode, change, subset, 'direct')
            contrasts.append(dict(identity, contrast='method-minus-direct', baseline='direct',
                                  metrics={m: collect(left, right, m) for m in METRICS}))
        if rule == 'presentation-tool':
            right = ('original', mode, change, subset, arm)
            contrasts.append(dict(identity, contrast='changed-rule-minus-original-rule', baseline='original',
                                  metrics={m: collect(left, right, m) for m in METRICS}))
    return dict(cells=cells, means=means, contrasts=contrasts,
        population='native weights normalized within named rule/change/support stratum; two saved draws averaged within equal paired lineages; empty strata explicit; cross-rule contrasts include changed native weights')


def controls():
    q = (1, 0, 2, 3, 4, 4); undo = (1, 0, 2, 3, 5, 4)
    return dict(V.controls(), **{
        'live:changed_endpoint': endpoint(q, RULES[0]) != endpoint(q, RULES[1]),
        'placebo:undo_stay': endpoint(undo, RULES[0]) == endpoint(undo, RULES[1]) == 2,
        'positive:wrong_law_floor': float((31/32*np.eye(8)[endpoint(q, RULES[0])] + 1/256)[endpoint(q, RULES[1])]) == 1/256,
        'placebo:empty_denominators': len(strata([], {})) == 15 and all(r['loss'] is None for r in strata([], {}))})


def run(root, plan, pulse):
    checks = controls(); write(root / 'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('transfer review controls failed')
    cfg = plan['design']; original = root / 'inputs/original'
    for name, digest in cfg['input_files'].items():
        if file_digest(root / 'inputs' / name) != digest: raise ValueError('review input changed')
    if file_digest(original / 'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan changed')
    design = read(original / 'PLAN.json')['design']; summary = read(original / 'SUMMARY.json')
    if design['arms'] != list(ARMS) or design['rules'] != list(RULES) or design['epsilon'] != 1/32:
        raise ValueError('target design differs')
    populations = {}; paths = 0; error = 0.; all_queries = set(); enumeration = []
    for lineage, rule in product(design['lineages'], RULES):
        pulse(phase='independent-transfer-paths', lineage=lineage, rule=rule)
        records = V.zipped(original / 'raw' / f'{lineage}-{rule}_points.json.gz')
        _, _, delta = V.population(records, lineage, rule); error = max(error, delta); paths += len(records)
        groups = {}
        for r in records:
            q = V.query(r); target = V.code(r['final']); all_queries.add(q)
            if endpoint(q, rule) != target: raise ValueError('native endpoint differs')
            if q not in groups: groups[q] = [target, 0.]
            if groups[q][0] != target: raise ValueError('query insufficiency')
            groups[q][1] += r['probability']
        error = max(error, V.near([sum(v[1] for v in groups.values())], [1.]))
        populations[lineage, rule] = groups
        enumeration.append(dict(lineage=lineage, rule=rule, paths=len(records), queries=len(groups)))
    if enumeration != read(original / 'ENUMERATION.json'): raise ValueError('enumeration differs')
    qs = sorted(all_queries)
    public = [dict(skill=q[0], belief_error=q[1], initial=list(V.ARTS[q[2]]), operations=[V.OPS[i] for i in q[3:]]) for q in qs]
    if public != read(original / 'reader/QUERIES.json'): raise ValueError('reader projection differs')
    truth = [dict(query=list(q), targets={rule: endpoint(q, rule) for rule in RULES}) for q in qs]
    if truth != read(original / 'QUERY_TRUTH.json'): raise ValueError('query truth differs')
    cells = []; checked_rows = 0; timing = read(original / 'TIMING.jsonl')['measurements']
    for draw in design['training_draws']:
        selected = read(root / 'inputs/training' / f'selection-{draw}.json')['paths'][:2048]
        for mode in ('original', 'composition-holdout'):
            pulse(phase='independent-transfer-forecasts', draw=draw, mode=mode)
            rr = selected if mode == 'original' else [r for r in selected if not V.held(V.query(r))]
            counts, direct = V.counts_from(rr); keys = sorted(direct)
            with np.load(original / 'inputs/models' / f'{draw}-{mode}.npz') as model:
                V.near(counts, model['transition_counts'], 0); V.near(keys, model['direct_keys'], 0)
                V.near([direct[k] for k in keys], model['direct_counts'], 0)
            predictions = {a: [] for a in V.ARMS}; diagnostics = []
            with np.load(original / 'forecasts' / f'{draw}-{mode}.npz') as saved, np.load(original / 'inputs/forecasts' / f'{draw}-{mode}.npz') as parent:
                V.near(saved['queries'], qs, 0); V.near(parent['queries'], qs, 0)
                if set(saved.files) != {'queries', 'uniforms', *V.ARMS, *('true-law-oracle:' + r for r in RULES)}:
                    raise ValueError('forecast schema differs')
                for i, q in enumerate(qs):
                    u = rng('v19-rollout-query', draw, *q).uniform(size=(16, 3))
                    V.near(u, saved['uniforms'][i], 0); V.near(u, parent['uniforms'][i], 0)
                    ps, diag, target = V.forecast(counts, direct, q, u); diagnostics.append(diag)
                    if target != endpoint(q, 'original'): raise ValueError('original target differs')
                    for arm, p in ps.items():
                        error = max(error, V.near(p, saved[arm][i], 1e-12), V.near(p, parent[arm][i], 1e-12))
                        predictions[arm].append(p)
                for rule in RULES:
                    p = 31/32*np.eye(8)[[endpoint(q, rule) for q in qs]] + 1/256
                    error = max(error, V.near(p, saved['true-law-oracle:' + rule], 1e-12))
                    predictions['true-law-oracle:' + rule] = p
            for folder in ('forecasts', 'inputs/forecasts'):
                if diagnostics != read(original / folder / f'{draw}-{mode}-support.json'): raise ValueError('support differs')
            rows = V.zipped(original / 'raw' / f'{draw}-{mode}-transfer_points.json.gz')
            lookup = {(r['lineage'], r['rule'], r['arm'], r['query_index']): r for r in rows}; consumed = set()
            if len(lookup) != len(rows): raise ValueError('duplicate scored row')
            for (lineage, rule), groups in populations.items():
                for arm in ARMS:
                    entries = []
                    for i, q in enumerate(qs):
                        if q not in groups: continue
                        target, mass = groups[q]; p = predictions['true-law-oracle:' + rule if arm == 'true-law-oracle' else arm][i]
                        diag = diagnostics[i]; changed = endpoint(q, 'original') != endpoint(q, 'presentation-tool')
                        metrics = dict(loss=float(-np.log(p[target])), squared_error=float(np.sum((p-np.eye(8)[target])**2)), true_probability=float(p[target]))
                        k = (lineage, rule, arm, i); row = lookup[k]; consumed.add(k)
                        if row['draw'] != draw or row['mode'] != mode or row['changed'] != changed or any(row[n] != v for n, v in diag.items()):
                            raise ValueError('scored row identity differs')
                        error = max(error, V.near([mass, *metrics.values()], [row['probability_mass'], *(row[m] for m in METRICS)]))
                        entries.append(dict(probability_mass=mass, changed=changed, **metrics, **diag))
                    cells.extend(strata(entries, dict(lineage=lineage, rule=rule, draw=draw, mode=mode, arm=arm)))
            if consumed != set(lookup): raise ValueError('unconsumed scored rows')
            checked_rows += len(rows)
            times = [r for r in timing if r['draw'] == draw and r['mode'] == mode]
            if len(times) != 2 or {r['phase'] for r in times} != {'parent-reconstruction', 'scoring'} or any(r['cpu_seconds'] < 0 for r in times):
                raise ValueError('timing completeness differs')
    key = lambda r: tuple(r[k] for k in ('lineage', 'draw', *AXES, 'arm'))
    lookup = {key(r): r for r in cells}; old = {key(r): r for r in summary['cells']}
    if len(old) != len(summary['cells']) or set(old) != set(lookup): raise ValueError('aggregate roster differs')
    for k, row in old.items():
        rebuilt = lookup[k]
        if row['queries'] != rebuilt['queries']: raise ValueError('query denominator differs')
        for metric in ('population_mass', *METRICS):
            if rebuilt[metric] is None:
                if row[metric] is not None: raise ValueError('empty conditional score differs')
            else: error = max(error, V.near([row[metric]], [rebuilt[metric]]))
    parent = read(original / 'inputs/SUMMARY.json')['cells']
    for row in parent:
        rebuilt = lookup[(row['lineage'], row['draw'], 'original', row['mode'], 'all', row['subset'], row['arm'])]
        if row['queries'] != rebuilt['queries']: raise ValueError('parent query denominator differs')
        error = max(error, V.near([row[k] for k in ('population_mass', 'loss', 'squared_error')], [rebuilt[k] for k in ('population_mass', 'loss', 'squared_error')]))
    receipt = read(original / 'PARENT_REPRODUCTION.json')
    if not receipt['passed'] or receipt['cells'] != len(parent) or receipt['forecasts'] != 2*len(design['training_draws']): raise ValueError('parent receipt differs')
    if summary['paths'] != paths or summary['rows'] != checked_rows or summary['queries'] != len(qs) or summary['fits'] != 0: raise ValueError('complete counts differ')
    write(root / 'INDEPENDENT_REGROUP.json', regroup(cells, design['lineages'], design['training_draws'], cfg))
    checks['positive:complete_reconstruction'] = True
    return dict(controls=checks, paths=paths, rows=checked_rows, cells=len(cells), queries=len(qs), parent_cells=len(parent), max_error=error,
                target_plan_sha256=cfg['target_plan_sha256'], scope='independent paths/counts/forecasts/support/scores and paired regroup; numerical event adjudication pending')
