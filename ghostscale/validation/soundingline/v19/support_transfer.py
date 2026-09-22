"""Frozen support-table transfer; common smoothing and explicit law privilege."""
import gzip
import time
import numpy as np
from ..v18_3.io import canonical, read, write, file_digest
from . import forward_support as S, rollout as R, rollout_transfer as T
from . import missing_tool as M, local_world as L

ARMS = S.ARMS + ('true-law-oracle',)
SUPPORT = ('all', 'query-seen', 'query-unseen', 'heldout-composition', 'heldout-seen-primitives')


def forecasts(counts, direct, queries, uniforms):
    table = R.normalized(counts)
    arrays = {a: [] for a in S.ARMS}
    for q, u in zip(queries, uniforms, strict=True):
        for arm in S.ARMS:
            if arm == 'direct': p = R.normalized(direct.get(q, np.ones(8)))
            elif arm == 'retrieval': p = T.retrieval(direct, q)
            elif arm == 'learned-exact': p = R.propagate(table, q)
            elif arm == 'oracle-exact': p = np.eye(8)[R.oracle(q)]
            else: p = S.empirical(R.sampled_paths(table, q, u[:int(arm.split('-')[1])]))
            arrays[arm].append(S.smooth(p))
    return {a: np.array(v) for a, v in arrays.items()}


def select(row, subset):
    return {'all': True, 'query-seen': row['query_seen'],
            'query-unseen': not row['query_seen'],
            'heldout-composition': row['withheld_composition'],
            'heldout-seen-primitives': row['withheld_composition'] and row['all_primitives_seen']}[subset]


def summarize(entries, identity):
    cells = []
    for change in ('all', 'changed', 'stay'):
        for subset in SUPPORT:
            chosen = [r for r in entries if select(r, subset) and
                      (change == 'all' or r['changed'] == (change == 'changed'))]
            mass = sum(r['probability_mass'] for r in chosen)
            # Empty support is an explicit denominator, never an omitted favorable cell.
            cells.append(dict(identity, change=change, subset=subset, queries=len(chosen),
                              population_mass=mass, **{k: sum(r['probability_mass'] * r[k] for r in chosen) / mass
                              if mass else None for k in ('loss', 'squared_error', 'true_probability')}))
    return cells


def controls():
    changed = (1, 0, 2, 3, 4, 4)
    stay = (1, 0, 2, 3, 5, 4)
    p = S.smooth(np.eye(8)[T.oracle(changed, 'original')])
    return {'live:changed_tool': T.oracle(changed, 'original') != T.oracle(changed, 'presentation-tool'),
            'placebo:undo_restores': T.oracle(stay, 'original') == T.oracle(stay, 'presentation-tool') == 2,
            'positive:wrong_law_finite': bool(p[T.oracle(changed, 'presentation-tool')] == 1 / 256),
            'positive:same_smoothing': np.array_equal(S.smooth(S.empirical([3])), S.smooth(S.empirical([3] * 16))),
            'positive:common_floor': bool(p.min() == 1 / 256 and np.isclose(p.sum(), 1)),
            'placebo:empty_subpopulation': all(x['loss'] is None and x['population_mass'] == 0
                                              for x in summarize([], {}))}


def run(root, plan, pulse):
    cfg = plan['design']; inputs = root / 'inputs'; checks = controls()
    if (not all(checks.values()) or cfg['arms'] != list(ARMS) or
            cfg['rules'] != list(M.RULES) or cfg['epsilon'] != S.EPSILON):
        raise ValueError('support transfer admission failed')
    for n, h in cfg['input_files'].items():
        if file_digest(inputs / n) != h: raise ValueError('frozen support input changed')
    for folder in ('raw', 'forecasts', 'reader'): (root / folder).mkdir()
    write(root / 'CONTROLS.json', checks)
    populations = {}; qs = None; timing = []; arrays_by_fit = {}; diagnostics = {}
    # Reconstruct every saved prediction before changing the execution law.
    for draw in cfg['training_draws']:
        for mode in ('original', 'composition-holdout'):
            start = time.process_time(); pulse(phase='parent-forecast-reconstruction', draw=draw, mode=mode)
            with np.load(inputs / 'models' / f'{draw}-{mode}.npz') as data:
                counts = data['transition_counts']; direct = {tuple(map(int, k)): v for k, v in
                           zip(data['direct_keys'], data['direct_counts'], strict=True)}
            with np.load(inputs / 'forecasts' / f'{draw}-{mode}.npz') as data:
                queries = [tuple(map(int, q)) for q in data['queries']]; uniforms = data['uniforms']
                if qs is None: qs = queries
                if qs != queries or len(qs) != cfg['queries']: raise ValueError('parent query roster changed')
                arrays = forecasts(counts, direct, qs, uniforms)
                if any(not np.allclose(arrays[a], data[a], atol=1e-12, rtol=0) for a in S.ARMS):
                    raise ValueError('parent forecasts do not reproduce')
            diag = [S.support(counts, direct, q) for q in qs]
            if diag != read(inputs / 'forecasts' / f'{draw}-{mode}-support.json'):
                raise ValueError('parent support does not reproduce')
            diagnostics[draw, mode] = diag; arrays_by_fit[draw, mode] = arrays
            for rule in M.RULES: arrays['true-law-oracle:' + rule] = S.smooth(np.eye(8)[[T.oracle(q, rule) for q in qs]])
            np.savez(root / 'forecasts' / f'{draw}-{mode}.npz', queries=np.array(qs), uniforms=uniforms, **arrays)
            write(root / 'forecasts' / f'{draw}-{mode}-support.json', diag)
            timing.append(dict(draw=draw, mode=mode, phase='parent-reconstruction', cpu_seconds=time.process_time()-start, fits=0))
    public = [R.visible(q) for q in qs]
    for p in public: R.validate_visible(p)
    write(root / 'reader/QUERIES.json', public)
    write(root / 'QUERY_TRUTH.json', [dict(query=list(q), targets={rule: T.oracle(q, rule) for rule in M.RULES}) for q in qs])
    enumeration = []
    for lineage in cfg['lineages']:
        for rule in M.RULES:
            pulse(phase='changed-tool-populations', lineage=lineage, rule=rule)
            rr = M.enumerate_rule(L.law(lineage), rule); groups = {}
            for r in rr:
                q = R.query(r); target = R.code(r['final'])
                if q not in qs or T.oracle(q, rule) != target: raise ValueError('changed native reference mismatch')
                if q in groups and groups[q][0] != target: raise ValueError('query insufficient')
                groups.setdefault(q, [target, 0.])[1] += r['probability']
            if abs(sum(v[1] for v in groups.values()) - 1) > 1e-10: raise ValueError('native mass')
            populations[lineage, rule] = groups
            (root / 'raw' / f'{lineage}-{rule}_points.json.gz').write_bytes(gzip.compress(canonical(rr), mtime=0))
            enumeration.append(dict(lineage=lineage, rule=rule, paths=len(rr), queries=len(groups)))
    cells = []; total_rows = 0
    for (draw, mode), arrays in arrays_by_fit.items():
        rows = []; start = time.process_time()
        for (lineage, rule), groups in populations.items():
            pulse(phase='support-transfer-scoring', lineage=lineage, rule=rule, draw=draw, mode=mode)
            for arm in ARMS:
                entries = []
                for i, q in enumerate(qs):
                    if q not in groups: continue
                    target, mass = groups[q]; p = arrays['true-law-oracle:' + rule if arm == 'true-law-oracle' else arm][i]
                    if (p < 1/256).any() or not np.isclose(p.sum(), 1): raise ValueError('invalid smoothed distribution')
                    entries.append(dict(lineage=lineage, rule=rule, draw=draw, mode=mode, arm=arm,
                        query_index=i, probability_mass=mass, changed=T.oracle(q, 'original') != T.oracle(q, 'presentation-tool'),
                        loss=float(-np.log(p[target])), squared_error=float(np.sum((p-np.eye(8)[target])**2)),
                        true_probability=float(p[target]), **diagnostics[draw, mode][i]))
                cells.extend(summarize(entries, dict(lineage=lineage, rule=rule, draw=draw, mode=mode, arm=arm)))
                rows.extend(entries)
        (root / 'raw' / f'{draw}-{mode}-transfer_points.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
        total_rows += len(rows); timing.append(dict(draw=draw, mode=mode, phase='scoring', cpu_seconds=time.process_time()-start))
    original = {(r['lineage'], r['draw'], r['mode'], r['arm'], r['subset']): r for r in cells
                if r['rule'] == 'original' and r['change'] == 'all' and r['arm'] in S.ARMS}
    parent = read(inputs / 'SUMMARY.json')['cells']
    for r in parent:
        z = original[r['lineage'], r['draw'], r['mode'], r['arm'], r['subset']]
        if z['queries'] != r['queries'] or any(abs(z[k]-r[k]) > 1e-10 for k in ('population_mass', 'loss', 'squared_error')):
            raise ValueError('parent original-law aggregate does not reproduce')
    write(root / 'PARENT_REPRODUCTION.json', dict(passed=True, cells=len(parent), forecasts=len(arrays_by_fit),
        scope='complete original-law score and forecast reproduction; parent endpoints are identity controls, not new replicates'))
    write(root / 'ENUMERATION.json', enumeration)
    write(root / 'TIMING.jsonl', dict(measurements=timing, accounting='new reconstruction/enumeration/scoring charged; no refit'))
    return dict(controls=checks, cells=cells, rows=total_rows, fits=0, queries=len(qs),
        paths=sum(r['paths'] for r in enumeration), parent_reproduction=True,
        scope='frozen mechanics transfer; original-law support strata, supplied skills/beliefs/operations; no inverse historical process claim')
