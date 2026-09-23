"""Independent hidden-metadata enumeration; imports no producer routines."""
from collections import defaultdict
from itertools import product
import math
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import support_review as V, transfer_review as T

VIEWS = ('full', 'omit-skill', 'omit-belief', 'omit-both')
AXES = ('rule', 'view', 'weighting', 'arm')
METRICS = ('finite_loss_contribution', 'infinite_loss_mass', 'squared_error',
           'true_probability', 'legal_ambiguous_mass', 'native_conditional_entropy')


def projection(q, view):
    if view not in VIEWS:
        raise ValueError('view')
    result = dict(initial=list(V.ARTS[q[2]]), operations=[V.OPS[o] for o in q[3:]])
    if view in ('full', 'omit-belief'):
        result['skill'] = q[0]
    if view in ('full', 'omit-skill'):
        result['belief_error'] = q[1]
    return result


def groups_for(queries, view):
    """Group by tuples, independently of the producer's content-hash grouping."""
    grouped = defaultdict(list)
    for i, q in enumerate(queries):
        key = (q[0] if view in ('full', 'omit-belief') else None,
               q[1] if view in ('full', 'omit-skill') else None, *q[2:])
        grouped[key].append(i)
    result = {}
    for key, indices in grouped.items():
        visible = projection(queries[indices[0]], view)
        legal = [(s, b, *key[2:]) for s, b in product(range(2), repeat=2)
                 if (key[0] is None or s == key[0]) and (key[1] is None or b == key[1])
                 and (s or 3 not in key[3:])]
        if not legal or any(queries[i] not in legal for i in indices):
            raise ValueError('legal metadata')
        result[digest(visible)] = dict(reader=visible, indices=indices,
                                     legal_completions=[list(q) for q in legal])
    return result


def entropy(probabilities):
    return math.fsum(-float(p)*math.log(float(p)) for p in probabilities if p > 0)


def reconstruct(queries, groups, mass, rule):
    mass = np.asarray(mass, dtype=float)
    if mass.shape != (len(queries),) or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError('population')
    V.near([math.fsum(mass)], [1.])
    targets = np.array([T.endpoint(q, rule) for q in queries])
    uniform = np.zeros((len(queries), 8)); native = uniform.copy(); records = []
    for key, group in groups.items():
        ids = group['indices']; legal = group['legal_completions']
        ends = [T.endpoint(q, rule) for q in legal]
        u = np.array([sum(e == t for e in ends)/len(ends) for t in range(8)])
        counts = np.array([math.fsum(mass[i] for i in ids if targets[i] == t) for t in range(8)])
        total = math.fsum(mass[i] for i in ids)
        n = counts/total if total > 0 else u.copy()
        uniform[ids] = u; native[ids] = n
        records.append(dict(reader_id=key, query_indices=ids, legal_completions=legal,
            legal_endpoints=ends, native_mass=total, endpoint_mass=counts.tolist(),
            uniform_prediction=u.tolist(), native_prediction=n.tolist(),
            legal_ambiguous=len(set(ends)) > 1, native_ambiguous=np.count_nonzero(counts) > 1,
            legal_entropy=entropy(u), native_entropy=entropy(n) if total else None,
            zero_native_mass=not bool(total), native_zero_mass_fallback=None if total else 'uniform-legal'))
    return targets, records, uniform, native


def scores(predicted, targets, weights):
    """Scalar proper scores; preserve infinite mass separately, without clipping."""
    finite = []; infinite = []; squared = []; truth = []
    for p, t, w in zip(predicted, targets, weights, strict=True):
        if not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError('prediction')
        V.near([math.fsum(p)], [1.])
        if p[t] == 0:
            infinite.append(float(w))
        else:
            finite.append(-float(w)*math.log(float(p[t])))
        squared.append(float(w)*math.fsum((float(v)-int(j == t))**2 for j, v in enumerate(p)))
        truth.append(float(w)*float(p[t]))
    return dict(finite_loss_contribution=math.fsum(finite), infinite_loss_mass=math.fsum(infinite),
                squared_error=math.fsum(squared), true_probability=math.fsum(truth))


def compare(a, b, tolerance=1e-12):
    """Schema-sensitive nested comparison: no unexamined group fields."""
    if isinstance(b, dict):
        if set(a) != set(b): raise ValueError('record schema')
        return max((compare(a[k], b[k], tolerance) for k in b), default=0.)
    if isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b): raise ValueError('record length')
        return max((compare(x, y, tolerance) for x, y in zip(a, b, strict=True)), default=0.)
    if b is None or isinstance(b, (str, bool, np.bool_, int, np.integer)):
        if a != b: raise ValueError('record identity/denominator')
        return 0.
    return V.near([a], [b], tolerance)


def regroup(cells, lineages, cfg):
    by_key = {(r['lineage'], *(r[k] for k in AXES)): r for r in cells}
    if len(by_key) != len(cells): raise ValueError('duplicate cells')
    estimates = []
    for key in sorted({tuple(r[k] for k in AXES) for r in cells}):
        comparisons = [('mean', None)]
        if key[1] != 'full': comparisons.append(('omission-minus-full', (key[0], 'full', *key[2:])))
        if key[3] == 'native-law': comparisons.append(('native-minus-uniform', (*key[:3], 'uniform-legal')))
        for label, base in comparisons:
            for metric in METRICS:
                values = [by_key[(lin, *key)][metric] - (by_key[(lin, *base)][metric] if base else 0.) for lin in lineages]
                estimates.append(dict(zip(AXES, key), contrast=label, metric=metric,
                                      **V.interval(values, cfg)))
    return dict(estimates=estimates, lineages=lineages, bootstrap_seed=cfg['bootstrap_seed'],
        bootstrap_resamples=cfg['bootstrap_resamples'], practical_margin_nats=.02,
        population='eight retained development laws weighted equally; native and equal-query estimands separate; no fits or fresh draws',
        limitation='finite loss contribution is not total loss when infinite_loss_mass is positive; intervals condition on fixed query roster and supplied laws')


def review(original, output, cfg, pulse=lambda **kw: None):
    design = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json')
    base = original/'inputs'; truth = read(base/'QUERY_TRUTH.json')
    queries = [tuple(row['query']) for row in truth]
    if design['views'] != list(VIEWS) or design['rules'] != list(T.RULES): raise ValueError('design')
    if len(set(queries)) != len(queries) or len(queries) != design['queries']: raise ValueError('queries')
    groups = {v: groups_for(queries, v) for v in VIEWS}
    if groups != read(base/'MEMBERSHIP.json') or groups != read(original/'evaluator/MEMBERSHIP.json'):
        raise ValueError('frozen membership')
    expected_reader = {k: row['reader'] for g in groups.values() for k, row in g.items()}
    if {p.stem for p in (original/'reader').glob('*.json')} != set(expected_reader): raise ValueError('reader coverage')
    for k, visible in expected_reader.items():
        if visible != read(original/'reader'/f'{k}.json'): raise ValueError('reader projection')
    if truth != [dict(query=list(q), targets={r: T.endpoint(q, r) for r in T.RULES}) for q in queries]:
        raise ValueError('query truth')
    index = {q: i for i, q in enumerate(queries)}; cells = []; groups_all = []; paths = 0; error = 0.; forecast_names = set()
    for lineage, rule in product(design['lineages'], T.RULES):
        pulse(phase='independent-metadata', lineage=lineage, rule=rule)
        records = V.zipped(base/'raw'/f'{lineage}-{rule}_points.json.gz'); paths += len(records)
        population_terms = [[] for _ in queries]
        for r in records:
            i = index[V.query(r)]; w = r['probability']
            if not math.isfinite(w) or w < 0 or V.code(r['final']) != T.endpoint(queries[i], rule):
                raise ValueError('native path')
            population_terms[i].append(w)
        mass = np.array([math.fsum(v) for v in population_terms]); error = max(error, V.near([mass.sum()], [1.]))
        for view in VIEWS:
            targets, rows, u, n = reconstruct(queries, groups[view], mass, rule)
            groups_all.extend(dict(lineage=lineage, rule=rule, view=view, **r) for r in rows)
            name = f'{lineage}-{rule}-{view}_points.npz'; forecast_names.add(name)
            with np.load(original/'forecasts'/name, allow_pickle=False) as z:
                if set(z.files) != {'queries', 'targets', 'mass', 'uniform', 'native'}: raise ValueError('forecast schema')
                error = max(error, V.near(z['queries'], queries, 0), V.near(z['targets'], targets, 0),
                            V.near(z['mass'], mass), V.near(z['uniform'], u), V.near(z['native'], n))
            for weighting, w in (('native', mass), ('equal-query', np.full(len(queries), 1/len(queries)))):
                for arm, p in (('uniform-legal', u), ('native-law', n)):
                    cells.append(dict(lineage=lineage, rule=rule, view=view, weighting=weighting, arm=arm,
                        groups=len(rows), queries=len(queries), zero_mass_groups=sum(r['zero_native_mass'] for r in rows),
                        legal_ambiguous_groups=sum(r['legal_ambiguous'] for r in rows),
                        native_ambiguous_groups=int(sum(r['native_ambiguous'] for r in rows)),
                        legal_ambiguous_mass=math.fsum(w[i] for r in rows if r['legal_ambiguous'] for i in r['query_indices']),
                        native_conditional_entropy=math.fsum(r['native_mass']*r['native_entropy'] for r in rows if r['native_mass']),
                        **scores(p, targets, w)))
    if {p.name for p in (original/'forecasts').glob('*.npz')} != forecast_names: raise ValueError('forecast coverage')
    error = max(error, compare(V.zipped(original/'evaluator/GROUPS_points.json.gz'), groups_all),
                compare(summary['cells'], cells))
    if summary['queries'] != len(queries) or summary['native_paths'] != paths or summary['group_rows'] != len(groups_all) or summary['fits'] != 0:
        raise ValueError('totals')
    checks = read(original/'CONTROLS.json')
    if not checks or not all(checks.values()) or checks != summary['controls']: raise ValueError('producer controls')
    timing = read(original/'TIMING.jsonl')
    if timing['fits'] != 0 or timing['cpu_seconds'] < 0: raise ValueError('timing')
    write(output/'INDEPENDENT_REGROUP.json', regroup(cells, design['lineages'], cfg))
    write(output/'RECONSTRUCTED_CELLS.json', cells)
    result = dict(passed=True, cells=len(cells), group_rows=len(groups_all), native_paths=paths,
        queries=len(queries), forecast_vectors=len(forecast_names)*len(queries)*2,
        reader_packets=len(expected_reader), max_error=error,
        scope='independent tuple groups,legal completions,validated independent mechanics,native mass sums,all distributions,entropy,proper scores,denominators and evidence projections; parent policy probabilities inherit validated source review')
    write(output/'NUMERICAL_REVIEW.json', result)
    return result


def run(root, plan, pulse):
    cfg = plan['design']; original = root/'inputs/original'
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('input binding')
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target binding')
    result = review(original, root, cfg, pulse)
    return dict(result, controls={'live:all_omission_distributions_rebuilt': True,
        'positive:complete_projection_and_population': True, 'placebo:full_metadata_and_zero_mass': True})
