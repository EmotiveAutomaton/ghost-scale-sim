"""Independent scalar reconstruction of actual versus stated reply reliability."""
from itertools import product
import math
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import noisy_review as N

ACCURACIES = (.5, .75, 1.)
REQUESTS = ('none', 'skill', 'belief')
AXES = ('rule', 'model', 'actual_accuracy', 'assumed_accuracy', 'request', 'weighting', 'cost')
METRICS = ('finite_loss_contribution', 'infinite_loss_mass', 'squared_error',
           'true_probability', 'residual_ambiguous_mass', 'net_finite_loss', 'request_rate')


def regroup(cells, lineages, cfg):
    lookup = {(r['lineage'], *(r[k] for k in AXES)): r for r in cells}
    if len(lookup) != len(cells): raise ValueError('duplicate cells')
    estimates = []
    for key in sorted({tuple(r[k] for k in AXES) for r in cells}):
        bases = [('mean', None)]
        if key[4] != 'none':
            bases.append(('minus-none', (*key[:4], 'none', *key[5:])))
        if key[3] != key[2]:
            bases.append(('minus-calibrated', (*key[:3], key[2], *key[4:])))
        if key[3] != .5:
            bases.append(('minus-ignore-reply', (*key[:3], .5, *key[4:])))
        for contrast, base in bases:
            for metric in METRICS:
                values = [lookup[(lin, *key)][metric] - (lookup[(lin, *base)][metric] if base else 0.) for lin in lineages]
                estimates.append(dict(zip(AXES, key), contrast=contrast, metric=metric,
                                      **N.D.V.V.interval(values, cfg)))
    return dict(estimates=estimates, lineages=lineages, bootstrap_seed=cfg['bootstrap_seed'],
                bootstrap_resamples=cfg['bootstrap_resamples'],
                population='paired development laws weighted equally; native and equal-query populations separate',
                limitation='fixed queries and supplied laws; finite contribution excludes infinite-loss mass; no fits or confirmation')


def review(original, output, cfg, pulse=lambda **kw: None):
    design = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json'); base = original/'inputs'
    for key, expected in [('actual_accuracies', ACCURACIES), ('assumed_accuracies', ACCURACIES),
                          ('requests', REQUESTS), ('models', N.MODELS), ('costs', N.COSTS), ('rules', N.D.V.T.RULES)]:
        if design[key] != list(expected): raise ValueError('design')
    laws = read(base/'DISCLOSURE_LAWS.json')
    lookup = {(r['lineage'], r['rule'], r['model'], r['reader_id']): r for r in laws}
    recorded = read(original/'evaluator/REPLY_LAWS.json')
    rows_lookup = {(r['lineage'], r['rule'], r['model'], r['assumed_accuracy'], r['request'], r['reader_id']): r for r in recorded}
    if len(lookup) != len(laws) or len(rows_lookup) != len(recorded): raise ValueError('duplicate laws')
    cells = []; used = set(); used_rows = set(); seen = set(); input_seen = set(); roster = None; error = 0.; fallbacks = 0
    near = N.D.V.V.near
    for lin, rule, model in product(design['lineages'], design['rules'], N.MODELS):
        pulse(phase='independent-reliability-mismatch', lineage=lin, rule=rule, model=model)
        name = f'{lin}-{rule}-{model}_points.npz'; input_seen.add(name)
        with np.load(base/'forecasts'/name, allow_pickle=False) as parent:
            qs = [tuple(map(int, q)) for q in parent['queries']]; n = len(qs)
            if n != design['queries'] or len(set(qs)) != n: raise ValueError('query roster')
            if roster is None:
                roster = qs; groups = N.D.V.groups_for(qs, 'omit-both')
                if groups != read(base/'MEMBERSHIP.json')['omit-both']: raise ValueError('membership')
            elif qs != roster: raise ValueError('query identity')
            targets = np.array([N.D.V.T.endpoint(q, rule) for q in qs]); mass = parent['mass'].copy()
            if mass.shape != (n,) or not np.isfinite(mass).all() or np.any(mass < 0): raise ValueError('population')
            error = max(error, near(parent['targets'], targets, 0), near([math.fsum(mass)], [1.]))
            index = {q: i for i, q in enumerate(qs)}
            for assumed in ACCURACIES:
                predictions = {r: np.zeros((n, 2, 8)) for r in REQUESTS}
                for key, group in groups.items():
                    ident = (lin, rule, model, key); law = lookup[ident]; used.add(ident)
                    legal = group['legal_completions']; ends = [N.D.V.T.endpoint(q, rule) for q in legal]
                    raw = [float(mass[index[tuple(q)]]) if tuple(q) in index else 0. for q in legal]; total = math.fsum(raw)
                    weights = [w/total for w in raw] if model == 'native-law' and total else [1/len(legal)]*len(legal)
                    if law['legal_completions'] != legal or law['endpoints'] != ends: raise ValueError('legal mechanics')
                    error = max(error, near(law['conditional_weights'], weights))
                    dist = N.conditional(legal, ends, weights, assumed); ids = group['indices']
                    predictions['none'][ids] = np.repeat(dist['prior'][None, :], 2, axis=0)
                    error = max(error, near(parent['none'][ids], np.tile(dist['prior'], (len(ids), 1))))
                    for request, axis in (('skill', 0), ('belief', 1)):
                        predictions[request][ids] = np.stack([dist['tables'][request][b] for b in (0, 1)])
                        identity = (lin, rule, model, assumed, request, key); used_rows.add(identity)
                        expected = dict(lineage=lin, rule=rule, model=model, assumed_accuracy=assumed, request=request, reader_id=key,
                                        modeled_reply_mass=[dist['reply_mass'][request][b] for b in (0, 1)],
                                        fallbacks=[dist['fallbacks'][request][b] for b in (0, 1)])
                        error = max(error, N.D.V.compare(rows_lookup[identity], expected))
                        fallbacks += sum(x is not None for x in expected['fallbacks'])
                        if assumed == .5:
                            for b in (0, 1): error = max(error, near(dist['tables'][request][b], dist['prior']))
                        if assumed == 1:
                            for i in ids: error = max(error, near(parent[request][i], dist['tables'][request][qs[i][axis]]))
                fname = f'{lin}-{rule}-{model}-assumed-{assumed:g}_points.npz'; seen.add(fname)
                with np.load(original/'forecasts'/fname, allow_pickle=False) as saved:
                    keys = {'queries', 'mass', 'targets', 'strings', *REQUESTS,
                            *(f'{r}-actual-{a:g}' for r, a in product(REQUESTS, ACCURACIES))}
                    if set(saved.files) != keys: raise ValueError('forecast schema')
                    for key, value in dict(queries=qs, mass=mass, targets=targets, strings=[0, 1]).items():
                        error = max(error, near(saved[key], value, 0))
                    for request in REQUESTS:
                        pred = predictions[request]; error = max(error, near(saved[request], pred))
                        for actual in ACCURACIES:
                            probabilities = np.tile([1., 0.], (n, 1)) if request == 'none' else np.array([
                                [actual if q[int(request == 'belief')] == bit else 1-actual for bit in (0, 1)] for q in qs])
                            error = max(error, near(saved[f'{request}-actual-{actual:g}'], probabilities, 0))
                            for weighting, weights in (('native', mass), ('equal-query', np.full(n, 1/n))):
                                result = N.scores(pred, probabilities, targets, weights); count = int(request != 'none')
                                for cost in N.COSTS:
                                    cells.append(dict(lineage=lin, rule=rule, model=model, actual_accuracy=actual, assumed_accuracy=assumed,
                                                      request=request, weighting=weighting, cost=cost, queries=n, groups=len(groups), request_rate=count,
                                                      net_finite_loss=result['finite_loss_contribution']+cost*count, **result))
    if used != set(lookup) or used_rows != set(rows_lookup): raise ValueError('law coverage')
    if {p.name for p in (original/'forecasts').glob('*.npz')} != seen or {p.name for p in (base/'forecasts').glob('*.npz')} != input_seen:
        raise ValueError('forecast coverage')
    error = max(error, N.D.V.compare(summary['cells'], cells))
    packets = {digest(g['reader']): g['reader'] for g in groups.values()}
    for group in groups.values():
        for field, assumed, bit in product(('skill', 'belief_error'), ACCURACIES, (0, 1)):
            packet = dict(group['reader'], requested_field=field, reply_value=bit, stated_accuracy=assumed); packets[digest(packet)] = packet
    if {p.name for p in (original/'reader').iterdir()} != {'REPLIES.json'} or read(original/'reader/REPLIES.json') != packets:
        raise ValueError('reader projection')
    checks = read(original/'CONTROLS.json')
    if not checks or not all(checks.values()) or checks != summary['controls']: raise ValueError('controls')
    if (summary['queries'] != len(roster) or summary['law_rows'] != len(used_rows)
        or summary['reader_packets'] != len(packets) or summary['fits'] != 0): raise ValueError('totals')
    timing = read(original/'TIMING.jsonl')
    if timing['fits'] != 0 or not math.isfinite(timing['cpu_seconds']) or timing['cpu_seconds'] < 0: raise ValueError('timing')
    write(output/'RECONSTRUCTED_CELLS.json', cells)
    write(output/'INDEPENDENT_REGROUP.json', regroup(cells, design['lineages'], cfg))
    result = dict(passed=True, cells=len(cells), law_rows=len(used_rows), queries=len(roster), reader_packets=len(packets),
                  forecast_vectors=len(seen)*len(roster)*6, max_error=error, zero_model_mass_fallbacks=fallbacks,
                  scope='independent scalar mechanics,law weights,Bayes sums,actual reply probabilities,realized-branch scores,identities,costs,roles and paired law strata; native masses inherit verified parent')
    write(output/'NUMERICAL_REVIEW.json', result); return result


def run(root, plan, pulse):
    cfg = plan['design']; original = root/'inputs/original'
    for name, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name) != h: raise ValueError('input binding')
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target binding')
    result = review(original, root, cfg, pulse)
    return dict(result, controls={'live:actual_and_assumed_channels_reconstructed': True,
                                 'positive:no_request_half_and_truthful_identities': True,
                                 'placebo:zero_support_and_explicit_infinite_loss': True})
