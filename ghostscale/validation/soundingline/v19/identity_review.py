"""Independent unknown-source Bayes reconstruction; no producer kernels."""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, file_digest, canonical
from .sufficient_review import structure, close, TOL
from .source_review import future_mass

ALPHAS = (0., .25, .5, .75, 1.)
IDENTITY = ('draw', 'initial_maker', 'kind', 'switched', 'duplicates')
KEY = ('lineage', 'evidence', 'length', 'checkpoint', 'alpha') + IDENTITY
METRICS = ('group_tv', 'max_future_difference', 'mean_future_tv')


def reciprocal_identity(coefficient, probability, count):
    """Check the defining unit-mass equation for unbounded reciprocals.

    Absolute error in a reciprocal has arbitrary scale near zero probability.
    The normalized identity and all downstream probabilities retain TOL. Exact
    support and finite/nonnegative coefficients are separately mandatory.
    """
    c = np.asarray(coefficient); p = np.asarray(probability)
    if c.shape != p.shape or not np.isfinite(c).all() or (c < 0).any(): raise ValueError('reciprocal shape/finite')
    if not np.array_equal(c > 0, p > 0): raise ValueError('reciprocal support')
    close(c*p*count[:, None, :], (p > 0).astype(float), 'reciprocal unit identity')


def reconstruct(w, st, law, ids, saved):
    """Scalar source normalizers; source-wise Bayes accumulation of posteriors.

    Accumulate unnormalized joint source/hypothesis weights for the exact account,
    separately normalized source posteriors for the uniform rival. Bincount over
    independent bit schedules rebuilds every future coordinate without the
    producer's sparse transition operator or grouping implementation.
    """
    w = np.asarray(w, float); law = np.asarray(law, float); ids = np.asarray(ids)
    if w.shape != (len(st['mapping']),) or (w < 0).any(): raise ValueError('weights')
    close(w.sum(), 1., 'weight mass')
    if law.shape != (16, 4, 8) or (law < 0).any(): raise ValueError('law')
    close(law.sum(-1), np.ones((16, 4)), 'law mass')
    if ids.ndim != 2 or ids.shape[1] != 3 or not len(ids) or ids.dtype.kind not in 'iu': raise ValueError('sources')
    if len(set(ids[:, 0])) != len(ids) or (ids[:, 0] < 1).any() or (ids[:, 0] > len(st['past'])).any() or (ids[:, 1] < 0).any() or (ids[:, 1] > 3).any() or (ids[:, 2] < 0).any() or (ids[:, 2] > 7).any(): raise ValueError('source bounds')
    count = len(ids); independent = np.empty((count, 8))
    likelihoods = []
    for s, (t, ctx, old) in enumerate(ids):
        states = st['past'][t-1]
        mass = np.bincount(states, weights=w, minlength=16)
        for e in range(8):
            independent[s, e] = math.fsum(float(mass[k])*float(law[k, ctx, e]) for k in range(16))
        if independent[s, old] <= 0: raise ValueError('old source unsupported')
        likelihoods.append(law[states, ctx].T)
    probabilities = np.empty((5, count, 8)); source_post = np.zeros_like(probabilities)
    coefficients = np.zeros_like(probabilities); report = np.empty((5, 8))
    entropy = np.empty((5, 8)); compatible = np.zeros((5, 8), int)
    for ai, a in enumerate(ALPHAS):
        for e in range(8):
            values = [math.fsum((a if e == old else 0., (1-a)*float(independent[s, e]))) for s, (_, _, old) in enumerate(ids)]
            total = math.fsum(values); report[ai, e] = total/count
            n = sum(v > 0 for v in values); compatible[ai, e] = n
            for s, v in enumerate(values):
                probabilities[ai, s, e] = v
                source_post[ai, s, e] = v/total if total else 0.
                coefficients[ai, s, e] = 1/(n*v) if v else 0.
            entropy[ai, e] = -math.fsum(q*math.log(q) for q in source_post[ai, :, e] if q > 0)
    possible = report > 0; recent_index = int(np.argmax(ids[:, 0])); recent_possible = probabilities[:, recent_index] > 0
    exact = np.zeros((5, 8, len(w))); uniform = np.zeros_like(exact); recent = np.zeros_like(exact)
    for s, (_, _, old) in enumerate(ids):
        # Rebuild each complete source's numerator independently, then accumulate.
        joint = np.stack([(1-a)*likelihoods[s] for a in ALPHAS])
        joint[:, old, :] += np.asarray(ALPHAS)[:, None]
        joint *= w
        exact += joint/count
        uniform += joint*coefficients[:, s, :, None]
        if s == recent_index:
            np.divide(joint, probabilities[:, s, :, None], out=recent, where=recent_possible[..., None])
    np.divide(exact, report[..., None], out=exact, where=possible[..., None])
    for values, mask in ((exact, possible), (uniform, possible), (recent, recent_possible)):
        close(values.sum(-1), mask.astype(float), 'posterior normalization')
    out = dict(source_rows=ids.copy(), independent_source_probability=independent,
               source_report_probability=probabilities, source_possible=probabilities > 0,
               report_probability=report, possible=possible, source_posterior=source_post,
               uniform_coefficients=coefficients, compatible_source_count=compatible,
               most_recent_possible=recent_possible, source_entropy=entropy)
    summary = dict(impossible_reports=(~possible).sum(-1),
                   expected_source_entropy=np.array([math.fsum(float(p)*float(h) for p, h in zip(ps, hs)) for ps, hs in zip(report, entropy)]),
                   expected_incompatible_source_fraction=np.array([math.fsum(float(p)*(1-int(n)/count) for p, n in zip(ps, ns)) for ps, ns in zip(report, compatible)]))
    for arm, rival, valid in (('uniform', uniform, possible), ('recent', recent, possible & recent_possible)):
        delta = (exact-rival).reshape(40, len(w))
        grouped = np.stack([np.bincount(st['mapping'], weights=d, minlength=len(st['members'])) for d in delta])
        forecasts = future_mass(delta, st) @ law.reshape(16, 32)
        metrics = dict(group_tv=(.5*abs(grouped).sum(-1)).reshape(5, 8),
                       max_future_difference=abs(forecasts).max(axis=(1, 2)).reshape(5, 8),
                       mean_future_tv=(.5*abs(forecasts.reshape(40, -1, 4, 8)).sum(-1).mean(axis=(1, 2))).reshape(5, 8))
        if np.any(metrics['max_future_difference'][valid] > metrics['group_tv'][valid]+TOL): raise ValueError('contraction')
        for key, v in metrics.items():
            out[arm+'_'+key] = np.where(valid, v, np.nan)
            summary[arm+'_supported_error_mass_'+key] = np.array([math.fsum(float(p)*float(x) for p, x, ok in zip(ps, xs, vs) if ok) for ps, xs, vs in zip(report, v, valid)])
            summary[arm+'_max_supported_'+key] = np.array([max((float(x) for x, ok in zip(xs, vs) if ok), default=0.) for xs, vs in zip(v, valid)])
        for label, mask in (('supported_mass', valid), ('support_failure_mass', possible & ~valid)):
            summary[arm+'_'+label] = np.array([math.fsum(float(p) for p, ok in zip(ps, ms) if ok) for ps, ms in zip(report, mask)])
    if set(out) != set(saved): raise ValueError('raw fields')
    for key, value in out.items():
        actual = np.asarray(saved[key])
        if value.shape != actual.shape: raise ValueError('shape '+key)
        if key == 'uniform_coefficients':
            reciprocal_identity(actual, probabilities, compatible)
        elif value.dtype.kind in 'biu':
            if not np.array_equal(value, actual): raise ValueError(key)
        else:
            if not np.array_equal(np.isnan(value), np.isnan(actual)): raise ValueError('undefined mask '+key)
            mask = ~np.isnan(value)
            close(value[mask], actual[mask], key)
    return summary


def controls():
    weights = (.2, .8); posteriors = (.9, .1)
    return {'live:source_likelihood_weighting_changes_answer': abs(sum(w*p for w, p in zip(weights, posteriors)) - sum(posteriors)/2) > .1,
            'placebo:equal_source_likelihood': sum(.5*p for p in posteriors) == sum(posteriors)/2,
            'positive:impossible_source_gets_zero_weight': 0./(.2+0.) == 0.}


def run(root, plan, pulse):
    cfg = plan['design']; checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('controls')
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('input binding')
    original = root/'inputs/original'; parent = root/'inputs/parent'
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan')
    design = read(original/'PLAN.json')['design']
    if design['alphas'] != list(ALPHAS) or not read(parent/'PARENT_REVIEW.json')['numerical_acceptance']: raise ValueError('design/parent')
    specs = read(parent/'SCHEDULES.json'); structures = {k:structure(v) for k, v in specs.items()}
    recorded = json.loads(gzip.decompress((original/'raw/source_identity_summary_points.json.gz').read_bytes()))
    keyed = {tuple(r[k] for k in KEY):r for r in recorded}
    if len(keyed) != len(recorded): raise ValueError('duplicate summary')
    timing = [json.loads(s) for s in (original/'TIMING.jsonl').read_text().splitlines()]
    timed = {(r['lineage'], r['evidence'], r['length'], r['checkpoint']):r for r in timing}
    if len(timed) != len(timing): raise ValueError('duplicate timing')
    all_rows = []; strata = defaultdict(list); used = set(); used_times = set(); paired = {}; unavailable = []; sources = 0
    for lineage in design['lineages']:
        law = np.asarray(read(parent/'aware/evaluator'/f'{lineage}-law.json'))
        for evidence in ('aware', 'omitted'):
            base = parent/evidence
            if not np.array_equal(law, read(base/'evaluator'/f'{lineage}-law.json')): raise ValueError('law pairing')
            maps = read(base/'evaluator'/f'{lineage}-joint-map.json')
            with np.load(base/'raw'/f'{lineage}-joint_points.npz', allow_pickle=False) as z: arrays = {n:z[n] for n in z.files if n.endswith('-unknown-time-type')}
            for key, spec in specs.items():
                cp = spec['checkpoint']; length = spec['length']; st = structures[key]; prefix = f'{lineage}-{evidence}-{key}'
                path = parent/'bindings'/(prefix+'-bindings.json')
                if not path.exists():
                    if (length, cp) not in ((128, 8), (128, 17), (128, 20)): raise ValueError('missing checkpoint')
                    unavailable.append(dict(lineage=lineage, evidence=evidence, length=length, checkpoint=cp)); continue
                binding = read(path); chosen = binding['rows']; ids = [tuple(r[k] for k in IDENTITY) for r in chosen]
                expected = set(product(design['draws'], range(16), ('purpose', 'skill'), (False, True), (False, True)))
                if len(ids) != len(expected) or set(ids) != expected: raise ValueError('paired roster')
                byrow = defaultdict(list)
                for i, t, ctx, e, sid in binding['sources']: byrow[i].append((t, ctx, e, sid))
                if set(byrow) != set(range(len(chosen))): raise ValueError('source row coverage')
                batch = {}; batch_keys = set(); total_sources = 0
                for i, (r, identity) in enumerate(zip(chosen, ids)):
                    if i % design['batch_rows'] == 0:
                        if batch_keys: raise ValueError('extra raw field')
                        filename = f'{prefix}-{i//design["batch_rows"]:03d}_points.npz'; used.add(filename)
                        with np.load(original/'raw'/filename, allow_pickle=False) as z: batch = {n:z[n] for n in z.files}
                        batch_keys = set(batch); pulse(phase='independent-source-identity-review', lineage=lineage, evidence=evidence, checkpoint=cp, row=i)
                    name, j = r['joint_array'], r['joint_row']; b = maps[name]
                    if b['hypotheses'] != spec['hypotheses'] or b['rows'][j] != [r['stream'], cp]: raise ValueError('posterior binding')
                    selected = byrow[i]
                    if len(selected) != r['report_sources'] or len({s[3] for s in selected}) != len(selected): raise ValueError('source roster')
                    pair = (lineage, key, *identity); witness = (r['stream'], selected)
                    if evidence == 'aware': paired[pair] = witness
                    elif paired[pair] != witness: raise ValueError('source pairing')
                    w = arrays[name][j]; current = st['signatures'][st['mapping'], 0]
                    close((np.bincount(current, weights=w, minlength=16) @ law.reshape(16, 32)).reshape(4, 8), r['forecast'], 'parent forecast')
                    tag = f'{i:03d}__'; names = {n for n in batch_keys if n.startswith(tag)}; saved = {n[len(tag):]:batch[n] for n in names}; batch_keys -= names
                    metrics = reconstruct(w, st, law, np.asarray([s[:3] for s in selected]), saved)
                    for ai, alpha in enumerate(ALPHAS):
                        result = dict(lineage=lineage, evidence=evidence, length=length, checkpoint=cp, alpha=alpha, **{k:r[k] for k in IDENTITY+('stream', 'report_sources')}, **{k:float(v[ai]) for k, v in metrics.items()})
                        old = keyed.pop(tuple(result[k] for k in KEY))
                        if set(old) != set(result): raise ValueError('summary fields')
                        for k, v in result.items():
                            if k in metrics: close(v, old[k], 'summary '+k)
                            elif v != old[k]: raise ValueError('summary identity')
                        all_rows.append(result); strata[(lineage, evidence, length, cp, r['draw'], alpha)].append(result)
                    total_sources += len(selected)
                if batch_keys: raise ValueError('extra raw field')
                tk = (lineage, evidence, length, cp); t = timed[tk]; used_times.add(tk)
                if t['posterior_rows'] != len(chosen) or t['sources'] != total_sources or not math.isfinite(t['cpu_seconds']) or t['cpu_seconds'] < 0: raise ValueError('timing coverage')
                sources += total_sources
    if keyed or used_times != set(timed) or used != {p.name for p in (original/'raw').glob('*.npz')}: raise ValueError('complete coverage')
    summary = read(original/'SUMMARY.json')
    for k, v in dict(rows=len(all_rows), posterior_rows=len(all_rows)//5, sources=sources, report_queries=len(all_rows)*8, unavailable_checkpoints=unavailable).items():
        if summary[k] != v: raise ValueError('summary '+k)
    grouped = []
    for key, rows in sorted(strata.items()):
        if len(rows) != 128 or len({tuple(r[k] for k in IDENTITY) for r in rows}) != 128: raise ValueError('stratum roster')
        grouped.append(dict(zip(('lineage', 'evidence', 'length', 'checkpoint', 'draw', 'alpha'), key), rows=128, **{k:math.fsum(r[k] for r in rows)/128 for k in metrics}))
    (root/'reconstructed').mkdir(exist_ok=True)
    (root/'reconstructed/summary_points.json.gz').write_bytes(gzip.compress(canonical(all_rows), mtime=0))
    write(root/'PAIRED_STRATA.json', grouped)
    write(root/'TIMING_REVIEW.json', dict(batches=len(timing), posterior_rows=len(all_rows)//5, sources=sources, cpu_seconds=math.fsum(t['cpu_seconds'] for t in timing), scope='producer source marginalization,all future differences,raw serialization;parent loading excluded;amortization not latency'))
    write(root/'EVIDENCE_ROLES.json', dict(reader='no new reader inputs', evaluator='independent scalar source Bayes factors,hypothesis updates and every future coordinate;undefined masks and support failures distinct;uniform source prior and source roster supplied'))
    return dict(passed=True, controls=checks, rows=len(all_rows), posterior_rows=len(all_rows)//5, sources=sources, report_queries=len(all_rows)*8, strata=len(grouped), raw_batches=len(used), numerical_acceptance=False, scope='all source normalizers,source weights,entropy,support masks,posteriors,future discrepancies and summaries reconstructed;separate original-row regroup and event adjudication required')
