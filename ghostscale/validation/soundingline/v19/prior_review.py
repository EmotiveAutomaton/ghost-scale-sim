"""Independent source-prior Bayes and proper-loss reconstruction."""
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
PRIORS = ('uniform', 'recency', 'early')
IDENTITY = ('draw', 'initial_maker', 'kind', 'switched', 'duplicates')
KEY = ('lineage', 'evidence', 'length', 'checkpoint', 'alpha') + IDENTITY


def reconstruct(w, st, law, ids, saved):
    """Source-wise scalar normalizers and joint Bayes numerators.

    Uses independent bit schedules and indexed future-state accumulation. Proper
    loss is summed over each possible one-hot outcome, separately from squared
    forecast error. No producer evaluation, grouping or sparse operator is used.
    """
    w = np.asarray(w, float); law = np.asarray(law, float); ids = np.asarray(ids)
    if w.shape != (len(st['mapping']),) or not np.isfinite(w).all() or (w < 0).any(): raise ValueError('weights')
    close(w.sum(), 1., 'weight mass')
    if law.shape != (16, 4, 8) or not np.isfinite(law).all() or (law < 0).any(): raise ValueError('law')
    close(law.sum(-1), np.ones((16, 4)), 'law mass')
    if ids.ndim != 2 or ids.shape[1] != 3 or not len(ids) or ids.dtype.kind not in 'iu': raise ValueError('sources')
    if len(set(ids[:, 0])) != len(ids) or (ids[:, 0] < 1).any() or (ids[:, 0] > len(st['past'])).any() or (ids[:, 1] < 0).any() or (ids[:, 1] > 3).any() or (ids[:, 2] < 0).any() or (ids[:, 2] > 7).any(): raise ValueError('source bounds')
    count = len(ids)
    prior = np.asarray([[1.]*count, [float(t) for t in ids[:, 0]], [1./int(t) for t in ids[:, 0]]])
    for row in prior: row /= math.fsum(row)
    independent = np.empty((count, 8)); averaged = np.zeros((3, 8, len(w))); copied = np.zeros((3, 8))
    for s, (t, ctx, old) in enumerate(ids):
        states = st['past'][t-1]; mass = np.bincount(states, weights=w, minlength=16)
        for e in range(8): independent[s, e] = math.fsum(float(mass[k])*float(law[k, ctx, e]) for k in range(16))
        if independent[s, old] <= 0: raise ValueError('old source unsupported')
        likelihood = law[states, ctx].T
        for r in range(3):
            averaged[r] += prior[r, s]*likelihood
            copied[r, old] += prior[r, s]
    source_probability = np.empty((5, count, 8))
    report = np.empty((3, 5, 8)); source_post = np.zeros((3, 5, count, 8)); entropy = np.empty((3, 5, 8))
    posterior = np.zeros((3, 5, 8, len(w)))
    for ai, a in enumerate(ALPHAS):
        for e in range(8):
            values = [math.fsum((a if e == old else 0., (1-a)*float(independent[s, e]))) for s, (_, _, old) in enumerate(ids)]
            source_probability[ai, :, e] = values
            for r in range(3):
                weighted = [float(prior[r, s])*v for s, v in enumerate(values)]
                den = math.fsum(weighted); report[r, ai, e] = den
                if den:
                    source_post[r, ai, :, e] = [v/den for v in weighted]
                    posterior[r, ai, e] = w*(a*copied[r, e]+(1-a)*averaged[r, e])/den
                entropy[r, ai, e] = -math.fsum(q*math.log(q) for q in source_post[r, ai, :, e] if q > 0)
    possible = report > 0
    if not np.array_equal(possible, np.broadcast_to(possible[0], possible.shape)): raise ValueError('positive prior support')
    close(posterior.sum(-1), possible.astype(float), 'posterior mass')
    forecasts = (future_mass(posterior.reshape(120, len(w)), st) @ law.reshape(16, 32)).reshape(3, 5, 8, -1, 4, 8)
    out = dict(source_rows=ids.copy(), source_prior=prior, source_report_probability=source_probability,
               report_probability=report, source_posterior=source_post, possible=possible, source_entropy=entropy)
    summary = {}
    prediction = forecasts[0]
    for ri, name in enumerate(PRIORS[1:], 1):
        truth = forecasts[ri]; delta = prediction-truth
        grouped = np.stack([np.bincount(st['mapping'], weights=d, minlength=len(st['members'])) for d in (posterior[0]-posterior[ri]).reshape(40, len(w))])
        # Explicit one-hot losses; keep endpoint and future-time multiplicities.
        before = np.zeros(truth.shape[:-1]); after = np.zeros_like(before)
        for e in range(8):
            target = np.eye(8)[e]
            before += truth[..., e]*np.sum((prediction-target)**2, axis=-1)
            after += truth[..., e]*np.sum((truth-target)**2, axis=-1)
        metrics = dict(group_tv=(.5*abs(grouped).sum(-1)).reshape(5, 8),
                       max_future_difference=abs(delta).max(axis=(-3, -2, -1)),
                       mean_future_tv=.5*abs(delta).sum(-1).mean(axis=(-2, -1)),
                       squared_forecast_difference=(delta*delta).sum(-1).mean(axis=(-2, -1)),
                       brier_regret=(before-after).mean(axis=(-2, -1)))
        valid = possible[ri]; probability = report[ri]
        close(metrics['squared_forecast_difference'][valid], metrics['brier_regret'][valid], 'proper-loss identity')
        if np.any(metrics['max_future_difference'][valid] > metrics['group_tv'][valid]+TOL): raise ValueError('contraction')
        summary[name+'_impossible_reports'] = (~valid).sum(-1)
        summary[name+'_supported_mass'] = np.asarray([math.fsum(float(p) for p, ok in zip(ps, vs) if ok) for ps, vs in zip(probability, valid)])
        summary[name+'_expected_source_entropy'] = np.asarray([math.fsum(float(p)*float(h) for p, h in zip(ps, hs)) for ps, hs in zip(probability, entropy[ri])])
        for key, value in metrics.items():
            out[name+'_'+key] = np.where(valid, value, np.nan)
            summary[name+'_expected_'+key] = np.asarray([math.fsum(float(p)*float(v) for p, v, ok in zip(ps, vs, mask) if ok) for ps, vs, mask in zip(probability, value, valid)])
            summary[name+'_max_supported_'+key] = np.asarray([max((float(v) for v, ok in zip(vs, mask) if ok), default=0.) for vs, mask in zip(value, valid)])
    if set(out) != set(saved): raise ValueError('raw fields')
    for key, value in out.items():
        actual = np.asarray(saved[key])
        if value.shape != actual.shape: raise ValueError('shape '+key)
        if value.dtype.kind in 'biu':
            if not np.array_equal(value, actual): raise ValueError(key)
        else:
            if not np.array_equal(np.isnan(value), np.isnan(actual)): raise ValueError('undefined mask '+key)
            mask = ~np.isnan(value); close(value[mask], actual[mask], key)
    return summary


def controls():
    return {'live:unequal_prior_changes_posterior': abs((.8*.9+.2*.1)-(.5*.9+.5*.1)) > .1,
            'placebo:equal_likelihood_prior_cancels': (.25*.5+.75*.5) == .5,
            'positive:one_hot_loss': abs((.7*(.3-1)**2+.3*.3**2)-(.7*(.7-1)**2+.3*.7**2)-(.3-.7)**2) < TOL}


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
    recorded = json.loads(gzip.decompress((original/'raw/source_prior_summary_points.json.gz').read_bytes()))
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
                        batch_keys = set(batch); pulse(phase='independent-source-prior-review', lineage=lineage, evidence=evidence, checkpoint=cp, row=i)
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
    for k, v in dict(rows=len(all_rows), posterior_rows=len(all_rows)//5, sources=sources, report_queries=len(all_rows)*8*2, unavailable_checkpoints=unavailable).items():
        if summary[k] != v: raise ValueError('summary '+k)
    grouped = []
    for key, rows in sorted(strata.items()):
        if len(rows) != 128 or len({tuple(r[k] for k in IDENTITY) for r in rows}) != 128: raise ValueError('stratum roster')
        grouped.append(dict(zip(('lineage', 'evidence', 'length', 'checkpoint', 'draw', 'alpha'), key), rows=128, **{k:math.fsum(r[k] for r in rows)/128 for k in metrics}))
    (root/'reconstructed').mkdir(exist_ok=True)
    (root/'reconstructed/summary_points.json.gz').write_bytes(gzip.compress(canonical(all_rows), mtime=0))
    write(root/'PAIRED_STRATA.json', grouped)
    write(root/'TIMING_REVIEW.json', dict(batches=len(timing), posterior_rows=len(all_rows)//5, sources=sources, cpu_seconds=math.fsum(t['cpu_seconds'] for t in timing), scope='producer source marginalization,all future differences,raw serialization;parent loading excluded;amortization not latency'))
    write(root/'EVIDENCE_ROLES.json', dict(reader='no new reader inputs', evaluator='independent scalar source Bayes factors,hypothesis updates and every future coordinate;undefined masks and support failures distinct;three source priors and source roster supplied'))
    return dict(passed=True, controls=checks, rows=len(all_rows), posterior_rows=len(all_rows)//5, sources=sources, report_queries=len(all_rows)*8*2, strata=len(grouped), raw_batches=len(used), numerical_acceptance=False, scope='all source priors,normalizers,source weights,entropy,support masks,posteriors,future discrepancies,one-hot proper losses and summaries reconstructed;separate original-row regroup and event adjudication required')
