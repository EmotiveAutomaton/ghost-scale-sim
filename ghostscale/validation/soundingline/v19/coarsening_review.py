"""Independent fixed-report Bayes and explicit one-hot loss reconstruction."""
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


def calculate(w, st, law, ids):
    """Sum source-wise factors, enumerate report members and one-hot losses.

    No producer content matrix, evaluator, summarizer or sparse transition
    operator is imported. Every future time and context retains its multiplicity.
    """
    w = np.asarray(w, float); law = np.asarray(law, float); ids = np.asarray(ids)
    if w.shape != (len(st['mapping']),) or not np.isfinite(w).all() or (w < 0).any(): raise ValueError('weights')
    close(w.sum(), 1., 'weight mass')
    if law.shape != (16, 4, 8) or not np.isfinite(law).all() or (law < 0).any(): raise ValueError('law')
    close(law.sum(-1), np.ones((16, 4)), 'law mass')
    if ids.ndim != 2 or ids.shape[1] != 3 or not len(ids) or ids.dtype.kind not in 'iu': raise ValueError('sources')
    if len(set(ids[:, 0])) != len(ids) or (ids[:, 0] < 1).any() or (ids[:, 0] > len(st['past'])).any() or (ids[:, 1] < 0).any() or (ids[:, 1] > 3).any() or (ids[:, 2] < 0).any() or (ids[:, 2] > 7).any(): raise ValueError('source bounds')
    count = len(ids); independent = np.empty((count, 8)); averaged = np.zeros((8, len(w)))
    for s, (t, ctx, old) in enumerate(ids):
        states = st['past'][t-1]; mass = np.bincount(states, weights=w, minlength=16)
        for e in range(8): independent[s, e] = math.fsum(float(mass[k])*float(law[k, ctx, e]) for k in range(16))
        if independent[s, old] <= 0: raise ValueError('old source unsupported')
        averaged += law[states, ctx].T / count
    members = [[e] for e in range(8)] + [[e for e in range(8) if (e >> bit) % 2 == value] for bit in range(3) for value in (0, 1)]
    fine_source = np.empty((5, count, 8)); report = np.empty((5, 14))
    source_post = np.zeros((5, count, 14)); posterior = np.zeros((5, 14, len(w))); entropy = np.zeros((5, 14))
    for ai, a in enumerate(ALPHAS):
        for s, (_, _, old) in enumerate(ids):
            for e in range(8): fine_source[ai, s, e] = math.fsum((a if e == old else 0., (1-a)*float(independent[s, e])))
        for r, endpoints in enumerate(members):
            source_likelihood = [math.fsum(float(fine_source[ai, s, e]) for e in endpoints) for s in range(count)]
            den = math.fsum(source_likelihood) / count; report[ai, r] = den
            if den:
                source_post[ai, :, r] = [p / (count*den) for p in source_likelihood]
                copy = sum(old in endpoints for _, _, old in ids) / count
                likelihood = sum((averaged[e] for e in endpoints), start=np.zeros(len(w)))
                posterior[ai, r] = w*(a*copy+(1-a)*likelihood)/den
                entropy[ai, r] = -math.fsum(q*math.log(q) for q in source_post[ai, :, r] if q > 0)
    possible = report > 0
    close(posterior.sum(-1), possible.astype(float), 'posterior mass')
    future = (future_mass(posterior.reshape(70, len(w)), st) @ law.reshape(16, 32)).reshape(5, 14, -1, 4, 8)
    baseline = (future_mass(w[None], st) @ law.reshape(16, 32)).reshape(-1, 4, 8)
    optimal = np.zeros(future.shape[:-1]); before = np.zeros_like(optimal); base_loss = np.zeros(baseline.shape[:-1])
    for e in range(8):
        one = np.eye(8)[e]
        optimal += future[..., e] * np.sum((future-one)**2, axis=-1)
        before += future[..., e] * np.sum((baseline-one)**2, axis=-1)
        base_loss += baseline[..., e] * np.sum((baseline-one)**2, axis=-1)
    optimal = optimal.mean(axis=(-2, -1)); gain = (before.mean(axis=(-2, -1))-optimal)
    direct_gain = ((future-baseline)**2).sum(-1).mean(axis=(-2, -1))
    close(gain[possible], direct_gain[possible], 'one-hot proper-loss gain')
    regret = np.empty((5, 3, 8))
    for bit in range(3):
        for e in range(8):
            coarse = 8+2*bit+((e >> bit) % 2)
            regret[:, bit, e] = ((future[:, e]-future[:, coarse])**2).sum(-1).mean(axis=(-2, -1))
    out = dict(source_rows=ids.copy(), fine_source_probability=fine_source, report_probability=report,
               source_posterior=source_post, possible=possible, source_entropy=np.where(possible, entropy, np.nan),
               future_optimal_squared_loss=np.where(possible, optimal, np.nan),
               no_report_squared_loss=np.full((5, 14), base_loss.mean()),
               future_squared_gain=np.where(possible, direct_gain, np.nan),
               fine_to_coarse_squared_regret=np.where(possible[:, None, :8], regret, np.nan))
    summary = {}
    for label, reports in [('fine', range(8)), ('bit0', range(8, 10)), ('bit1', range(10, 12)), ('bit2', range(12, 14))]:
        summary[label+'_supported_mass'] = np.asarray([math.fsum(float(p[r]) for r in reports) for p in report])
        summary[label+'_impossible_reports'] = np.asarray([sum(not mask[r] for r in reports) for mask in possible])
        for key, values in [('source_entropy', entropy), ('future_optimal_squared_loss', optimal), ('future_squared_gain', gain)]:
            summary[label+'_expected_'+key] = np.asarray([math.fsum(float(report[a, r])*float(values[a, r]) for r in reports if possible[a, r]) for a in range(5)])
        close(summary[label+'_supported_mass'], np.ones(5), 'report mass')
        close(base_loss.mean()-summary[label+'_expected_future_optimal_squared_loss'], summary[label+'_expected_future_squared_gain'], 'expected loss reduction')
    for bit in range(3):
        gap = summary['fine_expected_future_squared_gain']-summary[f'bit{bit}_expected_future_squared_gain']
        explicit = np.asarray([math.fsum(float(report[a, e])*float(regret[a, bit, e]) for e in range(8) if possible[a, e]) for a in range(5)])
        entropy_gap = summary[f'bit{bit}_expected_source_entropy']-summary['fine_expected_source_entropy']
        close(gap, explicit, 'fine/coarse proper-loss identity')
        if np.min(gap) < -TOL or np.min(entropy_gap) < -TOL: raise ValueError('coarsening monotonicity')
        summary[f'bit{bit}_lost_future_gain'] = gap
        summary[f'bit{bit}_fine_to_coarse_regret'] = explicit
        summary[f'bit{bit}_added_source_entropy'] = entropy_gap
    return out, summary


def reconstruct(w, st, law, ids, saved):
    out, summary = calculate(w, st, law, ids)
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
    spec = dict(hypotheses=[['none', 0, i] for i in range(16)], length=3, checkpoint=2, signatures=[[i, i] for i in range(16)], membership=list(range(16)))
    st = structure(spec); w = np.full(16, 1/16); law = np.full((16, 4, 8), 1/8)
    _, flat = calculate(w, st, law, [[1, 0, 0], [2, 1, 1]])
    law[:8] = np.eye(8)[0]; law[8:] = np.eye(8)[2]
    _, live = calculate(w, st, law, [[1, 0, 0], [2, 1, 2]])
    return {'live:discard_informative_bit': bool(live['bit0_lost_future_gain'][0] > .1),
            'placebo:constant_law': bool(abs(flat['fine_expected_future_squared_gain']).max() < TOL),
            'positive:retained_bit_sufficient': bool(abs(live['bit1_lost_future_gain']).max() < TOL),
            'placebo:certain_copy_no_state_gain': bool(abs(live['fine_expected_future_squared_gain'][-1]) < TOL)}


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
    recorded = json.loads(gzip.decompress((original/'raw/report_coarsening_summary_points.json.gz').read_bytes()))
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
                        batch_keys = set(batch); pulse(phase='independent-report-coarsening-review', lineage=lineage, evidence=evidence, checkpoint=cp, row=i)
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
    for k, v in dict(rows=len(all_rows), posterior_rows=len(all_rows)//5, sources=sources, report_queries=len(all_rows)*14, unavailable_checkpoints=unavailable).items():
        if summary[k] != v: raise ValueError('summary '+k)
    grouped = []
    for key, rows in sorted(strata.items()):
        if len(rows) != 128 or len({tuple(r[k] for k in IDENTITY) for r in rows}) != 128: raise ValueError('stratum roster')
        grouped.append(dict(zip(('lineage', 'evidence', 'length', 'checkpoint', 'draw', 'alpha'), key), rows=128, **{k:math.fsum(r[k] for r in rows)/128 for k in metrics}))
    (root/'reconstructed').mkdir(exist_ok=True)
    (root/'reconstructed/summary_points.json.gz').write_bytes(gzip.compress(canonical(all_rows), mtime=0))
    write(root/'PAIRED_STRATA.json', grouped)
    write(root/'TIMING_REVIEW.json', dict(batches=len(timing), posterior_rows=len(all_rows)//5, sources=sources, cpu_seconds=math.fsum(t['cpu_seconds'] for t in timing), scope='producer source marginalization,all future differences,raw serialization;parent loading excluded;amortization not latency'))
    write(root/'EVIDENCE_ROLES.json', dict(reader='no new reader inputs', evaluator='independent scalar source Bayes factors,hypothesis updates and every future coordinate;undefined masks and support failures distinct;equal source prior and complete source roster supplied;fixed endpoint and binary report alphabets'))
    return dict(passed=True, controls=checks, rows=len(all_rows), posterior_rows=len(all_rows)//5, sources=sources, report_queries=len(all_rows)*14, strata=len(grouped), raw_batches=len(used), numerical_acceptance=False, scope='all report probabilities,source weights,entropy,support masks,posteriors,future coordinates,one-hot proper losses,coarsening regrets and summaries reconstructed;separate original-row regroup and event adjudication required')
