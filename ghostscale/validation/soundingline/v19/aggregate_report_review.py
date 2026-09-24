"""Independent fixed-source-prior aggregate state reconstruction.

No producer acquisition, update, summary, grouping or transition kernel is used.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, file_digest, canonical
from .sufficient_review import structure, close, TOL

ALPHAS = (0., .25, .5, .75, 1.)
IDENTITY = ('draw', 'initial_maker', 'kind', 'switched', 'duplicates')
KEY = ('lineage', 'evidence', 'length', 'checkpoint', 'alpha') + IDENTITY


def project(weights, paths):
    """Index each independently decoded state change; no sparse operator."""
    _, times = paths.shape
    h, t = np.nonzero(paths[:, 1:] != paths[:, :-1]); t = t+1
    plus = t*16+paths[h, t]; minus = t*16+paths[h, t-1]
    rows = []
    for w in weights:
        mass = np.bincount(paths[:, 0], weights=w, minlength=times*16)
        mass += np.bincount(plus, weights=w[h], minlength=times*16)
        mass -= np.bincount(minus, weights=w[h], minlength=times*16)
        rows.append(mass.reshape(times, 16).cumsum(axis=0))
    return np.asarray(rows)


def calculate(w, st, law, ids):
    w = np.asarray(w, float); law = np.asarray(law, float); ids = np.asarray(ids)
    if w.shape != (len(st['mapping']),) or not np.isfinite(w).all() or (w < 0).any(): raise ValueError('weights')
    close(w.sum(), 1., 'weight mass')
    if law.shape != (16, 4, 8) or not np.isfinite(law).all() or (law < 0).any(): raise ValueError('law')
    close(law.sum(-1), np.ones((16, 4)), 'law mass')
    if ids.ndim != 2 or ids.shape[1] != 3 or not len(ids) or ids.dtype.kind not in 'iu': raise ValueError('sources')
    if len(set(ids[:, 0])) != len(ids) or (ids[:, 0] < 1).any() or (ids[:, 0] > len(st['past'])).any() or (ids[:, 1] < 0).any() or (ids[:, 1] > 3).any() or (ids[:, 2] < 0).any() or (ids[:, 2] > 7).any(): raise ValueError('source bounds')
    G = len(st['members']); N = len(ids)
    q = np.array([math.fsum(float(w[h]) for h in members) for members in st['members']])
    joint = np.zeros((G, 8)); factors = np.zeros((8, len(w))); counts = np.zeros(8, dtype=np.int32)
    for t, context, old in ids:
        past = st['past'][t-1]
        if math.fsum(float(w[h])*float(law[k, context, old]) for h, k in enumerate(past)) <= 0: raise ValueError('old source unsupported')
        # First retain each joint group/past-state mass, then contract with law.
        masses = np.bincount(st['mapping']*16+past, weights=w, minlength=G*16).reshape(G, 16)
        joint += (masses @ law[:, context])/N
        factors += law[past, context].T/N
        counts[old] += 1
    a = np.asarray(ALPHAS)[:, None, None]
    numer = a*(counts/N)[None, :, None]*q[None, None, :] + (1-a)*joint.T[None, :, :]
    # Separate full-hypothesis factors precede weighting and normalization.
    full = w[None, None, :]*(a*(counts/N)[None, :, None]+(1-a)*factors[None, :, :])
    prob = numer.sum(-1); ref = full.sum(-1); possible = prob > 0
    if not np.array_equal(possible, ref > 0): raise ValueError('support')
    close(prob.sum(-1), np.ones(5), 'report mass')
    post = np.divide(numer, prob[..., None], out=np.zeros_like(numer), where=possible[..., None])
    full_post = np.divide(full, ref[..., None], out=np.zeros_like(full), where=possible[..., None])
    ref_group = np.array([np.bincount(st['mapping'], weights=v, minlength=G) for v in full_post.reshape(40, len(w))]).reshape(5, 8, G)
    tv = .5*abs(post-ref_group).sum(-1)
    compact_future = (project(post.reshape(40, G), st['signatures']) @ law.reshape(16, 32)).reshape(5, 8, -1, 4, 8)
    full_future = (project(full_post.reshape(40, len(w)), st['signatures'][st['mapping']]) @ law.reshape(16, 32)).reshape(5, 8, -1, 4, 8)
    delta = compact_future-full_future
    maximum = abs(delta).max(axis=(-3, -2, -1))
    squared = (delta*delta).sum(-1).mean(axis=(-2, -1))
    if max(float(maximum.max()), float(tv.max()), float(abs(prob-ref).max())) > TOL: raise ValueError('independent sufficiency')
    if '_aggregate_review_costs' not in st:
        costs = []
        for past in st['past']:
            sets = [set() for _ in st['members']]
            for g, state in zip(st['mapping'], past): sets[g].add(int(state))
            costs.append((sum(len(s) for s in sets if len(s) > 1), sum(map(len, sets))))
        st['_aggregate_review_costs'] = costs
    mixed = sum(st['_aggregate_review_costs'][t-1][0] for t, _, _ in ids)
    cells = sum(st['_aggregate_review_costs'][t-1][1] for t, _, _ in ids)
    raw = dict(group_mass=q, joint_report_mass=joint, copy_counts=counts, source_rows=ids.astype(np.int32),
        report_probability=prob, reference_probability=ref, possible=possible,
        max_future_probability_error=np.where(possible, maximum, np.nan),
        future_squared_error=np.where(possible, squared, np.nan), updated_group_total_variation=np.where(possible, tv, np.nan),
        aggregate_float64_count=np.array(9*G), aggregate_int32_count=np.array(8),
        source_table_float64_count=np.array(G+mixed), source_table_int32_count=np.array(8+3*N+2*cells),
        full_hypothesis_float64_count=np.array(len(w)), shared_law_float64_count=np.array(512),
        shared_group_schedule_int32_count=np.array(st['signatures'].size),
        full_schedule_int32_count=np.array(len(w)*(len(st['past'])+st['signatures'].shape[1])))
    summary = dict(possible_reports=possible.sum(-1).astype(float), max_report_probability_error=abs(prob-ref).max(-1))
    for key in ('max_future_probability_error', 'future_squared_error', 'updated_group_total_variation'):
        summary['expected_'+key] = np.array([math.fsum(float(prob[i,e])*float(raw[key][i,e]) for e in range(8) if possible[i,e]) for i in range(5)])
        summary['maximum_'+key] = np.nanmax(raw[key], axis=-1)
    for key in raw:
        if key.endswith('_count'): summary[key] = np.full(5, raw[key], dtype=float)
    summary['aggregate_state_bytes'] = 8*summary['aggregate_float64_count']+4*summary['aggregate_int32_count']
    summary['source_table_state_bytes'] = 8*summary['source_table_float64_count']+4*summary['source_table_int32_count']
    summary['full_weight_state_bytes'] = 8*summary['full_hypothesis_float64_count']
    return raw, summary


def reconstruct(w, st, law, ids, saved):
    raw, summary = calculate(w, st, law, ids)
    if set(raw) != set(saved): raise ValueError('raw fields')
    for name, value in raw.items():
        actual = np.asarray(saved[name])
        if value.shape != actual.shape: raise ValueError('shape '+name)
        if value.dtype.kind in 'biu':
            if not np.array_equal(value, actual): raise ValueError(name)
        else:
            if not np.array_equal(np.isnan(value), np.isnan(actual)): raise ValueError('undefined '+name)
            mask = ~np.isnan(value); close(value[mask], actual[mask], name)
    return summary


def controls():
    spec = dict(hypotheses=[['none', 0, i] for i in range(16)], length=2, checkpoint=2, signatures=[[i] for i in range(16)], membership=list(range(16)))
    st = structure(spec); w = np.full(16, 1/16); law = np.full((16, 4, 8), 1/8); ids = [[1, 0, 0], [2, 1, 0]]
    flat, _ = calculate(w, st, law, ids)
    law[:8, 0] = np.eye(8)[0]; law[8:, 0] = np.eye(8)[1]
    live, _ = calculate(w, st, law, ids)
    return {'live:report_informative': bool(abs(live['joint_report_mass'][:, 0]-live['group_mass']*live['report_probability'][0, 0]).max() > .001),
        'placebo:constant_law': bool(abs(flat['joint_report_mass']-flat['group_mass'][:, None]/8).max() < TOL),
        'positive:independent_sufficiency': bool(np.nanmax(live['max_future_probability_error']) < TOL),
        'placebo:certain_copy': bool(np.nanmax(live['updated_group_total_variation'][-1]) < TOL)}


def run(root, plan, pulse):
    cfg = plan['design']; checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('controls')
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('input binding')
    original = root/'inputs/original'; parent = root/'inputs/parent'
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan')
    design = read(original/'PLAN.json')['design']
    if (design['alphas'] != list(ALPHAS) or design['report_state'] != 'fixed-uniform-source-joint-group-endpoint') or not read(parent/'PARENT_REVIEW.json')['numerical_acceptance']: raise ValueError('design/parent')
    specs = read(parent/'SCHEDULES.json'); structures = {k:structure(v) for k, v in specs.items()}
    recorded = json.loads(gzip.decompress((original/'raw/aggregate_report_summary_points.json.gz').read_bytes()))
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
                        batch_keys = set(batch); pulse(phase='independent-aggregate-report-review', lineage=lineage, evidence=evidence, checkpoint=cp, row=i)
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
    write(root/'TIMING_REVIEW.json', dict(batches=len(timing), posterior_rows=len(all_rows)//5, sources=sources, cpu_seconds=math.fsum(t['cpu_seconds'] for t in timing), scope='producer aggregate construction,all future differences,raw serialization;parent loading excluded;amortization not latency'))
    write(root/'EVIDENCE_ROLES.json', dict(reader='no new reader inputs', evaluator='independent source Bayes factors,joint group/report masses,copy counts,bit schedules,every future coordinate and explicit storage sets;exact support and undefined errors distinct;fixed uniform source prior only'))
    return dict(passed=True, controls=checks, rows=len(all_rows), posterior_rows=len(all_rows)//5, sources=sources, report_queries=len(all_rows)*8, strata=len(grouped), raw_batches=len(used), numerical_acceptance=False, scope='all independent joint group/report masses,copy histograms,report probabilities,support masks,posteriors,every future coordinate,total variation,storage and summaries reconstructed;separate original-row regroup and event adjudication required')
