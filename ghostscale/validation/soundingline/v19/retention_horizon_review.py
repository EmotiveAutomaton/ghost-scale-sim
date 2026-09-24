"""Independent reconstruction of time-localized source-retention errors."""
from collections import defaultdict
from itertools import product
from fractions import Fraction
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, file_digest, canonical
from .sufficient_review import structure, close, TOL

ALPHAS = (0., .25, .5, .75, 1.)
WINDOWS = ('full', 'recent_half', 'recent_quarter', 'spaced_half', 'spaced_quarter')
SPANS = ('current', 'near', 'far')
IDENTITY = ('draw', 'initial_maker', 'kind', 'switched', 'duplicates')
KEY = ('lineage', 'evidence', 'length', 'checkpoint', 'alpha') + IDENTITY


def project(weights, paths):
    """Accumulate group weights at independently reconstructed bit-state changes."""
    groups, times = paths.shape
    g, t = np.nonzero(paths[:, 1:] != paths[:, :-1]); t = t + 1
    plus = t*16 + paths[g, t]; minus = t*16 + paths[g, t-1]
    out = []
    for row in weights:
        mass = np.bincount(paths[:, 0], weights=row, minlength=times*16)
        mass += np.bincount(plus, weights=row[g], minlength=times*16)
        mass -= np.bincount(minus, weights=row[g], minlength=times*16)
        out.append(np.cumsum(mass.reshape(times, 16), axis=0))
    return np.asarray(out)


def select(ids, cp):
    """Select exact rational interval midpoints, independently of vector ranks.

    Ordering uses timestamps alone. The rank is the floor of the midpoint of
    each of k equal intervals on [0,n), evaluated with exact rational numbers.
    """
    times = [int(row[0]) for row in ids]
    recent = [[t > cp//2 for t in times], [t > 3*cp//4 for t in times]]
    masks = [[True]*len(ids), *recent]
    ordering = sorted(range(len(ids)), key=times.__getitem__)
    for mask in recent:
        k = sum(mask)
        ranks = [int(Fraction(len(ids), k)*(j+Fraction(1, 2))) for j in range(k)]
        selected = {ordering[j] for j in ranks}
        masks.append([i in selected for i in range(len(ids))])
    return np.asarray(masks)


def time_groups(times):
    if times < 1: raise ValueError('future times')
    # Classify each future offset independently of the producer's slices.
    groups = [[], [], []]
    for offset in range(times):
        group = 0 if offset == 0 else 1 if 2*offset <= times else 2
        groups[group].append(offset)
    return groups


def nullable_mean(values):
    values = list(values)
    missing = [v is None for v in values]
    if any(missing):
        if not all(missing): raise ValueError('mixed unavailable span')
        return None
    return math.fsum(values)/len(values)


def calculate(w, st, law, ids):
    w = np.asarray(w, float); law = np.asarray(law, float); ids = np.asarray(ids)
    if w.shape != (len(st['mapping']),) or not np.isfinite(w).all() or (w < 0).any(): raise ValueError('weights')
    close(w.sum(), 1., 'weight mass')
    if law.shape != (16, 4, 8) or not np.isfinite(law).all() or (law < 0).any(): raise ValueError('law')
    close(law.sum(-1), np.ones((16, 4)), 'law mass')
    if ids.ndim != 2 or ids.shape[1] != 3 or not len(ids) or ids.dtype.kind not in 'iu': raise ValueError('sources')
    cp = len(st['past']); count = len(ids); G = len(st['members'])
    if len(set(ids[:, 0])) != count or (ids[:, 0] < 1).any() or (ids[:, 0] > cp).any() or (ids[:, 1] < 0).any() or (ids[:, 1] > 3).any() or (ids[:, 2] < 0).any() or (ids[:, 2] > 7).any(): raise ValueError('source bounds')
    keep = select(ids, cp); windows = len(WINDOWS)
    q = np.bincount(st['mapping'], weights=w, minlength=G)
    factors = np.zeros((windows, 8, len(w))); histogram = np.zeros(8); contexts = np.zeros((windows, 4), dtype=np.int64)
    prior = np.array([[math.fsum(float(law[k, c, e]) for k in range(16))/16 for e in range(8)] for c in range(4)])
    for s, (t, c, old) in enumerate(ids):
        states = st['past'][t-1]; mass = np.bincount(states, weights=w, minlength=16)
        if math.fsum(float(mass[k])*float(law[k, c, old]) for k in range(16)) <= 0: raise ValueError('old source unsupported')
        histogram[old] += 1/count
        for wi in range(windows):
            if keep[wi, s]: factors[wi] += law[states, c].T/count
            else: factors[wi] += prior[c, :, None]/count; contexts[wi, c] += 1
    # Direct joint group/report numerators; no posterior-direction or sparse operator.
    joint = np.array([np.bincount(st['mapping'], weights=w*f, minlength=G) for f in factors.reshape(windows*8, len(w))]).reshape(windows, 8, G)
    a = np.asarray(ALPHAS)[None, :, None, None]
    numer = a*histogram[None, None, :, None]*q[None, None, None, :] + (1-a)*joint[:, None]
    report = numer.sum(-1); possible = report > 0
    close(report.sum(-1), np.ones((windows, 5)), 'report mass')
    post = np.divide(numer, report[..., None], out=np.zeros_like(numer), where=possible[..., None])
    close(post.sum(-1), possible.astype(float), 'posterior mass')
    times = st['signatures'].shape[1]
    independent = (project(joint.reshape(windows*8, G), st['signatures']) @ law.reshape(16, 32)).reshape(windows, 8, times, 4, 8)
    baseline = (project(q[None], st['signatures']) @ law.reshape(16, 32)).reshape(times, 4, 8)
    aa = np.asarray(ALPHAS)[None, :, None, None, None, None]
    future_numer = aa*histogram[None, None, :, None, None, None]*baseline[None, None, None] + (1-aa)*independent[:, None]
    future = np.divide(future_numer, report[..., None, None, None], out=np.zeros_like(future_numer), where=possible[..., None, None, None])
    truth = future[0:1]; delta = future-truth; both = possible & possible[0:1]
    # Explicit expected one-hot categorical losses, independently of squared distance.
    square = (future*future).sum(-1); true_square = (truth*truth).sum(-1)
    excess = np.zeros_like(square)
    for y in range(8):
        approximate_loss = square-2*future[..., y]+1
        optimal_loss = true_square-2*truth[..., y]+1
        excess += truth[..., y]*(approximate_loss-optimal_loss)
    regret = excess.mean(axis=(-2, -1)); direct = (delta*delta).sum(-1).mean(axis=(-2, -1))
    close(regret[both], direct[both], 'one-hot proper-loss regret')
    groups = time_groups(times)
    span_lengths = np.array([len(g) for g in groups], dtype=np.int32)
    span_regret = np.full((3, *regret.shape), np.nan)
    span_maximum = np.full_like(span_regret, np.nan)
    for si, offsets in enumerate(groups):
        if offsets:
            loss = excess[..., offsets, :].mean(axis=(-2, -1))
            distance = (delta[..., offsets, :, :]**2).sum(-1).mean(axis=(-2, -1))
            close(loss[both], distance[both], 'span one-hot proper-loss regret')
            span_regret[si] = np.where(both, loss, np.nan)
            span_maximum[si] = np.where(both, abs(delta[..., offsets, :, :]).max(axis=(-3, -2, -1)), np.nan)
    recombined = sum(np.nan_to_num(span_regret[i])*len(g) for i, g in enumerate(groups))/times
    close(recombined[both], regret[both], 'span weighted recombination')
    maximum = abs(delta).max(axis=(-3, -2, -1)); tv = .5*abs(post-post[0:1]).sum(-1)
    if np.any(maximum[both] > tv[both]+TOL): raise ValueError('total variation bound')
    if '_review_storage' not in st:
        counts = []
        for states in st['past']:
            members = [set() for _ in range(G)]
            for g, state in zip(st['mapping'], states): members[g].add(int(state))
            counts.append((sum(map(len, members)), sum(len(s) for s in members if len(s) > 1)))
        st['_review_storage'] = counts
    floats = []; integers = []
    for mask in keep:
        selected = [st['_review_storage'][t-1] for (t, _, _), kept in zip(ids, mask) if kept]
        floats.append(G+sum(m for _, m in selected))
        integers.append(12+3*int(mask.sum())+2*sum(n for n, _ in selected))
    out = dict(forecast_span_lengths=span_lengths, span_squared_regret=span_regret, span_max_future_probability_error=span_maximum, source_rows=ids.copy(), retained=keep, retained_sources=keep.sum(-1),
        forgotten_mass=np.array([sum(not k for k in mask)/count for mask in keep]), copy_histogram=histogram,
        forgotten_context_counts=contexts, report_probability=report, possible=possible, comparable=both,
        future_squared_regret=np.where(both, regret, np.nan), max_future_probability_error=np.where(both, maximum, np.nan),
        updated_group_total_variation=np.where(both, tv, np.nan), compact_float64_count=np.array(floats, dtype=np.int64),
        compact_int32_count=np.array(integers, dtype=np.int64), full_hypothesis_float64_count=np.full(windows, len(w), dtype=np.int64),
        shared_prior_float64_count=np.array(32, dtype=np.int64))
    summary = {}
    for wi, label in enumerate(WINDOWS):
        for suffix, supported in [('supported_mass', True), ('unsupported_mass', False)]:
            summary[label+'_'+suffix] = np.array([math.fsum(float(report[0, ai, e]) for e in range(8) if possible[wi, ai, e] == supported) for ai in range(5)])
        for suffix, value in [('expected_squared_regret', regret), ('expected_max_probability_error', maximum), ('expected_group_total_variation', tv)]:
            summary[label+'_'+suffix] = np.array([math.fsum(float(report[0, ai, e])*float(value[wi, ai, e]) for e in range(8) if both[wi, ai, e]) for ai in range(5)])
        for field in ('retained_sources', 'forgotten_mass', 'compact_float64_count', 'compact_int32_count', 'full_hypothesis_float64_count'):
            summary[label+'_'+field] = np.full(5, out[field][wi])
    for si, span in enumerate(SPANS):
        summary[span+'_steps'] = np.full(5, span_lengths[si])
        for wi, label in enumerate(WINDOWS):
            for suffix, values in [('expected_squared_regret', span_regret), ('expected_max_probability_error', span_maximum)]:
                summary[span+'_'+label+'_'+suffix] = (np.array([math.fsum(float(report[0, ai, e])*float(values[si, wi, ai, e]) for e in range(8) if both[wi, ai, e]) for ai in range(5)]) if span_lengths[si] else np.full(5, np.nan))
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
    spec = dict(hypotheses=[['none', 0, i] for i in range(16)], length=4, checkpoint=4, signatures=[[i] for i in range(16)], membership=list(range(16)))
    st = structure(spec); w = np.full(16, 1/16); law = np.full((16, 4, 8), 1/8)
    ids = [[1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 1, 0]]
    _, flat = calculate(w, st, law, ids)
    law[:8, 0] = np.eye(8)[0]; law[8:, 0] = np.eye(8)[1]
    raw, live = calculate(w, st, law, ids)
    return {'live:older_informative_time': bool(live['spaced_half_expected_squared_regret'][0] < live['recent_half_expected_squared_regret'][0]-.001),
        'placebo:constant_law': bool(abs(flat['spaced_half_expected_squared_regret']).max() < TOL),
        'positive:full_retention_identity': bool(abs(live['full_expected_squared_regret']).max() < TOL),
        'placebo:certain_copy_identity': bool(all(abs(live[k+'_expected_squared_regret'][-1]) < TOL for k in WINDOWS)),
        'positive:empty_spans_unavailable': bool(np.isnan(raw['span_squared_regret'][1:]).all() and np.isnan(raw['span_max_future_probability_error'][1:]).all()),
        'positive:current_span_complete': bool(np.array_equal(raw['forecast_span_lengths'], [1, 0, 0])),
        'positive:matched_counts': bool(np.array_equal(raw['retained_sources'][1:3], raw['retained_sources'][3:5]))}


def run(root, plan, pulse):
    cfg = plan['design']; checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('controls')
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('input binding')
    original = root/'inputs/original'; parent = root/'inputs/parent'
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan')
    design = read(original/'PLAN.json')['design']
    if (design['alphas'] != list(ALPHAS) or design['selection_rule'] != 'midpoint-sorted-time-ranks' or design['forecast_spans'] != list(SPANS)) or not read(parent/'PARENT_REVIEW.json')['numerical_acceptance']: raise ValueError('design/parent')
    specs = read(parent/'SCHEDULES.json'); structures = {k:structure(v) for k, v in specs.items()}
    recorded = json.loads(gzip.decompress((original/'raw/retention_horizon_summary_points.json.gz').read_bytes()))
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
                        batch_keys = set(batch); pulse(phase='independent-retention-horizon-review', lineage=lineage, evidence=evidence, checkpoint=cp, row=i)
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
                        result = dict(lineage=lineage, evidence=evidence, length=length, checkpoint=cp, alpha=alpha, **{k:r[k] for k in IDENTITY+('stream', 'report_sources')}, **{k:(None if np.isnan(v[ai]) else float(v[ai])) for k, v in metrics.items()})
                        old = keyed.pop(tuple(result[k] for k in KEY))
                        if set(old) != set(result): raise ValueError('summary fields')
                        for k, v in result.items():
                            if k in metrics:
                                if v is None or old[k] is None:
                                    if v is not None or old[k] is not None: raise ValueError('summary unavailable '+k)
                                else: close(v, old[k], 'summary '+k)
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
        grouped.append(dict(zip(('lineage', 'evidence', 'length', 'checkpoint', 'draw', 'alpha'), key), rows=128, **{k:nullable_mean(r[k] for r in rows) for k in metrics}))
    (root/'reconstructed').mkdir(exist_ok=True)
    (root/'reconstructed/summary_points.json.gz').write_bytes(gzip.compress(canonical(all_rows), mtime=0))
    write(root/'PAIRED_STRATA.json', grouped)
    write(root/'TIMING_REVIEW.json', dict(batches=len(timing), posterior_rows=len(all_rows)//5, sources=sources, cpu_seconds=math.fsum(t['cpu_seconds'] for t in timing), scope='producer source marginalization,all future differences,raw serialization;parent loading excluded;amortization not latency'))
    write(root/'EVIDENCE_ROLES.json', dict(reader='no new reader inputs', evaluator='independent source Bayes factors,joint group updates,bit schedules,every future coordinate and compact storage;undefined masks and support failures distinct;all original source mass retained in fixed time-spaced and recency rosters and original-prior remainder'))
    return dict(passed=True, controls=checks, rows=len(all_rows), posterior_rows=len(all_rows)//5, sources=sources, report_queries=len(all_rows)*8, strata=len(grouped), raw_batches=len(used), numerical_acceptance=False, scope='all three disjoint spans,empty masks,weighted regret recombination,window identities,mass,histograms,report probabilities,support masks,posteriors,future coordinates,one-hot proper losses,total variation,storage and summaries reconstructed;separate original-row regroup and event adjudication required')
