"""Fixed checkpoint truncation of complete supplied-law maker hypotheses.

This consumes retained posteriors; it does not perform a recursive approximation.
"""
from collections import defaultdict
import gzip
import json
import math
import time
import numpy as np
from ..v18_3.io import canonical, file_digest, read, write
from . import unknown_change as U, transient_filter as T

COUNTS = (1, 16, 64, 256, 'all')
ARM = 'unknown-time-type'


def truncate(weights, count):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not len(weights) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('invalid weights')
    if abs(math.fsum(weights) - 1) > 1e-10:
        raise ValueError('weights not normalized')
    if count == 'all':
        # Preserve exact parent addition order for the identity control.
        return np.arange(len(weights), dtype=np.int32), weights.copy(), 0.
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= len(weights):
        raise ValueError('invalid retained count')
    indices = np.argsort(-weights, kind='stable')[:count].astype(np.int32)
    kept = weights[indices].copy()
    mass = math.fsum(kept)
    if mass <= 0:
        raise ValueError('empty retained support')
    return indices, kept / mass, max(0., math.fsum(weights) - mass)


def project(hypotheses, step, indices, weights):
    states = U.state_indices(hypotheses, step)
    return np.bincount(states[indices], weights=weights, minlength=16)


def controls():
    weights = np.array([.5, .5, 0.])
    hs = [('none', 0, 0), ('none', 0, 1), ('none', 0, 2)]
    ix, kept, removed = truncate(weights, 1)
    point = project(hs, 1, ix, kept)
    full = project(hs, 1, *truncate(weights, 'all')[:2])
    # A known two-state ambiguity loses coverage under deterministic point choice.
    def credible(p):
        ordered = np.sort(p)[::-1]
        boundary = ordered[np.searchsorted(np.cumsum(ordered), .9)]
        return p >= boundary - 1e-14
    two = project(hs, 1, *truncate(weights, 2)[:2])
    return {'positive:all_identity': bool(np.array_equal(full, np.r_[weights, np.zeros(13)])),
        'placebo:zero_removal_identity': bool(np.array_equal(full, two)),
        'positive:normalized_nonnegative': bool(np.all(kept >= 0) and kept.sum() == 1),
        'positive:stable_tie': bool(ix.tolist() == [0] and removed == .5),
        'live:point_loses_ambiguity_coverage': bool(credible(full)[1] and not credible(point)[1])}


def gz(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def run(root, plan, pulse):
    cfg = plan['design']; checks = controls(); write(root / 'CONTROLS.json', checks)
    if not all(checks.values()) or cfg['retained_counts'] != list(COUNTS):
        raise ValueError('controls or count roster')
    for n, h in cfg['input_files'].items():
        if file_digest(root / 'inputs' / n) != h: raise ValueError('input differs')
    cells = []; total = identities = current_controls = 0
    for lineage in cfg['lineages']:
        rows = []; acc = defaultdict(list); selections = defaultdict(list); selection_weights = defaultdict(list)
        table = np.asarray(read(root / 'inputs/aware/evaluator' / f'{lineage}-law.json'))
        if table.shape != (16, 4, 8) or np.any(table < 0) or not np.allclose(table.sum(-1), 1, atol=1e-12, rtol=0):
            raise ValueError('invalid law')
        pairing = {}
        for evidence in ('aware', 'omitted'):
            parent = root / 'inputs' / evidence
            if not np.array_equal(table, read(parent / 'evaluator' / f'{lineage}-law.json')):
                raise ValueError('law pairing')
            oldrows = [r for r in gz(parent / 'raw' / f'{lineage}-forecasts_points.json.gz') if r['arm'] == ARM]
            mapping = read(parent / 'evaluator' / f'{lineage}-joint-map.json')
            with np.load(parent / 'raw' / f'{lineage}-joint_points.npz', allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files if k.endswith('-' + ARM)}
            expected = {(d, m, length, kind, switched, copied, step)
                for d in cfg['draws'] for m in range(16) for length in cfg['lengths']
                for kind in ('purpose', 'skill') for switched in (False, True)
                for copied in (False, True) for step in T.checkpoints(length)}
            seen = set(); used_joint = defaultdict(set)
            for i, old in enumerate(oldrows):
                pulse(phase='checkpoint-compression', lineage=lineage, evidence=evidence, row=i)
                key = tuple(old[k] for k in ('draw', 'initial_maker', 'length', 'kind', 'switched', 'duplicates', 'step'))
                if key in seen: raise ValueError('duplicate parent row')
                seen.add(key)
                if evidence == 'aware': pairing[key] = (old['stream'], old['actual_maker'])
                elif pairing[key] != (old['stream'], old['actual_maker']): raise ValueError('parent pairing')
                name = old['joint_array']; ji = old['joint_row']; meta = mapping[name]
                if ji in used_joint[name] or meta['rows'][ji] != [old['stream'], old['step']]:
                    raise ValueError('parent row mapping')
                used_joint[name].add(ji)
                hs, prior = U.hypotheses(ARM, old['length'], old['kind'])
                if meta['hypotheses'] != [list(h) for h in hs] or meta['prior'] != prior.tolist():
                    raise ValueError('hypothesis roster')
                full = arrays[name][ji]; full_indices = np.arange(len(full), dtype=np.int32)
                current = project(hs, old['step'], full_indices, full)
                base, forecast = T.score(table, current, old['actual_maker'])
                if not np.array_equal(current, old['posterior']) or not np.array_equal(forecast, old['forecast']) or any(base[m] != old[m] for m in T.METRICS):
                    raise ValueError('parent reconstruction')
                # Scalar contraction independently checks current-marginal forecast sufficiency.
                scalar = np.array([[math.fsum(current[m] * table[m, context, endpoint] for m in range(16)) for endpoint in range(8)] for context in range(4)])
                if not np.allclose(scalar, forecast, atol=1e-13, rtol=0): raise ValueError('marginal forecast control')
                current_controls += 1
                for count in COUNTS:
                    start = time.process_time()
                    ix, kept, removed = truncate(full, count)
                    posterior = project(hs, old['step'], ix, kept)
                    values, prediction = T.score(table, posterior, old['actual_maker'])
                    if count == 'all':
                        if not np.array_equal(posterior, current) or any(values[m] != base[m] for m in T.METRICS):
                            raise ValueError('all-hypothesis identity')
                        identities += 1; selection_key = None; selection_row = None
                    else:
                        selection_key = f'{evidence}-{name}-{count}'; selection_row = len(selections[selection_key])
                        selections[selection_key].append(ix); selection_weights[selection_key].append(kept)
                    with (root / 'TIMING.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(dict(lineage=lineage, evidence=evidence, row=i, count=count, cpu_seconds=time.process_time()-start), sort_keys=True)+'\n')
                    row = dict(evidence=evidence, count=count, stream=old['stream'], draw=old['draw'], initial_maker=old['initial_maker'], actual_maker=old['actual_maker'],
                        **{k:old[k] for k in ('length','kind','switched','duplicates','step')}, parent_joint_array=name, parent_joint_row=ji,
                        selection_array=selection_key, selection_row=selection_row, posterior=posterior.tolist(), forecast=prediction.tolist(),
                        removed_mass=removed, hypotheses=len(kept), weight_bytes=int(kept.nbytes), index_bytes=0 if count=='all' else int(ix.nbytes),
                        current_marginal_weight_bytes=16*8, **values)
                    rows.append(row)
                    acc[old['draw'],old['length'],old['kind'],old['switched'],old['duplicates'],old['step'],evidence,str(count)].append(row)
            if seen != expected: raise ValueError('parent denominator')
            if any(used_joint[k] != set(range(len(v))) for k,v in arrays.items()): raise ValueError('unused joint rows')
        raw=root/'raw';raw.mkdir(exist_ok=True)
        np.savez_compressed(raw/f'{lineage}-selection_points.npz', **{k:np.asarray(v,dtype=np.int32) for k,v in sorted(selections.items())})
        np.savez_compressed(raw/f'{lineage}-weight_points.npz', **{k:np.asarray(v,dtype=np.float64) for k,v in sorted(selection_weights.items())})
        (raw/f'{lineage}-forecasts_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
        total += len(rows)
        for key, rr in sorted(acc.items()):
            if len(rr) != 16: raise ValueError('maker denominator')
            cells.append(dict(lineage=lineage,**dict(zip(('draw','length','kind','switched','duplicates','step','evidence','count'),key)),makers=16,
                **{m:math.fsum(r[m] for r in rr)/16 for m in (*T.METRICS,'removed_mass','hypotheses','weight_bytes','index_bytes')}))
    checks.update(positive_parent_reconstruction=True, positive_all_hypothesis_identity=True, positive_current_marginal_forecast=True)
    write(root/'EVIDENCE_ROLES.json',dict(reader='unchanged bound parent inputs',scientific='checkpoint selection, retained weights, forecasts and scores',
        evaluator='parent laws, true makers and hypotheses; never reader inputs'))
    return dict(controls=checks,cells=cells,rows=total,parent_identity_rows=identities,current_marginal_controls=current_controls,fits=0,
        scope='checkpoint compression of retained supplied-law mixtures; not recursive sufficiency, learned memory or historical correspondence')
