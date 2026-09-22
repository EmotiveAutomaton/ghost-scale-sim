"""Independent audit of saved future-schedule grouping and its storage costs.

No producer roster, transition, grouping, forecasting or summary is imported.
Parent likelihood validity is inherited from the separately pinned reviews.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, file_digest


def roster(length):
    return ([['none', 0, m] for m in range(16)] +
            [[kind, t, m] for kind in ('purpose', 'skill')
             for t in range(8, length-7) for m in range(16)])


def schedule(h, start, end):
    kind, change, maker = h
    mask = {'none': 0, 'purpose': 8, 'skill': 4}[kind]
    return tuple(maker ^ (mask if t > change else 0) for t in range(start, end+1))


def partition(hs, start, end):
    members = defaultdict(list)
    for j, h in enumerate(hs):
        members[schedule(h, start, end)].append(j)
    signatures = sorted(members)
    inverse = [None]*len(hs)
    groups = []
    for k, s in enumerate(signatures):
        groups.append(members[s])
        for j in members[s]: inverse[j] = k
    return signatures, inverse, groups


def audit_arrays(hs, checkpoint, horizon, W, law, saved_w, saved_p, saved_errors):
    sig, inverse, groups = partition(hs, checkpoint, horizon)
    W = np.asarray(W, dtype=np.float64); law = np.asarray(law, dtype=np.float64)
    if W.ndim != 2 or W.shape[1] != len(hs) or not np.isfinite(W).all() or (W < 0).any():
        raise ValueError('parent weights')
    if not np.allclose(W.sum(1), 1., atol=1e-12, rtol=0): raise ValueError('parent mass')
    if law.shape != (16, 4, 8) or not np.isfinite(law).all() or (law < 0).any() or not np.allclose(law.sum(-1), 1., atol=1e-12, rtol=0):
        raise ValueError('law')
    compact = np.asarray([[math.fsum(float(w[j]) for j in ids) for ids in groups] for w in W])
    if saved_w.shape != compact.shape or saved_w.dtype != np.dtype('float64') or not np.isfinite(saved_w).all() or np.max(np.abs(compact-saved_w)) > 1e-12:
        raise ValueError('grouped weights')
    expected_shape = (len(W), horizon-checkpoint+1, 4, 8)
    if saved_p.shape != expected_shape or saved_errors.shape != expected_shape[:2]: raise ValueError('forecast shapes')
    if saved_p.dtype != np.dtype('float64') or saved_errors.dtype != np.dtype('float64') or not np.isfinite(saved_p).all() or not np.isfinite(saved_errors).all():
        raise ValueError('forecast dtype/finite')
    if (saved_errors < 0).any() or (saved_errors > 1e-12).any(): raise ValueError('reported errors')
    maximum = 0.
    for offset, t in enumerate(range(checkpoint, horizon+1)):
        state = np.asarray([schedule(h, t, t)[0] for h in hs])
        # Independent ungrouped state projection and matrix contraction.
        mass = np.column_stack([np.sum(W[:, state == m], axis=1) for m in range(16)])
        predicted = (mass @ law.reshape(16, 32)).reshape(len(W), 4, 8)
        error = float(np.max(np.abs(predicted-saved_p[:, offset])))
        if error > 1e-12: raise ValueError('future forecasts')
        maximum = max(maximum, error)
    return dict(weight_error=float(np.max(np.abs(compact-saved_w))), forecast_error=maximum,
                components=len(sig), rows=len(W), forecast_vectors=4*len(W)*(horizon-checkpoint+1))


def review(parent, output, pulse=lambda **kw: None):
    plan = read(parent/'PLAN.json'); cfg = plan['design']; summary = read(parent/'SUMMARY.json')
    meta = read(parent/'evaluator/SCHEDULES.json'); storage = []; available_keys = set(); cells = []; unavailable = []
    expected_keys = {f'{length}-{step}' for length in cfg['lengths'] for step in cfg['checkpoints'] if step <= length}
    if set(meta) != expected_keys: raise ValueError('schedule coverage')
    for length in cfg['lengths']:
        hs = roster(length)
        for step in cfg['checkpoints']:
            if step > length: continue
            key = f'{length}-{step}'; sig, inverse, groups = partition(hs, step, length)
            spec = dict(length=length, checkpoint=step, hypotheses=hs, signatures=[list(s) for s in sig],
                        membership=inverse, components=len(sig), original_components=len(hs),
                        weight_bytes=8*len(sig), original_weight_bytes=8*len(hs),
                        membership_int32_bytes=4*len(hs), schedule_int32_bytes=4*len(sig)*(length-step+1))
            if meta[key] != spec: raise ValueError('schedule membership or bytes')
            metadata = spec['membership_int32_bytes']+spec['schedule_int32_bytes']
            saving = spec['original_weight_bytes']-spec['weight_bytes']
            storage.append(dict(length=length, checkpoint=step, components=len(sig), original_components=len(hs),
                weight_bytes=spec['weight_bytes'], original_weight_bytes=spec['original_weight_bytes'],
                weight_fraction=len(sig)/len(hs), shared_membership_bytes=4*len(hs),
                shared_schedule_bytes=spec['schedule_int32_bytes'], shared_metadata_bytes=metadata,
                posterior_count_to_repay_metadata=None if saving == 0 else metadata//saving+1,
                comparison='quotient weights plus new int32 membership/schedule arrays against original weights alone; strict byte saving, common parent metadata excluded'))
    maximum_weight = maximum_forecast = 0.; paired_rows = 0
    for lineage in cfg['lineages']:
        pulse(phase='independent-quotient-reconstruction', lineage=lineage)
        saved_rows = json.loads(gzip.decompress((parent/'raw'/f'{lineage}-mapping_points.json.gz').read_bytes()))
        saved_arrays = np.load(parent/'raw'/f'{lineage}-quotient_points.npz', allow_pickle=False)
        wanted_rows = []; paired = {}; wanted_arrays = set()
        law = read(parent/'inputs/aware/evaluator'/f'{lineage}-law.json')
        for evidence in ('aware', 'omitted'):
            base = parent/'inputs'/evidence
            if read(base/'evaluator'/f'{lineage}-law.json') != law: raise ValueError('paired laws')
            parent_map = read(base/'evaluator'/f'{lineage}-joint-map.json')
            rows = json.loads(gzip.decompress((base/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
            rows = [r for r in rows if r['arm'] == 'unknown-time-type']
            with np.load(base/'raw'/f'{lineage}-joint_points.npz', allow_pickle=False) as z:
                parent_arrays = {n:z[n] for n in z.files if n.endswith('-unknown-time-type')}
                for length in cfg['lengths']:
                    hs = roster(length)
                    for step in cfg['checkpoints']:
                        if step > length: continue
                        selected = [r for r in rows if r['length'] == length and r['step'] == step]
                        if not selected:
                            unavailable.append(dict(lineage=lineage, evidence=evidence, length=length, checkpoint=step)); continue
                        available_keys.add(f'{length}-{step}')
                        keys = [tuple(r[k] for k in ('draw','initial_maker','kind','switched','duplicates')) for r in selected]
                        expected = set(product(cfg['draws'], range(16), ('purpose','skill'), (False,True), (False,True)))
                        if len(keys) != len(expected) or set(keys) != expected: raise ValueError('population coverage')
                        W = []
                        for r in selected:
                            n, j = r['joint_array'], r['joint_row']
                            if parent_map[n]['hypotheses'] != hs or parent_map[n]['rows'][j] != [r['stream'],step]: raise ValueError('parent row mapping')
                            identity = (length, step, *tuple(r[k] for k in ('draw','initial_maker','kind','switched','duplicates')))
                            witness = (r['stream'],r['actual_maker'])
                            if evidence == 'aware': paired[identity] = witness
                            elif paired.get(identity) != witness: raise ValueError('parent pairing')
                            else: paired_rows += 1
                            W.append(parent_arrays[n][j])
                        key = f'{evidence}-{length}-{step}'
                        wanted_arrays.update(key+s for s in ('-weights','-forecasts','-errors'))
                        result = audit_arrays(hs, step, length, W, law, saved_arrays[key+'-weights'], saved_arrays[key+'-forecasts'], saved_arrays[key+'-errors'])
                        maximum_weight = max(maximum_weight, result['weight_error']); maximum_forecast = max(maximum_forecast, result['forecast_error'])
                        if not np.allclose(saved_arrays[key+'-forecasts'][:,0], [r['forecast'] for r in selected], atol=1e-12, rtol=0): raise ValueError('parent forecasts')
                        cell = dict(lineage=lineage,evidence=evidence,length=length,checkpoint=step,rows=len(selected),
                            components=result['components'],original_components=len(hs),weight_bytes=8*result['components'],original_weight_bytes=8*len(hs),
                            forecast_vectors=result['forecast_vectors'], max_absolute_error=float(saved_arrays[key+'-errors'].max()))
                        cells.append(cell)
                        for j,r in enumerate(selected):
                            wanted_rows.append(dict(evidence=evidence,array=key,row=j,stream=r['stream'],
                                **{k:r[k] for k in ('draw','initial_maker','actual_maker','kind','switched','duplicates','length','step','joint_array','joint_row')},
                                components=result['components'],original_components=len(hs),weight_bytes=8*result['components'],original_weight_bytes=8*len(hs),
                                future_vectors=4*(length-step+1),max_absolute_forecast_error=float(saved_arrays[key+'-errors'][j].max())))
        if set(saved_arrays.files) != wanted_arrays or saved_rows != wanted_rows: raise ValueError('output row/array coverage')
        saved_arrays.close()
    if cells != summary['cells'] or unavailable != summary['unavailable_checkpoints']: raise ValueError('summary cells')
    if summary['rows'] != sum(r['rows'] for r in cells) or summary['forecast_vectors'] != sum(r['forecast_vectors'] for r in cells): raise ValueError('summary totals')
    if summary['max_absolute_error'] != max(r['max_absolute_error'] for r in cells): raise ValueError('summary error')
    for r in storage:
        key = f"{r['length']}-{r['checkpoint']}"
        count = sum(c['rows'] for c in cells if c['length']==r['length'] and c['checkpoint']==r['checkpoint'])
        r.update(available=key in available_keys,posterior_rows=count,
                 original_weight_total_bytes=count*r['original_weight_bytes'],
                 quotient_weight_total_bytes=count*r['weight_bytes'],
                 quotient_plus_shared_metadata_bytes=count*r['weight_bytes']+r['shared_metadata_bytes'])
    files = {p.relative_to(parent).as_posix():p.stat().st_size for folder in ('raw','evaluator') for p in (parent/folder).rglob('*') if p.is_file()}
    result = dict(passed=True, rows=summary['rows'], forecast_vectors=summary['forecast_vectors'], paired_source_rows=paired_rows,
        cells=len(cells), unavailable_checkpoints=unavailable, max_absolute_weight_error=maximum_weight,
        max_absolute_forecast_error=maximum_forecast, storage=storage, retained_artifact_bytes=files,
        original_weight_total_bytes=sum(r['original_weight_total_bytes'] for r in storage),
        quotient_weight_total_bytes=sum(r['quotient_weight_total_bytes'] for r in storage),
        shared_metadata_bytes_all_requested=sum(r['shared_metadata_bytes'] for r in storage),
        shared_metadata_bytes_used=sum(r['shared_metadata_bytes'] for r in storage if r['available']),
        scope='independent numerical identity on all saved rows; supplied laws; no online-update, historical process, reachable-simplex or human claim',
        limits='does not revalidate parent likelihoods; dtype storage is not Python workspace or archive size; no execution-speed advantage measured')
    write(output/'NUMERICAL_REVIEW.json', result)
    return result


def run(root, plan, pulse):
    for n, h in plan['design']['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('review input binding')
    result = review(root/'inputs/parent', root, pulse)
    return dict(result, controls={'live:all_saved_forecasts_reconstructed':result['max_absolute_forecast_error']<=1e-12,
        'positive:all_group_weights_reconstructed':result['max_absolute_weight_error']<=1e-12,
        'placebo:zero_weight_schedules_retained':True})
