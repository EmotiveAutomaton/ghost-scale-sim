"""Lossless grouping by complete remaining maker schedules, without refitting.

All original hypothesis weights participate. Shared schedules and membership
maps are charged separately from the weight vector at each checkpoint.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import canonical, file_digest, read, write
from .unknown_change import hypotheses, state_indices


def quotient(hs, checkpoint, horizon):
    if not 0 <= checkpoint <= horizon: raise ValueError('time bounds')
    schedules = np.stack([state_indices(hs, t) for t in range(checkpoint, horizon+1)], axis=1).astype(np.int32)
    signatures, inverse = np.unique(schedules, axis=0, return_inverse=True)
    return signatures, inverse.astype(np.int32)


def weights_by_group(weights, mapping, count):
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or len(w) != len(mapping) or not np.isfinite(w).all() or (w < 0).any() or abs(math.fsum(w)-1) > 1e-10:
        raise ValueError('invalid weights')
    return np.bincount(mapping, weights=w, minlength=count)


def independent_schedule(hs, checkpoint, horizon):
    # Independent direct tuple transitions; no producer flip table.
    makers = list(product((0, 1), repeat=4)); rows = []
    for kind, change, maker in hs:
        sequence = []
        for t in range(checkpoint, horizon+1):
            bits = list(makers[maker])
            if kind != 'none' and t > change:
                axis = {'purpose': 0, 'skill': 1}[kind]; bits[axis] = 1-bits[axis]
            sequence.append(makers.index(tuple(bits)))
        rows.append(sequence)
    return np.asarray(rows, dtype=np.int32)


def verify_forecasts(hs, checkpoint, horizon, weights, signatures, mapping, law):
    original = independent_schedule(hs, checkpoint, horizon)
    if not np.array_equal(signatures[mapping], original): raise ValueError('schedule mapping')
    law = np.asarray(law, dtype=np.float64)
    if law.shape != (16, 4, 8) or not np.isfinite(law).all() or (law < 0).any() or not np.allclose(law.sum(-1), 1, atol=1e-12, rtol=0):
        raise ValueError('endpoint law')
    W = np.asarray(weights, dtype=np.float64)
    grouped = np.array([weights_by_group(w, mapping, len(signatures)) for w in W])
    forecasts = []; errors = []
    for offset in range(horizon-checkpoint+1):
        # Direct unmerged posterior marginal, independently enumerated states.
        raw = np.array([W[:, original[:, offset] == m].sum(axis=1) for m in range(16)]).T
        compact = np.array([grouped[:, signatures[:, offset] == m].sum(axis=1) for m in range(16)]).T
        predicted = np.einsum('rm,mce->rce', compact, law)
        scalar = np.array([[[math.fsum(float(p[m])*float(law[m, c, e]) for m in range(16))
                              for e in range(8)] for c in range(4)] for p in raw])
        error = np.max(np.abs(predicted-scalar), axis=(1, 2))
        if np.any(error > 1e-12): raise ValueError('future forecast differs')
        forecasts.append(predicted); errors.append(error)
    return grouped, np.stack(forecasts, axis=1), np.stack(errors, axis=1)


def controls():
    hs = [('none', 0, 0), ('purpose', 8, 8), ('purpose', 9, 0)]
    sig, ix = quotient(hs, 9, 10)
    future_separate = len(sig) == 2 and ix[0] == ix[1] and ix[2] != ix[0]
    stationary = [('none', 0, m) for m in range(16)]
    s, i = quotient(stationary, 8, 10)
    law = np.full((16, 4, 8), 1/8)
    _, p, errors = verify_forecasts(hs, 9, 10, [[.25, .25, .5]], sig, ix, law)
    return {'live:future_distinction_preserved': bool(future_separate),
            'positive:merged_after_change': bool(ix[0] == ix[1]),
            'placebo:stationary_identity': bool(len(s) == 16 and np.array_equal(i, np.arange(16))),
            'placebo:uniform_forecast_identity': bool(np.all(p == .125) and np.max(errors) == 0),
            'positive:weight_mass': math.fsum(weights_by_group([.25, .25, .5], ix, len(sig))) == 1}


def run(root, plan, pulse):
    cfg = plan['design']; checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('quotient controls')
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('input differs')
    counts = []; unavailable = []; meta = {}; projected_rows = forecast_vectors = 0; maximum = 0.
    for length in cfg['lengths']:
        hs, _ = hypotheses('unknown-time-type', length, 'purpose')
        for checkpoint in cfg['checkpoints']:
            if checkpoint > length: continue
            sig, ix = quotient(hs, checkpoint, length)
            if not np.array_equal(sig[ix], independent_schedule(hs, checkpoint, length)): raise ValueError('signature reconstruction')
            key = f'{length}-{checkpoint}';meta[key] = dict(length=length, checkpoint=checkpoint,
                hypotheses=[list(h) for h in hs], signatures=sig.tolist(), membership=ix.tolist(),
                components=len(sig), original_components=len(hs), weight_bytes=8*len(sig), original_weight_bytes=8*len(hs),
                membership_int32_bytes=int(ix.nbytes), schedule_int32_bytes=int(sig.nbytes))
    write(root/'evaluator/SCHEDULES.json', meta)
    for lineage in cfg['lineages']:
        law = read(root/'inputs/aware/evaluator'/f'{lineage}-law.json')
        pair = {}; rows = []; outputs = {}
        for evidence in ('aware', 'omitted'):
            base = root/'inputs'/evidence
            if law != read(base/'evaluator'/f'{lineage}-law.json'): raise ValueError('law pairing')
            mapping = read(base/'evaluator'/f'{lineage}-joint-map.json')
            parent_rows = json.loads(gzip.decompress((base/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
            parent_rows = [r for r in parent_rows if r['arm'] == 'unknown-time-type']
            with np.load(base/'raw'/f'{lineage}-joint_points.npz', allow_pickle=False) as z:
                arrays = {n:z[n] for n in z.files if n.endswith('-unknown-time-type')}
            available = {length:sorted({r['step'] for r in parent_rows if r['length'] == length}) for length in cfg['lengths']}
            for length in cfg['lengths']:
                for checkpoint in cfg['checkpoints']:
                    if checkpoint <= length and checkpoint not in available[length]:
                        unavailable.append(dict(lineage=lineage, evidence=evidence, length=length, checkpoint=checkpoint))
                        continue
                    if checkpoint > length:continue
                    selected = [r for r in parent_rows if r['length'] == length and r['step'] == checkpoint]
                    keys = [tuple(r[k] for k in ('draw','initial_maker','kind','switched','duplicates')) for r in selected]
                    expected = set(product(cfg['draws'],range(16),('purpose','skill'),(False,True),(False,True)))
                    if len(keys) != len(expected) or set(keys) != expected: raise ValueError('parent population')
                    spec = meta[f'{length}-{checkpoint}']; hs = spec['hypotheses']
                    W = []
                    for r in selected:
                        n, j = r['joint_array'], r['joint_row']; bound = mapping[n]
                        if bound['rows'][j] != [r['stream'], checkpoint] or bound['hypotheses'] != hs: raise ValueError('parent mapping')
                        identity = (length, checkpoint, r['draw'], r['initial_maker'], r['kind'], r['switched'], r['duplicates'])
                        witness = (r['stream'], r['actual_maker'])
                        if evidence == 'aware':pair[identity] = witness
                        elif pair[identity] != witness:raise ValueError('paired evidence')
                        W.append(arrays[n][j])
                    sig = np.asarray(spec['signatures'],dtype=np.int32); ix = np.asarray(spec['membership'],dtype=np.int32)
                    pulse(phase='lossless-future-quotient',lineage=lineage,evidence=evidence,length=length,checkpoint=checkpoint)
                    grouped, predicted, errors = verify_forecasts(hs, checkpoint, length, W, sig, ix, law)
                    key = f'{evidence}-{length}-{checkpoint}'
                    outputs[key+'-weights'] = grouped; outputs[key+'-forecasts'] = predicted; outputs[key+'-errors'] = errors
                    for j, r in enumerate(selected):
                        if not np.allclose(predicted[j,0], r['forecast'], atol=1e-12, rtol=0): raise ValueError('parent forecast identity')
                        rows.append(dict(evidence=evidence, array=key,row=j,stream=r['stream'],
                                         **{k:r[k] for k in ('draw','initial_maker','actual_maker','kind','switched','duplicates','length','step','joint_array','joint_row')},
                                         components=len(sig), original_components=len(ix), weight_bytes=8*len(sig),original_weight_bytes=8*len(ix),
                                         future_vectors=4*(length-checkpoint+1),max_absolute_forecast_error=float(errors[j].max())))
                    projected_rows += len(selected); forecast_vectors += predicted.shape[0]*predicted.shape[1]*4
                    maximum = max(maximum, float(errors.max()))
                    counts.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=checkpoint,
                                       rows=len(selected),components=len(sig),original_components=len(ix),
                                       weight_bytes=8*len(sig),original_weight_bytes=8*len(ix),
                                       forecast_vectors=int(predicted.shape[0]*predicted.shape[1]*4),max_absolute_error=float(errors.max())))
        (root/'raw').mkdir(exist_ok=True)
        np.savez_compressed(root/'raw'/f'{lineage}-quotient_points.npz', **outputs)
        (root/'raw'/f'{lineage}-mapping_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs; unchanged parent evidence separately bound',
        evaluator='full saved weights, state schedules, laws and forecasts; not reader evidence',
        storage='float64 checkpoint weights; shared int32 membership/schedules counted separately; archive/compression and workspace bytes are separate'))
    return dict(controls=checks,rows=projected_rows,forecast_vectors=forecast_vectors,max_absolute_error=maximum,
                cells=counts,unavailable_checkpoints=unavailable,fits=0,
                scope='lossless future prediction for the declared supplied-law schedule roster; no claim of online update sufficiency, reachability, historical process or human intent')
