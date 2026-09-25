"""Quantize only the shared future endpoint law; retain exact report state."""
import numpy as np
import hashlib
from . import aggregate_report_state as base
from .retrospective_source import ALPHAS, transitions

TOL = base.TOL
DTYPES = ('float64', 'float32', 'float16')
MODES = ('direct', 'row-normalized')
VARIANTS = tuple((dtype, mode) for dtype in DTYPES for mode in MODES)


def reconstruct(stored, mode):
    law = np.asarray(stored, dtype=np.float64).copy()
    if law.shape != (16, 4, 8) or not np.isfinite(law).all() or (law < 0).any() or mode not in MODES:
        raise ValueError('law/mode')
    sums = law.sum(-1)
    failed = bool((sums == 0).any())
    if mode == 'row-normalized' and not failed:
        law /= sums[..., None]
    return law, failed


def forecast_metrics(occupancy, law, beta=None):
    """Enumerate every future coordinate, with an explicit expected loss change.

    For outcome e and output j, the squared-loss difference is
    delta_j * (prediction_j + reference_j - 2*I(j=e)). Sum against
    the reference outcome weights before averaging times and contexts. This
    form also evaluates direct forecasts whose mass is not exactly one.
    """
    def project(matrix):
        forecast = (occupancy @ matrix.reshape(16, 32)).reshape(*occupancy.shape[:-1], 4, 8)
        if beta is None:
            return forecast
        return (1-beta)[..., None, None, None]*forecast[0][None, None] + beta[..., None, None, None]*forecast[1:][None]
    truth = project(law)
    metrics = {k: [] for k in ('max_future_probability_error', 'future_squared_error',
        'expected_one_hot_loss_difference', 'max_future_normalization_drift',
        'lost_future_support_probability', 'lost_future_support_count')}
    failed = []; underflow = []; row_drift = []
    for dtype, mode in VARIANTS:
        stored = law.astype(dtype)
        approx, bad = reconstruct(stored, mode)
        # Propagate the law difference directly to retain small rounding errors.
        delta = project(approx-law)
        prediction = project(approx)
        loss_change = (delta*((prediction+truth)*truth.sum(-1, keepdims=True)-2*truth)).sum(-1)
        metrics['max_future_probability_error'].append(abs(delta).max(axis=(-3, -2, -1)))
        metrics['future_squared_error'].append((delta**2).sum(-1).mean(axis=(-2, -1)))
        metrics['expected_one_hot_loss_difference'].append(loss_change.mean(axis=(-2, -1)))
        metrics['max_future_normalization_drift'].append(abs(prediction.sum(-1)-1).max(axis=(-2, -1)))
        lost = (truth > 0) & (prediction == 0)
        metrics['lost_future_support_probability'].append((truth*lost).sum(-1).mean(axis=(-2, -1)))
        metrics['lost_future_support_count'].append(lost.sum(axis=(-3, -2, -1)))
        failed.append(bad)
        underflow.append(np.count_nonzero((law > 0) & (stored == 0)))
        row_drift.append(np.max(abs(stored.astype(float).sum(-1)-1)))
    return {k: np.asarray(v) for k, v in metrics.items()}, np.asarray(failed), np.asarray(underflow), np.asarray(row_drift)


def evaluate(w, st, law, ids, operator=None):
    law = np.asarray(law, dtype=np.float64)
    q, joint, counts, _ = base.acquire(w, st, law, ids)
    probability, possible, post = base.update(q, joint, counts)
    op = transitions(dict(future=st['signatures'])) if operator is None else operator
    independent_mass = joint.sum(0)
    independent = np.divide(joint.T, independent_mass[:, None], out=np.zeros_like(joint.T), where=independent_mass[:, None] > 0)
    beta = np.divide((1-np.asarray(ALPHAS)[:, None])*independent_mass, probability, out=np.zeros_like(probability), where=possible)
    components = np.concatenate([q[None], independent], axis=0)
    increments = (components @ op).reshape(9, st['future'].shape[1], 16)
    occupancy = increments.cumsum(axis=-2)
    # Integer support propagation identifies true zero cells, without a
    # tolerance, probability floor, or change to the stored posterior.
    support = ((components > 0).astype(np.int64) @ op).reshape(9, st['future'].shape[1], 16).cumsum(axis=-2) > 0
    occupancy = np.where(support, occupancy, 0.)
    if (occupancy < 0).any(): raise ValueError('negative propagated state')
    metrics, failed, underflow, drift = forecast_metrics(occupancy, law, beta)
    defined = possible[None, :, :] & ~failed[:, None, None]
    return dict(group_mass=q, joint_report_mass=joint, copy_counts=counts,
        source_rows=np.asarray(ids, dtype=np.int32), report_probability=probability,
        possible=possible, reconstruction_failed=failed, defined=defined,
        **{k: np.where(defined, v, np.nan) for k, v in metrics.items()},
        law_underflow_count=underflow, pre_repair_law_row_drift=drift,
        shared_law_bytes=np.array([512*np.dtype(d).itemsize for d, m in VARIANTS]),
        exact_state_bytes=np.array(9*len(q)*8+32),
        shared_schedule_bytes=np.array(st['signatures'].size*4))


METRICS = ('max_future_probability_error', 'future_squared_error',
    'expected_one_hot_loss_difference', 'max_future_normalization_drift',
    'lost_future_support_probability', 'lost_future_support_count')


def summarize(raw):
    p, possible = raw['report_probability'], raw['possible']
    if p.shape != (5, 8) or not np.isfinite(p).all() or (p < 0).any() or np.max(abs(p.sum(-1)-1)) > TOL or not np.array_equal(possible, p > 0):
        raise ValueError('report support')
    rebuilt, support, _ = base.update(raw['group_mass'], raw['joint_report_mass'], raw['copy_counts'])
    if not np.array_equal(support, possible) or np.max(abs(rebuilt-p)) > TOL:
        raise ValueError('exact state')
    if not np.array_equal(raw['copy_counts'], np.bincount(raw['source_rows'][:, 2], minlength=8)):
        raise ValueError('copy histogram')
    failed = raw['reconstruction_failed']
    if failed.shape != (6,): raise ValueError('failure shape')
    defined = possible[None, :, :] & ~failed[:, None, None]
    if not np.array_equal(defined, raw['defined']): raise ValueError('defined mask')
    result = dict(unusable_report_probability=(p[None, :, :]*~defined).sum(-1),
        reconstruction_failed=np.broadcast_to(failed[:, None], (6, 5)).astype(float))
    for field in METRICS:
        value = raw[field]
        if value.shape != (6, 5, 8) or not np.array_equal(np.isnan(value), ~defined) or not np.isfinite(value[defined]).all():
            raise ValueError('undefined metric')
        if field != 'expected_one_hot_loss_difference' and (value[defined] < 0).any():
            raise ValueError('negative error')
        result['defined_weighted_'+field] = (p[None, :, :]*np.nan_to_num(value)).sum(-1)
    for field in ('shared_law_bytes', 'law_underflow_count', 'pre_repair_law_row_drift'):
        if raw[field].shape != (6,) or not np.isfinite(raw[field]).all() or (raw[field] < 0).any(): raise ValueError('law diagnostics')
        result[field] = np.broadcast_to(raw[field][:, None], (6, 5)).astype(float)
    if not np.array_equal(raw['shared_law_bytes'], [512*np.dtype(d).itemsize for d, m in VARIANTS]) or raw['exact_state_bytes'] != 72*len(raw['group_mass'])+32:
        raise ValueError('storage')
    for field in ('exact_state_bytes', 'shared_schedule_bytes'):
        result[field] = np.full((6, 5), raw[field], dtype=float)
    return result


def export_raw(raw):
    """Do not duplicate exact state derived from the complete frozen inputs.

    Keep digest bindings to both exact arrays and preserve their full source
    posterior, law and source roster in the immutable inputs. A verifier must
    reconstruct these arrays before independently checking their arithmetic.
    """
    result = dict(raw)
    for field in ('group_mass', 'joint_report_mass'):
        value = result.pop(field)
        result[field+'_sha256'] = np.frombuffer(hashlib.sha256(value.astype('<f8').tobytes()).digest(), dtype=np.uint8)
    return result


def controls():
    from .reachable_retrospective import prepare, fixture
    st = prepare(fixture(), [.25, .25, .5]); w = np.array([.6, .1, .3])
    ids = np.array([[1, 0, 0], [2, 1, 1]])
    law = np.full((16, 4, 8), 1/8); flat = evaluate(w, st, law, ids)
    law[:, :, 0] += 0.00003; law[:, :, 1] -= 0.00003
    live = evaluate(w, st, law, ids)
    return {'live:law_rounding_detected': bool(np.nanmax(live['max_future_probability_error'][4:]) > 1e-6),
        'positive:float64_identity': bool(np.nanmax(live['max_future_probability_error'][0]) == 0),
        'placebo:exactly_representable_law': bool(np.nanmax(flat['max_future_probability_error']) == 0),
        'positive:exact_report_state': bool(np.array_equal(live['possible'], flat['possible']))}

import gzip
import json
import time
from itertools import product
from ..v18_3.io import read, write, canonical, file_digest
from .reachable_retrospective import prepare


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['alphas']!=list(ALPHAS) or cfg['report_state']!='exact-state-shared-future-law-precision' or cfg['storage_dtypes']!=list(DTYPES) or cfg['reconstruction_modes']!=list(MODES):raise ValueError('controls/design')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    if not read(root/'inputs/PARENT_REVIEW.json')['numerical_acceptance']:raise ValueError('parent acceptance')
    specs=read(root/'inputs/SCHEDULES.json');structures={}
    for key,spec in specs.items():
        hs=spec['hypotheses'];st=prepare(spec,[.5/16 if h[0]=='none' else .5/(len(hs)-16) for h in hs]);structures[key]=(st,transitions(dict(future=st['signatures'])))
    rows=[];unavailable=[];source_total=0;paired={};(root/'raw').mkdir(exist_ok=True)
    for lineage in cfg['lineages']:
        law=np.asarray(read(root/'inputs/aware/evaluator'/f'{lineage}-law.json'))
        (root/'evaluator').mkdir(exist_ok=True)
        np.savez_compressed(root/'evaluator'/f'{lineage}-stored-law_points.npz', **{dtype: law.astype(dtype) for dtype in DTYPES})
        for evidence in ('aware','omitted'):
            base=root/'inputs'/evidence
            if not np.array_equal(law,read(base/'evaluator'/f'{lineage}-law.json')):raise ValueError('law pairing')
            maps=read(base/'evaluator'/f'{lineage}-joint-map.json')
            with np.load(base/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as z:arrays={n:z[n] for n in z.files if n.endswith('-unknown-time-type')}
            for key,spec in specs.items():
                length,cp=spec['length'],spec['checkpoint'];prefix=f'{lineage}-{evidence}-{key}'
                binding_path=root/'inputs/bindings'/(prefix+'-bindings.json')
                if not binding_path.exists():
                    if (length,cp) not in ((128,8),(128,17),(128,20)):raise ValueError('missing checkpoint')
                    unavailable.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp));continue
                binding=read(binding_path);st,op=structures[key];chosen=binding['rows']
                ids=[tuple(r[k] for k in ('draw','initial_maker','kind','switched','duplicates')) for r in chosen]
                expected=set(product(cfg['draws'],range(16),('purpose','skill'),(False,True),(False,True)))
                if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('paired roster')
                byrow={i:[] for i in range(len(chosen))}
                for i,t,ctx,e,source_id in binding['sources']:byrow[i].append((t,ctx,e,source_id))
                start=time.process_time();chunks={}
                for index,(r,identity) in enumerate(zip(chosen,ids)):
                    if index%cfg['batch_rows']==0:
                        pulse(phase='shared-law-precision',lineage=lineage,evidence=evidence,checkpoint=cp,row=index)
                    name,j=r['joint_array'],r['joint_row'];bound=maps[name]
                    if bound['hypotheses']!=spec['hypotheses'] or bound['rows'][j]!=[r['stream'],cp]:raise ValueError('posterior binding')
                    sources=byrow[index]
                    if len(sources)!=r['report_sources'] or len({s[3] for s in sources})!=len(sources):raise ValueError('source roster')
                    pair=(lineage,key,*identity);witness=(r['stream'],sources)
                    if evidence=='aware':paired[pair]=witness
                    elif paired[pair]!=witness:raise ValueError('source pairing')
                    w=arrays[name][j]
                    baseline=(np.bincount(st['future'][:,0],weights=w,minlength=16)@law.reshape(16,32)).reshape(4,8)
                    if not np.allclose(baseline,r['forecast'],atol=TOL,rtol=0):raise ValueError('parent forecast identity')
                    raw=evaluate(w,st,law,np.asarray([s[:3] for s in sources]),op);metrics=summarize(raw)
                    for k,v in export_raw(raw).items():chunks[f'{index:03d}__'+k]=v
                    for vi,(dtype,mode) in enumerate(VARIANTS):
                        for ai,alpha in enumerate(ALPHAS):rows.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,alpha=alpha,storage_dtype=dtype,reconstruction=mode,**{k:r[k] for k in ('draw','initial_maker','kind','switched','duplicates','stream','report_sources')},**{k:float(v[vi,ai]) for k,v in metrics.items()}))
                    source_total+=len(sources)
                    if (index+1)%cfg['batch_rows']==0 or index==len(chosen)-1:
                        np.savez_compressed(root/'raw'/f'{prefix}-{index//cfg["batch_rows"]:03d}_points.npz',**chunks);chunks={}
                with (root/'TIMING.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,posterior_rows=len(chosen),sources=sum(map(len,byrow.values())),cpu_seconds=time.process_time()-start))+'\n')
    (root/'raw/shared_law_precision_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader input',evaluator='supplied fixed uniform source mixture,law,posterior and full source identities;exact aggregate state and integer copy histogram;only shared future law quantized;exact report law unchanged;explicit expected one-hot squared-loss change;future underflow and normalization diagnostics separate;undefined reports remain NaN',scope='single report under fixed source prior/content/law;not arbitrary reweighting,disclosed source,two dependent reports,minimal state or optimized compact latency'))
    return dict(controls=checks,posterior_rows=len(rows)//(len(ALPHAS)*len(VARIANTS)),rows=len(rows),sources=source_total,report_queries=len(rows)*8,unavailable_checkpoints=unavailable,numerical_acceptance=False,scope='fixed float64/32/16 shared future-law storage and direct/row-normalized reconstruction;all endpoint reports and future coordinates;no fitted model,observation or protected lineage')
