"""Exact retrospective provenance intervals from accepted lossless source factors."""
import gzip
import json
import time
from collections import defaultdict
import numpy as np
from ..v18_3.io import read, write, canonical, file_digest

INTERVALS = ((0., 1.), (.25, .75), (0., .5), (.5, 1.))
TOL = 1e-12


def factors(raw, intervals=INTERVALS):
    """Store the complete segment, support and probability envelopes.

    Only the old endpoint varies with alpha. Every other supported endpoint is
    the independent posterior. Baseline and signed future differences remain
    exactly reconstructible from the bound parent hypotheses/laws/source IDs.
    """
    bounds = np.asarray(intervals, float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds) or not np.isfinite(bounds).all() or (bounds < 0).any() or (bounds > 1).any() or (bounds[:, 0] > bounds[:, 1]).any():
        raise ValueError('interval bounds')
    p = np.asarray(raw['independent_report_probability'], float)
    ids = np.asarray(raw['source_rows'])
    if p.ndim != 2 or p.shape[1] != 8 or not len(p) or not np.isfinite(p).all() or (p < 0).any() or np.max(abs(p.sum(-1)-1)) > TOL:
        raise ValueError('report probabilities')
    if ids.shape != (len(p), 4) or ids.dtype.kind not in 'iu' or (ids[:, 3] < 0).any() or (ids[:, 3] >= 8).any():
        raise ValueError('source identity')
    old = ids[:, 3]; oldp = p[np.arange(len(p)), old]
    if (oldp <= 0).any(): raise ValueError('old source unsupported')
    indicators = np.eye(8)[old, None, :]
    lo, hi = bounds.T; mid = (lo+hi)/2
    prob_lo = lo[None, :, None]*indicators + (1-lo[None, :, None])*p[:, None, :]
    prob_hi = hi[None, :, None]*indicators + (1-hi[None, :, None])*p[:, None, :]
    prob_mid = mid[None, :, None]*indicators + (1-mid[None, :, None])*p[:, None, :]
    def rho(alpha):
        numerator = (1-alpha[None, :])*oldp[:, None]
        return numerator/(alpha[None, :]+numerator)
    low_rho, high_rho, mid_rho = rho(hi), rho(lo), rho(mid)
    spread = high_rho-low_rho
    radius = np.maximum(mid_rho-low_rho, high_rho-mid_rho)
    result = dict(source_rows=ids, independent_weight_lower=low_rho,
                  independent_weight_upper=high_rho, midpoint_independent_weight=mid_rho,
                  report_probability_lower=np.minimum(prob_lo, prob_hi),
                  report_probability_upper=np.maximum(prob_lo, prob_hi),
                  report_possible=(prob_lo > 0) | (prob_hi > 0),
                  lower_point_possible=prob_lo > 0, upper_point_possible=prob_hi > 0,
                  midpoint_possible=prob_mid > 0)
    for metric in ('group_tv', 'max_future_difference', 'mean_future_tv'):
        value = np.asarray(raw['independent_vs_copy_'+metric], float)
        if value.shape != (len(p),) or not np.isfinite(value).all() or (value < 0).any() or (value > 1+TOL).any():
            raise ValueError('discrepancy')
        result['old_endpoint_width_'+metric] = spread*value[:, None]
        result['old_endpoint_midpoint_worst_'+metric] = radius*value[:, None]
    return result


def measures(raw):
    count = raw['report_possible'].sum(-1)
    if (count == 0).any(): raise ValueError('empty report alphabet')
    out = {'supported_endpoints': count.astype(float),
           'impossible_endpoints': 8-count,
           'lower_point_unsupported_fraction': (raw['report_possible'] & ~raw['lower_point_possible']).sum(-1)/count,
           'upper_point_unsupported_fraction': (raw['report_possible'] & ~raw['upper_point_possible']).sum(-1)/count}
    for key, value in raw.items():
        if key.startswith('old_endpoint_'):
            out[key] = value
            out['equal_supported_endpoint_'+key] = value/count
    return out


def controls():
    raw = dict(source_rows=np.array([[0, 1, 0, 0]]), independent_report_probability=np.full((1, 8), .125),
               independent_vs_copy_group_tv=np.array([.5]), independent_vs_copy_max_future_difference=np.array([.2]), independent_vs_copy_mean_future_tv=np.array([.3]))
    full = factors(raw); zero = factors({**raw, 'independent_vs_copy_max_future_difference': np.zeros(1)})
    point = factors(raw, ((.5, .5),))
    return {'live:full_interval_nonzero_width': bool(full['old_endpoint_width_max_future_difference'][0, 0] == .2),
            'placebo:zero_future_difference': bool((zero['old_endpoint_width_max_future_difference'] == 0).all()),
            'positive:singleton_interval_zero_width': bool((point['old_endpoint_width_group_tv'] == 0).all()),
            'positive:copy_endpoint_support_excluded': bool(measures(full)['upper_point_unsupported_fraction'][0, 0] == 7/8)}


def run(root, plan, pulse):
    cfg = plan['design']; checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()) or cfg['intervals'] != [list(v) for v in INTERVALS]: raise ValueError('design/controls')
    for name, digest in cfg['input_files'].items():
        if file_digest(root/'inputs'/name) != digest: raise ValueError('input binding')
    base = root/'inputs/source'; parent = read(base/'PLAN.json'); complete = read(base/'COMPLETE.json')
    if file_digest(base/'PLAN.json') != cfg['parent_plan_sha256'] or complete['plan_sha256'] != cfg['parent_plan_sha256']: raise ValueError('parent plan')
    if not read(base/'FINAL_REVIEW.json')['numerical_acceptance']: raise ValueError('parent acceptance')
    expected = {n for n in complete['files'] if n.startswith('raw/') and n.endswith('.npz')}
    if expected != {p.relative_to(base).as_posix() for p in (base/'raw').glob('*.npz')}: raise ValueError('raw roster')
    (root/'raw').mkdir(exist_ok=True); rows=[]; all_sources=0; used=set()
    for path in sorted((base/'evaluator').glob('*-bindings.json')):
        bindings = read(path); prefix = path.stem[:-len('-bindings')]
        lineage, evidence, length, checkpoint = prefix.split('-')
        per_row=defaultdict(lambda:defaultdict(list)); seen=[]; started=time.process_time()
        files=sorted((base/'raw').glob(prefix+'-*_points.npz'))
        if not files: raise ValueError('missing source factors')
        for file in files:
            pulse(phase='retrospective-provenance-interval',lineage=int(lineage),evidence=evidence,checkpoint=int(checkpoint))
            used.add(file.relative_to(base).as_posix())
            with np.load(file,allow_pickle=False) as z: inherited={n:z[n] for n in z.files}
            result=factors(inherited); stats=measures(result)
            np.savez_compressed(root/'raw'/file.name,**result)
            for index,row in enumerate(result['source_rows'][:,0]):
                seen.append(result['source_rows'][index].tolist())
                for key,value in stats.items():per_row[int(row)][key].append(value[index])
        if seen != [r[:4] for r in bindings['sources']]: raise ValueError('source order/identity')
        for index,record in enumerate(bindings['rows']):
            if any(len(v)!=record['report_sources'] for v in per_row[index].values()): raise ValueError('source count')
            means={k:np.mean(v,axis=0) for k,v in per_row[index].items()}
            for i,interval in enumerate(INTERVALS):
                rows.append(dict(lineage=int(lineage),evidence=evidence,length=int(length),checkpoint=int(checkpoint),interval=list(interval),
                    **{k:record[k] for k in ('draw','initial_maker','kind','switched','duplicates','stream','report_sources')},
                    **{k:float(v[i]) for k,v in means.items()}))
        all_sources+=len(seen)
        with (root/'TIMING.jsonl').open('a',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(dict(lineage=int(lineage),evidence=evidence,length=int(length),checkpoint=int(checkpoint),sources=len(seen),posterior_rows=len(bindings['rows']),cpu_seconds=time.process_time()-started))+'\n')
    if used != expected: raise ValueError('unused source factors')
    prior=read(base/'SUMMARY.json')
    if all_sources!=prior['sources'] or len(rows)!=prior['posterior_rows']*len(INTERVALS): raise ValueError('population')
    (root/'raw/interval_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='lossless source-conditioned posterior segments;bound parent hypotheses and laws reconstruct all coordinate extrema;coordinate envelope is not an independently selectable rectangular joint forecast',population='sources averaged within posterior;equal supported-endpoint summary is descriptive counting,not a calibrated report distribution;endpoint support failures remain separate',factorization='old endpoint:base+rho*(independent-base),rho in stored closed interval;other possible endpoints:independent only;impossible alpha/report pairs excluded'))
    return dict(controls=checks,posterior_rows=prior['posterior_rows'],sources=all_sources,rows=len(rows),intervals=[list(v) for v in INTERVALS],source_interval_endpoint_queries=all_sources*len(INTERVALS)*8,unavailable_checkpoints=prior['unavailable_checkpoints'],numerical_acceptance=False,scope='exact supplied provenance-interval envelopes and fixed-point worst-case discrepancies;no learned provenance or historical correspondence')
