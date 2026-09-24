"""Scalar provenance-interval reconstruction from independently accepted factors.

No producer kernels are imported. The parent factors already bind a reconstruction
of every future coordinate from hypothesis weights and supplied native laws.
"""
from collections import defaultdict
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, file_digest

TOL = 1e-12
INTERVALS = ((0., 1.), (.25, .75), (0., .5), (.5, 1.))
IDENTITY = ('draw', 'initial_maker', 'kind', 'switched', 'duplicates')
METRICS = ('group_tv', 'max_future_difference', 'mean_future_tv')


def close(actual, expected, name):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.max(abs(a-b), initial=0) > TOL:
        raise ValueError(name)


def reconstruct(parent, saved, intervals=INTERVALS):
    """Bayes copy fraction, rather than the producer's independent fraction.

    All future signed differences share one scalar. Consequently the extrema of
    each coordinate, total variation and largest discrepancy scale by the same
    segment length. Rectangular combinations are never treated as joint states.
    """
    if not intervals or any(len(x)!=2 or not all(math.isfinite(a) for a in x) or not 0<=x[0]<=x[1]<=1 for x in intervals):
        raise ValueError('interval bounds')
    p = np.asarray(parent['independent_report_probability'], float)
    ids = np.asarray(parent['source_rows'])
    if p.ndim!=2 or p.shape[1]!=8 or not len(p) or (p<0).any(): raise ValueError('probability shape')
    close(p.sum(-1), np.ones(len(p)), 'probability mass')
    if ids.shape!=(len(p),4) or ids.dtype.kind not in 'iu' or (ids[:,3]<0).any() or (ids[:,3]>=8).any(): raise ValueError('source identity')
    if not np.array_equal(saved['source_rows'], ids): raise ValueError('source rows')
    shape=(len(p),len(intervals)); out={'source_rows':ids.copy()}
    for key in ('independent_weight_lower','independent_weight_upper','midpoint_independent_weight'):
        out[key]=np.empty(shape)
    for key in ('report_probability_lower','report_probability_upper'):
        out[key]=np.empty(shape+(8,))
    for key in ('report_possible','lower_point_possible','upper_point_possible','midpoint_possible'):
        out[key]=np.zeros(shape+(8,),bool)
    for metric in METRICS:
        for prefix in ('old_endpoint_width_','old_endpoint_midpoint_worst_'):
            out[prefix+metric]=np.empty(shape)
        value=np.asarray(parent['independent_vs_copy_'+metric])
        if value.shape!=(len(p),) or not np.isfinite(value).all() or (value<0).any() or (value>1+TOL).any(): raise ValueError('parent discrepancy')
    for n, (_,_,_,old) in enumerate(ids):
        if p[n,old]<=0: raise ValueError('old report unsupported')
        for j,(lo,hi) in enumerate(intervals):
            mid=(lo+hi)/2
            copies=[]
            for a in (lo,hi,mid):
                mass=math.fsum((a,(1-a)*float(p[n,old])))
                copies.append(a/mass)
            low_copy,high_copy,mid_copy=copies
            out['independent_weight_lower'][n,j]=1-high_copy
            out['independent_weight_upper'][n,j]=1-low_copy
            out['midpoint_independent_weight'][n,j]=1-mid_copy
            for endpoint in range(8):
                probs=[math.fsum((a if endpoint==old else 0.,(1-a)*float(p[n,endpoint]))) for a in (lo,hi,mid)]
                out['report_probability_lower'][n,j,endpoint]=min(probs[:2])
                out['report_probability_upper'][n,j,endpoint]=max(probs[:2])
                out['report_possible'][n,j,endpoint]=max(probs[:2])>0
                for k,v in zip(('lower_point_possible','upper_point_possible','midpoint_possible'),probs):out[k][n,j,endpoint]=v>0
            for metric in METRICS:
                distance=float(parent['independent_vs_copy_'+metric][n])
                out['old_endpoint_width_'+metric][n,j]=(high_copy-low_copy)*distance
                out['old_endpoint_midpoint_worst_'+metric][n,j]=max(abs(mid_copy-low_copy),abs(high_copy-mid_copy))*distance
    if set(saved)!=set(out):raise ValueError('raw fields')
    for key,value in out.items():
        if value.dtype.kind in 'biu':
            if not np.array_equal(value,saved[key]):raise ValueError(key)
        else:close(value,saved[key],key)
    stats={k:np.empty(shape) for k in ('supported_endpoints','impossible_endpoints','lower_point_unsupported_fraction','upper_point_unsupported_fraction')}
    for key in out:
        if key.startswith('old_endpoint_'):
            stats[key]=out[key]
            stats['equal_supported_endpoint_'+key]=np.empty(shape)
    for n in range(shape[0]):
        for j in range(shape[1]):
            supported=[e for e in range(8) if out['report_possible'][n,j,e]]
            count=len(supported)
            if count==0:raise ValueError('empty support')
            stats['supported_endpoints'][n,j]=count;stats['impossible_endpoints'][n,j]=8-count
            for point in ('lower','upper'):
                stats[point+'_point_unsupported_fraction'][n,j]=sum(not out[point+'_point_possible'][n,j,e] for e in supported)/count
            for key in out:
                if key.startswith('old_endpoint_'):stats['equal_supported_endpoint_'+key][n,j]=out[key][n,j]/count
    return stats


def controls():
    p=.2; a=.5; copy=a/(a+(1-a)*p)
    return {'live:copy_fraction_changes_segment':copy>a,
            'placebo:constant_future_direction_zero_width':(1.-0.)*0.==0.,
            'positive:copy_only_excludes_other_endpoints':(1.-1.)*.8==0.}


def rowkey(row):
    return tuple(row[k] for k in ('lineage','evidence','length','checkpoint')+IDENTITY)+(tuple(row['interval']),)


def run(root,plan,pulse):
    cfg=plan['design']; checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for name,digest in cfg['input_files'].items():
        if file_digest(root/'inputs'/name)!=digest:raise ValueError('input binding')
    original=root/'inputs/original'; parent=root/'inputs/parent'
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('target plan')
    design=read(original/'PLAN.json')['design']
    if design['intervals']!=[list(x) for x in INTERVALS]:raise ValueError('interval roster')
    if file_digest(parent/'PLAN.json')!=design['parent_plan_sha256'] or read(parent/'COMPLETE.json')['plan_sha256']!=design['parent_plan_sha256']:raise ValueError('parent plan')
    if not read(parent/'FINAL_REVIEW.json')['numerical_acceptance']:raise ValueError('parent acceptance')
    recorded=json.loads(gzip.decompress((original/'raw/interval_summary_points.json.gz').read_bytes()))
    keyed={rowkey(r):r for r in recorded}
    if len(keyed)!=len(recorded):raise ValueError('duplicate row')
    timing=[json.loads(s) for s in (original/'TIMING.jsonl').read_text().splitlines()]
    timed={(r['lineage'],r['evidence'],r['length'],r['checkpoint']):r for r in timing}
    if len(timed)!=len(timing):raise ValueError('duplicate timing')
    rows=[];strata=defaultdict(list);used=set();used_times=set();sources=0
    for path in sorted((parent/'evaluator').glob('*-bindings.json')):
        prefix=path.stem[:-len('-bindings')];lineage,evidence,length,checkpoint=prefix.split('-')
        binding=read(path);acc=defaultdict(lambda:defaultdict(list));seen=[]
        files=sorted((parent/'raw').glob(prefix+'-*_points.npz'))
        if not files:raise ValueError('missing factors')
        for file in files:
            pulse(phase='independent-provenance-interval-review',lineage=int(lineage),evidence=evidence,checkpoint=int(checkpoint))
            with np.load(file,allow_pickle=False) as z:base={k:z[k] for k in z.files}
            with np.load(original/'raw'/file.name,allow_pickle=False) as z:saved={k:z[k] for k in z.files}
            values=reconstruct(base,saved);used.add(file.name)
            for index,source in enumerate(base['source_rows']):
                seen.append(source.tolist())
                for key,value in values.items():acc[int(source[0])][key].append(value[index])
        if seen!=[r[:4] for r in binding['sources']]:raise ValueError('source inventory')
        for row,r in enumerate(binding['rows']):
            if not acc[row] or any(len(v)!=r['report_sources'] for v in acc[row].values()):raise ValueError('source denominator')
            for j,interval in enumerate(INTERVALS):
                result=dict(lineage=int(lineage),evidence=evidence,length=int(length),checkpoint=int(checkpoint),interval=list(interval),**{k:r[k] for k in IDENTITY+('stream','report_sources')})
                for key,value in acc[row].items():result[key]=math.fsum(float(v[j]) for v in value)/r['report_sources']
                old=keyed.pop(rowkey(result))
                if set(old)!=set(result):raise ValueError('row fields')
                for key,value in result.items():
                    if key in acc[row]:close(value,old[key],'summary '+key)
                    elif value!=old[key]:raise ValueError('row identity')
                rows.append(result);strata[(int(lineage),evidence,int(length),int(checkpoint),r['draw'],interval)].append(result)
        tkey=(int(lineage),evidence,int(length),int(checkpoint));t=timed[tkey];used_times.add(tkey)
        if t['sources']!=len(seen) or t['posterior_rows']!=len(binding['rows']) or not math.isfinite(t['cpu_seconds']) or t['cpu_seconds']<0:raise ValueError('timing population')
        sources+=len(seen)
    if keyed or used_times!=set(timed):raise ValueError('row/timing coverage')
    if used!={p.name for p in (original/'raw').glob('*.npz')} or used!={p.name for p in (parent/'raw').glob('*.npz')}:raise ValueError('raw coverage')
    summary=read(original/'SUMMARY.json');prior=read(parent/'SUMMARY.json')
    for key,value in dict(sources=sources,rows=len(rows),posterior_rows=len(rows)//4,source_interval_endpoint_queries=sources*32,intervals=[list(x) for x in INTERVALS],unavailable_checkpoints=prior['unavailable_checkpoints']).items():
        if summary[key]!=value:raise ValueError('summary '+key)
    if sources!=prior['sources'] or len(rows)//4!=prior['posterior_rows']:raise ValueError('parent population')
    grouped=[]
    metric_keys=tuple(acc[0])
    for key,values in sorted(strata.items()):
        if len(values)!=128:raise ValueError('stratum denominator')
        if len({tuple(r[k] for k in IDENTITY) for r in values})!=128:raise ValueError('stratum roster')
        grouped.append(dict(zip(('lineage','evidence','length','checkpoint','draw','interval'),key),rows=128,**{k:math.fsum(r[k] for r in values)/128 for k in metric_keys}))
    (root/'reconstructed').mkdir(exist_ok=True)
    (root/'reconstructed/summary_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'PAIRED_STRATA.json',grouped)
    write(root/'TIMING_REVIEW.json',dict(batches=len(timing),sources=sources,cpu_seconds=math.fsum(r['cpu_seconds'] for r in timing),scope='producer scalar factors,serialization and source accumulation;parent loading excluded;amortization not latency'))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='scalar interval factors independently reconstructed from accepted all-future signed directions;coordinate envelopes form one posterior segment,not arbitrary rectangular combinations;equal endpoint weights are descriptive counts'))
    return dict(passed=True,controls=checks,sources=sources,rows=len(rows),posterior_rows=len(rows)//4,strata=len(grouped),raw_batches=len(used),source_interval_endpoint_queries=sources*32,numerical_acceptance=False,scope='all scalar bounds,probability envelopes,masks,worst point discrepancies and summaries reconstructed;accepted parent supplies independently verified all-future directions;separate original-row regroup and event adjudication remain required')
