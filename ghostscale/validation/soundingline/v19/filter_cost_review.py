"""Independent retained accounting reconstruction; no producer arithmetic or filters."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import gzip
import hashlib
import json
import math
import statistics

ARMS = ('static', 'reset-16', 'known-time-type', 'unknown-time', 'unknown-time-type')
METRICS = ('endpoint_loss', 'state_loss', 'true_state_mass', 'coverage90', 'size90')
AXES = ('lineage', 'draw', 'length', 'kind', 'switched', 'quarter', 'duplicates', 'step', 'arm')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def zipped(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def equal(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError('field roster differs')
    for k, v in expected.items():
        if isinstance(v, float):
            if not math.isfinite(actual[k]) or abs(actual[k]-v) > 1e-12:
                raise ValueError('numerical disagreement: '+k)
        elif actual[k] != v:
            raise ValueError('identity disagreement: '+k)


def reconstruct(stream, step, arm):
    observed = {}; unique = []
    for t, row in enumerate(stream['observations'][:step], 1):
        if row['step'] != t:
            raise ValueError('stream order')
        key = row['source_id']; value = (row['source_step'], row['context'], row['endpoint'])
        if key in observed and observed[key] != value:
            raise ValueError('source conflict')
        if key not in observed:
            observed[key] = value; unique.append(row)
    retained = unique[-16:] if arm == 'reset-16' else unique
    hs = [('none', 0, m) for m in range(16)]
    if arm not in ('static', 'reset-16'):
        kinds = ('purpose', 'skill') if arm == 'unknown-time-type' else (stream['kind'],)
        times = (stream.get('switch_at', stream['length']//2),) if arm == 'known-time-type' else tuple(range(8, stream['length']-7))
        hs.extend(product(kinds, times, range(16)))
    h = len(hs)
    encoded = json.dumps(retained, sort_keys=True, separators=(',', ':')).encode()
    values = dict(hypothesis_weights=h, weight_bytes=h*8, current_maker_weights=16,
        current_maker_bytes=128, supplied_sources=len(unique), retained_sources=len(retained),
        retained_source_json_bytes=len(encoded), likelihood_factor_applications=len(retained)*h,
        projection_contributions=0 if arm in ('static', 'reset-16') else h,
        normalization_weights=h, endpoint_multiplications_per_context=16*8,
        endpoint_additions_per_context=15*8, endpoint_law_bytes=16*4*8*8,
        checkpoint_forecast_bytes=4*8*8)
    return values, [list(x) for x in hs]


def verify(packet):
    plan = read(packet/'PLAN.json'); done = read(packet/'COMPLETE.json')
    assert sha(packet/'PLAN.json') == done['plan_sha256']
    assert sha(packet/'SOURCE.zip') == plan['source_archive_sha256']
    for n, h in done['files'].items():
        assert sha(packet/n) == h, n
    for n, h in plan['design']['input_files'].items():
        assert sha(packet/'inputs'/n) == h, n
    actual_cells = read(packet/'COST_CELLS.json')
    cell_index = {(r['packet'], *(r[k] for k in AXES)): r for r in actual_cells}
    assert len(cell_index) == len(actual_cells)
    actual_times = read(packet/'TIMING_SUMMARY.json')
    tax = ('packet','execution','length','kind','switched','quarter','duplicates','arm')
    time_index = {tuple(r[k] for k in tax): r for r in actual_times}
    assert len(time_index) == len(actual_times)
    rows_total = cells_total = calls_total = 0; scenarios = set(); observations = 0
    timing_display = []
    for item in plan['design']['packets']:
        name = item['name']; parent = packet/'inputs'/name
        pp = read(parent/'PLAN.json'); pd = read(parent/'COMPLETE.json')
        assert sha(parent/'PLAN.json') == pd['plan_sha256']
        assert sha(parent/'SOURCE.zip') == pp['source_archive_sha256']
        for n in ('SUMMARY.json',): assert sha(parent/n) == pd['files'][n]
        source_cells = read(parent/'SUMMARY.json')['cells']
        source_index = {tuple(r.get(k, 2 if k=='quarter' else True) for k in AXES):r for r in source_cells}
        produced = zipped(packet/(name+'_points.json.gz')); index = defaultdict(list)
        for r in produced: index[r['lineage']].append(r)
        groups = defaultdict(list); streams = {}
        for lineage in pp['design']['lineages']:
            for n in (f'raw/{lineage}-observations_points.json.gz',f'raw/{lineage}-forecasts_points.json.gz',f'evaluator/{lineage}-joint-map.json'):
                assert sha(parent/n) == pd['files'][n]
            ss = zipped(parent/'raw'/f'{lineage}-observations_points.json.gz'); streams[lineage] = ss
            rr = zipped(parent/'raw'/f'{lineage}-forecasts_points.json.gz')
            mp = read(parent/'evaluator'/f'{lineage}-joint-map.json')
            expected = {(i,t,a) for i,s in enumerate(ss) for t in ((8,16,17,20,32) if s['length']==32 else (16,32,64,65,80,128)) for a in ARMS}
            assert len(rr)==len(expected) and {(r['stream'],r['step'],r['arm']) for r in rr}==expected
            assert len(index[lineage]) == len(rr)
            observations += sum(len(s['observations']) for s in ss)
            cache = {}
            for raw, result in zip(rr,index[lineage]):
                s = ss[raw['stream']]; count, hs = reconstruct(s,raw['step'],raw['arm'])
                key = raw['joint_array']
                if key not in cache:
                    assert mp[key]['hypotheses']==hs; cache[key]=True
                assert mp[key]['rows'][raw['joint_row']]==[raw['stream'],raw['step']]
                meta = dict(lineage=lineage,draw=s['draw'],maker=s['maker'],length=s['length'],kind=s['kind'],
                    switched=s.get('switched',True),quarter=s.get('quarter',2),duplicates=s['duplicates'],step=raw['step'],arm=raw['arm'])
                reconstructed = dict(meta,**count,**{m:raw[m] for m in METRICS})
                equal(result,reconstructed)
                groups[tuple(meta[k] for k in AXES)].append(reconstructed)
                scenarios.add((len(hs),count['retained_sources'],count['projection_contributions']))
                rows_total += 1
        for key, records in groups.items():
            assert len(records)==16 and {r['maker'] for r in records}==set(range(16))
            expected = dict(packet=name,**dict(zip(AXES,key)),makers=16,
                **{k:statistics.mean(r[k] for r in records) for k in (*METRICS,*count)})
            equal(cell_index.pop((name,*key)),expected)
            original = source_index.pop(key)
            assert all(abs(expected[k]-original[k])<1e-12 for k in METRICS)
            cells_total += 1
        assert not source_index
        for execution in ('original','replay','portable'):
            timing_file = parent/'timings'/f'{execution}.jsonl'
            td = read(parent/'timings'/f'{execution}-COMPLETE.json')
            assert td['plan_sha256']==pd['plan_sha256'] and td['files']==pd['files']
            assert sha(timing_file)==td['execution_measurements']['TIMING.jsonl']
            rows = [json.loads(x) for x in timing_file.read_text(encoding='utf-8').splitlines()]
            expected = {(l,i,a) for l,ss in streams.items() for i in range(len(ss)) for a in ARMS}
            collected = defaultdict(list)
            for r in rows:
                identity=(r['lineage'],r['stream'],r['arm'])
                assert identity in expected; expected.remove(identity)
                v=r['cpu_seconds']; assert math.isfinite(v) and v>=0
                s=streams[r['lineage']][r['stream']]
                key=(name,execution,s['length'],s['kind'],s.get('switched',True),s.get('quarter',2),s['duplicates'],r['arm'])
                collected[key].append(v)
            assert not expected
            for key, vv in collected.items():
                expected=dict(zip(tax,key),calls=len(vv),total_cpu_seconds=math.fsum(vv),mean_cpu_seconds=statistics.mean(vv),
                    median_cpu_seconds=statistics.median(vv),minimum_cpu_seconds=min(vv),maximum_cpu_seconds=max(vv))
                equal(time_index.pop(key),expected)
            calls_total+=len(rows)
            timing_display.append(dict(packet=name,execution=execution,calls=len(rows),zero_duration_calls=sum(r['cpu_seconds']==0 for r in rows),
                total_cpu_seconds=math.fsum(r['cpu_seconds'] for r in rows),
                minimum_positive_cpu_seconds=min(r['cpu_seconds'] for r in rows if r['cpu_seconds']>0)))
    assert not cell_index and not time_index
    surfaces=read(packet/'SYMBOLIC_SCENARIOS.json')['cells']
    for cell in surfaces:
        h,s,p=(cell[k] for k in ('hypothesis_weights','retained_sources','projection_contributions'))
        assert (h,s,p) in scenarios; scenarios.remove((h,s,p))
        expected={(q,r) for q in (1,4,16,64) for r in (.25,1.,4.)}
        for row in cell['scenarios']:
            q,r=row['queries'],row['query_to_update_unit_ratio'];expected.remove((q,r))
            u=h*s+p
            equal(row,dict(queries=q,query_to_update_unit_ratio=r,cached_units=float(u+q*r*128),reread_units=float(q*u+q*r*128),
                saved_units=(q-1)*u,equality_query_count=1,first_strictly_cheaper_integer_queries=2))
        assert not expected
    assert not scenarios
    summary=read(packet/'SUMMARY.json')
    assert (rows_total,cells_total,calls_total,len(actual_times))==(summary['rows'],summary['cells'],summary['timing_calls'],summary['timing_cells'])
    return dict(passed=True,rows=rows_total,cells=cells_total,timing_calls=calls_total,timing_cells=len(actual_times),
        scenario_cells=len(surfaces),observations_counting_parent_reuse=observations,timing_resolution=timing_display,
        scope='Independent counts, source grouping, retained means, native timing coverage and symbolic arithmetic; no new forecasts or empirical query benchmark')


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=verify(a.packet)
    a.output.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
    print(json.dumps(result))
