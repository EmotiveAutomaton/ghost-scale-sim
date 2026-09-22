"""Retained filter work/storage accounting. Never executes scientific forecasts."""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import statistics
from ..v18_3.io import canonical, file_digest, read, write

ARMS = ('static', 'reset-16', 'known-time-type', 'unknown-time', 'unknown-time-type')
METRICS = ('endpoint_loss', 'state_loss', 'true_state_mass', 'coverage90', 'size90')


def hypotheses(arm, length, kind, switch_at=None):
    if arm not in ARMS or length not in (32, 128) or kind not in ('purpose', 'skill'):
        raise ValueError('outside frozen roster')
    hs = [('none', 0, m) for m in range(16)]
    if arm in ('static', 'reset-16'):
        return hs
    times = [length//2 if switch_at is None else switch_at] if arm == 'known-time-type' else range(8, length-7)
    kinds = ('purpose', 'skill') if arm == 'unknown-time-type' else (kind,)
    return hs + [(k, t, m) for k, t, m in product(kinds, times, range(16))]


def sources(rows):
    seen = {}; unique = []
    for row in rows:
        identity = row['source_step'], row['context'], row['endpoint']
        if row['source_id'] in seen:
            if seen[row['source_id']] != identity:
                raise ValueError('conflicting source')
        else:
            seen[row['source_id']] = identity; unique.append(row)
    return unique


def counts(observations, arm, length, kind, step, switch_at=None):
    if [r['step'] for r in observations] != list(range(1, len(observations)+1)) or not 1 <= step <= len(observations):
        raise ValueError('invalid stream/checkpoint')
    supplied = sources(observations[:step])
    retained = supplied[-16:] if arm == 'reset-16' else supplied
    if arm not in ARMS or length not in (32,128) or kind not in ('purpose','skill'):
        raise ValueError('outside frozen roster')
    h = 16 if arm in ('static','reset-16') else (32 if arm=='known-time-type' else 16*(1+(length-15)*(2 if arm=='unknown-time-type' else 1)))
    projection = h if arm not in ('static', 'reset-16') else 0
    return dict(hypothesis_weights=h, weight_bytes=8*h, current_maker_weights=16,
        current_maker_bytes=128, supplied_sources=len(supplied), retained_sources=len(retained),
        retained_source_json_bytes=len(canonical(retained)), likelihood_factor_applications=h*len(retained),
        projection_contributions=projection, normalization_weights=h,
        endpoint_multiplications_per_context=128, endpoint_additions_per_context=120,
        endpoint_law_bytes=4096, checkpoint_forecast_bytes=256)


def scenarios(count):
    # Abstract unit accounting: one factor application and one projection contribution
    # each cost one update unit. An endpoint multiplication costs r units. This is
    # not a CPU model, and omits normalization, source parsing and additions.
    update = count['likelihood_factor_applications'] + count['projection_contributions']
    return [dict(queries=q, query_to_update_unit_ratio=r,
        cached_units=update+q*r*128, reread_units=q*(update+r*128),
        saved_units=(q-1)*update, equality_query_count=1,
        first_strictly_cheaper_integer_queries=2)
        for q, r in product((1, 4, 16, 64), (.25, 1., 4.))]


def timing_records(rows, streams):
    expected = {(l, i, a) for l, ss in streams.items() for i in range(len(ss)) for a in ARMS}
    seen = set(); groups = defaultdict(list)
    for r in rows:
        key = r['lineage'], r['stream'], r['arm']
        if key not in expected or key in seen or not math.isfinite(r['cpu_seconds']) or r['cpu_seconds'] < 0:
            raise ValueError('timing identity or duration')
        seen.add(key); s = streams[r['lineage']][r['stream']]
        group = (s['length'], s['kind'], s.get('switched', True), s.get('quarter', 2), s['duplicates'], r['arm'])
        groups[group].append(r['cpu_seconds'])
    if seen != expected:
        raise ValueError('missing timing records')
    return [dict(zip(('length','kind','switched','quarter','duplicates','arm'), k),
        calls=len(v), total_cpu_seconds=math.fsum(v), mean_cpu_seconds=statistics.mean(v),
        median_cpu_seconds=statistics.median(v), minimum_cpu_seconds=min(v), maximum_cpu_seconds=max(v))
        for k, v in sorted(groups.items())]


def controls():
    a = dict(step=1, source_step=1, source_id='a', context=0, endpoint=0)
    duplicate = [a, dict(a, step=2)]
    distinct = [a, dict(a, step=2, source_id='b')]
    old = [dict(a, step=i, source_step=i, source_id=str(i)) for i in range(1, 33)]
    x = counts(duplicate, 'unknown-time-type', 32, 'purpose', 2)
    y = counts(distinct, 'unknown-time-type', 32, 'purpose', 2)
    return {'live:distinct_sources_increase_factor_work':y['likelihood_factor_applications']==2*x['likelihood_factor_applications'],
        'placebo:renamed_same_equivalence_same_work':counts([dict(r,source_id='renamed') for r in duplicate], 'static',32,'purpose',2)['likelihood_factor_applications']==16,
        'positive:static_sixteen':counts(old,'static',32,'purpose',32)['hypothesis_weights']==16,
        'positive:unknown_roster':x['hypothesis_weights']==16*(1+2*17),
        'positive:long_unknown_roster':len(hypotheses('unknown-time-type',128,'skill'))==16*(1+2*113),
        'positive:reset_limit':counts(old,'reset-16',32,'purpose',32)['retained_sources']==16,
        'positive:one_query_identity':all(r['saved_units']==0 for r in scenarios(x) if r['queries']==1)}


def zipped(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def run(root, plan, pulse):
    cfg=plan['design']; checks=controls(); write(root/'CONTROLS.json',checks)
    if not all(checks.values()): raise ValueError('cost controls failed')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h: raise ValueError('retained input changed')
    summaries=[]; timing=[]; scenario_rows={}; total_rows=0
    for packet in cfg['packets']:
        name=packet['name']; parent=root/'inputs'/name
        original_plan=read(parent/'PLAN.json'); done=read(parent/'COMPLETE.json')
        if file_digest(parent/'PLAN.json')!=done['plan_sha256']: raise ValueError('parent plan')
        streams={}; raw=[]; acc=defaultdict(list)
        for lineage in original_plan['design']['lineages']:
            pulse(phase='retained-cost-accounting',packet=name,lineage=lineage)
            ss=zipped(parent/'raw'/f'{lineage}-observations_points.json.gz'); streams[lineage]=ss
            rr=zipped(parent/'raw'/f'{lineage}-forecasts_points.json.gz')
            mapping=read(parent/'evaluator'/f'{lineage}-joint-map.json')
            seen=set(); roster_checked=set()
            for row in rr:
                key=row['stream'],row['step'],row['arm']
                if key in seen: raise ValueError('duplicate forecast row')
                seen.add(key); s=ss[row['stream']]
                switch=s.get('switch_at',s['length']//2)
                if row['joint_array'] not in roster_checked:
                    hs=hypotheses(row['arm'],s['length'],s['kind'],switch)
                    actual_hs=mapping[row['joint_array']]['hypotheses']
                    if [list(x) for x in hs]!=actual_hs: raise ValueError('hypothesis roster differs')
                    roster_checked.add(row['joint_array'])
                if mapping[row['joint_array']]['rows'][row['joint_row']]!=[row['stream'],row['step']]: raise ValueError('map row differs')
                count=counts(s['observations'],row['arm'],s['length'],s['kind'],row['step'],switch)
                metadata=dict(lineage=lineage,draw=s['draw'],maker=s['maker'],length=s['length'],kind=s['kind'],
                    switched=s.get('switched',True),quarter=s.get('quarter',2),duplicates=s['duplicates'],step=row['step'],arm=row['arm'])
                result=dict(metadata,**count,**{m:row[m] for m in METRICS});raw.append(result)
                axes=('draw','length','kind','switched','quarter','duplicates','step','arm')
                acc[lineage,*(metadata[k] for k in axes)].append(result)
                structural=tuple(count[k] for k in ('hypothesis_weights','retained_sources','projection_contributions'))
                scenario_rows[structural]=dict(hypothesis_weights=structural[0],retained_sources=structural[1],
                    projection_contributions=structural[2],scenarios=scenarios(count))
            if len(rr)!=sum(len([8,16,17,20,32] if s['length']==32 else [16,32,64,65,80,128])*5 for s in ss):
                raise ValueError('forecast denominator')
        for key, rows in sorted(acc.items()):
            if len(rows)!=16 or {r['maker'] for r in rows}!=set(range(16)): raise ValueError('maker roster')
            names=('lineage','draw','length','kind','switched','quarter','duplicates','step','arm')
            summaries.append(dict(packet=name,**dict(zip(names,key)),makers=16,
                **{m:math.fsum(r[m] for r in rows)/16 for m in (*METRICS,*count.keys())}))
        parent_cells=read(parent/'SUMMARY.json')['cells']
        def identity(r):return tuple(r[k] for k in ('lineage','draw','length','kind','duplicates','step','arm'))+(r.get('quarter',2),r.get('switched',True))
        pindex={identity(r):r for r in parent_cells}
        selected=[r for r in summaries if r['packet']==name]
        if len(pindex)!=len(parent_cells) or len(selected)!=len(pindex):raise ValueError('summary denominator')
        for r in selected:
            if any(abs(r[m]-pindex[identity(r)][m])>1e-12 for m in METRICS):raise ValueError('retained score changed')
        (root/f'{name}_points.json.gz').write_bytes(gzip.compress(canonical(raw),mtime=0));total_rows+=len(raw)
        for execution in ('original','replay','portable'):
            path=parent/'timings'/f'{execution}.jsonl'
            records=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
            timing.extend(dict(packet=name,execution=execution,**r) for r in timing_records(records,streams))
    write(root/'COST_CELLS.json',summaries)
    write(root/'TIMING_SUMMARY.json',timing)
    write(root/'SYMBOLIC_SCENARIOS.json',dict(role='accounting scenarios, not measured break-even; same retained law and evidence; cached posterior versus full prefix recomputation for each query',
        omitted_costs='normalization, source parsing, additions, allocation and hardware; weight bytes exclude prior/indices/laws/temporary buffers',
        cells=[scenario_rows[k] for k in sorted(scenario_rows)]))
    return dict(controls=checks,rows=total_rows,cells=len(summaries),timing_cells=len(timing),
        timing_calls=sum(r['calls'] for r in timing),packets=[p['name'] for p in cfg['packets']],fits=0,
        scope='retained supplied-law filter cost; measured whole-call timing kept separate from symbolic work and storage; no learned memory or process correspondence')
