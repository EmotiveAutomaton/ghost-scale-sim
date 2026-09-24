"""Independent enumeration and scalar kernels for retrospective certificates.

No producer, transition-table, or producer aggregation code is imported. Native
comparisons are reconstructed by complete integer multiplicities; scalar small
fixtures additionally expand every hypothesis/report/future-coordinate case.
"""
from collections import Counter, defaultdict
from itertools import combinations
import math
import numpy as np
from ..v18_3.io import read, write, file_digest

TOL = 1e-12


def state_at(h, t):
    kind, change, initial = h
    if kind not in ('none', 'purpose', 'skill') or not 0 <= initial < 16:
        raise ValueError('hypothesis')
    return initial ^ ({'purpose': 8, 'skill': 4}.get(kind, 0) if t > change else 0)


def classify(rows, width):
    keys = sorted(set(rows)); lookup = {key: i for i, key in enumerate(keys)}
    ids = [lookup[row] for row in rows]; count = Counter(ids)
    return dict(classes=[list(k) for k in keys], membership=ids,
                multiplicity=[count[i] for i in range(len(keys))])


def reconstruct(spec):
    hs = spec['hypotheses']; cp = spec['checkpoint']; horizon = spec['length']
    if not 1 <= cp <= horizon or not hs: raise ValueError('schedule bounds')
    paths = [tuple(state_at(h, t) for t in range(cp, horizon+1)) for h in hs]
    sig = [tuple(s) for s in spec['signatures']]; mapping = spec['membership']
    if len(mapping) != len(hs) or len(set(sig)) != len(sig): raise ValueError('group coverage')
    if set(mapping) != set(range(len(sig))): raise ValueError('membership range')
    if any(sig[g] != path for g, path in zip(mapping, paths)): raise ValueError('schedule mapping')
    members = defaultdict(list)
    for i, g in enumerate(mapping): members[g].append(i)
    past = [[state_at(h, t) for h in hs] for t in range(1, cp+1)]
    pairs = []; groups = []
    for g in range(len(sig)):
        for i, j in combinations(members[g], 2):
            if any(row[i] != row[j] for row in past): pairs.append((i,j)); groups.append(g)
    if any(len(set(sig[g])) != 1 for g in groups): raise ValueError('merged group not stationary')
    masks = [sum(2**s for s in set(path)) for path in sig]
    blocks = []; pair_maps = []; third_maps = []
    for t, row in enumerate(past, 1):
        pb = classify([(g,row[i],row[j]) for (i,j),g in zip(pairs,groups)], 3)
        tb = classify([(g,row[i]) for i,g in enumerate(mapping)], 2)
        pair_maps.append(pb); third_maps.append(tb)
        # Independently count EVERY hypothesis in each outside group, rather
        # than recycling the producer's third-class multiplicities.
        by_group = defaultdict(Counter)
        for i,g in enumerate(mapping): by_group[g][(row[i],masks[g])] += 1
        total = sum(by_group.values(), Counter())
        for (g,a,b), pair_count in zip(pb['classes'],pb['multiplicity']):
            outside = total - by_group[g]
            if sum(outside.values()) != len(hs)-len(members[g]): raise ValueError('third denominator')
            for (z,mask), count in sorted(outside.items()):
                blocks.append((t,g,a,b,z,mask,pair_count*count))
    expected = cp * sum(len(hs)-len(members[g]) for g in groups)
    if sum(b[-1] for b in blocks) != expected: raise ValueError('comparison denominator')
    arrays = dict(pairs=np.array(pairs,np.int32).reshape(-1,2),pair_groups=np.array(groups,np.int32),
                  past=np.array(past,np.int32),schedules=np.array(sig,np.int32),mapping=np.array(mapping,np.int32),
                  masks=np.array(masks,np.int32),blocks=np.array(blocks,np.int64).reshape(-1,7))
    meta = dict(pair_classes_by_time=pair_maps,third_classes_by_time=third_maps,pair_count=len(pairs),
                pair_third_time_comparisons=expected,report_comparisons=32*expected)
    return arrays, meta


def check_structure(arrays, meta, retained_arrays, retained_meta):
    if set(arrays) != set(retained_arrays): raise ValueError('structure field roster')
    for key, value in arrays.items():
        other = retained_arrays[key]
        if value.shape != other.shape or not np.array_equal(value,other): raise ValueError('structure '+key)
    for key,value in meta.items():
        if value != retained_meta[key]: raise ValueError('bindings '+key)


def scalar_kernel(law):
    law=np.asarray(law,float)
    if law.shape != (16,4,8) or not np.isfinite(law).all() or (law < 0).any(): raise ValueError('endpoint law')
    if any(abs(math.fsum(row)-1)>TOL for row in law.reshape(-1,8)): raise ValueError('endpoint mass')
    shape=(16,16,16,4,8)
    result={k:np.empty(shape,bool if k=='feasible' else float) for k in
            ('first_group_weight','second_group_weight','first_report_probability','second_report_probability','feasible','quotient_distance')}
    algebra_error=0.
    for a in range(16):
        for b in range(16):
            for z in range(16):
                for c in range(4):
                    for e in range(8):
                        x,y,v=(float(law[s,c,e]) for s in (a,b,z)); d1=x+v;d2=y+v
                        q1=x/d1 if d1 else math.nan;q2=y/d2 if d2 else math.nan
                        valid=d1>0 and d2>0;delta=abs(q1-q2) if valid else 0.
                        if valid:
                            alternate=(v/d1)*(abs(x-y)/d2)
                            algebra_error=max(algebra_error,abs(delta-alternate))
                        index=(a,b,z,c,e)
                        for key,val in zip(result,(q1,q2,.5*d1,.5*d2,valid,delta)):result[key][index]=val
    if algebra_error>TOL: raise ValueError('independent Bayes algebra')
    return result,algebra_error


def check_kernel(expected, actual):
    if set(expected)!=set(actual):raise ValueError('kernel field roster')
    for key,value in expected.items():
        if value.shape!=actual[key].shape or not np.array_equal(value,actual[key],equal_nan=True):
            raise ValueError('kernel '+key)


def reconstruct_row(arrays, law, kernel):
    # Collapse across past times only AFTER their original pair/third bindings
    # have been independently checked. All integer multiplicities remain exact.
    weights=Counter()
    for _,g,a,b,z,mask,mult in arrays['blocks'].tolist():weights[(g,a,b,z,mask)]+=mult
    row=dict(report_comparisons=0,both_impossible=0,one_impossible=0,both_possible=0,
             exact_update_witnesses=0,tolerance_update_witnesses=0,
             exact_forecast_witnesses=0,tolerance_forecast_witnesses=0,
             max_quotient_distance=0.,max_forecast_coordinate_difference=0.)
    scales={}
    for g,_,_,_,mask in weights:
        key=(g,mask)
        if key not in scales:
            s=int(arrays['schedules'][g,0])
            scales[key]=max(abs(float(law[s,c,e])-float(law[v,c,e])) for v in range(16)
                            if mask&(2**v) for c in range(4) for e in range(8))
    for (g,a,b,z,mask),count in weights.items():
        ix=(a,b,z);p=kernel['first_report_probability'][ix];q=kernel['second_report_probability'][ix]
        valid=kernel['feasible'][ix];delta=kernel['quotient_distance'][ix];forecast=delta*scales[g,mask]
        row['report_comparisons']+=32*count
        row['both_impossible']+=int(np.count_nonzero((p==0)&(q==0)))*count
        row['one_impossible']+=int(np.count_nonzero((p==0)^(q==0)))*count
        row['both_possible']+=int(np.count_nonzero(valid))*count
        for name,values in (('update',delta),('forecast',forecast)):
            row['exact_'+name+'_witnesses']+=int(np.count_nonzero(valid&(values>0)))*count
            row['tolerance_'+name+'_witnesses']+=int(np.count_nonzero(valid&(values>TOL)))*count
        row['max_quotient_distance']=max(row['max_quotient_distance'],float(delta.max()))
        row['max_forecast_coordinate_difference']=max(row['max_forecast_coordinate_difference'],float(forecast.max()))
    if row['report_comparisons']!=sum(row[k] for k in ('both_impossible','one_impossible','both_possible')):
        raise ValueError('report denominator')
    return row


def check_row(expected, actual):
    for key,value in expected.items():
        if key.startswith('max_'):
            if not math.isfinite(actual[key]) or abs(value-actual[key])>TOL:raise ValueError('summary '+key)
        elif value!=actual[key]:raise ValueError('summary '+key)


def controls():
    spec=dict(hypotheses=[['none',0,0],['purpose',1,8],['none',0,1]],length=3,checkpoint=2,
              signatures=[[0,0],[1,1]],membership=[0,0,1])
    arrays,meta=reconstruct(spec);law=np.full((16,4,8),.125)
    flat=reconstruct_row(arrays,law,scalar_kernel(law)[0])
    law[0]=[.75,.25,0,0,0,0,0,0];law[8]=[.25,.75,0,0,0,0,0,0];law[1]=[.5,.5,0,0,0,0,0,0]
    changed=reconstruct_row(arrays,law,scalar_kernel(law)[0])
    return {'live:past_report_changes_future_quotient':changed['tolerance_forecast_witnesses']>0,
            'placebo:uniform_law':flat['exact_forecast_witnesses']==flat['exact_update_witnesses']==0,
            'positive:complete_denominator':meta['report_comparisons']==64,
            'positive:impossible_reports':changed['both_impossible']>0}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('review controls')
    for name,digest in cfg['input_files'].items():
        if file_digest(root/'inputs'/name)!=digest:raise ValueError('input binding')
    original=root/'inputs/original';parent=root/'inputs/parent'
    original_plan=read(original/'PLAN.json');design=original_plan['design']
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('target plan')
    specs=read(parent/'SCHEDULES.json')
    if specs!=read(original/'evaluator/SCHEDULES.json'):raise ValueError('schedule input binding')
    expected={f'{h}-{cp}' for h in design['lengths'] for cp in design['checkpoints'] if cp<=h}
    if set(specs)!=expected:raise ValueError('complete structural roster')
    points=read(original/'retrospective_points.json');summary=read(original/'SUMMARY.json')
    if points!=summary['cells']:raise ValueError('summary points')
    keyed={(r['lineage'],r['length'],r['checkpoint']):r for r in points}
    identities={(law,s['length'],s['checkpoint']) for law in design['lineages'] for s in specs.values()}
    if set(keyed)!=identities or len(keyed)!=len(points):raise ValueError('summary roster')
    structures={};structure_checks=[];(root/'reconstructed').mkdir(exist_ok=True)
    for key,spec in specs.items():
        pulse(phase='independent-pair-third-enumeration',checkpoint=key)
        arrays,meta=reconstruct(spec)
        with np.load(original/'evaluator'/f'{key}-structure.npz',allow_pickle=False) as z:retained={k:z[k] for k in z.files}
        check_structure(arrays,meta,retained,read(original/'evaluator'/f'{key}-bindings.json'))
        np.savez_compressed(root/'reconstructed'/f'{key}-structure.npz',**arrays)
        write(root/'reconstructed'/f'{key}-bindings.json',meta)
        structures[key]=arrays;structure_checks.append(dict(key=key,pairs=meta['pair_count'],report_comparisons=meta['report_comparisons'],factor_blocks=len(arrays['blocks']),passed=True))
    rows=[];kernel_checks=[]
    for lineage in design['lineages']:
        pulse(phase='independent-scalar-Bayes-kernel',lineage=lineage)
        law=np.array(read(parent/f'{lineage}-law.json'),float);kernel,error=scalar_kernel(law)
        with np.load(original/'evaluator'/f'{lineage}-kernel.npz',allow_pickle=False) as z:check_kernel(kernel,{k:z[k] for k in z.files})
        np.savez_compressed(root/'reconstructed'/f'{lineage}-kernel.npz',**kernel)
        kernel_checks.append(dict(lineage=lineage,cells=16**3*32,algebra_error=error,passed=True))
        for key,spec in specs.items():
            pulse(phase='independent-complete-summary',lineage=lineage,checkpoint=key)
            row=dict(lineage=lineage,length=spec['length'],checkpoint=spec['checkpoint'],**reconstruct_row(structures[key],law,kernel))
            check_row(row,keyed[lineage,spec['length'],spec['checkpoint']]);rows.append(row)
    write(root/'RECONSTRUCTED_POINTS.json',rows);write(root/'STRUCTURE_CHECKS.json',structure_checks);write(root/'KERNEL_CHECKS.json',kernel_checks)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='complete independent pair/third memberships,kernels,all80law/checkpoint rows;full-simplex witnesses only;not reachable-posterior evidence'))
    return dict(passed=True,controls=checks,structural_cells=len(structures),lineages=len(design['lineages']),summary_rows=len(rows),
                report_comparisons=sum(r['report_comparisons'] for r in rows),max_algebra_error=max(r['algebra_error'] for r in kernel_checks),
                scope='independent full-simplex factor and scalar Bayes reconstruction;separate event-owned regroup and final adjudication required',
                numerical_acceptance=False,reachable_posterior_claim=False)
