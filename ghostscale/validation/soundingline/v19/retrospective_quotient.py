"""Complete full-simplex retrospective witnesses, with lossless factorization.

This is an evaluator certificate, not a learned decoder or reachable posterior.
Pair and third-hypothesis identities are retained; no witnesses are sampled.
"""
from itertools import combinations
import math
import numpy as np
from ..v18_3.io import read, write, file_digest
from .future_quotient import independent_schedule

TOLERANCE = 1e-12


def structure(spec):
    hs = spec['hypotheses']; checkpoint = spec['checkpoint']; horizon = spec['length']
    schedules = np.asarray(spec['signatures'], dtype=np.int32)
    mapping = np.asarray(spec['membership'], dtype=np.int32)
    if mapping.shape != (len(hs),) or schedules.ndim != 2 or schedules.shape[1] != horizon-checkpoint+1:
        raise ValueError('schedule shape')
    if np.any(mapping < 0) or np.any(mapping >= len(schedules)) or set(mapping) != set(range(len(schedules))):
        raise ValueError('membership range')
    if not np.array_equal(schedules[mapping], independent_schedule(hs, checkpoint, horizon)):
        raise ValueError('schedule mapping')
    if len(np.unique(schedules, axis=0)) != len(schedules): raise ValueError('duplicate groups')
    past = independent_schedule(hs, 1, checkpoint).T
    pairs = []; groups = []
    for g in range(len(schedules)):
        members = np.flatnonzero(mapping == g)
        for a,b in combinations(members.tolist(), 2):
            if np.any(past[:,a] != past[:,b]): pairs.append((a,b));groups.append(g)
    pairs = np.asarray(pairs, dtype=np.int32).reshape(-1,2)
    groups = np.asarray(groups, dtype=np.int32)
    # In this single-change roster, any merged future group is stationary.
    if any(np.any(schedules[g] != schedules[g,0]) for g in set(groups)):
        raise ValueError('nonstationary merged group needs a different factorization')
    masks = np.array([sum(1 << int(s) for s in set(row)) for row in schedules], dtype=np.int32)
    blocks = []; pair_bindings = []; third_bindings = []
    for ti, states in enumerate(past):
        # These two maps retain every original member of the implicit Cartesian
        # comparison. Numerics may collapse only equal likelihood/future-max types.
        if len(pairs):
            descriptors = np.column_stack((groups,states[pairs[:,0]],states[pairs[:,1]]))
            pc,pi,pm = np.unique(descriptors,axis=0,return_inverse=True,return_counts=True)
        else:
            pc=np.empty((0,3),np.int32);pi=np.empty(0,np.int32);pm=np.empty(0,np.int64)
        tc,ti_map,tm = np.unique(np.column_stack((mapping,states)),axis=0,return_inverse=True,return_counts=True)
        pair_bindings.append(dict(classes=pc.tolist(),membership=pi.tolist(),multiplicity=pm.tolist()))
        third_bindings.append(dict(classes=tc.tolist(),membership=ti_map.tolist(),multiplicity=tm.tolist()))
        for (g,a,b),pair_count in zip(pc,pm):
            other = tc[:,0] != g
            types = np.column_stack((tc[other,1],masks[tc[other,0]]))
            unique,inverse = np.unique(types,axis=0,return_inverse=True)
            mult = np.bincount(inverse,weights=tm[other],minlength=len(unique)).astype(np.int64)
            if int(mult.sum()) != len(hs)-int(np.sum(mapping==g)):raise ValueError('third coverage')
            for (z,mask),third_count in zip(unique,mult):
                blocks.append((ti+1,int(g),int(a),int(b),int(z),int(mask),int(pair_count)*int(third_count)))
    blocks=np.asarray(blocks,dtype=np.int64).reshape(-1,7)
    expected=checkpoint*sum(len(hs)-int(np.sum(mapping==g)) for g in groups)
    if int(blocks[:,6].sum()) != expected:raise ValueError('pair coverage')
    return dict(pairs=pairs,pair_groups=groups,past=past,schedules=schedules,mapping=mapping,masks=masks,blocks=blocks),dict(
        pair_classes_by_time=pair_bindings,third_classes_by_time=third_bindings,
        pair_count=len(pairs),pair_third_time_comparisons=int(expected),
        report_comparisons=int(expected)*32,
        scope='all unordered divergent-past within-group pairs;every third hypothesis outside that group;all past steps and32reports')


def likelihood_kernel(law):
    law=np.asarray(law,dtype=np.float64)
    if law.shape!=(16,4,8) or not np.isfinite(law).all() or np.any(law<0) or np.any(law>1+TOLERANCE) or not np.allclose(law.sum(-1),1,rtol=0,atol=TOLERANCE):
        raise ValueError('endpoint law')
    a=law[:,None,None];b=law[None,:,None];z=law[None,None,:]
    da=np.broadcast_to(a+z,(16,16,16,4,8));db=np.broadcast_to(b+z,da.shape)
    qa=np.divide(np.broadcast_to(a,da.shape),da,out=np.full_like(da,np.nan),where=da>0)
    qb=np.divide(np.broadcast_to(b,db.shape),db,out=np.full_like(db,np.nan),where=db>0)
    valid=(da>0)&(db>0)
    return dict(first_group_weight=qa,second_group_weight=qb,
                first_report_probability=.5*da,second_report_probability=.5*db,
                feasible=valid,quotient_distance=np.where(valid,np.abs(qa-qb),0.))


def evaluate(arrays, law, kernel=None):
    kernel=likelihood_kernel(law) if kernel is None else kernel
    law=np.asarray(law,float);blocks=arrays['blocks'];sig=arrays['schedules']
    result=dict(report_comparisons=0,both_impossible=0,one_impossible=0,both_possible=0,
                exact_update_witnesses=0,tolerance_update_witnesses=0,
                exact_forecast_witnesses=0,tolerance_forecast_witnesses=0,
                max_quotient_distance=0.,max_forecast_coordinate_difference=0.)
    # Retain one primitive response maximum per future-state mask and current
    # group; the complete schedules and law reconstruct every future forecast.
    response={}
    for mask in np.unique(blocks[:,5]) if len(blocks) else []:
        states=[s for s in range(16) if int(mask)&(1<<s)]
        response[int(mask)]=np.max(np.abs(law[:,None]-law[states][None]),axis=(1,2,3))
    for start in range(0,len(blocks),4096):
        block=blocks[start:start+4096];g,a,b,z,mask,mult=(block[:,i] for i in range(1,7))
        first=kernel['first_report_probability'][a,b,z];second=kernel['second_report_probability'][a,b,z]
        valid=kernel['feasible'][a,b,z];delta=kernel['quotient_distance'][a,b,z]
        scale=np.array([response[int(m)][int(sig[x,0])] for x,m in zip(g,mask)])
        forecast=delta*scale[:,None,None]
        count=lambda v:int(np.dot(v.sum((1,2)).astype(np.int64),mult))
        result['report_comparisons']+=32*int(mult.sum())
        result['both_impossible']+=count((first==0)&(second==0))
        result['one_impossible']+=count((first==0)^(second==0))
        result['both_possible']+=count(valid)
        for prefix,values in [('update',delta),('forecast',forecast)]:
            result['exact_'+prefix+'_witnesses']+=count(valid&(values>0))
            result['tolerance_'+prefix+'_witnesses']+=count(valid&(values>TOLERANCE))
        if len(block):
            result['max_quotient_distance']=max(result['max_quotient_distance'],float(delta.max()))
            result['max_forecast_coordinate_difference']=max(result['max_forecast_coordinate_difference'],float(forecast.max()))
    if result['report_comparisons'] != sum(result[k] for k in ('both_impossible','one_impossible','both_possible')):raise ValueError('possibility denominator')
    return result


def fixture():
    hs=[('none',0,0),('purpose',1,8),('none',0,1)]
    return dict(hypotheses=hs,length=3,checkpoint=2,signatures=[[0,0],[1,1]],membership=[0,0,1])


def controls():
    a,m=structure(fixture());law=np.full((16,4,8),1/8)
    flat=evaluate(a,law)
    law[0,:,:]=[.75,.25,0,0,0,0,0,0];law[8,:,:]=[.25,.75,0,0,0,0,0,0];law[1,:,:]=[.5,.5,0,0,0,0,0,0]
    changed=evaluate(a,law)
    return {'live:retrospective_update_witness':changed['tolerance_update_witnesses']>0,
            'live:retrospective_forecast_witness':changed['tolerance_forecast_witnesses']>0,
            'placebo:uniform_law':flat['exact_update_witnesses']==flat['exact_forecast_witnesses']==0,
            'positive:complete_pair_denominator':m['report_comparisons']==64,
            'positive:impossible_report_retained':changed['both_impossible']>0}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('retrospective controls')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    specs=read(root/'inputs/SCHEDULES.json')
    expected={f'{h}-{s}' for h in cfg['lengths'] for s in cfg['checkpoints'] if s<=h}
    if set(specs)!=expected:raise ValueError('schedule roster')
    structures={};counts={};(root/'evaluator').mkdir(exist_ok=True)
    for key,spec in specs.items():
        pulse(phase='complete-retrospective-structure',checkpoint=key)
        arr,meta=structure(spec);structures[key]=arr;counts[key]=meta
        np.savez_compressed(root/'evaluator'/f'{key}-structure.npz',**arr)
        write(root/'evaluator'/f'{key}-bindings.json',meta)
    write(root/'evaluator/SCHEDULES.json',specs)
    rows=[]
    for lineage in cfg['lineages']:
        law=read(root/'inputs'/f'{lineage}-law.json');kernel=likelihood_kernel(law)
        np.savez_compressed(root/'evaluator'/f'{lineage}-kernel.npz',**kernel)
        for key,arr in structures.items():
            pulse(phase='full-simplex-retrospective-witnesses',lineage=lineage,checkpoint=key)
            row=evaluate(arr,law,kernel)
            if row['report_comparisons']!=counts[key]['report_comparisons']:raise ValueError('full denominator')
            rows.append(dict(lineage=lineage,length=specs[key]['length'],checkpoint=specs[key]['checkpoint'],**row))
    write(root/'retrospective_points.json',rows)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs;no teacher label or posterior supplied to a reader',evaluator='supplied endpoint laws,schedules,pair/third memberships and likelihood kernels;full-simplex mixtures are not reachable-posterior claims',reconstruction='each pair uses half mass on its first member plus each third member outside the future group;exchange paired member for second mixture;kernel retains two quotient weights,report probabilities and impossible flags;remaining quotient mass is in third group;all future forecasts follow original complete schedules and laws',factorization='past state likelihood types and complete future-state masks preserve every pair/third/report multiplicity;mask reduction used only for maximum over all future coordinates,not for individual forecast identity'))
    return dict(controls=checks,cells=rows,lineages=len(cfg['lineages']),schedule_cells=len(specs),fits=0,tolerance=TOLERANCE,
                claim='supplied-law full-simplex representational certificate only;not reachable posterior,learned access,historical process or human intent',
                arithmetic='exact means binary64 nonzero under the factored update-difference formula;1e-12 tolerance reported separately;not rational-arithmetic certification',
                independent_acceptance=False)
