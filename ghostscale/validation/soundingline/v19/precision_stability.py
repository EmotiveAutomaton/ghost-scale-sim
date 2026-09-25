"""Exact interval bounds for finite precision choices under arithmetic variation."""
from itertools import product
import math
import struct
import gzip
import numpy as np
from .schedule_envelope_review import reconstruct as scalar_law
from ..v18_3.io import read, write, canonical, file_digest

ASSIGNMENTS=tuple(product(range(3),repeat=4))
VERTICES=tuple(tuple(13 if c==v else 1 for c in range(4)) for v in range(4))
BUDGETS=(1280,1536,2048)
LENGTHS=(16,64,128)
DTYPES=('float16','float32','float64')


def intervals(first,second):
    a=np.asarray(first,dtype=float);b=np.asarray(second,dtype=float)
    if a.shape!=(4,3) or b.shape!=(4,3) or not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():
        raise ValueError('finite nonnegative four-context/three-precision inputs required')
    pairs=[(min(float(x),float(y)).as_integer_ratio(),max(float(x),float(y)).as_integer_ratio()) for x,y in zip(a.flat,b.flat)]
    denominator=max(d for pair in pairs for _,d in pair)
    assert all(denominator%d==0 for pair in pairs for _,d in pair)
    lower=tuple(tuple(pairs[3*c+k][0][0]*(denominator//pairs[3*c+k][0][1]) for k in range(3)) for c in range(4))
    upper=tuple(tuple(pairs[3*c+k][1][0]*(denominator//pairs[3*c+k][1][1]) for k in range(3)) for c in range(4))
    return lower,upper,denominator


def pair_bounds(lower,upper,first,second,vertices=VERTICES):
    """Minimax interval relaxation; preserves common coordinates and cancellation."""
    low=[];high=[]
    for va in vertices:
        lows=[];highs=[]
        for vb in vertices:
            minimum=maximum=0
            for c,(a,b) in enumerate(zip(first,second)):
                if a==b:
                    coefficient=va[c]-vb[c]
                    minimum+=coefficient*(lower[c][a] if coefficient>=0 else upper[c][a])
                    maximum+=coefficient*(upper[c][a] if coefficient>=0 else lower[c][a])
                else:
                    minimum+=va[c]*lower[c][a]-vb[c]*upper[c][b]
                    maximum+=va[c]*upper[c][a]-vb[c]*lower[c][b]
            lows.append(minimum);highs.append(maximum)
        low.append(lows);high.append(highs)
    lower_bound=min(max(low[v][w] for v in range(len(vertices))) for w in range(len(vertices)))
    upper_bound=max(min(row) for row in high)
    assert lower_bound<=upper_bound
    return dict(lower_numerator=lower_bound,upper_numerator=upper_bound,
                vertex_lower_numerators=low,vertex_upper_numerators=high)


def costs():
    return tuple(sum(128*struct.calcsize(('e','f','d')[k]) for k in a) for a in ASSIGNMENTS)


def evaluate(first,second,recorded_choices,budgets=BUDGETS,charges=None):
    lower,upper,denominator=intervals(first,second)
    actual=costs()
    if charges is not None and tuple(charges)!=actual:raise ValueError('corrupt byte charges')
    pairs=[];summaries=[]
    for budget,chosen in zip(budgets,recorded_choices,strict=True):
        feasible=[i for i,c in enumerate(actual) if c<=budget]
        if not chosen or any(i not in feasible for i in chosen):raise ValueError('recorded choice infeasible')
        lookup={}
        for i in feasible:
            for j in feasible:
                bound=pair_bounds(lower,upper,ASSIGNMENTS[i],ASSIGNMENTS[j])
                lookup[i,j]=bound
                pairs.append(dict(budget=budget,first_assignment=i,second_assignment=j,**bound))
        possible=[i for i in feasible if all(lookup[i,j]['lower_numerator']<=0 for j in chosen)]
        guaranteed=[i for i in feasible if all(lookup[i,j]['upper_numerator']<=0 for j in feasible)]
        for n in LENGTHS:
            scale=n//16
            summaries.append(dict(budget=budget,length=n,recorded_choices=list(chosen),
                feasible_assignments=feasible,not_ruled_out=possible,guaranteed_optima=guaranteed,
                recorded_regret_upper_numerators=[max(0,max(lookup[i,j]['upper_numerator'] for j in feasible))*scale for i in chosen],
                denominator=denominator))
    return dict(lower_numerators=lower,upper_numerators=upper,denominator=denominator,
        charges=actual,pairs=pairs,summaries=summaries)


def controls():
    zero=np.zeros((4,3));choices=[[0],[0],[0]];null=evaluate(zero,zero,choices)
    lower=((0,2,4),)*4
    strict=pair_bounds(lower,lower,(0,)*4,(1,)*4)
    return {'live:strict_margin':strict['upper_numerator']==-32,
        'placebo:zero_regret':all(not any(r['recorded_regret_upper_numerators']) for r in null['summaries']),
        'positive:all_null_ties':all(r['guaranteed_optima']==r['feasible_assignments'] for r in null['summaries']),
        'positive:identity':pair_bounds(lower,lower,(1,)*4,(1,)*4)['upper_numerator']==0}


def run(root,plan,pulse):
    cfg=plan['design']
    if cfg['lineages']!=list(range(190000,190008)) or cfg['budgets']!=list(BUDGETS) or cfg['lengths']!=list(LENGTHS) or cfg['assignments']!=[list(a) for a in ASSIGNMENTS] or cfg['count_vertices']!=[list(v) for v in VERTICES]:raise ValueError('design')
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    (root/'raw').mkdir(exist_ok=True);summaries=[];pair_count=0
    for lineage in cfg['lineages']:
        pulse(phase='precision-stability',lineage=lineage)
        law=read(root/'inputs'/f'{lineage}-law.json')
        scalar=np.array([scalar_law(law,dtype)['context_log_ranges'] for dtype in DTYPES]).T
        with np.load(root/'inputs'/f'{lineage}-robust-allocation_points.npz',allow_pickle=False) as parent:
            first=parent['context_log_ranges'].copy();charges=parent['stored_law_bytes'].tolist()
            chosen=[np.flatnonzero(row).tolist() for row in parent['selected']]
        result=evaluate(first,scalar,chosen,charges=charges)
        result.update(lineage=lineage,producer_ranges=first.tolist(),scalar_ranges=scalar.tolist())
        (root/'raw'/f'{lineage}-precision_stability_points.json.gz').write_bytes(gzip.compress(canonical(result),mtime=0))
        pair_count+=len(result['pairs'])
        summaries.extend(dict(lineage=lineage,**r) for r in result['summaries'])
    write(root/'STRATA.json',summaries)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='supplied laws,producer and scalar arithmetic ranges,recorded robust choices and true storage charges',scope='conservative exact bounds within the box between two floating arithmetic evaluations;not certified containment of exact real-valued law ranges or attained inference error'))
    return dict(controls=checks,lineages=8,rows=len(summaries),all_feasible_ordered_pairs=pair_count,numerical_acceptance=False,
        scope='arithmetic decision sensitivity;no real-number certificate,learned access,forecast accuracy or process correspondence')
