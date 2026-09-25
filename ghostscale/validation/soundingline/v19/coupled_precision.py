"""Exact one-parameter arithmetic sensitivity of finite precision choices."""
from fractions import Fraction as F
from itertools import product, combinations
import math
import gzip
import json
import numpy as np
from .schedule_envelope_review import reconstruct as scalar_law
from ..v18_3.io import read, write, canonical, file_digest

ASSIGNMENTS = tuple(product(range(3), repeat=4))
BUDGETS = (1280, 1536, 2048)
LENGTHS = (16, 64, 128)


def crossing(a, b):
    return None if a[1] == b[1] else F(b[0]-a[0], a[1]-b[1])


def at(line, weight):
    return line[0]+line[1]*weight


def solve(lines, feasible, recorded):
    """Partition all vertex and objective intersections, retaining exact ties."""
    if not recorded or not set(recorded) <= set(feasible):
        raise ValueError('recorded choices must be feasible')
    vertex_points = {F(0), F(1)}; vertex_events = []
    for i in feasible:
        for v, w in combinations(range(len(lines[i])), 2):
            t = crossing(lines[i][v], lines[i][w])
            if t is not None and 0 <= t <= 1:
                vertex_points.add(t); vertex_events.append((i, v, w, t))
    ordered = sorted(vertex_points); objective_events = []; points = set(ordered)
    for left, right in zip(ordered[:-1], ordered[1:]):
        midpoint = (left+right)/2
        active = {i: max(lines[i], key=lambda line: at(line, midpoint)) for i in feasible}
        for i, j in combinations(feasible, 2):
            t = crossing(active[i], active[j])
            if t is not None and left < t < right:
                points.add(t); objective_events.append((left, right, i, j, t))
    points = sorted(points)
    def evaluate(t):
        values = [max(at(line, t) for line in lines[i]) for i in feasible]
        optimum = min(values)
        return values, [i for i, value in zip(feasible, values) if value == optimum]
    boundary = []; regrets = {i: F(0) for i in recorded}; winners = set(); always = set(feasible)
    for t in points:
        values, choices = evaluate(t); lookup = dict(zip(feasible, values)); best = min(values)
        for i in recorded:regrets[i] = max(regrets[i], lookup[i]-best)
        boundary.append(dict(weight=t, objectives=values, choices=choices))
        winners.update(choices); always.intersection_update(choices)
    segments = []
    for left, right in zip(points[:-1], points[1:]):
        _, choices = evaluate((left+right)/2)
        segments.append(dict(left=left, right=right, choices=choices))
        winners.update(choices); always.intersection_update(choices)
    return dict(feasible=feasible, recorded=recorded, vertex_events=vertex_events,
        objective_events=objective_events, boundaries=boundary, segments=segments,
        possible_optima=sorted(winners), guaranteed_optima=sorted(always),
        recorded_maximum_regrets=[regrets[i] for i in recorded])


def rational_json(value):
    if isinstance(value, F):return [value.numerator, value.denominator]
    if isinstance(value, dict):return {k:rational_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):return [rational_json(v) for v in value]
    return value


def evaluate(first, second, recorded, box_summaries=None):
    a = np.asarray(first, dtype=float); b = np.asarray(second, dtype=float)
    if a.shape != (4,3) or b.shape != (4,3) or not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():
        raise ValueError('finite nonnegative four-context/three-precision arrays required')
    fa = [[F(float(x)) for x in row] for row in a]; fb = [[F(float(x)) for x in row] for row in b]
    denominator = math.lcm(*(x.denominator for row in fa+fb for x in row))
    lines = []
    for assignment in ASSIGNMENTS:
        vertices = []
        for v in range(4):
            intercept = sum((13 if c==v else 1)*fa[c][assignment[c]] for c in range(4))
            slope = sum((13 if c==v else 1)*(fb[c][assignment[c]]-fa[c][assignment[c]]) for c in range(4))
            vertices.append((int(intercept*denominator), int(slope*denominator)))
        lines.append(vertices)
    charges = [sum((256,512,1024)[k] for k in assignment) for assignment in ASSIGNMENTS]
    problems = []; summaries = []
    for budget, chosen in zip(BUDGETS, recorded, strict=True):
        feasible = [i for i, cost in enumerate(charges) if cost<=budget]
        result = solve(lines, feasible, chosen);problems.append(dict(budget=budget, **result))
        for length in LENGTHS:
            regrets = [value*F(length//16, denominator) for value in result['recorded_maximum_regrets']]
            row = dict(budget=budget,length=length,recorded_choices=chosen,possible_optima=result['possible_optima'],guaranteed_optima=result['guaranteed_optima'],recorded_maximum_regrets=regrets,segments=len(result['segments']))
            if box_summaries is not None:
                box = next(r for r in box_summaries if r['budget']==budget and r['length']==length)
                if box['recorded_choices'] != chosen:raise ValueError('box choice mismatch')
                upper = [F(n, box['denominator']) for n in box['recorded_regret_upper_numerators']]
                if any(value>bound for value,bound in zip(regrets,upper,strict=True)):raise ValueError('coupled regret exceeds containing box')
                if not set(result['possible_optima']) <= set(box['not_ruled_out']):raise ValueError('box excluded a path optimum')
                if not set(box['guaranteed_optima']) <= set(result['guaranteed_optima']):raise ValueError('box guaranteed choice lost on path')
                row.update(box_regret_upper_bounds=upper,box_not_ruled_out=box['not_ruled_out'],box_guaranteed_optima=box['guaranteed_optima'])
            summaries.append(row)
    return rational_json(dict(denominator=denominator,lines=lines,charges=charges,problems=problems,summaries=summaries))


def controls():
    reversal = solve([[(0,1)],[(1,-1)]],[0,1],[0])
    null = solve([[(1,0)],[(1,0)]],[0,1],[0])
    return {'live:known_reversal': reversal['possible_optima']==[0,1] and reversal['recorded_maximum_regrets']==[F(1)],
        'placebo:identical_choices':null['guaranteed_optima']==[0,1] and null['recorded_maximum_regrets']==[F(0)],
        'positive:exact_midpoint_tie':any(r['weight']==F(1,2) and r['choices']==[0,1] for r in reversal['boundaries'])}


def run(root, plan, pulse):
    cfg=plan['design']
    if cfg['lineages']!=list(range(190000,190008)) or cfg['budgets']!=list(BUDGETS) or cfg['lengths']!=list(LENGTHS) or cfg['assignments']!=[list(a) for a in ASSIGNMENTS]:raise ValueError('design')
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    (root/'raw').mkdir(exist_ok=True);summaries=[];segments=0;boundaries=0
    for lineage in cfg['lineages']:
        pulse(phase='coupled-precision',lineage=lineage)
        law=read(root/'inputs'/f'{lineage}-law.json')
        scalar=np.array([scalar_law(law,dtype)['context_log_ranges'] for dtype in ('float16','float32','float64')]).T
        with np.load(root/'inputs'/f'{lineage}-robust-allocation_points.npz',allow_pickle=False) as parent:
            first=parent['context_log_ranges'].copy();chosen=[np.flatnonzero(row).tolist() for row in parent['selected']]
            charges=parent['stored_law_bytes'].tolist()
        box=json.loads(gzip.decompress((root/'inputs'/f'{lineage}-precision_stability_points.json.gz').read_bytes()))
        result=evaluate(first,scalar,chosen,box['summaries'])
        if result['charges']!=charges:raise ValueError('corrupt byte charges')
        result.update(lineage=lineage,producer_ranges=first.tolist(),scalar_ranges=scalar.tolist())
        (root/'raw'/f'{lineage}-coupled_precision_points.json.gz').write_bytes(gzip.compress(canonical(result),mtime=0))
        summaries.extend(dict(lineage=lineage,**r) for r in result['summaries'])
        segments+=sum(len(r['segments']) for r in result['problems']);boundaries+=sum(len(r['boundaries']) for r in result['problems'])
    write(root/'STRATA.json',summaries)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='supplied laws and exact piecewise-linear comparisons along the common arithmetic interpolation',scope='one-parameter sensitivity model;not a real-valued accuracy certificate or attained forecast error'))
    return dict(controls=checks,lineages=8,rows=len(summaries),segments=segments,boundaries=boundaries,numerical_acceptance=False,scope='coupled arithmetic sensitivity;no forecast accuracy,learned access or process correspondence')
