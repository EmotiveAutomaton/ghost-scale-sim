"""Independent exact reconstruction of the shared arithmetic path."""
from fractions import Fraction as F
from itertools import product, combinations
from collections import defaultdict
import gzip
import json
import math
import numpy as np
from .schedule_envelope_review import reconstruct as scalar_law
from .robust_precision_review import reconstruct as parent_choices
from ..v18_3.io import read


def encode(x):
    if isinstance(x, F): return [x.numerator, x.denominator]
    if isinstance(x, dict): return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [encode(v) for v in x]
    return x


def reconstruct_partition(lines, feasible, recorded):
    """Use explicit endpoint differences and sign changes, not the producer solver."""
    assert recorded and set(recorded) <= set(feasible)
    def value(line, t): return line[0] * (1-t) + (line[0]+line[1]) * t
    def root(a, b):
        d0 = value(a, F(0))-value(b, F(0))
        d1 = value(a, F(1))-value(b, F(1))
        return None if d0 == d1 else F(d0, d0-d1)
    cuts = {F(0), F(1)}; vertices = []
    for i in feasible:
        for v, w in combinations(range(len(lines[i])), 2):
            t = root(lines[i][v], lines[i][w])
            if t is not None and 0 <= t <= 1:
                cuts.add(t); vertices.append((i, v, w, t))
    intersections = []; allcuts = set(cuts)
    ordered = sorted(cuts)
    for lo, hi in zip(ordered, ordered[1:]):
        # A convex maximum is affine between all of its vertex crossings.
        # Recover that affine function from its two boundary objective values.
        affine = {}
        for i in feasible:
            y0 = max(value(line, lo) for line in lines[i])
            y1 = max(value(line, hi) for line in lines[i])
            slope = (y1-y0)/(hi-lo)
            affine[i] = (y0-slope*lo, slope)
        for i, j in combinations(feasible, 2):
            t = root(affine[i], affine[j])
            if t is not None and lo < t < hi:
                allcuts.add(t); intersections.append((lo, hi, i, j, t))
    def objectives(t): return [max(value(line, t) for line in lines[i]) for i in feasible]
    boundaries = []; optima = []
    for t in sorted(allcuts):
        vals = objectives(t)
        choices = [i for i, v in zip(feasible, vals) if v == min(vals)]
        boundaries.append(dict(weight=t, objectives=vals, choices=choices)); optima.append(set(choices))
    segments = []
    for a, b in zip(boundaries, boundaries[1:]):
        # All objectives are affine here; identical boundary pairs characterize
        # persistent ties. A convex combination evaluates the open interval.
        vals = [(x+y)/2 for x, y in zip(a['objectives'], b['objectives'])]
        choices = [i for i, v in zip(feasible, vals) if v == min(vals)]
        segments.append(dict(left=a['weight'], right=b['weight'], choices=choices)); optima.append(set(choices))
    regret = [max(r['objectives'][feasible.index(i)]-min(r['objectives']) for r in boundaries) for i in recorded]
    return dict(feasible=feasible, recorded=recorded, vertex_events=vertices,
        objective_events=intersections, boundaries=boundaries, segments=segments,
        possible_optima=sorted(set.union(*optima)), guaranteed_optima=sorted(set.intersection(*optima)),
        recorded_maximum_regrets=regret)


def reconstruct(first, second, recorded, box):
    pairs = [[(F(float(a)), F(float(b))) for a, b in zip(x, y)] for x, y in zip(first, second)]
    denominator = math.lcm(*(p.denominator for row in pairs for pair in row for p in pair))
    assignments = list(product(range(3), repeat=4)); lines = []
    charges = [sum((256, 512, 1024)[p] for p in a) for a in assignments]
    for a in assignments:
        vertices = []
        for v in range(4):
            endpoints = [sum((1+12*(c == v))*pairs[c][a[c]][side] for c in range(4)) for side in (0, 1)]
            vertices.append((int(endpoints[0]*denominator), int((endpoints[1]-endpoints[0])*denominator)))
        lines.append(vertices)
    problems = []; strata = []
    for budget, choices in zip((1280, 1536, 2048), recorded):
        feasible = [i for i, cost in enumerate(charges) if cost <= budget]
        result = reconstruct_partition(lines, feasible, choices)
        problems.append(dict(budget=budget, **result))
        for length in (16, 64, 128):
            regrets = [v*F(length, 16*denominator) for v in result['recorded_maximum_regrets']]
            bound = next(r for r in box if (r['budget'], r['length']) == (budget, length))
            upper = [F(x, bound['denominator']) for x in bound['recorded_regret_upper_numerators']]
            assert bound['recorded_choices'] == choices
            assert all(a <= b for a, b in zip(regrets, upper, strict=True))
            assert set(result['possible_optima']) <= set(bound['not_ruled_out'])
            assert set(bound['guaranteed_optima']) <= set(result['guaranteed_optima'])
            strata.append(dict(budget=budget, length=length, recorded_choices=choices,
                possible_optima=result['possible_optima'], guaranteed_optima=result['guaranteed_optima'],
                recorded_maximum_regrets=regrets, segments=len(result['segments']),
                box_regret_upper_bounds=upper, box_not_ruled_out=bound['not_ruled_out'], box_guaranteed_optima=bound['guaranteed_optima']))
    return encode(dict(denominator=denominator, lines=lines, charges=charges, problems=problems, summaries=strata))


def controls():
    r = reconstruct_partition([[(0, 4), (4, -4)], [(3, 0)]], [0, 1], [0, 1])
    z = reconstruct_partition([[(1, 0)], [(1, 0)]], [0, 1], [0])
    return {'live:interior_switch': [x['choices'] for x in r['segments']] == [[1], [0], [0], [1]],
        'positive:interior_regret': r['recorded_maximum_regrets'] == [1, 1],
        'placebo:identical_path': z['guaranteed_optima'] == [0, 1] and z['recorded_maximum_regrets'] == [0]}


def verify(root):
    cfg = read(root/'PLAN.json')['design']; strata = []; groups = defaultdict(list)
    assert cfg['lineages'] == list(range(190000, 190008))
    assert cfg['assignments'] == [list(a) for a in product(range(3), repeat=4)]
    assert cfg['budgets'] == [1280,1536,2048] and cfg['lengths'] == [16,64,128]
    assert cfg['count_vertices'] == [[13 if c == v else 1 for c in range(4)] for v in range(4)]
    boundaries = segments = vertex_events = objective_events = 0
    for lineage in cfg['lineages']:
        law = read(root/'inputs'/f'{lineage}-law.json')
        scalar = np.array([scalar_law(law, d)['context_log_ranges'] for d in ('float16','float32','float64')]).T
        with np.load(root/'inputs'/f'{lineage}-robust-allocation_points.npz', allow_pickle=False) as parent:
            first = parent['context_log_ranges'].copy(); selected = parent_choices(first)['selected']
            assert np.array_equal(selected, parent['selected'])
            choices = [np.flatnonzero(row).tolist() for row in selected]
            box = json.loads(gzip.decompress((root/'inputs'/f'{lineage}-precision_stability_points.json.gz').read_bytes()))
            expected = reconstruct(first, scalar, choices, box['summaries'])
            assert expected['charges'] == parent['stored_law_bytes'].tolist()
        expected.update(lineage=lineage, producer_ranges=first.tolist(), scalar_ranges=scalar.tolist())
        actual = json.loads(gzip.decompress((root/'raw'/f'{lineage}-coupled_precision_points.json.gz').read_bytes()))
        assert actual == expected, lineage
        for p in expected['problems']:
            boundaries += len(p['boundaries']); segments += len(p['segments'])
            vertex_events += len(p['vertex_events']); objective_events += len(p['objective_events'])
        for row in expected['summaries']:
            strata.append(dict(lineage=lineage, **row)); groups[row['budget'],row['length']].append(row)
    assert read(root/'STRATA.json') == strata and len(strata) == 72
    summary = read(root/'SUMMARY.json')
    assert summary['rows'] == 72 and summary['lineages'] == 8 and summary['boundaries'] == boundaries and summary['segments'] == segments
    cells = []
    for (budget,length), rows in groups.items():
        assert len(rows) == 8
        regrets = [max(F(*v) for v in r['recorded_maximum_regrets']) for r in rows]
        cells.append(dict(budget=budget,length=length,laws=8,
            laws_with_guaranteed_optimum=sum(bool(r['guaranteed_optima']) for r in rows),
            laws_with_all_recorded_choices_guaranteed=sum(set(r['recorded_choices']) <= set(r['guaranteed_optima']) for r in rows),
            laws_with_fewer_possible_choices_than_box=sum(len(r['possible_optima']) < len(r['box_not_ruled_out']) for r in rows),
            mean_possible_choices=sum(len(r['possible_optima']) for r in rows)/8,
            maximum_regret=float(max(regrets)),mean_regret=float(sum(regrets)/8)))
    base = [r for r in strata if r['length'] == 16]
    return dict(passed=True,numerical_acceptance=True,original_rows=72,structural_problems=24,
        boundaries=boundaries,segments=segments,vertex_events=vertex_events,objective_events=objective_events,
        law_strata=strata,equal_law_cells=cells,
        problems_with_guaranteed_optimum=sum(bool(r['guaranteed_optima']) for r in base),
        problems_with_all_recorded_choices_guaranteed=sum(set(r['recorded_choices']) <= set(r['guaranteed_optima']) for r in base),
        problems_with_fewer_possible_choices_than_box=sum(len(r['possible_optima']) < len(r['box_not_ruled_out']) for r in base),
        maximum_regret_at128=max(r['maximum_regret'] for r in cells if r['length'] == 128),
        checks='independent scalar laws,parent float choices,rational line coefficients,all vertex and objective intersections,boundary values,open-interval ties,regret maxima,box comparisons,72strata and9equal-law cells',
        limitation='shared one-dimensional interpolation between two arithmetic evaluations; not an exact-real certificate or process correspondence')
