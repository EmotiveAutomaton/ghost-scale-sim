"""Independent rational reconstruction of arithmetic-box decision bounds."""
from fractions import Fraction as F
from itertools import product
from collections import defaultdict
import gzip
import json
import math
import numpy as np
from .schedule_envelope_review import reconstruct as scalar_law
from .robust_precision_review import reconstruct as parent_choices
from ..v18_3.io import read


def reconstruct(first, second, choices, budgets=(1280, 1536, 2048)):
    endpoints = [(F(float(a)), F(float(b))) for a, b in zip(np.ravel(first), np.ravel(second), strict=True)]
    assert len(endpoints) == 12 and all(a >= 0 and b >= 0 for a, b in endpoints)
    lo = [min(a, b) for a, b in endpoints]
    hi = [max(a, b) for a, b in endpoints]
    denominator = math.lcm(*(x.denominator for x in lo + hi))
    assignments = list(product(range(3), repeat=4))
    charges = [sum((256, 512, 1024)[k] for k in a) for a in assignments]
    # Sparse vectors over the twelve independent uncertain coordinates.
    vectors = [[{3*c+a[c]: 13 if c == v else 1 for c in range(4)} for v in range(4)] for a in assignments]
    pairs = []; summaries = []
    for budget, chosen in zip(budgets, choices, strict=True):
        feasible = [i for i, cost in enumerate(charges) if cost <= budget]
        assert chosen and len(chosen) == len(set(chosen)) and set(chosen) <= set(feasible)
        bounds = {}
        for i, j in product(feasible, repeat=2):
            lows = []; highs = []
            for v in vectors[i]:
                lowrow = []; highrow = []
                for w in vectors[j]:
                    coefficients = {k: v.get(k, 0)-w.get(k, 0) for k in v.keys() | w.keys()}
                    lower = sum(t*(lo[k] if t >= 0 else hi[k]) for k, t in coefficients.items())
                    upper = sum(t*(hi[k] if t >= 0 else lo[k]) for k, t in coefficients.items())
                    lowrow.append(int(lower*denominator)); highrow.append(int(upper*denominator))
                lows.append(lowrow); highs.append(highrow)
            lower = min(map(max, zip(*lows))); upper = max(map(min, highs))
            assert lower <= upper
            bounds[i, j] = (lower, upper)
            pairs.append(dict(budget=budget, first_assignment=i, second_assignment=j,
                lower_numerator=lower, upper_numerator=upper,
                vertex_lower_numerators=lows, vertex_upper_numerators=highs))
        possible = [i for i in feasible if all(bounds[i, j][0] <= 0 for j in chosen)]
        guaranteed = [i for i in feasible if all(bounds[i, j][1] <= 0 for j in feasible)]
        for length in (16, 64, 128):
            summaries.append(dict(budget=budget, length=length, recorded_choices=chosen,
                feasible_assignments=feasible, not_ruled_out=possible, guaranteed_optima=guaranteed,
                recorded_regret_upper_numerators=[max(0, *(bounds[i, j][1] for j in feasible))*(length//16) for i in chosen],
                denominator=denominator))
    return dict(lower_numerators=[[int(x*denominator) for x in lo[3*c:3*c+3]] for c in range(4)],
        upper_numerators=[[int(x*denominator) for x in hi[3*c:3*c+3]] for c in range(4)],
        denominator=denominator, charges=charges, pairs=pairs, summaries=summaries)


def controls():
    a = np.array([[3., 2., 1.]]*4)
    strict = reconstruct(a, a, [[0]], budgets=(1280,))
    pair = next(r for r in strict['pairs'] if r['first_assignment'] == 0 and r['second_assignment'] == 1)
    null = reconstruct(a*0, a*0, [[0]], budgets=(1280,))
    return {'live:exact_strict_difference': pair['lower_numerator'] == pair['upper_numerator'] == 1,
            'placebo:zero_regret': all(not any(r['recorded_regret_upper_numerators']) for r in null['summaries'])}


def verify(root):
    cfg = read(root/'PLAN.json')['design']
    assert cfg['lineages'] == list(range(190000, 190008))
    assert cfg['assignments'] == [list(a) for a in product(range(3), repeat=4)]
    assert cfg['budgets'] == [1280, 1536, 2048] and cfg['lengths'] == [16, 64, 128]
    assert cfg['count_vertices'] == [[13 if c == v else 1 for c in range(4)] for v in range(4)]
    strata = []; total_pairs = 0; groups = defaultdict(list)
    for lineage in cfg['lineages']:
        law = read(root/'inputs'/f'{lineage}-law.json')
        scalar = np.array([scalar_law(law, dtype)['context_log_ranges'] for dtype in ('float16', 'float32', 'float64')]).T
        with np.load(root/'inputs'/f'{lineage}-robust-allocation_points.npz', allow_pickle=False) as parent:
            first = parent['context_log_ranges'].copy()
            selected = parent_choices(first)['selected']
            assert np.array_equal(selected, parent['selected'])
            chosen = [np.flatnonzero(row).tolist() for row in selected]
            expected = reconstruct(first, scalar, chosen)
            assert expected['charges'] == parent['stored_law_bytes'].tolist()
        actual = json.loads(gzip.decompress((root/'raw'/f'{lineage}-precision_stability_points.json.gz').read_bytes()))
        expected.update(lineage=lineage, producer_ranges=first.tolist(), scalar_ranges=scalar.tolist())
        assert actual == expected, lineage
        total_pairs += len(expected['pairs'])
        for row in expected['summaries']:
            strata.append(dict(lineage=lineage, **row))
            groups[row['budget'], row['length']].append(row)
    assert read(root/'STRATA.json') == strata and len(strata) == 72 and total_pairs == 9360
    summary = read(root/'SUMMARY.json')
    assert summary['rows'] == 72 and summary['lineages'] == 8 and summary['all_feasible_ordered_pairs'] == total_pairs
    cells = []
    for (budget, length), rows in groups.items():
        assert len(rows) == 8
        regrets = [max(F(n, r['denominator']) for n in r['recorded_regret_upper_numerators']) for r in rows]
        cells.append(dict(budget=budget, length=length, laws=8,
            laws_with_guaranteed_optimum=sum(bool(r['guaranteed_optima']) for r in rows),
            laws_with_all_recorded_choices_guaranteed=sum(set(r['recorded_choices']) <= set(r['guaranteed_optima']) for r in rows),
            mean_unresolved_choices=sum(len(r['not_ruled_out']) for r in rows)/8,
            maximum_regret_upper_bound=float(max(regrets)), mean_regret_upper_bound=float(sum(regrets)/8)))
    base = [r for r in strata if r['length'] == 16]
    return dict(passed=True, numerical_acceptance=True, original_rows=72, structural_problems=24,
        ordered_pairs=total_pairs, vertex_contrasts=16*total_pairs, law_strata=strata, equal_law_cells=cells,
        problems_with_guaranteed_optimum=sum(bool(r['guaranteed_optima']) for r in base),
        problems_with_all_recorded_choices_guaranteed=sum(set(r['recorded_choices']) <= set(r['guaranteed_optima']) for r in base),
        maximum_regret_upper_bound_at128=max(r['maximum_regret_upper_bound'] for r in cells if r['length'] == 128),
        checks='independent scalar law reconstruction; parent float choices; Fraction endpoints and sparse-coordinate linear extrema; all9360pairs and149760vertex contrasts; all72strata and9equal-law cells',
        limitation='conservative bounds within the box between two arithmetic evaluations; no certificate of exact-real range containment or attained forecast error')
