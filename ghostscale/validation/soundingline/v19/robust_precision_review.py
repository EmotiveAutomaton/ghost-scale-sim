"""Independent scalar enumeration for the frozen robust precision comparison."""
import itertools
import math
import struct
from collections import defaultdict
import numpy as np
from .schedule_envelope_review import reconstruct as reconstruct_law
from .likelihood_envelope_review import check_arrays, read_gzip
from ..v18_3.io import read


def reconstruct(ranges, budgets=(1280, 1536, 2048)):
    choices = list(itertools.product(range(3), repeat=4))
    vertices = [[13 if c == v else 1 for c in range(4)] for v in range(4)]
    lengths = [16, 64, 128]
    costs = [sum(128*struct.calcsize(('e', 'f', 'd')[d]) for d in a) for a in choices]
    totals = np.array([[[math.fsum(float(ranges[c, a[c]])*counts[c] for c in range(4))*(n//16)
                        for counts in vertices] for n in lengths] for a in choices])
    worst = np.array([[max(row) for row in a] for a in totals])
    balanced = np.array([[math.fsum(float(ranges[c, a[c]]) for c in range(4))*(n//4)
                         for n in lengths] for a in choices])
    selected = np.zeros((len(budgets), 81), dtype=bool)
    bal_selected = selected.copy(); vertex_selected = np.zeros((len(budgets), 4, 81), dtype=bool)
    best = np.full((len(budgets), 3, 4), np.nan)
    regrets = np.full((len(budgets), 81, 3, 4), np.nan)
    for k, budget in enumerate(budgets):
        feasible = [i for i, cost in enumerate(costs) if cost <= budget]
        if not feasible: continue
        robust_min = min(worst[i, 0] for i in feasible)
        balanced_min = min(balanced[i, 0] for i in feasible)
        for i in feasible:
            selected[k, i] = worst[i, 0] == robust_min
            bal_selected[k, i] = balanced[i, 0] == balanced_min
        for v in range(4):
            for j in range(3):
                best[k, j, v] = min(totals[i, j, v] for i in feasible)
                for i in range(81):
                    x, y = float(totals[i, j, v]), float(best[k, j, v])
                    regrets[k, i, j, v] = math.nan if math.isinf(x) and math.isinf(y) else x-y
            vertex_selected[k, v] = [i in feasible and totals[i, 0, v] == best[k, 0, v] for i in range(81)]
    def bound(a):
        return np.array([1. if math.isinf(x) else -math.expm1(-x/2)/(1+math.exp(-x/2)) for x in a.flat]).reshape(a.shape)
    return dict(assignments=np.array(choices, dtype=np.int64), stored_law_bytes=np.array(costs, dtype=np.int64),
        context_log_ranges=np.array(ranges, dtype=float), count_vertices=np.array(vertices, dtype=np.int64),
        lengths=np.array(lengths), budgets=np.array(budgets, dtype=np.int64), vertex_log_ranges=totals,
        vertex_posterior_tv_bounds=bound(totals), worst_log_ranges=worst, worst_posterior_tv_bounds=bound(worst),
        balanced_log_ranges=balanced, selected=selected, balanced_selected=bal_selected, vertex_selected=vertex_selected,
        vertex_optimum_log_ranges=best, vertex_regrets=regrets,
        uniform=np.array([all(d == a[0] for d in a) for a in choices]))


def verify(root):
    cfg = read(root/'PLAN.json')['design']
    assert cfg['lineages'] == list(range(190000, 190008))
    assert cfg['lengths'] == [16, 64, 128] and cfg['budgets'] == [1280, 1536, 2048]
    assert cfg['assignments'] == [list(x) for x in itertools.product(range(3), repeat=4)]
    assert cfg['count_vertices'] == [[13 if c == v else 1 for c in range(4)] for v in range(4)]
    rows = read_gzip(root/'raw/robust_context_precision_points.json.gz')
    lookup = {(r['lineage'], r['budget'], r['length']): r for r in rows}
    assert len(rows) == len(lookup) == 72
    strata = []; groups = defaultdict(list); sensitivity = []
    discrete = ('selected', 'balanced_selected', 'vertex_selected')
    for lineage in cfg['lineages']:
        ranges = []
        for dtype in ('float16', 'float32', 'float64'):
            law = reconstruct_law(read(root/'inputs'/f'{lineage}-law.json'), dtype)
            with np.load(root/'raw'/f'{lineage}-{dtype}_points.npz', allow_pickle=False) as raw: check_arrays(raw, law)
            ranges.append(law['context_log_ranges'])
        independent = reconstruct(np.array(ranges).T)
        with np.load(root/'raw'/f'{lineage}-robust-allocation_points.npz', allow_pickle=False) as raw:
            check_arrays({k: raw[k] for k in raw if k not in discrete}, {k: v for k, v in independent.items() if k not in discrete})
            exact = reconstruct(raw['context_log_ranges'])
            for name in ('assignments', 'stored_law_bytes', 'count_vertices', 'lengths', 'budgets', 'uniform', *discrete):
                assert np.array_equal(raw[name], exact[name]), name
            for name in discrete:
                for index in np.ndindex(raw[name].shape[:-1]):
                    chosen = np.flatnonzero(raw[name][index]).tolist()
                    scalar = np.flatnonzero(independent[name][index]).tolist()
                    if chosen != scalar:
                        objective = independent['worst_log_ranges'][:, -1] if name == 'selected' else independent['balanced_log_ranges'][:, -1] if name == 'balanced_selected' else independent['vertex_log_ranges'][:, -1, index[1]]
                        regret = max(float(objective[i]) for i in chosen)-min(float(objective[i]) for i in scalar)
                        assert abs(regret) <= 2e-13
                        sensitivity.append(dict(lineage=lineage, selection=name, budget=cfg['budgets'][index[0]],
                            vertex=index[1] if len(index)>1 else None, recorded_choices=chosen, scalar_choices=scalar,
                            maximum_log_range_regret_at128=regret))
        a = exact
        for k, budget in enumerate(cfg['budgets']):
            chosen = np.flatnonzero(a['selected'][k]).tolist(); balanced = np.flatnonzero(a['balanced_selected'][k]).tolist()
            uniform = np.flatnonzero(a['uniform'] & (a['stored_law_bytes'] <= budget)).tolist()
            vertices = [np.flatnonzero(a['vertex_selected'][k, v]).tolist() for v in range(4)]
            for j, length in enumerate(cfg['lengths']):
                expected = dict(lineage=lineage, budget=budget, length=length, chosen_assignment_indices=chosen,
                    lexical_first_assignment=chosen[0], selected_bytes=[int(a['stored_law_bytes'][i]) for i in chosen],
                    balanced_assignment_indices=balanced, vertex_assignment_indices=vertices,
                    minimum_worst_posterior_tv_bound=float(a['worst_posterior_tv_bounds'][chosen[0], j]),
                    balanced_worst_bound_min=min(float(a['worst_posterior_tv_bounds'][i, j]) for i in balanced),
                    balanced_worst_bound_max=max(float(a['worst_posterior_tv_bounds'][i, j]) for i in balanced),
                    selected_vertex_regrets=a['vertex_regrets'][k, chosen, j, :].tolist(), feasible_uniform_indices=uniform,
                    best_uniform_worst_bound=min(float(a['worst_posterior_tv_bounds'][i, j]) for i in uniform),
                    feasible_assignments=int((a['stored_law_bytes'] <= budget).sum()))
                actual = lookup.pop((lineage, budget, length)); assert set(actual) == set(expected)
                for name, value in expected.items():
                    if name == 'selected_vertex_regrets': assert np.allclose(actual[name], value, rtol=2e-13, atol=2e-13)
                    elif isinstance(value, float): assert math.isclose(actual[name], value, rel_tol=2e-13, abs_tol=2e-13), name
                    else: assert actual[name] == value, name
                assert expected['minimum_worst_posterior_tv_bound'] <= expected['balanced_worst_bound_min']+2e-13
                assert expected['minimum_worst_posterior_tv_bound'] <= expected['best_uniform_worst_bound']+2e-13
                strata.append(expected); groups[budget, length].append(expected)
    assert not lookup
    cells = []
    for (budget, length), rr in groups.items():
        assert len(rr) == 8
        cells.append(dict(budget=budget, length=length, laws=8,
            mean_selected_bound=math.fsum(r['minimum_worst_posterior_tv_bound'] for r in rr)/8,
            maximum_selected_bound=max(r['minimum_worst_posterior_tv_bound'] for r in rr),
            maximum_balanced_bound=max(r['balanced_worst_bound_max'] for r in rr),
            maximum_uniform_bound=max(r['best_uniform_worst_bound'] for r in rr),
            laws_improved_over_balanced=sum(r['minimum_worst_posterior_tv_bound'] < r['balanced_worst_bound_min']-2e-13 for r in rr),
            laws_improved_over_uniform=sum(r['minimum_worst_posterior_tv_bound'] < r['best_uniform_worst_bound']-2e-13 for r in rr)))
    return dict(passed=True, numerical_acceptance=True, original_rows=72, law_strata=strata, equal_law_cells=cells,
        raw_arrays=32, assignments=648, scalar_tie_sensitivity=sensitivity,
        tie_scope='Recorded computed-float ties checked exactly from independently verified ranges; scalar sensitivity retained. No exact-real optimizer certification.',
        checks='independent scalar laws and exponential bounds;all648assignments,four vertices,three lengths,actual bytes,robust/balanced/vertex ties,regrets,uniform baselines;72original rows,nine equal-law cells')
