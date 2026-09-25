"""Independent scalar regret selection and complete original-row reconstruction."""
import itertools
import math
from collections import defaultdict
import numpy as np
from .robust_precision_review import reconstruct as absolute_reconstruct
from .schedule_envelope_review import reconstruct as reconstruct_law
from .likelihood_envelope_review import check_arrays, read_gzip
from ..v18_3.io import read


def reconstruct(ranges, budgets=(1280, 1536, 2048)):
    a = absolute_reconstruct(ranges, budgets)
    defined = np.zeros((len(budgets), 81), dtype=bool)
    worst = np.full((len(budgets), 81, 3), np.nan)
    selected = defined.copy()
    for k, budget in enumerate(budgets):
        for i in range(81):
            defined[k, i] = all(not math.isnan(float(x)) for x in a['vertex_regrets'][k, i].flat)
            for j in range(3):
                values = [float(x) for x in a['vertex_regrets'][k, i, j]]
                if not any(math.isnan(x) for x in values): worst[k, i, j] = max(values)
        eligible = [i for i in range(81) if defined[k, i] and a['stored_law_bytes'][i] <= budget]
        if eligible:
            best = min(worst[k, i, 0] for i in eligible)
            for i in eligible: selected[k, i] = worst[k, i, 0] == best
    a.update(regret_defined=defined, worst_vertex_regret=worst, regret_selected=selected)
    return a


def verify(root):
    cfg = read(root/'PLAN.json')['design']
    assert cfg['lineages'] == list(range(190000, 190008))
    assert cfg['lengths'] == [16, 64, 128] and cfg['budgets'] == [1280, 1536, 2048]
    assert cfg['assignments'] == [list(x) for x in itertools.product(range(3), repeat=4)]
    assert cfg['count_vertices'] == [[13 if c == v else 1 for c in range(4)] for v in range(4)]
    rows = read_gzip(root/'raw/context_precision_regret_points.json.gz')
    lookup = {(r['lineage'], r['budget'], r['length']): r for r in rows}
    assert len(rows) == len(lookup) == 72
    strata = []; groups = defaultdict(list); sensitivity = []
    discrete = ('selected', 'balanced_selected', 'vertex_selected', 'regret_selected')
    for lineage in cfg['lineages']:
        ranges = []
        for dtype in ('float16', 'float32', 'float64'):
            law = reconstruct_law(read(root/'inputs'/f'{lineage}-law.json'), dtype)
            with np.load(root/'raw'/f'{lineage}-{dtype}_points.npz', allow_pickle=False) as raw: check_arrays(raw, law)
            ranges.append(law['context_log_ranges'])
        independent = reconstruct(np.array(ranges).T)
        with np.load(root/'raw'/f'{lineage}-regret-allocation_points.npz', allow_pickle=False) as raw:
            check_arrays({k: raw[k] for k in raw if k not in discrete}, {k: v for k, v in independent.items() if k not in discrete})
            exact = reconstruct(raw['context_log_ranges'])
            for name in ('assignments', 'stored_law_bytes', 'count_vertices', 'lengths', 'budgets', 'uniform', 'regret_defined', *discrete):
                assert np.array_equal(raw[name], exact[name]), name
            for name in discrete:
                for index in np.ndindex(raw[name].shape[:-1]):
                    chosen = np.flatnonzero(raw[name][index]).tolist()
                    scalar = np.flatnonzero(independent[name][index]).tolist()
                    if chosen != scalar:
                        k = index[0]
                        objective = (independent['worst_vertex_regret'][k, :, -1] if name == 'regret_selected'
                            else independent['worst_log_ranges'][:, -1] if name == 'selected'
                            else independent['balanced_log_ranges'][:, -1] if name == 'balanced_selected'
                            else independent['vertex_log_ranges'][:, -1, index[1]])
                        regret = max(float(objective[i]) for i in chosen)-min(float(objective[i]) for i in scalar)
                        assert abs(regret) <= 2e-13
                        sensitivity.append(dict(lineage=lineage, selection=name, budget=cfg['budgets'][k],
                            vertex=index[1] if len(index)>1 else None, recorded_choices=chosen, scalar_choices=scalar,
                            maximum_objective_regret_at128=regret))
        a = exact
        for k, budget in enumerate(cfg['budgets']):
            chosen = np.flatnonzero(a['regret_selected'][k]).tolist()
            absolute = np.flatnonzero(a['selected'][k]).tolist(); balanced = np.flatnonzero(a['balanced_selected'][k]).tolist()
            uniform = np.flatnonzero(a['uniform'] & (a['stored_law_bytes'] <= budget)).tolist()
            vertices = [np.flatnonzero(a['vertex_selected'][k, v]).tolist() for v in range(4)]
            for j, length in enumerate(cfg['lengths']):
                expected = dict(lineage=lineage, budget=budget, length=length, chosen_assignment_indices=chosen,
                    lexical_first_assignment=chosen[0], selected_bytes=[int(a['stored_law_bytes'][i]) for i in chosen],
                    absolute_assignment_indices=absolute, balanced_assignment_indices=balanced, vertex_assignment_indices=vertices,
                    minimum_worst_vertex_regret=float(a['worst_vertex_regret'][k, chosen[0], j]),
                    absolute_worst_regret_min=min(float(a['worst_vertex_regret'][k, i, j]) for i in absolute),
                    absolute_worst_regret_max=max(float(a['worst_vertex_regret'][k, i, j]) for i in absolute),
                    balanced_worst_regret_min=min(float(a['worst_vertex_regret'][k, i, j]) for i in balanced),
                    balanced_worst_regret_max=max(float(a['worst_vertex_regret'][k, i, j]) for i in balanced),
                    selected_absolute_bound_min=min(float(a['worst_posterior_tv_bounds'][i, j]) for i in chosen),
                    selected_absolute_bound_max=max(float(a['worst_posterior_tv_bounds'][i, j]) for i in chosen),
                    selected_vertex_regrets=a['vertex_regrets'][k, chosen, j, :].tolist(), feasible_uniform_indices=uniform,
                    best_uniform_worst_regret=min(float(a['worst_vertex_regret'][k, i, j]) for i in uniform),
                    feasible_assignments=int((a['stored_law_bytes'] <= budget).sum()))
                actual = lookup.pop((lineage, budget, length)); assert set(actual) == set(expected)
                for name, value in expected.items():
                    if name == 'selected_vertex_regrets': assert np.allclose(actual[name], value, rtol=2e-13, atol=2e-13)
                    elif isinstance(value, float): assert math.isclose(actual[name], value, rel_tol=2e-13, abs_tol=2e-13), name
                    else: assert actual[name] == value, name
                assert expected['minimum_worst_vertex_regret'] <= min(expected['absolute_worst_regret_min'], expected['balanced_worst_regret_min'], expected['best_uniform_worst_regret'])+2e-13
                expected['absolute_optimum_bound'] = float(a['worst_posterior_tv_bounds'][absolute[0], j])
                expected['regret_selection_absolute_cost'] = expected['selected_absolute_bound_max']-expected['absolute_optimum_bound']
                strata.append(expected); groups[budget, length].append(expected)
    assert not lookup
    cells = []
    for (budget, length), rr in groups.items():
        assert len(rr) == 8
        cells.append(dict(budget=budget, length=length, laws=8,
            mean_selected_regret=math.fsum(r['minimum_worst_vertex_regret'] for r in rr)/8,
            maximum_selected_regret=max(r['minimum_worst_vertex_regret'] for r in rr),
            maximum_absolute_choice_regret=max(r['absolute_worst_regret_max'] for r in rr),
            maximum_uniform_regret=max(r['best_uniform_worst_regret'] for r in rr),
            maximum_absolute_bound_cost=max(r['regret_selection_absolute_cost'] for r in rr),
            laws_improved_over_absolute=sum(r['minimum_worst_vertex_regret'] < r['absolute_worst_regret_min']-2e-13 for r in rr),
            laws_improved_over_balanced=sum(r['minimum_worst_vertex_regret'] < r['balanced_worst_regret_min']-2e-13 for r in rr)))
    return dict(passed=True, numerical_acceptance=True, original_rows=72, law_strata=strata, equal_law_cells=cells,
        raw_arrays=32, assignments=648, scalar_tie_sensitivity=sensitivity,
        tie_scope='Recorded computed-float ties checked exactly from independently verified ranges; scalar sensitivity retained. No exact-real optimizer certification.',
        checks='independent scalar laws and exponential bounds;all648assignments,four vertices,three lengths,actual bytes,regret/absolute/balanced/vertex ties,regrets,undefined handling,absolute tradeoffs,uniform baselines;72original rows,nine equal-law cells')
