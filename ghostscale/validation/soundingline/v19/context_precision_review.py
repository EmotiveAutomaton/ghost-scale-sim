"""Independent enumeration and scalar law reconstruction for precision allocation."""
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
    costs = [sum(16*8*struct.calcsize(('e', 'f', 'd')[d]) for d in a) for a in choices]
    scores = [math.fsum(float(ranges[c, a[c]]) for c in range(4)) for a in choices]
    totals = [[score*(n//4) for n in (8, 32, 128)] for score in scores]
    bounds = [[1. if math.isinf(t) else -math.expm1(-t/2)/(1+math.exp(-t/2)) for t in row] for row in totals]
    selected = []
    for budget in budgets:
        feasible = [i for i, cost in enumerate(costs) if cost <= budget]
        best = min((scores[i] for i in feasible), default=None)
        selected.append([i in feasible and scores[i] == best for i in range(81)])
    dominated = [any(costs[j] <= costs[i] and scores[j] <= scores[i] and
                     (costs[j] < costs[i] or scores[j] < scores[i]) for j in range(81)) for i in range(81)]
    return dict(assignments=np.array(choices, dtype=np.int64), stored_law_bytes=np.array(costs, dtype=np.int64),
        accumulated_log_ranges=np.array(totals), posterior_tv_bounds=np.array(bounds),
        budgets=np.array(budgets, dtype=np.int64), selected=np.array(selected, dtype=bool),
        uniform=np.array([all(d == a[0] for d in a) for a in choices]), dominated=np.array(dominated),
        context_log_ranges=np.array(ranges))


def verify(root):
    cfg = read(root/'PLAN.json')['design']
    assert cfg['lineages'] == list(range(190000, 190008))
    assert cfg['lengths'] == [8, 32, 128] and cfg['budgets'] == [1280, 1536, 2048]
    assert cfg['assignments'] == [list(x) for x in itertools.product(range(3), repeat=4)]
    rows = read_gzip(root/'raw/context_precision_points.json.gz')
    lookup = {(r['lineage'], r['budget'], r['length']): r for r in rows}
    assert len(rows) == len(lookup) == 72
    strata = []; groups = defaultdict(list); tie_sensitivity = []
    for lineage in cfg['lineages']:
        ranges = []
        for dtype in ('float16', 'float32', 'float64'):
            law = reconstruct_law(read(root/'inputs'/f'{lineage}-law.json'), dtype)
            with np.load(root/'raw'/f'{lineage}-{dtype}_points.npz', allow_pickle=False) as raw:
                check_arrays(raw, law)
            ranges.append(law['context_log_ranges'])
        independent = reconstruct(np.array(ranges).T)
        with np.load(root/'raw'/f'{lineage}-allocation_points.npz', allow_pickle=False) as raw:
            # Numeric objectives reconstruct independently. Discrete computed-float
            # choices need not survive an alternate normalization summation order.
            check_arrays({k: raw[k] for k in raw if k not in ('selected', 'dominated')},
                         {k: v for k, v in independent.items() if k not in ('selected', 'dominated')})
            # Replay the recorded float tie rule exactly after independently checking
            # every underlying range; a tolerance does not redefine a selected tie.
            exact_choices = reconstruct(raw['context_log_ranges'])
            for name in ('assignments', 'stored_law_bytes', 'budgets', 'selected', 'uniform', 'dominated'):
                assert np.array_equal(raw[name], exact_choices[name]), name
            for k, budget in enumerate(cfg['budgets']):
                selected = np.flatnonzero(raw['selected'][k]).tolist()
                scalar_selected = np.flatnonzero(independent['selected'][k]).tolist()
                if selected != scalar_selected:
                    regret = max(float(independent['accumulated_log_ranges'][i, -1]) for i in selected)-min(
                        float(independent['accumulated_log_ranges'][i, -1]) for i in scalar_selected)
                    assert regret <= 2e-13
                    tie_sensitivity.append(dict(lineage=lineage, budget=budget, recorded_choices=selected,
                        scalar_choices=scalar_selected, maximum_log_range_regret_at128=regret))
        for k, budget in enumerate(cfg['budgets']):
            a = exact_choices; chosen = np.flatnonzero(a['selected'][k]).tolist()
            uniform = np.flatnonzero(a['uniform'] & (a['stored_law_bytes'] <= budget)).tolist()
            for j, length in enumerate(cfg['lengths']):
                expected = dict(lineage=lineage, budget=budget, length=length,
                    chosen_assignment_indices=chosen, lexical_first_assignment=chosen[0],
                    selected_bytes=[int(a['stored_law_bytes'][i]) for i in chosen],
                    minimum_posterior_tv_bound=float(a['posterior_tv_bounds'][chosen[0], j]),
                    feasible_uniform_indices=uniform,
                    best_uniform_bound=min(float(a['posterior_tv_bounds'][i, j]) for i in uniform),
                    feasible_assignments=int((a['stored_law_bytes'] <= budget).sum()))
                actual = lookup.pop((lineage, budget, length)); assert set(actual) == set(expected)
                for name, value in expected.items():
                    if isinstance(value, float): assert math.isclose(actual[name], value, rel_tol=2e-13, abs_tol=2e-13), name
                    else: assert actual[name] == value, name
                assert expected['minimum_posterior_tv_bound'] <= expected['best_uniform_bound']+2e-13
                strata.append(expected); groups[budget, length].append(expected)
    assert not lookup
    cells = []
    for (budget, length), rr in groups.items():
        assert len(rr) == 8
        cells.append(dict(budget=budget, length=length, laws=8,
            mean_selected_bound=math.fsum(r['minimum_posterior_tv_bound'] for r in rr)/8,
            maximum_selected_bound=max(r['minimum_posterior_tv_bound'] for r in rr),
            mean_uniform_bound=math.fsum(r['best_uniform_bound'] for r in rr)/8,
            maximum_uniform_bound=max(r['best_uniform_bound'] for r in rr),
            strictly_improved_laws=sum(r['minimum_posterior_tv_bound'] < r['best_uniform_bound'] for r in rr)))
    return dict(passed=True, numerical_acceptance=True, original_rows=72, law_strata=strata,
        equal_law_cells=cells, raw_arrays=32, assignments=648,
        scalar_tie_sensitivity=tie_sensitivity,
        tie_scope='Recorded computed-float ties reconstructed exactly from independently checked ranges; scalar summation may break ties. No certified real-number optimizer identities.',
        checks='scalar binary casts, normalization, support, likelihood ranges; exhaustive independent81assignments per law; actual bytes, all ties, dominance, uniform baselines;72original rows and9equal-law cells')
