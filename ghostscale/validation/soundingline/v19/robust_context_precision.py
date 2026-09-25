"""Finite minimax precision allocation under a frozen convex count set."""
import gzip
import math
import numpy as np
from .context_precision_allocation import ASSIGNMENTS, DTYPES, BUDGETS, CONTEXT_BYTES, validate_costs
from .schedule_likelihood_envelope import evaluate as envelope
from ..v18_3.io import read, write, canonical, file_digest

LENGTHS = (16, 64, 128)
VERTICES = tuple(tuple(13 if c == v else 1 for c in range(4)) for v in range(4))


def allocate(ranges, budgets=BUDGETS):
    r = np.asarray(ranges, dtype=float)
    if r.shape != (4, 3) or np.isnan(r).any() or (r < 0).any(): raise ValueError('ranges')
    if any(type(b) is not int or b < 0 for b in budgets): raise ValueError('budgets')
    costs = np.array([sum(CONTEXT_BYTES[d] for d in a) for a in ASSIGNMENTS], dtype=np.int64)
    validate_costs(ASSIGNMENTS, costs)
    totals = np.array([[[math.fsum(float(r[c, a[c]])*counts[c] for c in range(4))*(n//16)
                        for counts in VERTICES] for n in LENGTHS] for a in ASSIGNMENTS])
    worst = totals.max(-1); balanced = np.array([[math.fsum(float(r[c, a[c]]) for c in range(4))*(n//4)
                                               for n in LENGTHS] for a in ASSIGNMENTS])
    selected = np.zeros((len(budgets), 81), dtype=bool); balanced_selected = selected.copy()
    vertex_selected = np.zeros((len(budgets), 4, 81), dtype=bool)
    vertex_best = np.full((len(budgets), 3, 4), np.nan)
    # Regret is undefined for infinity minus infinity, retained as NaN.
    regrets = np.full((len(budgets), 81, 3, 4), np.nan)
    for k, budget in enumerate(budgets):
        feasible = costs <= budget
        if not feasible.any(): continue
        selected[k] = feasible & (worst[:, 0] == worst[feasible, 0].min())
        balanced_selected[k] = feasible & (balanced[:, 0] == balanced[feasible, 0].min())
        for v in range(4):
            vertex_selected[k, v] = feasible & (totals[:, 0, v] == totals[feasible, 0, v].min())
        vertex_best[k] = totals[feasible].min(0)
        with np.errstate(invalid='ignore'): regrets[k] = totals-vertex_best[k]
    return dict(assignments=np.array(ASSIGNMENTS, dtype=np.int64), stored_law_bytes=costs,
        context_log_ranges=r, count_vertices=np.array(VERTICES, dtype=np.int64), lengths=np.array(LENGTHS),
        budgets=np.array(budgets, dtype=np.int64), vertex_log_ranges=totals,
        vertex_posterior_tv_bounds=np.tanh(totals/4), worst_log_ranges=worst,
        worst_posterior_tv_bounds=np.tanh(worst/4), balanced_log_ranges=balanced,
        selected=selected, balanced_selected=balanced_selected, vertex_selected=vertex_selected,
        vertex_optimum_log_ranges=vertex_best, vertex_regrets=regrets,
        uniform=np.array([len(set(a)) == 1 for a in ASSIGNMENTS]))


def controls():
    zero = allocate(np.zeros((4, 3)))
    heterogeneous = allocate(np.array([[6, 0, 0], [10, 5, 0], [0, 0, 0], [0, 0, 0.]]))
    robust = np.flatnonzero(heterogeneous['selected'][0])[0]
    balanced = np.flatnonzero(heterogeneous['balanced_selected'][0])[0]
    return {'live:uncertainty_changes_optimum': bool(heterogeneous['worst_log_ranges'][robust, 0] < heterogeneous['worst_log_ranges'][balanced, 0]),
        'placebo:exact_cast_zero': not bool(zero['worst_posterior_tv_bounds'].any()),
        'positive:balanced_is_convex_mean': bool(np.array_equal(np.mean(VERTICES, axis=0), [4]*4)),
        'negative:infeasible_bytes': not bool(allocate(np.zeros((4, 3)), (1023,))['selected'].any())}


def run(root, plan, pulse):
    cfg = plan['design']
    if (cfg['storage_dtypes'] != list(DTYPES) or cfg['lengths'] != list(LENGTHS)
        or cfg['budgets'] != list(BUDGETS) or cfg['assignments'] != [list(a) for a in ASSIGNMENTS]
        or cfg['count_vertices'] != [list(v) for v in VERTICES]): raise ValueError('frozen variants')
    checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('controls')
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('input binding')
    (root/'raw').mkdir(exist_ok=True); rows = []
    for lineage in cfg['lineages']:
        pulse(phase='robust-context-precision', lineage=lineage)
        law = read(root/'inputs'/f'{lineage}-law.json'); ranges = []
        for dtype in DTYPES:
            e = envelope(law, dtype); ranges.append(e['context_log_ranges'])
            np.savez_compressed(root/'raw'/f'{lineage}-{dtype}_points.npz', **e)
        a = allocate(np.array(ranges).T)
        np.savez_compressed(root/'raw'/f'{lineage}-robust-allocation_points.npz', **a)
        for k, budget in enumerate(BUDGETS):
            chosen = np.flatnonzero(a['selected'][k]).tolist()
            balanced = np.flatnonzero(a['balanced_selected'][k]).tolist()
            uniform = np.flatnonzero(a['uniform'] & (a['stored_law_bytes'] <= budget)).tolist()
            vertices = [np.flatnonzero(a['vertex_selected'][k, v]).tolist() for v in range(4)]
            for j, length in enumerate(LENGTHS):
                rows.append(dict(lineage=lineage, budget=budget, length=length,
                    chosen_assignment_indices=chosen, lexical_first_assignment=chosen[0],
                    selected_bytes=[int(a['stored_law_bytes'][i]) for i in chosen],
                    balanced_assignment_indices=balanced, vertex_assignment_indices=vertices,
                    minimum_worst_posterior_tv_bound=float(a['worst_posterior_tv_bounds'][chosen[0], j]),
                    balanced_worst_bound_min=min(float(a['worst_posterior_tv_bounds'][i, j]) for i in balanced),
                    balanced_worst_bound_max=max(float(a['worst_posterior_tv_bounds'][i, j]) for i in balanced),
                    selected_vertex_regrets=a['vertex_regrets'][k, chosen, j, :].tolist(),
                    feasible_uniform_indices=uniform,
                    best_uniform_worst_bound=min(float(a['worst_posterior_tv_bounds'][i, j]) for i in uniform),
                    feasible_assignments=int((a['stored_law_bytes'] <= budget).sum())))
    (root/'raw/robust_context_precision_points.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
    write(root/'EVIDENCE_ROLES.json', dict(reader='no new reader inputs', evaluator='supplied law;finite precision library;frozen convex query-count set',
        scope='finite worst-case bound allocation;computed-float ties;not attained inference error,learned access,process correspondence or human intent'))
    return dict(controls=checks, lineages=len(cfg['lineages']), assignments=81, vertices=4, rows=len(rows), numerical_acceptance=False,
        scope='same-prior same-transition supplied-law bound;law bytes exclude posterior state and schedules')
