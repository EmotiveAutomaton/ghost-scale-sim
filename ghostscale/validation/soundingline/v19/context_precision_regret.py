"""Finite regret allocation relative to query-informed precision choices."""
import gzip
import numpy as np
from .robust_context_precision import allocate as absolute_allocate, LENGTHS, VERTICES
from .context_precision_allocation import ASSIGNMENTS, DTYPES, BUDGETS
from .schedule_likelihood_envelope import evaluate as envelope
from ..v18_3.io import write, read, canonical, file_digest


def allocate(ranges, budgets=BUDGETS):
    a = absolute_allocate(ranges, budgets)
    defined = ~np.isnan(a['vertex_regrets']).any(axis=(2, 3))
    worst = np.max(a['vertex_regrets'], axis=-1)
    selected = np.zeros((len(budgets), 81), dtype=bool)
    for k, budget in enumerate(budgets):
        eligible = defined[k] & (a['stored_law_bytes'] <= budget)
        if eligible.any(): selected[k] = eligible & (worst[k, :, 0] == worst[k, eligible, 0].min())
    a.update(regret_defined=defined, worst_vertex_regret=worst, regret_selected=selected)
    return a


def controls():
    zero = allocate(np.zeros((4, 3)))
    x = allocate(np.array([[6, 0, 0], [10, 5, 0], [0, 0, 0], [0, 0, 0.]]))
    robust = np.flatnonzero(x['selected'][0]); regret = np.flatnonzero(x['regret_selected'][0])
    return {'live:regret_differs_from_absolute': bool(x['worst_vertex_regret'][0, regret[0], 0] < min(x['worst_vertex_regret'][0, i, 0] for i in robust)),
        'placebo:zero_ranges_zero_regret': not bool(zero['worst_vertex_regret'].any()),
        'negative:undefined_is_not_zero': not bool(allocate(np.full((4, 3), np.inf))['regret_selected'].any()),
        'negative:infeasible_budget': not bool(allocate(np.zeros((4, 3)), (1023,))['regret_selected'].any())}


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
        pulse(phase='context-precision-regret', lineage=lineage)
        law = read(root/'inputs'/f'{lineage}-law.json'); ranges = []
        for dtype in DTYPES:
            e = envelope(law, dtype); ranges.append(e['context_log_ranges'])
            np.savez_compressed(root/'raw'/f'{lineage}-{dtype}_points.npz', **e)
        a = allocate(np.array(ranges).T)
        if not np.isfinite(a['worst_vertex_regret']).all(): raise ValueError('actual law regret must be finite')
        np.savez_compressed(root/'raw'/f'{lineage}-regret-allocation_points.npz', **a)
        for k, budget in enumerate(BUDGETS):
            chosen = np.flatnonzero(a['regret_selected'][k]).tolist()
            absolute = np.flatnonzero(a['selected'][k]).tolist(); balanced = np.flatnonzero(a['balanced_selected'][k]).tolist()
            uniform = np.flatnonzero(a['uniform'] & (a['stored_law_bytes'] <= budget)).tolist()
            vertices = [np.flatnonzero(a['vertex_selected'][k, v]).tolist() for v in range(4)]
            for j, length in enumerate(LENGTHS):
                rows.append(dict(lineage=lineage, budget=budget, length=length, chosen_assignment_indices=chosen,
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
                    feasible_assignments=int((a['stored_law_bytes'] <= budget).sum())))
    (root/'raw/context_precision_regret_points.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
    write(root/'EVIDENCE_ROLES.json', dict(reader='no new reader inputs', evaluator='supplied law;finite precision library;frozen convex count set',
        scope='minimax log-range regret against query-informed feasible optima;not nonlinear posterior-loss regret,attained error,learned access,process correspondence or human intent'))
    return dict(controls=checks, lineages=len(cfg['lineages']), assignments=81, vertices=4, rows=len(rows), numerical_acceptance=False,
        scope='same-prior same-transition supplied-law log-range regret;law bytes exclude posterior state and schedules')
