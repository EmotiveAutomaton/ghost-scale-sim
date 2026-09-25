"""Exhaustive finite precision allocation for a supplied observation law."""
import gzip
import itertools
import math
import numpy as np
from .schedule_likelihood_envelope import evaluate as envelope
from ..v18_3.io import read, write, canonical, file_digest

DTYPES = ('float16', 'float32', 'float64')
LENGTHS = (8, 32, 128)
BUDGETS = (1280, 1536, 2048)
ASSIGNMENTS = tuple(itertools.product(range(3), repeat=4))
CONTEXT_BYTES = (256, 512, 1024)  # 16 states x 8 endpoints x item size.


def validate_costs(assignments, costs):
    if len(assignments) != len(costs): raise ValueError('cost inventory')
    for assignment, cost in zip(assignments, costs):
        actual = sum(np.empty((16, 8), dtype=DTYPES[d]).nbytes for d in assignment)
        if cost != actual: raise ValueError('stored law byte charge')


def allocate(ranges, budgets=BUDGETS):
    r = np.asarray(ranges, dtype=float)
    if r.shape != (4, 3) or np.isnan(r).any() or (r < 0).any(): raise ValueError('ranges')
    if any(type(b) is not int or b < 0 for b in budgets): raise ValueError('budgets')
    costs = np.array([sum(CONTEXT_BYTES[i] for i in a) for a in ASSIGNMENTS], dtype=np.int64)
    validate_costs(ASSIGNMENTS, costs)
    # Fixed balanced schedule, with every context observed equally often.
    totals = np.array([[math.fsum(float(r[c, a[c]]) for c in range(4))*(n//4)
                       for n in LENGTHS] for a in ASSIGNMENTS])
    bounds = np.tanh(totals/4)
    selected = np.zeros((len(budgets), 81), dtype=bool)
    uniform = np.array([len(set(a)) == 1 for a in ASSIGNMENTS])
    dominated = np.array([any(costs[j] <= costs[i] and totals[j, 0] <= totals[i, 0]
                          and (costs[j] < costs[i] or totals[j, 0] < totals[i, 0])
                          for j in range(81)) for i in range(81)])
    for k, budget in enumerate(budgets):
        feasible = costs <= budget
        if feasible.any(): selected[k] = feasible & (totals[:, 0] == np.min(totals[feasible, 0]))
    return dict(assignments=np.array(ASSIGNMENTS, dtype=np.int64), stored_law_bytes=costs,
                accumulated_log_ranges=totals, posterior_tv_bounds=bounds,
                budgets=np.array(budgets, dtype=np.int64), selected=selected,
                uniform=uniform, dominated=dominated, context_log_ranges=r)


def controls():
    r = np.array([[10, 9, 0], [6, 0, 0], [6, 0, 0], [0, 0, 0]], dtype=float)
    a = allocate(r)
    best = min(a['accumulated_log_ranges'][a['selected'][2], 0])
    return {'live:non_greedy_optimum': best == 12.,
            'placebo:exact_cast_zero': not allocate(np.zeros((4, 3)))['posterior_tv_bounds'].any(),
            'positive:all_assignments_counted': len(ASSIGNMENTS) == 81,
            'negative:insufficient_bytes': not allocate(r, (1023,))['selected'].any()}


def run(root, plan, pulse):
    cfg = plan['design']
    if (cfg['storage_dtypes'] != list(DTYPES) or cfg['lengths'] != list(LENGTHS)
        or cfg['budgets'] != list(BUDGETS) or cfg['assignments'] != [list(a) for a in ASSIGNMENTS]):
        raise ValueError('frozen variants')
    checks = {k: bool(v) for k, v in controls().items()}; write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('controls')
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('input binding')
    (root/'raw').mkdir(exist_ok=True); rows = []
    for lineage in cfg['lineages']:
        pulse(phase='context-precision-allocation', lineage=lineage)
        law = read(root/'inputs'/f'{lineage}-law.json'); ranges = []
        for dtype in DTYPES:
            e = envelope(law, dtype); ranges.append(e['context_log_ranges'])
            np.savez_compressed(root/'raw'/f'{lineage}-{dtype}_points.npz', **e)
        a = allocate(np.array(ranges).T)
        np.savez_compressed(root/'raw'/f'{lineage}-allocation_points.npz', **a)
        for k, budget in enumerate(BUDGETS):
            chosen = np.flatnonzero(a['selected'][k]).tolist()
            feasible_uniform = np.flatnonzero(a['uniform'] & (a['stored_law_bytes'] <= budget)).tolist()
            for j, length in enumerate(LENGTHS):
                rows.append(dict(lineage=lineage, budget=budget, length=length,
                    chosen_assignment_indices=chosen, lexical_first_assignment=chosen[0],
                    selected_bytes=[int(a['stored_law_bytes'][i]) for i in chosen],
                    minimum_posterior_tv_bound=float(a['posterior_tv_bounds'][chosen[0], j]),
                    feasible_uniform_indices=feasible_uniform,
                    best_uniform_bound=min(float(a['posterior_tv_bounds'][i, j]) for i in feasible_uniform),
                    feasible_assignments=int((a['stored_law_bytes'] <= budget).sum())))
    (root/'raw/context_precision_points.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
    write(root/'EVIDENCE_ROLES.json', dict(reader='no new reader inputs', evaluator='supplied law and exhaustive finite allocation library',
        scope='minimum conservative balanced-schedule bound over81assignments;not realized inference error,learned access,process correspondence or human intent'))
    return dict(controls=checks, lineages=len(cfg['lineages']), assignments=81, rows=len(rows), numerical_acceptance=False,
        scope='finite supplied-law storage allocation;law bytes exclude schedules and posterior state')
