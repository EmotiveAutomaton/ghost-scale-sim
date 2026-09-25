"""Outcome-independent query schedules for the supplied-law rounding bound."""
import gzip
import numpy as np
from .law_likelihood_envelope import evaluate as law_envelope, DTYPES, LENGTHS, fixture
from ..v18_3.io import read, write, canonical, file_digest

ORDERS = ((0, 1, 2, 3), (3, 2, 1, 0))


def schedule_bound(context_ranges, schedule):
    ranges = np.asarray(context_ranges, dtype=float)
    if ranges.shape != (4,) or np.isnan(ranges).any() or (ranges < 0).any():
        raise ValueError('context ranges')
    if not schedule or any(type(c) is not int or c not in range(4) for c in schedule):
        raise ValueError('schedule')
    total = sum(float(ranges[c]) for c in schedule)
    return total, float(np.tanh(total/4))


def evaluate(law, dtype):
    r = law_envelope(law, dtype)
    context_ranges = np.array([max(r['log_ranges'][c, r['supported_observations'][c]]) for c in range(4)])
    attaining = r['supported_observations'] & (r['log_ranges'] == context_ranges[:, None])
    totals = np.empty((2, 3)); bounds = totals.copy()
    for i, order in enumerate(ORDERS):
        for j, length in enumerate(LENGTHS):
            schedule = list(order)*(length//4)
            totals[i, j], bounds[i, j] = schedule_bound(context_ranges, schedule)
            r[f'schedule_{i}_{length}'] = np.array(schedule, dtype=np.int64)
    r.update(context_log_ranges=context_ranges, attaining_endpoints=attaining,
             schedule_log_ranges=totals, schedule_posterior_tv_bounds=bounds)
    return r


def controls():
    _, known = schedule_bound([0., .01, .02, .03], [0, 1, 2, 3]*2)
    _, reverse = schedule_bound([0., .01, .02, .03], [3, 2, 1, 0]*2)
    return {'live:heterogeneous_context_tightens': known < np.tanh(8*.03/4),
            'placebo:order_count_identity': abs(known-reverse) < 1e-15,
            'positive:constant_range_identity': abs(schedule_bound([.01]*4, [0, 1, 2, 3])[1]-np.tanh(.01)) < 1e-15,
            'negative:undersized_bound': known > 0.}


def run(root, plan, pulse):
    cfg = plan['design']
    if (cfg['storage_dtypes'] != list(DTYPES) or cfg['lengths'] != list(LENGTHS)
        or cfg['context_orders'] != [list(x) for x in ORDERS]):
        raise ValueError('frozen variants')
    checks = {k: bool(v) for k, v in controls().items()}; write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('controls')
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('input binding')
    rows = []; (root/'raw').mkdir(exist_ok=True)
    for lineage in cfg['lineages']:
        pulse(phase='schedule-likelihood-envelope', lineage=lineage)
        for dtype in DTYPES:
            r = evaluate(read(root/'inputs'/f'{lineage}-law.json'), dtype)
            np.savez_compressed(root/'raw'/f'{lineage}-{dtype}_points.npz', **r)
            for i, order in enumerate(ORDERS):
                for j, length in enumerate(LENGTHS):
                    total = float(r['schedule_log_ranges'][i, j]); bound = float(r['schedule_posterior_tv_bounds'][i, j])
                    universal = float(r['posterior_tv_bounds'][j])
                    rows.append(dict(lineage=lineage, storage_dtype=dtype, length=length, order=i,
                        accumulated_log_ratio_range=None if np.isinf(total) else total,
                        unbounded=bool(np.isinf(total)), schedule_posterior_tv_bound=bound,
                        universal_posterior_tv_bound=universal, bound_reduction=universal-bound,
                        lost_support_count=int(r['lost_support'].sum()), stored_law_bytes=int(r['stored'].nbytes)))
    (root/'raw/schedule_likelihood_points.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
    write(root/'EVIDENCE_ROLES.json', dict(reader='no new reader inputs', evaluator='supplied original/rounded laws and fixed context schedules', scope='same-prior same-transition conservative posterior bound;equal counts cannot establish order effects;not attained error,learned access,process correspondence or human intent'))
    return dict(controls=checks, lineages=len(cfg['lineages']), variants=3, schedules=2,
                rows=len(rows), numerical_acceptance=False, scope='query-composition sensitivity of a supplied-law ruler;not actual inference error')
