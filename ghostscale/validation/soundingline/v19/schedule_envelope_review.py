"""Scalar verification of fixed schedule bounds, independent of the producer."""
import math
from collections import defaultdict
import numpy as np
from .likelihood_envelope_review import reconstruct as law_reconstruct, check_arrays, read_gzip
from ..v18_3.io import read


def reconstruct(law, dtype):
    r = law_reconstruct(law, dtype)
    ranges = np.array([max(float(r['log_ranges'][c, e]) for e in range(8)
                           if r['supported_observations'][c, e]) for c in range(4)])
    attaining = np.zeros((4, 8), dtype=bool)
    for c in range(4):
        for e in range(8):
            attaining[c, e] = bool(r['supported_observations'][c, e] and r['log_ranges'][c, e] == ranges[c])
    totals = np.empty((2, 3)); bounds = totals.copy()
    for order in range(2):
        for j, length in enumerate((8, 32, 128)):
            schedule = [(step % 4 if order == 0 else 3-step % 4) for step in range(length)]
            r[f'schedule_{order}_{length}'] = np.array(schedule, dtype=np.int64)
            total = math.fsum(float(ranges[c]) for c in schedule)
            totals[order, j] = total
            bounds[order, j] = 1. if math.isinf(total) else -math.expm1(-total/2)/(1+math.exp(-total/2))
    r.update(context_log_ranges=ranges, attaining_endpoints=attaining,
             schedule_log_ranges=totals, schedule_posterior_tv_bounds=bounds)
    return r


def verify(root):
    cfg = read(root/'PLAN.json')['design']
    assert cfg['lineages'] == list(range(190000, 190008))
    assert cfg['context_orders'] == [[0, 1, 2, 3], [3, 2, 1, 0]]
    rows = read_gzip(root/'raw/schedule_likelihood_points.json.gz')
    lookup = {(x['lineage'], x['storage_dtype'], x['order'], x['length']): x for x in rows}
    assert len(rows) == len(lookup) == 144
    strata = []; groups = defaultdict(list)
    for lineage in cfg['lineages']:
        for dtype in ('float64', 'float32', 'float16'):
            r = reconstruct(read(root/'inputs'/f'{lineage}-law.json'), dtype)
            with np.load(root/'raw'/f'{lineage}-{dtype}_points.npz', allow_pickle=False) as actual:
                check_arrays(actual, r)
                for key in r:
                    if key.startswith('schedule_') and r[key].dtype.kind == 'i':
                        assert np.array_equal(actual[key], r[key]), key
            assert np.allclose(r['schedule_posterior_tv_bounds'][0], r['schedule_posterior_tv_bounds'][1], atol=2e-13, rtol=0)
            assert np.all(r['schedule_posterior_tv_bounds'] <= r['posterior_tv_bounds']+2e-13)
            for order in range(2):
                for j, length in enumerate((8, 32, 128)):
                    total = float(r['schedule_log_ranges'][order, j]); bound = float(r['schedule_posterior_tv_bounds'][order, j])
                    universal = float(r['posterior_tv_bounds'][j])
                    expected = dict(lineage=lineage, storage_dtype=dtype, length=length, order=order,
                        accumulated_log_ratio_range=None if math.isinf(total) else total,
                        unbounded=math.isinf(total), schedule_posterior_tv_bound=bound,
                        universal_posterior_tv_bound=universal, bound_reduction=universal-bound,
                        lost_support_count=int(r['lost_support'].sum()), stored_law_bytes=int(r['stored'].nbytes))
                    actual = lookup.pop((lineage, dtype, order, length)); assert set(actual) == set(expected)
                    for k, v in expected.items():
                        if isinstance(v, float): assert math.isclose(actual[k], v, rel_tol=2e-13, abs_tol=2e-13), k
                        else: assert actual[k] == v, k
                    strata.append(expected); groups[dtype, order, length].append(expected)
    assert not lookup
    cells = []
    for (dtype, order, length), rr in groups.items():
        assert len(rr) == 8
        cells.append(dict(storage_dtype=dtype, order=order, length=length, laws=8,
            mean_schedule_bound=math.fsum(r['schedule_posterior_tv_bound'] for r in rr)/8,
            minimum_schedule_bound=min(r['schedule_posterior_tv_bound'] for r in rr),
            maximum_schedule_bound=max(r['schedule_posterior_tv_bound'] for r in rr),
            mean_universal_bound=math.fsum(r['universal_posterior_tv_bound'] for r in rr)/8,
            mean_bound_reduction=math.fsum(r['bound_reduction'] for r in rr)/8,
            lost_support_count=sum(r['lost_support_count'] for r in rr), stored_law_bytes=rr[0]['stored_law_bytes']))
    return dict(passed=True, numerical_acceptance=True, original_rows=144, law_strata=strata,
        equal_law_cells=cells, raw_arrays=24,
        checks='independent scalar casts, normalization, ratios, support, context extrema and attaining endpoints; integer schedules, fsum, exponential bound, all144rows and18equal-law cells;paired order equality and universal comparison')
