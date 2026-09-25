"""Independent scalar reconstruction of the supplied-law likelihood ruler."""
import gzip
import math
import struct
from collections import defaultdict
import numpy as np
from ..v18_3.io import read


def reconstruct(law, dtype):
    a = np.asarray(law, dtype=float)
    if a.shape != (16, 4, 8) or dtype not in ('float64', 'float32', 'float16'):
        raise ValueError('law/design')
    fmt = {'float64': 'd', 'float32': 'f', 'float16': 'e'}[dtype]
    cast = np.array([struct.unpack('<'+fmt, struct.pack('<'+fmt, float(x)))[0]
                     for x in a.flat]).reshape(a.shape)
    ref = np.empty_like(a); rounded = np.empty_like(a)
    for s in range(16):
        for c in range(4):
            row = list(a[s, c]); mass = sum(row); cmass = sum(cast[s, c])
            if any(not math.isfinite(x) or x < 0 for x in row) or abs(mass-1) > 1e-12 or cmass <= 0:
                raise ValueError('invalid law row')
            ref[s, c] = [x/mass for x in row]
            rounded[s, c] = [x/cmass for x in cast[s, c]]
    positive = ref > 0; lost = positive & (rounded == 0)
    ratios = np.full(a.shape, np.nan); logs = ratios.copy()
    low = np.full((4, 8), np.nan); high = low.copy(); spans = low.copy()
    supported = np.zeros((4, 8), dtype=bool); unbounded = supported.copy()
    for c in range(4):
        for e in range(8):
            values = []
            for s in range(16):
                if not positive[s, c, e]: continue
                r = rounded[s, c, e]/ref[s, c, e]
                ratios[s, c, e] = r
                logs[s, c, e] = math.log(r) if r else -math.inf
                values.append(logs[s, c, e])
            if not values: continue
            supported[c, e] = True; unbounded[c, e] = any(x == -math.inf for x in values)
            low[c, e] = min(values); high[c, e] = max(values)
            spans[c, e] = math.inf if unbounded[c, e] else max(values)-min(values)
    maximum = max(spans[supported])
    # Algebraically equivalent exponential formula, without the producer's tanh.
    bounds = [1. if math.isinf(maximum) else -math.expm1(-n*maximum/2)/(1+math.exp(-n*maximum/2))
              for n in (8, 32, 128)]
    return dict(original=a, reference=ref, stored=cast.astype(dtype), rounded=rounded,
        reference_normalization_delta=ref-a, rounded_normalization_delta=rounded-cast,
        positive_reference=positive, lost_support=lost, ratios=ratios, log_ratios=logs,
        supported_observations=supported, unbounded_observations=unbounded,
        lower=low, upper=high, log_ranges=spans, maximum_log_range=np.array(maximum),
        lengths=np.array([8, 32, 128]), posterior_tv_bounds=np.array(bounds))


def check_arrays(actual, expected):
    assert set(actual) == set(expected), 'raw field inventory'
    for name, want in expected.items():
        got = actual[name]
        assert got.shape == want.shape and got.dtype == want.dtype, name
        if name in ('original', 'stored', 'lengths') or want.dtype == bool:
            assert np.array_equal(got, want), name
        else:
            assert np.allclose(got, want, rtol=2e-13, atol=2e-13, equal_nan=True), name


def verify(root):
    plan = read(root/'PLAN.json'); rows = read_gzip(root/'raw/likelihood_envelope_points.json.gz')
    cfg = plan['design']; assert cfg['lineages'] == list(range(190000, 190008))
    assert len(rows) == 72
    lookup = {(r['lineage'], r['storage_dtype'], r['length']): r for r in rows}
    assert len(lookup) == 72
    groups = defaultdict(list); strata = []
    for lineage in cfg['lineages']:
        for dtype in ('float64', 'float32', 'float16'):
            r = reconstruct(read(root/'inputs'/f'{lineage}-law.json'), dtype)
            with np.load(root/'raw'/f'{lineage}-{dtype}_points.npz', allow_pickle=False) as raw:
                check_arrays(raw, r)
            maximum = float(r['maximum_log_range'])
            for i, length in enumerate((8, 32, 128)):
                expected = dict(lineage=lineage, storage_dtype=dtype, length=length,
                    maximum_log_ratio_range=None if math.isinf(maximum) else maximum,
                    accumulated_log_ratio_range=None if math.isinf(maximum) else length*maximum,
                    unbounded=math.isinf(maximum), posterior_tv_bound=float(r['posterior_tv_bounds'][i]),
                    lost_support_count=int(r['lost_support'].sum()), supported_observations=int(r['supported_observations'].sum()),
                    stored_law_bytes=int(r['stored'].nbytes), maximum_reference_normalization_change=float(abs(r['reference_normalization_delta']).max()))
                actual = lookup.pop((lineage, dtype, length)); assert set(actual) == set(expected)
                for k, v in expected.items():
                    if isinstance(v, float): assert math.isclose(actual[k], v, rel_tol=2e-13, abs_tol=2e-13), k
                    else: assert actual[k] == v, k
                strata.append(expected); groups[dtype, length].append(expected)
    assert not lookup
    cells = []
    for (dtype, length), rr in groups.items():
        assert len(rr) == 8
        cells.append(dict(storage_dtype=dtype, length=length, laws=8,
            mean_posterior_tv_bound=math.fsum(r['posterior_tv_bound'] for r in rr)/8,
            minimum_posterior_tv_bound=min(r['posterior_tv_bound'] for r in rr),
            maximum_posterior_tv_bound=max(r['posterior_tv_bound'] for r in rr),
            lost_support_count=sum(r['lost_support_count'] for r in rr),
            unbounded_laws=sum(r['unbounded'] for r in rr), stored_law_bytes=rr[0]['stored_law_bytes']))
    return dict(passed=True, numerical_acceptance=True, original_rows=72,
        law_strata=strata, equal_law_cells=cells, raw_arrays=24,
        checks='independent struct casts, scalar normalization/logs, all supports and extrema, exponential bound, all original rows and equal-law regroup')


def read_gzip(path):
    import json
    return json.loads(gzip.decompress(path.read_bytes()))
