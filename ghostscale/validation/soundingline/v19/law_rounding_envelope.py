"""Finite supplied-law rounding bounds, independent of a realized history."""
import gzip
from itertools import combinations
import numpy as np
from ..v18_3.io import read, write, canonical, file_digest

DTYPES = ('float64', 'float32', 'float16')
MODES = ('direct', 'row-normalized')


def evaluate(law, dtype, mode):
    law = np.asarray(law, dtype=np.float64)
    if (law.shape != (16, 4, 8) or not np.isfinite(law).all()
            or (law < 0).any() or np.max(abs(law.sum(-1)-1)) > 1e-12
            or dtype not in DTYPES or mode not in MODES):
        raise ValueError('law/design')
    stored = law.astype(dtype)
    reconstructed = stored.astype(np.float64)
    mass = reconstructed.sum(-1)
    if (mass == 0).any():
        raise ValueError('empty positive law row; no floor permitted')
    if mode == 'row-normalized':
        reconstructed /= mass[..., None]
    delta = reconstructed-law
    lower, upper = delta.min(0), delta.max(0)
    row_squared = (delta*delta).sum(-1)
    squared_bound = row_squared.max(0)
    lost = (law > 0) & (stored == 0)
    # Enumerate outcomes explicitly. Direct forecasts need not sum to one.
    one_hot_loss_change = np.zeros((16, 4))
    for outcome in range(8):
        target = np.eye(8)[outcome]
        change = delta*(reconstructed+law-2*target)
        one_hot_loss_change += law[:, :, outcome]*change.sum(-1)
    return dict(original=law, stored=stored, reconstructed=reconstructed,
        delta=delta, lower=lower, upper=upper,
        lower_attainers=delta == lower, upper_attainers=delta == upper,
        row_squared=row_squared, squared_bound=squared_bound,
        squared_attainers=row_squared == squared_bound,
        row_mass_drift=reconstructed.sum(-1)-1,
        cast_row_mass_drift=mass-1, lost_support=lost,
        lost_support_mass=(law*lost).sum(-1),
        expected_one_hot_loss_change=one_hot_loss_change)


def check_mixture(raw, weights):
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (16,) or not np.isfinite(weights).all()
            or (weights < 0).any() or abs(weights.sum()-1) > 1e-14):
        raise ValueError('normalized nonnegative mixture required')
    error = np.einsum('s,sce->ce', weights, raw['delta'])
    slack = 32*np.finfo(float).eps*max(1e-300, float(abs(raw['delta']).max()))
    squared_slack = 64*np.finfo(float).eps*max(1e-300, float(raw['row_squared'].max()))
    return bool(np.all(error >= raw['lower']-slack)
        and np.all(error <= raw['upper']+slack)
        and np.all((error*error).sum(-1) <= raw['squared_bound']+squared_slack))


def fixture():
    law = np.full((16, 4, 8), 1/8)
    for state in range(16):
        for context in range(4):
            shift = (state-7)*0.0000031+(context-1)*0.0000013
            law[state, context, 0] += shift
            law[state, context, 1] -= shift
    return law


def controls():
    raw = evaluate(fixture(), 'float16', 'direct')
    exact = evaluate(np.full((16, 4, 8), 1/8), 'float16', 'direct')
    broken = dict(raw, upper=raw['upper']-1e-4)
    return {'live:rounding_detected': bool(abs(raw['delta']).max() > 0),
        'placebo:exactly_representable': bool(abs(exact['delta']).max() == 0),
        'positive:all_vertices': all(check_mixture(raw, w) for w in np.eye(16)),
        'negative:undersized_bound_rejected': not check_mixture(broken, np.full(16, 1/16))}


def run(root, plan, pulse):
    cfg = plan['design']
    if cfg['storage_dtypes'] != list(DTYPES) or cfg['reconstruction_modes'] != list(MODES):
        raise ValueError('frozen variants')
    checks = controls()
    write(root/'CONTROLS.json', checks)
    if not all(checks.values()):
        raise ValueError('controls')
    for name, digest in cfg['input_files'].items():
        if file_digest(root/'inputs'/name) != digest:
            raise ValueError('input binding')
    rows = []
    (root/'raw').mkdir(exist_ok=True)
    for lineage in cfg['lineages']:
        pulse(phase='law-rounding-envelope', lineage=lineage)
        law = read(root/'inputs'/f'{lineage}-law.json')
        for dtype in DTYPES:
            for mode in MODES:
                raw = evaluate(law, dtype, mode)
                if not all(check_mixture(raw, w) for w in np.eye(16)):
                    raise ValueError('vertex bound')
                if not check_mixture(raw, np.full(16, 1/16)):
                    raise ValueError('uniform mixture bound')
                for a, b in combinations(range(16), 2):
                    for weight in (.25, .5, .75):
                        w = np.zeros(16); w[a] = weight; w[b] = 1-weight
                        if not check_mixture(raw, w):
                            raise ValueError('rational two-state bound')
                np.savez_compressed(root/'raw'/f'{lineage}-{dtype}-{mode}_points.npz', **raw)
                for context in range(4):
                    rows.append(dict(lineage=lineage, storage_dtype=dtype, reconstruction=mode,
                        context=context, maximum_absolute_coordinate_error=float(abs(raw['delta'][:, context]).max()),
                        squared_forecast_bound=float(raw['squared_bound'][context]),
                        maximum_row_mass_drift=float(abs(raw['row_mass_drift'][:, context]).max()),
                        cast_maximum_row_mass_drift=float(abs(raw['cast_row_mass_drift'][:, context]).max()),
                        lost_support_count=int(raw['lost_support'][:, context].sum()),
                        maximum_lost_support_mass=float(raw['lost_support_mass'][:, context].max()),
                        maximum_vertex_expected_loss_change=float(raw['expected_one_hot_loss_change'][:, context].max()),
                        stored_law_bytes=int(raw['stored'].nbytes)))
    (root/'raw/law_rounding_summary_points.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
    write(root/'EVIDENCE_ROLES.json', dict(reader='no new reader inputs',
        evaluator='supplied laws, stored arrays, exact rounding differences, attaining maker states and fixed mixture checks',
        scope='finite algebraic ruler over all state mixtures; no realized-history, learning, process correspondence or human-intent claim'))
    return dict(controls=checks, lineages=len(cfg['lineages']), rows=len(rows), variants=6,
        mixture_checks_per_variant=377, numerical_acceptance=False,
        scope='supplied-law storage bounds on existing development laws; no new fits or protected lineages')
