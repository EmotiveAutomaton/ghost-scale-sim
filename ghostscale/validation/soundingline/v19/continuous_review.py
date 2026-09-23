"""Independent scalar reconstruction of binary-channel continuum certificates.

No production continuum routine is imported. Native population mass is inherited
from its verified parent; legal mechanics and all conditional sums are rebuilt.
"""
from fractions import Fraction
from itertools import product
import math
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import input_privilege_review as V

FIELDS = ('skill', 'belief')
MODELS = ('uniform-legal', 'native-law')
GRID = tuple(Fraction(i, 32) for i in range(16, 33))


def scalar(legal, endpoints, weights, axis, bit, reliability):
    """Sum likelihood-weighted individual hypotheses, not endpoint coefficients."""
    r = float(reliability)
    if not math.isfinite(r) or not .5 <= r <= 1:
        raise ValueError('reliability')
    if not legal or len(legal) != len(endpoints) or len(legal) != len(weights):
        raise ValueError('law shape')
    if axis not in (0, 1) or bit not in (0, 1):
        raise ValueError('field/reply')
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('weights')
    V.V.near([math.fsum(weights)], [1.])
    terms = [float(w)*(r if q[axis] == bit else 1-r)
             for q, w in zip(legal, weights, strict=True)]
    z = math.fsum(terms)
    if not z:
        return None, z
    return [math.fsum(w for e, w in zip(endpoints, terms) if e == y)/z
            for y in range(8)], z


def rational_checks():
    """Exact fractions test the identity; they do not replace its algebraic proof."""
    cases = [((1, 2), (3, 4)), ((0, 0), (1, 2)), ((1, 2), (0, 0)),
             ((1, 2), (2, 4)), ((1, 0), (0, 3))]
    checked = impossible = 0
    for hs, ms in cases:
        h = tuple(map(Fraction, hs)); m = tuple(map(Fraction, ms))
        H, M = sum(h), sum(m)
        lo = [(x+y)/(H+M) for x, y in zip(h, m)]
        hi = [x/H for x in h] if H else [x/M for x in m]
        for r in GRID:
            z = r*H+(1-r)*M
            if not z:
                if r != 1 or H != 0: raise ValueError('singular rational case')
                impossible += 1
                continue
            a = (2*r-1)*H/z
            if not 0 <= a <= 1: raise ValueError('rational mixing weight')
            p = [(r*x+(1-r)*y)/z for x, y in zip(h, m)]
            if p != [(1-a)*x+a*y for x, y in zip(lo, hi)]:
                raise ValueError('rational interpolation')
            checked += 1
    return dict(equalities=checked, impossible=impossible)


def review(original, output, finite, pulse=lambda **kw: None):
    cfg = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json')
    base = original/'inputs'; fcfg = read(finite/'PLAN.json')['design']
    if (cfg['fields'] != list(FIELDS) or cfg['models'] != list(MODELS)
        or cfg['rules'] != list(V.T.RULES) or cfg['grid_denominator'] != 32
        or cfg['grid_numerators'] != list(range(16, 33))):
        raise ValueError('design')
    for key in ('lineages', 'rules', 'models', 'queries', 'fields', 'input_files'):
        if cfg[key] != fcfg[key]: raise ValueError('finite reference population')
    if fcfg['reliabilities'] != [.5, .75, 1.]: raise ValueError('finite reference channels')
    laws = read(base/'DISCLOSURE_LAWS.json')
    lookup = {(r['lineage'], r['rule'], r['model'], r['reader_id']): r for r in laws}
    if len(lookup) != len(laws): raise ValueError('duplicate law')
    near = V.V.near; error = 0.; finite_error = 0.; cells = []; seen = set(); used = set()
    count = singular_count = grid_count = finite_count = 0; roster = None
    rational = rational_checks()
    vector_keys = {'hit', 'miss', 'low', 'high_limit', 'lower', 'upper'}
    scalar_keys = {'hit_mass', 'miss_mass', 'diameter'}
    boolean_keys = {'high_compatible', 'collapsed_binary64'}
    grid_keys = {'grid_forecasts', 'grid_mass', 'grid_weight', 'grid_compatible', 'grid_interpolation_error'}
    for lin, rule, model in product(cfg['lineages'], cfg['rules'], MODELS):
        pulse(phase='independent-continuum', lineage=lin, rule=rule, model=model)
        name = f'{lin}-{rule}-{model}_points.npz'; seen.add(name)
        with np.load(base/'forecasts'/name, allow_pickle=False) as parent, \
             np.load(original/'certificates'/name, allow_pickle=False) as saved, \
             np.load(finite/'sets'/name, allow_pickle=False) as prior:
            qs = [tuple(map(int, q)) for q in parent['queries']]; n = len(qs)
            if n != cfg['queries'] or len(set(qs)) != n: raise ValueError('query roster')
            if roster is None: roster = qs
            elif roster != qs: raise ValueError('query identity')
            mass = parent['mass']
            if mass.shape != (n,) or not np.isfinite(mass).all() or np.any(mass < 0):
                raise ValueError('native mass')
            error = max(error, near([math.fsum(mass)], [1.]),
                        near(parent['targets'], [V.T.endpoint(q, rule) for q in qs], 0))
            groups = V.groups_for(qs, 'omit-both'); ids = sorted(groups)
            if read(base/'MEMBERSHIP.json')['omit-both'] != groups: raise ValueError('membership')
            shape = (len(ids), 2, 2); index = {q: i for i, q in enumerate(qs)}
            if set(saved.files) != vector_keys | scalar_keys | boolean_keys | grid_keys:
                raise ValueError('certificate schema')
            for k in saved.files:
                expected = (*shape, 8) if k in vector_keys else shape
                if k in grid_keys: expected = (*shape, 17, 8) if k == 'grid_forecasts' else (*shape, 17)
                if saved[k].shape != expected or not np.isfinite(saved[k]).all():
                    raise ValueError('certificate shape/nonfinite')
            diameters = []; binary_collapsed = tolerant_collapsed = singular = valid_count = 0
            max_interpolation_error = 0.
            # NPZ members are cached once, avoiding repeated decompression per cell.
            d = {k: saved[k] for k in saved.files}; f = {k: prior[k] for k in prior.files}
            error = max(error, near(f['queries'], qs, 0), near(f['mass'], mass, 0))
            for gi, key in enumerate(ids):
                ident = (lin, rule, model, key); used.add(ident); law = lookup[ident]
                g = groups[key]; legal = g['legal_completions']; ends = [V.T.endpoint(q, rule) for q in legal]
                raw = [float(mass[index[tuple(q)]]) if tuple(q) in index else 0. for q in legal]
                total = math.fsum(raw)
                weights = [w/total for w in raw] if model == 'native-law' and total else [1/len(legal)]*len(legal)
                if law['legal_completions'] != legal or law['endpoints'] != ends: raise ValueError('legal mechanics')
                error = max(error, near(law['conditional_weights'], weights))
                for axis, bit in product((0, 1), repeat=2):
                    ix = gi, axis, bit
                    # Use exact sums of binary64 weights to check saved partitions.
                    h, m = [[float(sum((Fraction(float(w)) for q, e, w in zip(legal, ends, law['conditional_weights'])
                                  if (q[axis] == bit) == match and e == y), Fraction()))
                             for y in range(8)] for match in (True, False)]
                    H, M = math.fsum(h), math.fsum(m)
                    error = max(error, near(d['hit'][ix], h, 0), near(d['miss'][ix], m, 0),
                                near([d['hit_mass'][ix], d['miss_mass'][ix]], [H, M], 0))
                    lo, _ = scalar(legal, ends, weights, axis, bit, .5)
                    hi, high_mass = scalar(legal, ends, weights, axis, bit, 1.)
                    compatible = high_mass > 0
                    if not compatible: hi = lo
                    lower = [min(x, y) for x, y in zip(lo, hi)]
                    upper = [max(x, y) for x, y in zip(lo, hi)]
                    diameter = .5*math.fsum(abs(x-y) for x, y in zip(lo, hi))
                    for k, v in [('low', lo), ('high_limit', hi), ('lower', lower), ('upper', upper), ('diameter', diameter)]:
                        error = max(error, near(d[k][ix], v))
                    error = max(error, near(d['high_compatible'][ix], compatible, 0))
                    # Exact equality is a stored-representation property, kept distinct
                    # from the independent scalar numerical agreement above.
                    exact = bool(np.array_equal(d['low'][ix], d['high_limit'][ix]))
                    error = max(error, near(d['collapsed_binary64'][ix], exact, 0))
                    binary_collapsed += int(exact); tolerant_collapsed += int(diameter <= 1e-12)
                    singular += int(not compatible); diameters.append(diameter)
                    for ri, r in enumerate(GRID):
                        j = (*ix, ri); p, z = scalar(legal, ends, weights, axis, bit, r)
                        valid = p is not None
                        error = max(error, near(d['grid_mass'][j], z), near(d['grid_compatible'][j], valid, 0))
                        if not valid:
                            if r != 1 or compatible: raise ValueError('unexpected singularity')
                            for k in ('grid_forecasts', 'grid_weight', 'grid_interpolation_error'):
                                error = max(error, near(d[k][j], np.zeros_like(d[k][j]), 0))
                            continue
                        a = float(2*r-1)*high_mass/z
                        if not 0 <= a <= 1: raise ValueError('mixing weight')
                        error = max(error, near(d['grid_forecasts'][j], p), near(d['grid_weight'][j], a))
                        interpolated = [(1-a)*x+a*y for x, y in zip(lo, hi)]
                        error = max(error, near(p, interpolated))
                        if any(v < l-1e-12 or v > u+1e-12 for v, l, u in zip(p, lower, upper)):
                            raise ValueError('coordinate containment')
                        # Recompute the recorded binary64 error from its stored operands.
                        stored_a = d['grid_weight'][j]
                        measured = float(np.max(np.abs(d['grid_forecasts'][j]-((1-stored_a)*d['low'][ix]+stored_a*d['high_limit'][ix]))))
                        error = max(error, near(d['grid_interpolation_error'][j], measured, 0))
                        if measured > 1e-12: raise ValueError('interpolation error')
                        max_interpolation_error = max(max_interpolation_error, measured)
                        valid_count += 1
                    for ai, ri in enumerate((0, 8, 16)):
                        j = (*ix, ri); is_valid = bool(d['grid_compatible'][j])
                        finite_error = max(finite_error, near(f['compatible'][ix][ai], is_valid, 0),
                                           near(f['modeled_reply_mass'][ix][ai], d['grid_mass'][j]))
                        if is_valid:
                            finite_error = max(finite_error, near(f['candidates'][ix][ai], d['grid_forecasts'][j]))
                            finite_count += 1
                    for k in ('lower', 'upper', 'diameter'):
                        finite_error = max(finite_error, near(f[k][ix], d[k][ix]))
            cell = dict(lineage=lin, rule=rule, model=model, certificates=len(diameters),
                        singular_endpoints=singular, collapsed_binary64=binary_collapsed,
                        collapsed_tolerance=tolerant_collapsed, grid_valid=valid_count,
                        maximum_diameter=max(diameters), maximum_interpolation_error=max_interpolation_error)
            cells.append(cell); count += len(diameters); singular_count += singular; grid_count += valid_count
    if used != set(lookup): raise ValueError('law coverage')
    for folder in (base/'forecasts', original/'certificates', finite/'sets'):
        if {p.name for p in folder.glob('*.npz')} != seen: raise ValueError('file coverage')
    error = max(error, V.compare(summary['cells'], cells))
    packets = {}
    for g in groups.values():
        for field, bit in product(('skill', 'belief_error'), (0, 1)):
            p = dict(g['reader'], requested_field=field, reply_value=bit, reliability_interval=[.5, 1.])
            if set(p) != {'initial', 'operations', 'requested_field', 'reply_value', 'reliability_interval'}:
                raise ValueError('reader allowlist')
            packets[digest(p)] = p
    if read(original/'reader/REPLIES.json') != packets or {p.name for p in (original/'reader').iterdir()} != {'REPLIES.json'}:
        raise ValueError('reader projection')
    expected_index = dict(group_ids=ids, fields=list(FIELDS), replies=[0, 1],
        grid=[[r.numerator, r.denominator] for r in GRID],
        singular='truthful endpoint excluded when hit mass is zero; high_limit is then the left limit',
        unavailable_grid_vector='zero placeholder, accompanied by grid_compatible=false; never a forecast',
        theorem='nonnegative affine endpoint mass gives denominator-weighted convex interpolation; exact real algebra')
    if read(original/'evaluator/INDEX.json') != expected_index: raise ValueError('index')
    if read(original/'EVIDENCE_ROLES.json') != dict(reader='initial artifact, proposed operations, requested bit, reply, interval only',
        evaluator='law coefficients, endpoints, certificates, probabilities, group identities; not reader inputs',
        inference='conditional forecast segment, not a chosen point forecast or decision; no new observations'):
        raise ValueError('evidence roles')
    checks = read(original/'CONTROLS.json')
    if set(checks) != {'live:nonconstant_segment', 'placebo:constant_forecast', 'positive:singular_endpoint_excluded',
                      'positive:singular_limit_constant', 'positive:exact_rational_identity'} or not all(checks.values()) or summary['controls'] != checks:
        raise ValueError('controls')
    if (summary['certificates'] != count or summary['singular_endpoints'] != singular_count
        or summary['reader_packets'] != len(packets) or summary['fits'] != 0): raise ValueError('totals')
    result = dict(passed=True, cells=len(cells), certificates=count, singular_endpoints=singular_count,
        compatible_grid_vectors=grid_count, finite_reference_vectors=finite_count, law_rows=len(used),
        reader_packets=len(packets), max_error=error, finite_reference_max_error=finite_error, rational_checks=rational,
        scope='complete scalar conditional and legal-mechanics reconstruction; native mass inherited from verified parent; finite grid validates implementation, real algebra supplies continuum proof; no learned access')
    write(output/'RECONSTRUCTED_CELLS.json', cells); write(output/'NUMERICAL_REVIEW.json', result)
    return result


def run(root, plan, pulse):
    cfg = plan['design']; base = root/'inputs'
    for n, h in cfg['input_files'].items():
        if file_digest(base/n) != h: raise ValueError('input binding')
    if file_digest(base/'original/PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target binding')
    if file_digest(base/'finite/PLAN.json') != cfg['finite_plan_sha256']: raise ValueError('finite binding')
    return dict(review(base/'original', root, base/'finite', pulse), controls={
        'live:complete_scalar_reconstruction': True, 'placebo:constant_and_singular_limits': True,
        'positive:finite_set_agreement_and_reader_allowlists': True})
