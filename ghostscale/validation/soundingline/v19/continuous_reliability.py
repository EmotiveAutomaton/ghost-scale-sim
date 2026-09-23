"""Conditional forecast segment for a binary channel with reliability in [1/2,1].

The algebraic certificate is in CONTINUOUS_RELIABILITY_ADMISSION_PROTOCOL.md.
The finite grid checks the implementation, not the continuum theorem.
"""
from fractions import Fraction
from itertools import product
import math
import time
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import metadata_disclosure as D

FIELDS = ('skill', 'belief')
GRID = tuple(Fraction(i, 32) for i in range(16, 33))


def coefficients(legal, endpoints, weights, field, reply):
    if field not in FIELDS or reply not in (0, 1): raise ValueError('field/reply')
    w = np.asarray(weights, dtype=float)
    if (len(legal) != len(endpoints) or len(legal) != len(w) or not len(w)
        or not np.isfinite(w).all() or np.any(w < 0)
        or abs(math.fsum(w)-1) > 1e-12): raise ValueError('weights')
    axis = FIELDS.index(field)
    if any(len(q) != 6 or q[axis] not in (0, 1) for q in legal): raise ValueError('legal bits')
    if any(e not in range(8) for e in endpoints): raise ValueError('endpoints')
    # Positive partitions avoid cancellation in the affine form.
    hit = np.array([math.fsum(float(x) for q, e, x in zip(legal, endpoints, w)
                             if q[axis] == reply and e == j) for j in range(8)])
    miss = np.array([math.fsum(float(x) for q, e, x in zip(legal, endpoints, w)
                              if q[axis] != reply and e == j) for j in range(8)])
    return hit, miss


def certify(hit, miss):
    h = np.asarray(hit, float); m = np.asarray(miss, float)
    if (h.shape != (8,) or m.shape != (8,) or not np.isfinite(h).all()
        or not np.isfinite(m).all() or np.any(h < 0) or np.any(m < 0)):
        raise ValueError('endpoint masses')
    H = math.fsum(h); M = math.fsum(m)
    if H+M <= 0: raise ValueError('empty observation law')
    low = (h+m)/(H+M)
    # This is a limit, never an arbitrary fallback at an impossible observation.
    high = h/H if H > 0 else m/M
    return dict(hit=h, miss=m, hit_mass=H, miss_mass=M,
                low=low, high_limit=high, high_compatible=H > 0,
                lower=np.minimum(low, high), upper=np.maximum(low, high),
                diameter=float(.5*np.abs(low-high).sum()),
                collapsed_binary64=bool(np.array_equal(low, high)))


def evaluate(certificate, reliability):
    r = float(reliability)
    if not math.isfinite(r) or not .5 <= r <= 1: raise ValueError('reliability')
    h, m = certificate['hit'], certificate['miss']
    mass = r*certificate['hit_mass']+(1-r)*certificate['miss_mass']
    if mass == 0: return None, 0., None
    direct = (r*h+(1-r)*m)/mass
    weight = (2*r-1)*certificate['hit_mass']/mass
    return direct, mass, weight


def rational_controls():
    # Independent exact-rational expansion, including a singular endpoint.
    cases = [([Fraction(1, 8), Fraction(3, 8)], [Fraction(1, 4), Fraction(1, 4)]),
             ([Fraction(0), Fraction(0)], [Fraction(1, 3), Fraction(2, 3)]),
             ([Fraction(1), Fraction(0)], [Fraction(0), Fraction(0)])]
    equalities = singular = 0
    for h, m in cases:
        H, M = sum(h), sum(m); low = [(x+y)/(H+M) for x, y in zip(h, m)]
        high = [x/H for x in h] if H else [x/M for x in m]
        for r in GRID:
            z = r*H+(1-r)*M
            if not z: singular += 1; continue
            a = (2*r-1)*H/z
            assert 0 <= a <= 1
            assert [(r*x+(1-r)*y)/z for x, y in zip(h, m)] == [(1-a)*x+a*y for x, y in zip(low, high)]
            equalities += 1
    return equalities == 50 and singular == 1


def controls():
    h = np.zeros(8); h[0] = .5; m = np.zeros(8); m[1] = .5
    c = certify(h, m); singular = certify(h*0, m*2)
    p, _, a = evaluate(c, Fraction(3, 4))
    null = certify(h, h)
    return {'live:nonconstant_segment': c['diameter'] == .5 and a == .5 and np.array_equal(p[:2], [.75, .25]),
            'placebo:constant_forecast': null['collapsed_binary64'],
            'positive:singular_endpoint_excluded': not singular['high_compatible'] and evaluate(singular, 1)[0] is None,
            'positive:singular_limit_constant': bool(np.array_equal(singular['low'], singular['high_limit'])),
            'positive:exact_rational_identity': rational_controls()}


def run(root, plan, pulse):
    cfg = plan['design']; base = root/'inputs'; start = time.process_time(); checks = controls()
    if (cfg['fields'] != list(FIELDS) or cfg['models'] != list(D.MODELS)
        or cfg['grid_denominator'] != 32 or cfg['grid_numerators'] != list(range(16, 33))
        or not all(checks.values())): raise ValueError('continuous design')
    for n, h in cfg['input_files'].items():
        if file_digest(base/n) != h: raise ValueError('input binding')
    groups = read(base/'MEMBERSHIP.json')['omit-both']; ids = sorted(groups)
    laws = read(base/'DISCLOSURE_LAWS.json')
    lookup = {(r['lineage'], r['rule'], r['model'], r['reader_id']): r for r in laws}
    if len(lookup) != len(laws): raise ValueError('duplicate laws')
    for folder in ('reader', 'certificates', 'evaluator'): (root/folder).mkdir()
    packets = {}
    for g in groups.values():
        for field, bit in product(('skill', 'belief_error'), (0, 1)):
            p = dict(g['reader'], requested_field=field, reply_value=bit, reliability_interval=[.5, 1.])
            packets[digest(p)] = p
    write(root/'reader/REPLIES.json', packets); write(root/'CONTROLS.json', checks)
    write(root/'evaluator/INDEX.json', dict(group_ids=ids, fields=FIELDS, replies=[0, 1],
          grid=[[r.numerator, r.denominator] for r in GRID],
          singular='truthful endpoint excluded when hit mass is zero; high_limit is then the left limit',
          unavailable_grid_vector='zero placeholder, accompanied by grid_compatible=false; never a forecast',
          theorem='nonnegative affine endpoint mass gives denominator-weighted convex interpolation; exact real algebra'))
    used = set(); roster = None; cells = []; count = singular_count = 0
    for lin, rule, model in product(cfg['lineages'], cfg['rules'], cfg['models']):
        pulse(phase='continuous-reliability-certificate', lineage=lin, rule=rule, model=model)
        with np.load(base/'forecasts'/f'{lin}-{rule}-{model}_points.npz', allow_pickle=False) as parent:
            qs = [tuple(map(int, q)) for q in parent['queries']]
            if len(qs) != cfg['queries'] or len(set(qs)) != len(qs): raise ValueError('query roster')
            if roster is None: roster = qs
            elif qs != roster: raise ValueError('query identity')
            if D.P.membership(qs)['omit-both'] != groups: raise ValueError('membership')
            if not np.array_equal(parent['targets'], [D.P.T.oracle(q, rule) for q in qs]): raise ValueError('targets')
        shape = (len(ids), 2, 2)
        arrays = {k: np.zeros((*shape, 8)) for k in ('hit', 'miss', 'low', 'high_limit', 'lower', 'upper')}
        arrays.update({k: np.zeros(shape) for k in ('hit_mass', 'miss_mass', 'diameter')})
        arrays.update({k: np.zeros(shape, bool) for k in ('high_compatible', 'collapsed_binary64')})
        arrays.update(grid_forecasts=np.zeros((*shape, len(GRID), 8)), grid_mass=np.zeros((*shape, len(GRID))),
                      grid_weight=np.zeros((*shape, len(GRID))), grid_compatible=np.zeros((*shape, len(GRID)), bool),
                      grid_interpolation_error=np.zeros((*shape, len(GRID))))
        for gi, key in enumerate(ids):
            ident = (lin, rule, model, key); law = lookup[ident]; used.add(ident); g = groups[key]
            legal, ends = law['legal_completions'], law['endpoints']
            if legal != g['legal_completions'] or ends != [D.P.T.oracle(q, rule) for q in legal]: raise ValueError('legal mechanics')
            for fi, field in enumerate(FIELDS):
                for bit in (0, 1):
                    cert = certify(*coefficients(legal, ends, law['conditional_weights'], field, bit))
                    ix = gi, fi, bit
                    for k, v in cert.items(): arrays[k][ix] = v
                    for ri, reliability in enumerate(GRID):
                        p, mass, a = evaluate(cert, reliability); j = (*ix, ri)
                        arrays['grid_mass'][j] = mass
                        if p is None: continue
                        interpolated = (1-a)*cert['low']+a*cert['high_limit']
                        error = float(np.max(np.abs(p-interpolated)))
                        if (error > 1e-12 or not 0 <= a <= 1 or np.any(p < cert['lower']-1e-12)
                            or np.any(p > cert['upper']+1e-12)): raise ValueError('certificate implementation')
                        arrays['grid_forecasts'][j] = p; arrays['grid_weight'][j] = a
                        arrays['grid_compatible'][j] = True; arrays['grid_interpolation_error'][j] = error
                    count += 1; singular_count += int(not cert['high_compatible'])
        np.savez_compressed(root/'certificates'/f'{lin}-{rule}-{model}_points.npz', **arrays)
        cells.append(dict(lineage=lin, rule=rule, model=model, certificates=int(np.prod(shape)),
                          singular_endpoints=int((~arrays['high_compatible']).sum()),
                          collapsed_binary64=int(arrays['collapsed_binary64'].sum()),
                          collapsed_tolerance=int((arrays['diameter'] <= 1e-12).sum()),
                          grid_valid=int(arrays['grid_compatible'].sum()),
                          maximum_diameter=float(arrays['diameter'].max()),
                          maximum_interpolation_error=float(arrays['grid_interpolation_error'].max())))
    if used != set(lookup): raise ValueError('law coverage')
    write(root/'EVIDENCE_ROLES.json', dict(reader='initial artifact, proposed operations, requested bit, reply, interval only',
          evaluator='law coefficients, endpoints, certificates, probabilities, group identities; not reader inputs',
          inference='conditional forecast segment, not a chosen point forecast or decision; no new observations'))
    write(root/'TIMING.jsonl', dict(cpu_seconds=time.process_time()-start, fits=0))
    return dict(controls=checks, cells=cells, certificates=count, singular_endpoints=singular_count,
                reader_packets=len(packets), fits=0,
                scope='supplied binary-channel continuum certificate; grid is numerical control, not continuum proof; no learned access or historical-process claim')
