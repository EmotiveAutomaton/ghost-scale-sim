"""Finite compatible forecast sets for an unspecified binary reply channel."""
from itertools import product, combinations
import time
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import metadata_disclosure as D
from .noisy_disclosure import channel, validate_channel

ACCURACIES = (.5, .75, 1.)
FIELDS = ('skill', 'belief')
PAIRS = tuple(combinations(range(3), 2))


def describe(candidates, masses):
    """Fallback vectors remain recorded but never enter the compatible set."""
    p = np.asarray(candidates, float); m = np.asarray(masses, float)
    if (p.shape != (3, 8) or m.shape != (3,) or not np.isfinite(p).all()
        or not np.isfinite(m).all() or np.any(p < 0) or np.any(m < 0)
        or np.any(m > 1+1e-12) or not np.allclose(p.sum(1), 1, atol=1e-12, rtol=0)):
        raise ValueError('candidate set')
    valid = m > 0
    if not valid.any(): raise ValueError('empty compatible set')
    lower = p[valid].min(0); upper = p[valid].max(0)
    distances = np.array([.5*np.abs(p[i]-p[j]).sum() if valid[i] and valid[j] else -1.
                          for i, j in PAIRS])
    diameter = max(0., float(distances.max()))
    return dict(compatible=valid, lower=lower, upper=upper, pairwise_tv=distances,
                diameter=diameter, max_range=float((upper-lower).max()),
                sum_range=float((upper-lower).sum()), collapsed_exact=bool(np.all(p[valid] == p[valid][0])),
                collapsed_tolerance=diameter <= 1e-12, excluded_candidates=int((~valid).sum()))


def controls():
    p = np.zeros((3, 8)); p[:, 0] = 1
    null = describe(p, [.5, .75, 1])
    p[1] = 0; p[1, 1] = 1; p[2] = .5*(p[0]+p[1])
    live = describe(p, [.5, .5, .5]); excluded = describe(p, [.5, 0, 0])
    return {'live:nontrivial_complete_set': live['diameter'] == 1,
            'placebo:constant_endpoint_collapse': null['collapsed_exact'],
            'positive:impossible_candidates_excluded': excluded['diameter'] == 0 and excluded['excluded_candidates'] == 2,
            'positive:envelope_is_not_joint_forecast': bool(live['upper'].sum() == 2 and live['lower'].sum() == 0)}


def run(root, plan, pulse):
    cfg = plan['design']; base = root/'inputs'; start = time.process_time(); checks = controls()
    if (cfg['reliabilities'] != list(ACCURACIES) or cfg['fields'] != list(FIELDS)
        or cfg['models'] != list(D.MODELS) or not all(checks.values())): raise ValueError('set design')
    for n, h in cfg['input_files'].items():
        if file_digest(base/n) != h: raise ValueError('input binding')
    groups = read(base/'MEMBERSHIP.json')['omit-both']; ids = sorted(groups)
    laws = read(base/'DISCLOSURE_LAWS.json')
    lookup = {(r['lineage'], r['rule'], r['model'], r['reader_id']): r for r in laws}
    if len(lookup) != len(laws): raise ValueError('duplicate laws')
    for folder in ('reader', 'sets', 'evaluator'): (root/folder).mkdir()
    packets = {}
    for g in groups.values():
        for field, bit in product(('skill', 'belief_error'), (0, 1)):
            packet = dict(g['reader'], requested_field=field, reply_value=bit, reliability_candidates=list(ACCURACIES))
            packets[digest(packet)] = packet
    write(root/'reader/REPLIES.json', packets); write(root/'CONTROLS.json', checks)
    write(root/'evaluator/INDEX.json', dict(group_ids=ids, fields=FIELDS, replies=[0, 1], reliabilities=ACCURACIES,
          pair_indices=PAIRS, missing_distance=-1, note='zero modeled reply mass excludes candidate; fallback vector retained'))
    cells = []; used = set(); roster = None; total_sets = 0; fallbacks = []
    metrics = ('diameter', 'max_range', 'sum_range', 'collapsed_exact', 'collapsed_tolerance', 'excluded_candidates')
    for lin, rule, model in product(cfg['lineages'], cfg['rules'], D.MODELS):
        pulse(phase='finite-reliability-sets', lineage=lin, rule=rule, model=model)
        with np.load(base/'forecasts'/f'{lin}-{rule}-{model}_points.npz', allow_pickle=False) as parent:
            qs = [tuple(map(int, q)) for q in parent['queries']]; mass = parent['mass'].copy(); n = len(qs)
            if n != cfg['queries'] or len(set(qs)) != n: raise ValueError('query roster')
            if roster is None: roster = qs
            elif roster != qs: raise ValueError('query identity')
            if D.P.membership(qs)['omit-both'] != groups: raise ValueError('membership')
            if not np.array_equal(parent['targets'], [D.P.T.oracle(q, rule) for q in qs]): raise ValueError('targets')
            if mass.shape != (n,) or np.any(mass < 0) or not np.isfinite(mass).all() or not np.isclose(mass.sum(), 1, atol=1e-12, rtol=0): raise ValueError('mass')
            shape = (len(ids), 2, 2)
            saved = dict(candidates=np.zeros((*shape, 3, 8)), modeled_reply_mass=np.zeros((*shape, 3)),
                         compatible=np.zeros((*shape, 3), bool), lower=np.zeros((*shape, 8)), upper=np.zeros((*shape, 8)),
                         pairwise_tv=np.zeros((*shape, 3)), queries=np.array(qs), mass=mass)
            values = {k: np.zeros(shape) for k in metrics}; query_group = np.full(n, -1, dtype=np.int32)
            for gi, key in enumerate(ids):
                g = groups[key]; ident = (lin, rule, model, key); law = lookup[ident]; used.add(ident)
                legal = law['legal_completions']; ends = law['endpoints']
                if legal != g['legal_completions'] or ends != [D.P.T.oracle(q, rule) for q in legal]: raise ValueError('legal mechanics')
                channels = [channel([tuple(q) for q in legal], ends, law['conditional_weights'], a) for a in ACCURACIES]
                for ch in channels: validate_channel(ch)
                query_group[g['indices']] = gi
                for fi, field in enumerate(FIELDS):
                    for bit in (0, 1):
                        p = np.stack([ch['tables'][field][bit] for ch in channels])
                        m = np.array([ch['reply_mass'][field][bit] for ch in channels]); desc = describe(p, m)
                        saved['candidates'][gi, fi, bit] = p; saved['modeled_reply_mass'][gi, fi, bit] = m
                        for k in ('compatible', 'lower', 'upper', 'pairwise_tv'): saved[k][gi, fi, bit] = desc[k]
                        for k in metrics: values[k][gi, fi, bit] = desc[k]
                        for ai, ch in enumerate(channels):
                            if m[ai] == 0:
                                fallbacks.append(dict(lineage=lin, rule=rule, model=model, reader_id=key,
                                    field=field, reply=bit, reliability=ACCURACIES[ai], convention=ch['fallbacks'][field][bit]))
                        total_sets += 1
            if np.any(query_group < 0): raise ValueError('query coverage')
            saved.update(values); saved['query_group'] = query_group
            # Actual channels weight evaluator summaries only; no winner or point forecast is chosen.
            for fi, field in enumerate(FIELDS):
                truth = np.array([q[fi] for q in qs])
                for actual in ACCURACIES:
                    probs = np.where(truth[:, None] == np.arange(2), actual, 1-actual)
                    for weighting, w in (('native', mass), ('equal-query', np.full(n, 1/n))):
                        scores = {k: float(np.sum(w[:, None]*probs*v[query_group, fi])) for k, v in values.items()}
                        actual_index = ACCURACIES.index(actual)
                        excluded = ~saved['compatible'][query_group, fi, :, actual_index]
                        scores['actual_candidate_excluded_mass'] = float(np.sum(w[:, None]*probs*excluded))
                        cells.append(dict(lineage=lin, rule=rule, model=model, field=field,
                                          actual_accuracy=actual, weighting=weighting, queries=n, groups=len(ids), **scores))
            np.savez_compressed(root/'sets'/f'{lin}-{rule}-{model}_points.npz', **saved)
    if used != set(lookup): raise ValueError('law coverage')
    write(root/'evaluator/FALLBACKS.json', fallbacks)
    write(root/'TIMING.jsonl', dict(cpu_seconds=time.process_time()-start, fits=0, scope='exhaustive finite channel sets; no fitted or selected reliability'))
    return dict(controls=checks, cells=cells, queries=cfg['queries'], sets=total_sets,
                reader_packets=len(packets), excluded_candidates=len(fallbacks), fits=0,
                scope='finite supplied-law compatible forecast sets; coordinate envelope is not a joint forecast; no continuum,calibration,learned-access or historical-process claim')
