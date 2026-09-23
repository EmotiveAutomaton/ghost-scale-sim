"""Independent scalar checks of finite reliability sets and population summaries."""
from itertools import product, combinations
import math
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import noisy_review as N

ACCURACIES = (.5, .75, 1.)
FIELDS = ('skill', 'belief')
METRICS = ('diameter', 'max_range', 'sum_range', 'collapsed_exact',
           'collapsed_tolerance', 'excluded_candidates', 'actual_candidate_excluded_mass')
AXES = ('rule', 'model', 'field', 'actual_accuracy', 'weighting')


def regroup(cells, lineages, cfg):
    lookup = {(r['lineage'], *(r[k] for k in AXES)): r for r in cells}
    if len(lookup) != len(cells): raise ValueError('duplicate cells')
    estimates = []
    for key in sorted({tuple(r[k] for k in AXES) for r in cells}):
        for metric in METRICS:
            values = [lookup[(lin, *key)][metric] for lin in lineages]
            estimates.append(dict(zip(AXES, key), metric=metric, **N.D.V.V.interval(values, cfg)))
    return dict(estimates=estimates, lineages=lineages, bootstrap_seed=cfg['bootstrap_seed'],
                bootstrap_resamples=cfg['bootstrap_resamples'],
                scope='paired development-law means; native and equal-query populations distinct; no reliability prior')


def review(original, output, cfg, pulse=lambda **kw: None):
    design = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json'); base = original/'inputs'
    for key, expected in [('reliabilities', ACCURACIES), ('fields', FIELDS), ('models', N.MODELS), ('rules', N.D.V.T.RULES)]:
        if design[key] != list(expected): raise ValueError('design')
    laws = read(base/'DISCLOSURE_LAWS.json')
    lookup = {(r['lineage'], r['rule'], r['model'], r['reader_id']): r for r in laws}
    if len(lookup) != len(laws): raise ValueError('duplicate laws')
    near = N.D.V.V.near; error = 0.; cells = []; seen = set(); used = set(); fallbacks = []; roster = None; count = 0
    for lin, rule, model in product(design['lineages'], design['rules'], N.MODELS):
        pulse(phase='independent-finite-reliability', lineage=lin, rule=rule, model=model)
        name = f'{lin}-{rule}-{model}_points.npz'; seen.add(name)
        with np.load(base/'forecasts'/name, allow_pickle=False) as parent, np.load(original/'sets'/name, allow_pickle=False) as saved:
            qs = [tuple(map(int, q)) for q in parent['queries']]; n = len(qs); mass = parent['mass'].copy()
            if n != design['queries'] or len(set(qs)) != n: raise ValueError('query roster')
            if roster is None: roster = qs
            elif roster != qs: raise ValueError('query identity')
            if mass.shape != (n,) or np.any(mass < 0) or not np.isfinite(mass).all(): raise ValueError('population')
            error = max(error, near([math.fsum(mass)], [1.]), near(parent['targets'], [N.D.V.T.endpoint(q, rule) for q in qs], 0))
            groups = N.D.V.groups_for(qs, 'omit-both'); ids = sorted(groups); index = {q: i for i, q in enumerate(qs)}
            if read(base/'MEMBERSHIP.json')['omit-both'] != groups: raise ValueError('membership')
            expected_keys = {'candidates', 'modeled_reply_mass', 'compatible', 'lower', 'upper', 'pairwise_tv', 'queries', 'mass', 'query_group', *METRICS[:-1]}
            if set(saved.files) != expected_keys: raise ValueError('set schema')
            error = max(error, near(saved['queries'], qs, 0), near(saved['mass'], mass, 0))
            shape = (len(ids), 2, 2); values = {k: np.zeros(shape) for k in METRICS[:-1]}; masks = np.zeros((*shape, 3), bool)
            query_group = np.full(n, -1, int)
            for gi, key in enumerate(ids):
                g = groups[key]; ident = (lin, rule, model, key); used.add(ident); law = lookup[ident]
                legal = g['legal_completions']; ends = [N.D.V.T.endpoint(q, rule) for q in legal]
                raw = [float(mass[index[tuple(q)]]) if tuple(q) in index else 0. for q in legal]; total = math.fsum(raw)
                weights = [w/total for w in raw] if model == 'native-law' and total else [1/len(legal)]*len(legal)
                if law['legal_completions'] != legal or law['endpoints'] != ends: raise ValueError('legal mechanics')
                error = max(error, near(law['conditional_weights'], weights)); query_group[g['indices']] = gi
                channels = [N.conditional(legal, ends, weights, a) for a in ACCURACIES]
                for fi, field in enumerate(FIELDS):
                    for bit in (0, 1):
                        vectors = [ch['tables'][field][bit] for ch in channels]
                        masses = [ch['reply_mass'][field][bit] for ch in channels]; valid = [i for i,m in enumerate(masses) if m > 0]
                        if not valid: raise ValueError('empty compatible set')
                        masks[gi,fi,bit] = [m > 0 for m in masses]
                        error = max(error, near(saved['candidates'][gi,fi,bit], vectors), near(saved['modeled_reply_mass'][gi,fi,bit], masses),
                                    near(saved['compatible'][gi,fi,bit], masks[gi,fi,bit], 0))
                        lower = [min(vectors[i][y] for i in valid) for y in range(8)]
                        upper = [max(vectors[i][y] for i in valid) for y in range(8)]
                        pairs = [.5*math.fsum(abs(float(vectors[i][y])-float(vectors[j][y])) for y in range(8)) if i in valid and j in valid else -1.
                                 for i,j in combinations(range(3),2)]
                        diameter = max(0., *pairs); widths = [u-l for u,l in zip(upper,lower,strict=True)]
                        for metric, value in [('diameter',diameter),('max_range',max(widths)),('sum_range',math.fsum(widths)),
                                              ('collapsed_tolerance',diameter<=1e-12),('excluded_candidates',3-len(valid))]:
                            values[metric][gi,fi,bit] = value
                        # Exact binary64 equality is a representation fact, separate from scalar reconstruction tolerance.
                        stored = saved['candidates'][gi,fi,bit]
                        values['collapsed_exact'][gi,fi,bit] = all(all(stored[i,y]==stored[valid[0],y] for y in range(8)) for i in valid)
                        for metric, value in [('lower',lower),('upper',upper),('pairwise_tv',pairs)]: error=max(error,near(saved[metric][gi,fi,bit],value))
                        for ai,ch in enumerate(channels):
                            if not masses[ai]: fallbacks.append(dict(lineage=lin,rule=rule,model=model,reader_id=key,field=field,reply=bit,reliability=ACCURACIES[ai],convention=ch['fallbacks'][field][bit]))
                        count += 1
            error=max(error,near(saved['query_group'],query_group,0))
            if np.any(query_group<0):raise ValueError('query coverage')
            for metric,v in values.items():error=max(error,near(saved[metric],v,0 if metric in ('collapsed_exact','collapsed_tolerance','excluded_candidates') else 1e-12))
            for fi,field in enumerate(FIELDS):
                for ai,actual in enumerate(ACCURACIES):
                    for weighting,w in [('native',mass),('equal-query',[1/n]*n)]:
                        scores={}
                        for metric in METRICS:
                            scores[metric]=math.fsum(float(w[i])*(actual if q[fi]==bit else 1-actual)*
                                (float(not masks[query_group[i],fi,bit,ai]) if metric==METRICS[-1] else float(values[metric][query_group[i],fi,bit]))
                                for i,q in enumerate(qs) for bit in (0,1))
                        cells.append(dict(lineage=lin,rule=rule,model=model,field=field,actual_accuracy=actual,weighting=weighting,queries=n,groups=len(ids),**scores))
    if used!=set(lookup) or {p.name for p in (original/'sets').glob('*.npz')}!=seen or {p.name for p in (base/'forecasts').glob('*.npz')}!=seen:raise ValueError('coverage')
    error=max(error,N.D.V.compare(summary['cells'],cells))
    if read(original/'evaluator/FALLBACKS.json')!=fallbacks:raise ValueError('fallbacks')
    packets={}
    for g in groups.values():
        for field,bit in product(('skill','belief_error'),(0,1)):
            packet=dict(g['reader'],requested_field=field,reply_value=bit,reliability_candidates=list(ACCURACIES));packets[digest(packet)]=packet
    if read(original/'reader/REPLIES.json')!=packets or {p.name for p in (original/'reader').iterdir()}!={'REPLIES.json'}:raise ValueError('reader projection')
    expected_index=dict(group_ids=ids,fields=list(FIELDS),replies=[0,1],reliabilities=list(ACCURACIES),pair_indices=[list(p) for p in combinations(range(3),2)],missing_distance=-1,note='zero modeled reply mass excludes candidate; fallback vector retained')
    if read(original/'evaluator/INDEX.json')!=expected_index:raise ValueError('index')
    checks=read(original/'CONTROLS.json')
    if not checks or not all(checks.values()) or checks!=summary['controls']:raise ValueError('controls')
    if (summary['sets']!=count or summary['reader_packets']!=len(packets) or summary['queries']!=len(roster)
        or summary['excluded_candidates']!=len(fallbacks) or summary['fits']!=0):raise ValueError('totals')
    write(output/'RECONSTRUCTED_CELLS.json',cells);write(output/'INDEPENDENT_REGROUP.json',regroup(cells,design['lineages'],cfg))
    result=dict(passed=True,cells=len(cells),sets=count,law_rows=len(used),reader_packets=len(packets),excluded_candidates=len(fallbacks),max_error=error,
        scope='scalar mechanics and conditional laws,complete candidate masks,envelopes,pairs,populations and roles; native mass inherited from verified parent; finite set only')
    write(output/'NUMERICAL_REVIEW.json',result);return result


def run(root,plan,pulse):
    cfg=plan['design'];original=root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('target binding')
    return dict(review(original,root,cfg,pulse),controls={'live:complete_scalar_candidate_reconstruction':True,'positive:impossible_support_excluded':True,'placebo:complete_reader_allowlist_and_population_reconstruction':True})
