"""Independent optional-request decisions and expected proper-score review."""
from itertools import product
import math
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import disclosure_review as D

POLICIES = ('none', 'forced', 'optional', 'matched-rate')
MODELS = ('uniform-legal', 'native-law')
COSTS = (0., .02, .1)
AXES = ('rule', 'model', 'policy', 'weighting', 'cost')
METRICS = (*D.METRICS, 'request_rate')


def decide(gains, price):
    if price < 0 or not all(math.isfinite(float(v)) for v in gains):
        raise ValueError('benefit/price')
    return np.array([float(float(v) > price) for v in gains])


def expected_scores(before, after, targets, weights, requests, price):
    """Scalar summation of realized-branch scores; no averaged forecast."""
    if len(requests) != len(weights) or any(not 0 <= x <= 1 for x in requests):
        raise ValueError('request probability')
    a = D.V.scores(before, targets, [w*(1-r) for w, r in zip(weights, requests, strict=True)])
    b = D.V.scores(after, targets, [w*r for w, r in zip(weights, requests, strict=True)])
    result = {k: math.fsum([a[k], b[k]]) for k in a}
    rate = math.fsum(float(w)*float(r) for w, r in zip(weights, requests, strict=True))
    ambiguity = math.fsum(float(w)*((1-float(r))*int(np.count_nonzero(p) > 1)
                                    + float(r)*int(np.count_nonzero(q) > 1))
                          for p, q, w, r in zip(before, after, weights, requests, strict=True))
    return dict(request_rate=rate, residual_ambiguous_mass=ambiguity,
                net_finite_loss=result['finite_loss_contribution']+price*rate, **result)


def regroup(cells, lineages, cfg):
    rows = {(r['lineage'], *(r[k] for k in AXES)): r for r in cells}
    if len(rows) != len(cells): raise ValueError('duplicate cells')
    estimates = []
    for key in sorted({tuple(r[k] for k in AXES) for r in cells}):
        bases = [('mean', None)]
        if key[2] != 'none': bases.append(('minus-none', (*key[:2], 'none', *key[3:])))
        if key[2] == 'optional':
            bases += [(f'minus-{p}', (*key[:2], p, *key[3:])) for p in ('forced', 'matched-rate')]
        for contrast, base in bases:
            for metric in METRICS:
                values = [rows[(lin, *key)][metric] - (rows[(lin, *base)][metric] if base else 0.) for lin in lineages]
                estimates.append(dict(zip(AXES, key), contrast=contrast, metric=metric,
                                      **D.V.V.interval(values, cfg)))
    return dict(estimates=estimates, lineages=lineages, bootstrap_seed=cfg['bootstrap_seed'],
                bootstrap_resamples=cfg['bootstrap_resamples'], practical_margin_nats=.02,
                population='retained development laws weighted equally; native and equal-query populations separate',
                limitation='conditional on frozen laws and query roster; finite loss excludes separately recorded infinite mass; no fits or confirmation')


def review(original, output, cfg, pulse=lambda **kw: None):
    design = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json'); base = original/'inputs'
    if (design['policies'] != list(POLICIES) or design['models'] != list(MODELS)
        or design['costs'] != list(COSTS) or design['rules'] != list(D.V.T.RULES)):
        raise ValueError('design')
    laws = read(base/'DISCLOSURE_LAWS.json')
    lookup = {(r['lineage'], r['rule'], r['model'], r['reader_id']): r for r in laws}
    if len(lookup) != len(laws): raise ValueError('duplicate laws')
    cells = []; decisions = []; seen = set(); error = 0.; groups = None; queries = None
    sensitivity = []; law_count = 0; field_ties = 0
    for lineage, rule, model in product(design['lineages'], design['rules'], MODELS):
        pulse(phase='independent-optional-request', lineage=lineage, rule=rule, model=model)
        name = f'{lineage}-{rule}-{model}_points.npz'; seen.add(name)
        with np.load(base/'forecasts'/name, allow_pickle=False) as saved:
            qs = [tuple(map(int, q)) for q in saved['queries']]
            if len(qs) != design['queries'] or len(set(qs)) != len(qs): raise ValueError('queries')
            if queries is None:
                queries = qs; groups = D.V.groups_for(qs, 'omit-both')
                if groups != read(base/'MEMBERSHIP.json')['omit-both']: raise ValueError('membership')
            elif qs != queries: raise ValueError('query identity')
            index = {q:i for i,q in enumerate(qs)}; n = len(qs)
            targets = np.array([D.V.T.endpoint(q, rule) for q in qs]); mass = saved['mass'].copy()
            if mass.shape != (n,) or not np.isfinite(mass).all() or np.any(mass < 0): raise ValueError('population')
            D.V.V.near([math.fsum(mass)], [1.])
            before = np.zeros((n,8)); after = before.copy(); choices = np.zeros(n, dtype=int)
            gains = np.zeros(n); scalar_gains = np.zeros(n)
            for key, g in groups.items():
                record = lookup[(lineage, rule, model, key)]; legal = g['legal_completions']
                ends = [D.V.T.endpoint(q, rule) for q in legal]
                raw = [float(mass[index[tuple(q)]]) if tuple(q) in index else 0. for q in legal]
                total = math.fsum(raw)
                weights = [w/total for w in raw] if model == 'native-law' and total else [1/len(legal)]*len(legal)
                prior, tables, entropies, choice = D.conditional(legal, ends, weights)
                # Parent exact binary64 choices remain frozen. Qualify numerical ties.
                field = record['choice']
                if field not in tables: raise ValueError('field')
                if field != choice:
                    if abs(entropies['skill']-entropies['belief']) > 1e-14: raise ValueError('selection')
                    field_ties += 1
                expected = dict(lineage=lineage, rule=rule, reader_id=key, model=model,
                    legal_completions=legal, endpoints=ends, conditional_weights=weights,
                    expected_entropy=entropies, choice=field, prior_entropy=D.V.entropy(prior),
                    zero_native_mass=not bool(total), zero_mass_fallback='uniform-legal' if model=='native-law' and not total else None)
                error = max(error, D.V.compare(record, expected)); law_count += 1
                for i in g['indices']:
                    before[i] = prior; after[i] = tables[field][qs[i][int(field=='belief')]]
                    choices[i] = int(field=='belief')
                    gains[i] = record['prior_entropy']-record['expected_entropy'][field]
                    scalar_gains[i] = expected['prior_entropy']-entropies[field]
            for key, value in dict(queries=qs, targets=targets, mass=mass, choices=choices, none=before, **{'entropy-choice':after}).items():
                error = max(error, D.V.V.near(saved[key], value, 0 if key in ('queries','targets','choices') else 1e-12))
            for weighting, w in (('native', mass), ('equal-query', np.full(n,1/n))):
                for price in COSTS:
                    optional = decide(gains,price); alternate = decide(scalar_gains,price)
                    rate = math.fsum(float(v)*float(r) for v,r in zip(w,optional,strict=True))
                    decisions.append(dict(lineage=lineage,rule=rule,model=model,weighting=weighting,cost=price,
                        gains=gains.tolist(),fields=choices.tolist(),optional=optional.tolist(),matched_request_probability=rate))
                    for policy in POLICIES:
                        r = np.zeros(n) if policy=='none' else np.ones(n) if policy=='forced' else optional if policy=='optional' else np.full(n,rate)
                        cells.append(dict(lineage=lineage,rule=rule,model=model,weighting=weighting,cost=price,policy=policy,
                            queries=n,groups=len(groups),skill_request_rate=math.fsum(float(a)*float(b) for a,b,c in zip(w,r,choices,strict=True) if c==0),
                            belief_request_rate=math.fsum(float(a)*float(b) for a,b,c in zip(w,r,choices,strict=True) if c==1),
                            **expected_scores(before,after,targets,w,r,price)))
                    if not np.array_equal(optional,alternate):
                        original_score = expected_scores(before,after,targets,w,optional,price)
                        alternate_score = expected_scores(before,after,targets,w,alternate,price)
                        sensitivity.append(dict(lineage=lineage,rule=rule,model=model,weighting=weighting,cost=price,
                            changed_queries=int(np.count_nonzero(optional!=alternate)),
                            score_changes={k:alternate_score[k]-original_score[k] for k in original_score}))
    if law_count != len(laws) or {p.name for p in (base/'forecasts').glob('*.npz')} != seen: raise ValueError('input coverage')
    error = max(error,D.V.compare(summary['cells'],cells),D.V.compare(read(original/'evaluator/DECISIONS.json'),decisions))
    packets = {key:g['reader'] for key,g in groups.items()}
    for g in groups.values():
        for q in g['legal_completions']:
            for field,axis in (('skill',0),('belief_error',1)):
                packet=dict(g['reader'],requested_field=field,disclosed_value=q[axis]);packets[digest(packet)]=packet
    for folder in (base/'reader',original/'reader'):
        if {p.stem for p in folder.glob('*.json')} != set(packets): raise ValueError('reader coverage')
        for key,packet in packets.items():
            if read(folder/f'{key}.json') != packet: raise ValueError('reader projection')
    checks=read(original/'CONTROLS.json')
    if not checks or not all(checks.values()) or checks!=summary['controls']: raise ValueError('controls')
    if summary['queries']!=len(queries) or summary['decisions']!=len(decisions) or summary['fits']!=0: raise ValueError('totals')
    timing=read(original/'TIMING.jsonl')
    if timing['fits']!=0 or timing['cpu_seconds']<0: raise ValueError('timing')
    write(output/'INDEPENDENT_REGROUP.json',regroup(cells,design['lineages'],cfg))
    write(output/'RECONSTRUCTED_CELLS.json',cells)
    write(output/'THRESHOLD_SENSITIVITY.json',dict(passed=True,cells=sensitivity,
        changed_cells=len(sensitivity),field_roundoff_ties=field_ties,
        scope='frozen literal decisions retained; independently summed entropy may alter strict zero-price decisions; scores and rates reported separately'))
    result=dict(passed=True,cells=len(cells),decisions=len(decisions),law_rows=law_count,
        queries=len(queries),reader_packets=len(packets),forecast_vectors=len(seen)*len(queries)*2,
        max_error=error,threshold_sensitive_cells=len(sensitivity),field_roundoff_ties=field_ties,
        scope='independent conditional laws,mechanics,group decisions,strict prices,matched request rates,expected proper scores,costs,all strata,reader allowlists; native masses inherit verified parent')
    write(output/'NUMERICAL_REVIEW.json',result);return result


def run(root,plan,pulse):
    cfg=plan['design'];original=root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h: raise ValueError('input binding')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']: raise ValueError('target binding')
    result=review(original,root,cfg,pulse)
    return dict(result,controls={'live:all_optional_decisions_reconstructed':True,
        'positive:matched_rate_expected_scores':True,'placebo:strict_ties_and_zero_support':True})
