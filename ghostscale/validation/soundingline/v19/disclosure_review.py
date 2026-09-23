"""Independent reconstruction of fixed metadata requests and proper scores."""
from itertools import product
import math
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import input_privilege_review as V

POLICIES = ('none', 'skill', 'belief', 'entropy-choice', 'equal-request', 'all-fields')
MODELS = ('uniform-legal', 'native-law')
COSTS = (0., .02, .1)
AXES = ('rule', 'model', 'policy', 'weighting', 'cost')
METRICS = ('finite_loss_contribution', 'infinite_loss_mass', 'squared_error',
           'true_probability', 'residual_ambiguous_mass', 'net_finite_loss',
           'skill_request_rate', 'belief_request_rate')


def conditional(legal, ends, weights):
    """Scalar law sums; no producer selection or scoring code."""
    def law(ids):
        mass = math.fsum(weights[i] for i in ids)
        return np.array([math.fsum(weights[i] for i in ids if ends[i] == t)/mass
                         if mass else sum(ends[i] == t for i in ids)/len(ids)
                         for t in range(8)])
    prior = law(range(len(legal))); tables = {}; expected = {}
    for name, axis in (('skill', 0), ('belief', 1)):
        tables[name] = {}; terms = []
        for value in sorted({q[axis] for q in legal}):
            ids = [i for i, q in enumerate(legal) if q[axis] == value]
            tables[name][value] = law(ids)
            terms.append(math.fsum(weights[i] for i in ids)*V.entropy(tables[name][value]))
        expected[name] = math.fsum(terms)
    choice = 'skill' if expected['skill'] <= expected['belief'] else 'belief'
    return prior, tables, expected, choice


def regroup(cells, lineages, cfg):
    lookup = {(r['lineage'], *(r[k] for k in AXES)): r for r in cells}
    if len(lookup) != len(cells): raise ValueError('duplicate cells')
    estimates = []
    for key in sorted({tuple(r[k] for k in AXES) for r in cells}):
        bases = [('mean', None)]
        if key[2] != 'none': bases.append(('minus-none', (*key[:2], 'none', *key[3:])))
        if key[2] == 'entropy-choice': bases.append(('minus-equal-request', (*key[:2], 'equal-request', *key[3:])))
        for label, base in bases:
            for metric in METRICS:
                values = [lookup[(lin, *key)][metric] - (lookup[(lin, *base)][metric] if base else 0.) for lin in lineages]
                estimates.append(dict(zip(AXES, key), contrast=label, metric=metric, **V.V.interval(values, cfg)))
    return dict(estimates=estimates, lineages=lineages, bootstrap_seed=cfg['bootstrap_seed'],
                bootstrap_resamples=cfg['bootstrap_resamples'], practical_margin_nats=.02,
                population='eight retained development laws; native and equal-query populations separate; no new fits or draws',
                limitation='finite contribution excludes infinite-loss mass; supplied-law selection; intervals condition on retained queries and laws')


def review(original, output, cfg, pulse=lambda **kw: None):
    design = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json')
    if (design['policies'] != list(POLICIES) or design['models'] != list(MODELS)
        or design['costs'] != list(COSTS) or design['rules'] != list(V.T.RULES)): raise ValueError('design')
    base = original/'inputs'; truth = read(base/'QUERY_TRUTH.json'); qs = [tuple(r['query']) for r in truth]
    if len(qs) != design['queries'] or len(set(qs)) != len(qs): raise ValueError('queries')
    groups = V.groups_for(qs, 'omit-both')
    if groups != read(base/'MEMBERSHIP.json')['omit-both']: raise ValueError('membership')
    if truth != [dict(query=list(q), targets={r: V.T.endpoint(q, r) for r in V.T.RULES}) for q in qs]: raise ValueError('truth')
    packets = {key: row['reader'] for key, row in groups.items()}
    for row in groups.values():
        for q in row['legal_completions']:
            for field, axis in (('skill', 0), ('belief_error', 1)):
                packet = dict(row['reader'], requested_field=field, disclosed_value=q[axis]); packets[digest(packet)] = packet
    if {p.stem for p in (original/'reader').glob('*.json')} != set(packets): raise ValueError('reader coverage')
    for key, packet in packets.items():
        if packet != read(original/'reader'/f'{key}.json'): raise ValueError('reader projection')
    index = {q:i for i,q in enumerate(qs)}; cells = []; laws = []; paths = 0; error = 0.; names = set(); tie_roundoff = 0
    for lineage, rule in product(design['lineages'], design['rules']):
        pulse(phase='independent-disclosure', lineage=lineage, rule=rule)
        targets = np.array([V.T.endpoint(q, rule) for q in qs]); terms = [[] for _ in qs]
        records = V.V.zipped(base/'raw'/f'{lineage}-{rule}_points.json.gz'); paths += len(records)
        for row in records:
            i = index[V.V.query(row)]; w = row['probability']
            if not math.isfinite(w) or w < 0 or V.V.code(row['final']) != targets[i]: raise ValueError('native path')
            terms[i].append(w)
        mass = np.array([math.fsum(t) for t in terms]); V.V.near([mass.sum()], [1.])
        for model in MODELS:
            pred = {p:np.zeros((len(qs),8)) for p in ('none','skill','belief','entropy-choice','all-fields')}
            choices = np.zeros(len(qs), dtype=np.int8)
            name = f'{lineage}-{rule}-{model}_points.npz'; names.add(name)
            with np.load(original/'forecasts'/name, allow_pickle=False) as saved:
                if set(saved.files) != {'queries','targets','mass','choices',*pred}: raise ValueError('forecast schema')
                for key, g in groups.items():
                    legal = g['legal_completions']; ends = [V.T.endpoint(q,rule) for q in legal]
                    raw = [float(mass[index[tuple(q)]]) if tuple(q) in index else 0. for q in legal]; total = math.fsum(raw)
                    weights = [w/total for w in raw] if model == 'native-law' and total else [1/len(legal)]*len(legal)
                    prior, tables, entropy, choice = conditional(legal,ends,weights)
                    # Preserve a declared exact tie separately from floating arithmetic.
                    observed = {int(saved['choices'][i]) for i in g['indices']}
                    expected = int(choice == 'belief')
                    if observed != {expected}:
                        if len(observed) != 1 or not observed <= {0,1} or abs(entropy['skill']-entropy['belief']) > 1e-14:
                            raise ValueError('selection')
                        tie_roundoff += 1; choice = 'belief' if next(iter(observed)) else 'skill'
                    for i in g['indices']:
                        q = qs[i]; pred['none'][i] = prior
                        pred['skill'][i] = tables['skill'][q[0]]; pred['belief'][i] = tables['belief'][q[1]]
                        pred['entropy-choice'][i] = tables[choice][q[int(choice == 'belief')]]
                        pred['all-fields'][i,targets[i]] = 1.; choices[i] = int(choice == 'belief')
                    laws.append(dict(lineage=lineage,rule=rule,reader_id=key,model=model,legal_completions=legal,
                        endpoints=ends,conditional_weights=weights,expected_entropy=entropy,choice=choice,
                        prior_entropy=V.entropy(prior),zero_native_mass=not bool(total),
                        zero_mass_fallback='uniform-legal' if model=='native-law' and not total else None))
                for k,v in dict(queries=qs,targets=targets,mass=mass,choices=choices,**pred).items():
                    error = max(error,V.V.near(saved[k],v,0 if k in ('queries','targets','choices') else 1e-12))
            for weighting, weights in (('native',mass),('equal-query',np.full(len(qs),1/len(qs)))):
                for policy in POLICIES:
                    ps = [pred['skill'],pred['belief']] if policy=='equal-request' else [pred[policy]]
                    scores = [V.scores(p,targets,weights) for p in ps]
                    metrics = {k:math.fsum(s[k] for s in scores)/len(scores) for k in scores[0]}
                    ambiguity = math.fsum(math.fsum(float(w) for p,w in zip(pmat,weights) if np.count_nonzero(p)>1) for pmat in ps)/len(ps)
                    fields = 0 if policy=='none' else 2 if policy=='all-fields' else 1
                    skill = math.fsum(w for w,v in zip(weights,choices) if v==0) if policy=='entropy-choice' else .5 if policy=='equal-request' else float(policy in ('skill','all-fields'))
                    belief = math.fsum(w for w,v in zip(weights,choices) if v==1) if policy=='entropy-choice' else .5 if policy=='equal-request' else float(policy in ('belief','all-fields'))
                    for cost in COSTS:
                        cells.append(dict(lineage=lineage,rule=rule,model=model,policy=policy,weighting=weighting,cost=cost,
                            queries=len(qs),groups=len(groups),fields_requested=fields,skill_request_rate=skill,belief_request_rate=belief,
                            residual_ambiguous_mass=ambiguity,net_finite_loss=metrics['finite_loss_contribution']+cost*fields,**metrics))
    if {p.name for p in (original/'forecasts').glob('*.npz')} != names: raise ValueError('forecast coverage')
    # Producer writes group laws in group order within each model, as reconstructed above.
    error = max(error,V.compare(read(original/'evaluator/DISCLOSURE_LAWS.json'),laws),V.compare(summary['cells'],cells))
    if summary['queries']!=len(qs) or summary['native_paths']!=paths or summary['law_rows']!=len(laws) or summary['fits']!=0: raise ValueError('totals')
    checks = read(original/'CONTROLS.json')
    if not checks or not all(checks.values()) or summary['controls']!=checks: raise ValueError('controls')
    write(output/'INDEPENDENT_REGROUP.json',regroup(cells,design['lineages'],cfg))
    write(output/'RECONSTRUCTED_CELLS.json',cells)
    result = dict(passed=True,cells=len(cells),law_rows=len(laws),native_paths=paths,queries=len(qs),
        forecast_vectors=len(names)*len(qs)*5,reader_packets=len(packets),max_error=error,roundoff_tied_selections=tie_roundoff,
        scope='independent legal completions, scalar conditional laws, visible-group choices, zero support fallback, expected randomized proper scores, all costs, strata and reader allowlists; native policy weights inherit validated parent')
    write(output/'NUMERICAL_REVIEW.json',result); return result


def run(root,plan,pulse):
    cfg=plan['design']; original=root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h: raise ValueError('input binding')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']: raise ValueError('target binding')
    result=review(original,root,cfg,pulse)
    return dict(result,controls={'live:complete_disclosure_reconstruction':True,
        'positive:conditional_scores_and_costs':True,'placebo:no_request_and_zero_mass':True})
