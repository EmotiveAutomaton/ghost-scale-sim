"""Independent goal decisions, native scores and paired lineage estimates.

Integer goal labels and scalar sums replace producer tensor operations. Saved
binary64 probabilities and marginals are checked before exact tie adjudication.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from .joint_factorization_review import ARMS, FIELDS, close, expand, references
from .witnessed_goal_review import witness, reconstruct, validate

GOALS = tuple(product(range(3), repeat=3))
FORECASTS = ('restricted', 'goal-product')
CHOICES = ('joint', 'coordinate')
METRICS = ('goal_accuracy', 'path_accuracy', 'top_incompatible',
           'forecast_hamming_loss', 'forecast_path_loss', 'loss')
DECISION_FIELDS = {'joint', 'coordinate', 'marginals', 'joint_ties', 'coordinate_ties',
                   'joint_margin', 'coordinate_margins', 'hamming_risks'}


def decide(q, saved=None):
    q = np.asarray(q)
    if q.shape != (27,) or not np.isfinite(q).all() or np.any(q < 0): raise ValueError('goal forecast')
    close(math.fsum(q.tolist()), 1., 4e-12)
    marg = np.array([[math.fsum(float(q[k]) for k, g in enumerate(GOALS) if g[t] == v)
                      for v in range(3)] for t in range(3)])
    error = 0.
    executed = marg
    if saved is not None:
        if set(saved) != DECISION_FIELDS: raise ValueError('decision fields')
        executed = np.asarray(saved['marginals'])
        error = close(marg, executed, 1e-12)
        if np.any(executed < 0): raise ValueError('saved marginal')
    # max keeps the first integer in an exact tie. No tolerance changes a mode.
    joint = max(range(27), key=lambda k: float(q[k]))
    digits = [max(range(3), key=lambda v: float(executed[t, v])) for t in range(3)]
    coordinate = 9*digits[0] + 3*digits[1] + digits[2]
    ordered = sorted(q.tolist()); mm = [sorted(m.tolist()) for m in executed]
    risks = np.array([math.fsum(1.-float(marg[t, g[t]]) for t in range(3))/3 for g in GOALS])
    result = dict(joint=joint, coordinate=coordinate, marginals=executed,
        joint_ties=sum(float(v) == float(q[joint]) for v in q),
        coordinate_ties=np.array([sum(float(v) == float(executed[t, digits[t]]) for v in executed[t]) for t in range(3)]),
        joint_margin=ordered[-1]-ordered[-2],
        coordinate_margins=np.array([m[-1]-m[-2] for m in mm]), hamming_risks=risks)
    if risks[coordinate] > min(risks)+1e-12: raise ValueError('coordinate objective')
    if saved is not None:
        for k, value in result.items():
            if k in ('joint', 'coordinate', 'joint_ties', 'coordinate_ties'):
                if not np.array_equal(value, saved[k]): raise ValueError('exact choice or tie count')
            else: error = max(error, close(value, saved[k], 1e-12))
    return result, error


def scores(q, decision, reference):
    """One frame, sparse target labels, all paired lineages; no producer score."""
    labels, target, mass = reference
    labels = [int(k) for k in labels]
    if any(q[k] <= 0 for k in labels): raise ValueError('zero true probability')
    logq = np.array([-math.log(float(q[k])) for k in labels])
    result = {}
    for which in CHOICES:
        chosen = decision[which]
        hit = np.array([sum(a == b for a, b in zip(GOALS[k], GOALS[chosen]))/3 for k in labels])
        probability = target[:, labels.index(chosen)] if chosen in labels else np.zeros(len(mass))
        scalar_risk = math.fsum(float(q[k])*sum(a != b for a, b in zip(GOALS[k], GOALS[chosen]))/3 for k in range(27))
        close(scalar_risk, decision['hamming_risks'][chosen], 1e-12)
        values = dict(loss=target @ logq, goal_accuracy=target @ hit,
            path_accuracy=probability, top_incompatible=probability == 0,
            forecast_hamming_loss=scalar_risk, forecast_path_loss=1-float(q[chosen]),
            decision_disagreement=decision['joint'] != decision['coordinate'],
            joint_tie=decision['joint_ties'] > 1,
            coordinate_tie=any(v > 1 for v in decision['coordinate_ties']),
            joint_margin=decision['joint_margin'], coordinate_margin=min(decision['coordinate_margins']))
        result[which] = {k: mass*v for k, v in values.items()}
    return result


def controls():
    q = np.zeros(27); q[[0, 12, 10, 4]] = [.30, .26, .26, .18]
    d, _ = decide(q)
    ref = (np.array([0, 4, 10, 12]), q[[0, 4, 10, 12]][None], np.ones(1))
    v = scores(q, d, ref); u, _ = decide(np.full(27, 1/27))
    return {'live:opposing_objectives': d['joint'] == 0 and d['coordinate'] == 9,
            'positive:outside_support': bool(v['coordinate']['top_incompatible'][0] == 1),
            'positive:native_objectives': bool(v['coordinate']['goal_accuracy'][0] > v['joint']['goal_accuracy'][0]
                                             and v['coordinate']['path_accuracy'][0] < v['joint']['path_accuracy'][0]),
            'placebo:prediction_identity': bool(v['coordinate']['loss'][0] == v['joint']['loss'][0]),
            'positive:numeric_ties': u['joint'] == u['coordinate'] == 0 and u['joint_ties'] == 27}


def regroup(rows, cfg):
    idx = {tuple(r[k] for k in FIELDS): r for r in rows}
    ls, ds, ss, bs = (cfg[k] for k in ('development_lineages', 'training_draws', 'fit_seeds', 'budgets'))
    arms = tuple(a+'-'+f+'-'+d for a, f, d in product(ARMS, FORECASTS, CHOICES))
    if len(idx) != len(rows) or set(idx) != set(product(cfg['tiers'], bs, arms, ls, ds, ss)): raise ValueError('stratum roster')
    if len(bs) < 2 or any(b >= c for b, c in zip(bs, bs[1:])): raise ValueError('budget order')
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(ls), size=(cfg['bootstrap_resamples'], len(ls)))
    counts = np.array([np.bincount(s, minlength=len(ls)) for s in samples])
    mean = lambda v: math.fsum(v)/len(v)
    def estimate(values):
        line = [mean([values[l, d, s] for d, s in product(ds, ss)]) for l in ls]
        boot = counts @ np.array(line)/len(ls)
        return dict(mean=mean(line), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)),
            lineage_values=line, draw_means=[mean([values[l, d, s] for l, s in product(ls, ss)]) for d in ds],
            feature_seed_means=[mean([values[l, d, s] for l, d in product(ls, ds)]) for s in ss])
    means = []; contrasts = []; areas = []; metrics = sorted(set(rows[0])-set(FIELDS)-{'frames'})
    for tier in cfg['tiers']:
        for arm, budget in product(arms, bs):
            rr = [idx[tier, budget, arm, l, d, s] for l, d, s in product(ls, ds, ss)]
            means.append(dict(tier=tier, arm=arm, budget=budget, **{k: mean([r[k] for r in rr]) for k in metrics}))
        for arm, forecast, metric in product(ARMS, FORECASTS, METRICS):
            a = arm+'-'+forecast+'-coordinate'; b = arm+'-'+forecast+'-joint'; curves = {}
            for budget in bs:
                values = {(l, d, s): idx[tier, budget, a, l, d, s][metric]-idx[tier, budget, b, l, d, s][metric]
                          for l, d, s in product(ls, ds, ss)}
                curves[budget] = values
                contrasts.append(dict(tier=tier, budget=budget, arm=a, baseline=b, metric=metric, **estimate(values)))
            area = {key: math.fsum((curves[x][key]+curves[y][key])*.5*math.log(y/x) for x, y in zip(bs, bs[1:]))/math.log(bs[-1]/bs[0])
                    for key in product(ls, ds, ss)}
            areas.append(dict(tier=tier, arm=a, baseline=b, metric=metric, **estimate(area)))
    return dict(contrasts=contrasts, normalized_log_budget_area=areas, means=means)


def compare_groups(actual, expected):
    error = 0.
    if set(actual) != set(expected): raise ValueError('summary groups')
    for name in actual:
        if len(actual[name]) != len(expected[name]): raise ValueError('summary roster')
        for a, b in zip(actual[name], expected[name]):
            if set(a) != set(b): raise ValueError('summary fields')
            for k in a:
                if k in ('tier', 'budget', 'arm', 'baseline', 'metric'):
                    if a[k] != b[k]: raise ValueError('summary identity')
                else: error = max(error, close(a[k], b[k]))
    return error


def run(root, plan, pulse):
    checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('known-answer controls')
    cfg = plan['design']; inputs = root/'inputs'; original = inputs/'original'; parent = inputs/'parent'
    for n, h in cfg['input_files'].items():
        if file_digest(inputs/n) != h: raise ValueError('input binding')
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan')
    design = read(original/'PLAN.json')['design']; prior = read(parent/'parent/PLAN.json')['design']
    for k in ('tiers', 'budgets', 'training_draws', 'fit_seeds', 'development_lineages'):
        if design[k] != prior[k]: raise ValueError('inherited population')
    if design['tiers'] != ['E2-full']: raise ValueError('complete witnesses')
    packets = read(original/'reader/PACKETS.json')
    if packets != read(parent/'reader/PACKETS.json'): raise ValueError('changed reader')
    keys = sorted(packets['packets']); operations = []
    for k in keys:
        p = packets['packets'][k]; validate(p)
        if digest(p) != k: raise ValueError('packet identity')
        operations.append(witness(p))
    refs = read(original/'evaluator/REFERENCES.json')
    if refs != read(parent/'evaluator/REFERENCES.json'): raise ValueError('changed references')
    ls = design['development_lineages']; full_refs = references(refs, 'E2-full', keys, ls); reference = []
    for op, (labels, target, mass) in zip(operations, full_refs):
        if any(int(k)%216 != op for k in labels): raise ValueError('native operation support')
        reference.append((labels//216, target, mass))
    raw = json.loads(gzip.decompress((original/'goal_decision_points.json.gz').read_bytes()))
    idx = {tuple(r[k] for k in FIELDS): r for r in raw}; regroup(raw, design)
    old = json.loads(gzip.decompress((parent/'parent/goal_mass_points.json.gz').read_bytes()))
    old_idx = {tuple(r[k] for k in FIELDS): r for r in old}
    expected = set(product(design['tiers'], design['budgets'], tuple(a+'-'+f for a in ARMS for f in ('restricted', 'mass-matched', 'goal-product')),
                           ls, design['training_draws'], design['fit_seeds']))
    if len(old_idx) != len(old) or set(old_idx) != expected: raise ValueError('parent roster')
    native = read(original/'evaluator/NATIVE_SCORES.json'); ni = {(r['lineage'], r['decision']): r for r in native}
    if len(ni) != len(native) or set(ni) != set(product(ls, CHOICES)): raise ValueError('native score roster')
    error = dict(probability=0., decision=0., score=0., parent=0.)
    def compare_row(values, saved, identity):
        if set(saved) != set(values)|set(identity) or any(saved[k] != v for k, v in identity.items()): raise ValueError('row identity')
        return close([values[k] for k in values], [saved[k] for k in values])
    def load_saved(path, native=False):
        with np.load(path, allow_pickle=False) as z: v = {k: z[k] for k in z.files}
        if set(v) != DECISION_FIELDS|{'probabilities'}| (set() if native else {'operations'}): raise ValueError('saved fields')
        shapes = dict(probabilities=(len(keys),27), joint=(len(keys),), coordinate=(len(keys),), marginals=(len(keys),3,3),
            joint_ties=(len(keys),), coordinate_ties=(len(keys),3), joint_margin=(len(keys),), coordinate_margins=(len(keys),3), hamming_risks=(len(keys),27))
        if any(v[k].shape != shape for k, shape in shapes.items()): raise ValueError('saved shapes')
        if not native and not np.array_equal(v['operations'], operations): raise ValueError('saved operations')
        return v
    def checked(q, saved, i):
        actual = saved['probabilities'][i]
        error['probability'] = max(error['probability'], close(q, actual, 1e-12))
        if not np.array_equal(q == 0, actual == 0): raise ValueError('changed support')
        d, e = decide(actual, {k:saved[k][i] for k in DECISION_FIELDS}); error['decision'] = max(error['decision'], e)
        return actual, d
    def finish(totals): return {k: np.sum(v, axis=0) for k,v in totals.items()}
    native_rows = []
    for li, l in enumerate(ls):
        pulse(phase='independent-native-goal-decisions', lineage=l)
        saved = load_saved(original/'evaluator'/f'native-{l}.npz', True)
        totals = {d:defaultdict(list) for d in CHOICES}
        for i, (labels, target, mass) in enumerate(reference):
            q = np.zeros(27); q[labels] = target[li]; q, dec = checked(q, saved, i)
            values = scores(q, dec, (labels, target[li:li+1], mass[li:li+1]))
            for d in CHOICES:
                for k, v in values[d].items(): totals[d][k].append(v)
        for d in CHOICES:
            values = {k:float(v[0]) for k,v in finish(totals[d]).items()}; identity = dict(tier='E2-full', lineage=l, decision=d)
            error['score'] = max(error['score'], compare_row(values, ni[l,d], identity)); native_rows.append(dict(identity, **values))
    rows = []; frames = reproduced = 0
    for draw, seed, budget, arm in product(design['training_draws'], design['fit_seeds'], design['budgets'], ARMS):
        pulse(phase='independent-learned-goal-decisions', draw=draw, seed=seed, budget=budget, arm=arm)
        stem = f'{draw}-E2-full-{seed}-{budget}-{arm}'
        if read(parent/'forecasts'/(stem+'-frames.json')) != keys: raise ValueError('frame roster')
        with np.load(parent/'forecasts'/(stem+'.npz'), allow_pickle=False) as z:
            if set(z.files) != {'probabilities','alphabet'}: raise ValueError('forecast fields')
            p, alphabet = z['probabilities'], z['alphabet']
        with np.load(parent/'masses'/(stem+'.npz'), allow_pickle=False) as z:
            if set(z.files) != {'goals','normalizer','original','product','operations'}: raise ValueError('mass fields')
            goals, norm, ops = z['goals'], z['normalizer'], z['operations']
        if len(p) != len(keys) or goals.shape != (len(keys),3,3) or norm.shape != (len(keys),) or not np.array_equal(ops, operations): raise ValueError('inherited shape')
        saved = {f:load_saved(original/'decisions'/(stem+'-'+f+'.npz')) for f in FORECASTS}
        totals = {(f,d):defaultdict(list) for f,d in product(FORECASTS,CHOICES)}
        for i, ref in enumerate(reference):
            q = expand(p[i], alphabet, budget); restricted, fact, e = reconstruct(q, operations[i], goals[i], norm[i])
            error['probability'] = max(error['probability'], e)
            for f, full in zip(FORECASTS, (restricted, fact)):
                q, dec = checked(full[operations[i]::216], saved[f], i)
                values = scores(q, dec, ref)
                for d in CHOICES:
                    for k,v in values[d].items(): totals[f,d][k].append(v)
                frames += 1
        for f,d in product(FORECASTS,CHOICES):
            values = finish(totals[f,d])
            for li,l in enumerate(ls):
                computed = {k:float(v[li]) for k,v in values.items()}
                identity = dict(tier='E2-full',budget=budget,arm=arm+'-'+f+'-'+d,lineage=l,draw=draw,seed=seed,frames=len(keys))
                error['score'] = max(error['score'],compare_row(computed,idx[tuple(identity[k] for k in FIELDS)],identity))
                error['parent'] = max(error['parent'],close(computed['loss'],old_idx['E2-full',budget,arm+'-'+f,l,draw,seed]['loss']))
                reproduced += int(d == 'joint'); rows.append(dict(identity,**computed))
    summary = read(original/'SUMMARY.json')
    for k,v in dict(rows=len(rows),frame_forecasts=frames,packets=len(keys),native_rows=len(native_rows),parent_loss_cells=reproduced).items():
        if summary[k] != v: raise ValueError('coverage')
    groups = regroup(rows,design); original_groups = regroup(raw,design); expected = {k:summary[k] for k in groups}
    error['regroup'] = compare_groups(groups,expected); error['original_regroup'] = compare_groups(original_groups,expected)
    write(root/'INDEPENDENT_REGROUP.json',groups); write(root/'ORIGINAL_ROW_REGROUP.json',original_groups)
    write(root/'NATIVE_RECONSTRUCTION.json',native_rows)
    (root/'reconstructed_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    result = dict(passed=True,rows=len(rows),frame_forecasts=frames,packets=len(keys),native_rows=len(native_rows),parent_loss_cells=reproduced,
        **{'max_'+k+'_error':v for k,v in error.items()},controls=checks,
        scope='independent scalar goal decisions and scores; saved binary64 probabilities/marginals verified before exact ties; inherited native targets and fits; no historical correspondence claim')
    write(root/'INDEPENDENT_REVIEW.json',result)
    return result
