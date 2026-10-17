"""N4c finite intervention model. Mechanisms are stipulated, not learned here."""
import math
import numpy as np
from .common import rng, normalize, canonical
from . import ledger

MECHANISMS = ('goal-directed', 'action-habit', 'acquired-end', 'habitual-goal')
RELIABILITY = np.linspace(.6, .95, 16)
# Original need, candidate-end deficit, original means available. Reset trials.
CONTEXTS = ((1, 1, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1), (0, 1, 0), (1, 1, 0))
PROBES = ('baseline', 'sham', 'A', 'B-satisfied', 'B-disturbed-alternative', 'alternative-control')


def variants():
    return [dict(id=f'N4c-{m}-{n}', family='N4c', settings=dict(mechanism=m, history=n))
            for m in MECHANISMS for n in (0, 8, 32)]


def intended(mechanism, context):
    need, end, available = context
    if mechanism == 'action-habit': return 3 + int(available)
    target = end if mechanism == 'acquired-end' else need
    goal = 1 if mechanism == 'habitual-goal' else int(bool(target))
    action = (1 if available else 2) if target else 0
    return 3 * goal + action


def laws(contexts):
    """Mechanism x reliability x probe x joint response."""
    out = np.broadcast_to((1-RELIABILITY)[None, :, None, None]/6,
                          (4, 16, len(contexts), 6)).copy()
    for m, mechanism in enumerate(MECHANISMS):
        for q, context in enumerate(contexts):
            out[m, :, q, intended(mechanism, context)] += RELIABILITY
    return out


def sample(v, phase, world, case):
    n = v['settings']['history']; m = MECHANISMS.index(v['settings']['mechanism'])
    r = rng('N4c-paired', n, phase, world, case)
    reliability = RELIABILITY[world % 16]
    base = np.full(6, (1-reliability)/6); base[4] += reliability
    history = r.choice(6, size=n, p=base).tolist()
    order = r.permutation(6).tolist()
    contexts = [list(CONTEXTS[i]) for i in order]
    probabilities = np.full((6,6), (1-reliability)/6)
    for q, context in enumerate(contexts):
        probabilities[q, intended(MECHANISMS[m], context)] += reliability
    uniforms = r.random(6)
    y = [int(np.searchsorted(np.cumsum(p), u)) for p, u in zip(probabilities, uniforms, strict=True)]
    return dict(history=history, query=contexts, probe_order=order), dict(y=y, mechanism=m, law=world % 16)


def features(v, p):
    x = np.zeros((32, 7))
    for i, value in enumerate(p['history']): x[i, value] = 1; x[i, 6] = 1
    return np.r_[x.ravel(), np.asarray(p['query']).ravel()]


def posterior(p, y=None, subset=None, action_only=False):
    base = laws([CONTEXTS[0]])[:, :, 0]
    logmass = np.full((4, 16), -math.log(64))
    for value in p['history']: logmass += np.log(base[:, :, value])
    if y is not None:
        probs = laws(p['query'])
        if action_only: probs = probs.reshape(4, 16, 6, 2, 3).sum(-2)
        for i, value in enumerate(y):
            if subset is None or p['probe_order'][i] in subset:
                logmass += np.log(probs[:, :, i, value % 3 if action_only else value])
    return normalize(logmass.ravel()).reshape(4, 16)


def independent(p, y=None, subset=None, action_only=False):
    """Separately enumerated likelihood products, without the production law builder."""
    mass = np.ones((4, 16))/64
    def probability(m, rel, context, observed):
        need, end, available = context
        if m == 0: goal, action = int(bool(need)), (1 if available else 2) if need else 0
        elif m == 1: goal, action = 1, int(available)
        elif m == 2: goal, action = int(bool(end)), (1 if available else 2) if end else 0
        else: goal, action = 1, (1 if available else 2) if need else 0
        target = action if action_only else 3*goal+action
        return (1-rel)/(3 if action_only else 6) + rel*(target == observed)
    for value in p['history']:
        mass *= (1-RELIABILITY)[None, :]/6 + RELIABILITY[None, :]*(value == 4)
        mass /= mass.sum()
    if y is not None:
        for i, value in enumerate(y):
            if subset is not None and p['probe_order'][i] not in subset: continue
            for m in range(4):
                for law, rel in enumerate(RELIABILITY):
                    mass[m, law] *= probability(m, rel, p['query'][i], value % 3 if action_only else value)
            mass /= mass.sum()
    return mass


def evaluate(v, p, t):
    before = posterior(p); probs = laws(p['query'])
    pred = {'exact': np.einsum('ml,mlqc->qc', before, probs)}
    for m, name in enumerate(MECHANISMS):
        pred[name] = np.einsum('l,lqc->qc', before[m]/before[m].sum(), probs[m])
    detail = dict(before=before.sum(1).tolist(), after={})
    diagnostics = dict(pre_probe_mechanism_loss=-math.log(detail['before'][t['mechanism']]),
                       probe_label_used_for_forecast=False)
    for name, subset, action_only in (('all', None, False), ('A', {2}, False),
            ('B', {3, 4}, False), ('sham', {0, 1}, False), ('action-only', None, True)):
        mass = posterior(p, t['y'], subset, action_only)
        rebuilt = independent(p, t['y'], subset, action_only)
        if not np.allclose(mass, rebuilt, atol=1e-12, rtol=0):
            raise ValueError('N4c independent posterior differs')
        detail['after'][name] = mass.sum(1).tolist()
        diagnostics[name+'_gain'] = diagnostics['pre_probe_mechanism_loss'] + math.log(mass.sum(1)[t['mechanism']])
    diagnostics['post_probe_mechanism_loss'] = diagnostics['pre_probe_mechanism_loss']-diagnostics['all_gain']
    return pred, detail, diagnostics


def emit(p, schema, identity):
    e = ledger.Emission(schema, identity)
    for i, value in enumerate(p['history']): e.decision('history', i, list(range(6)), value)
    for i, context in enumerate(p['query']):
        e.decision('query', i, list(range(6)), None, observed=dict(context=context, probe=p['probe_order'][i]))
    return e.finish()


def decode(bundle, schema):
    ledger.validate(bundle, schema); history=[]; query=[]; order=[]
    rows = bundle['rows']['Decision']
    for channel in ('history', 'query'):
        selected = sorted((r for r in rows if r['context']['channel']==channel), key=lambda r:r['context']['index'])
        for row in selected:
            if channel=='history':
                history.append(next(o['position'] for o in row['options'] if o['option_id']==row['taken']))
            else:
                query.append(row['context']['observed']['context']);order.append(row['context']['observed']['probe'])
    return dict(history=history, query=query, probe_order=order)


def controls():
    signatures=[]
    for m in range(4):
        p=dict(history=[4]*32,query=[list(c) for c in CONTEXTS],probe_order=list(range(6)))
        t=dict(y=[intended(MECHANISMS[m], c) for c in CONTEXTS],mechanism=m)
        _, d, diag = evaluate({}, p, t)
        if not np.allclose(d['before'], .25, atol=1e-12, rtol=0): raise ValueError('baseline mechanism leak')
        if not np.allclose(d['after']['sham'], .25, atol=1e-12, rtol=0): raise ValueError('sham mechanism leak')
        if np.argmax(d['after']['all'])!=m or d['after']['all'][m]<.9: raise ValueError('positive recovery failed')
        signatures.append(t['y'])
    p,t=sample(variants()[0], 'known-answer',15,0)
    if not np.array_equal(features(variants()[0],p), features(variants()[-1],p)): raise ValueError('hidden-label feature leak')
    if not np.allclose(laws(CONTEXTS).sum(-1),1): raise ValueError('probabilities invalid')
    # Without observed goal selection, instrumental and habitual-goal rules alias.
    action=laws(CONTEXTS).reshape(4,16,6,2,3).sum(-2)
    if not np.array_equal(action[0],action[3]): raise ValueError('action-only alias broken')
    return dict(passed=True, baseline_aliased=True, sham_aliased=True, four_rule_positive=True,
                action_only_cannot_separate_goal_habit=True, independent_mass_agrees=True,
                evaluator_label_excluded=True, miniature_architecture_untested=True)
