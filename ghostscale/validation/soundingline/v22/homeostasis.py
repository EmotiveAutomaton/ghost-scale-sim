"""B1: finite set points, choice cost and explicit power-likelihood heuristic."""
import itertools
import numpy as np
from scipy.special import logsumexp
from .common import rng, normalize

LAWS = ((2, 2, .8), (2, 2, 3.), (4, 2, .8), (4, 2, 3.))


def states(d):
    return np.asarray(list(itertools.product((-1., 0., 1.), repeat=d)))


def choice(hstar, h, effects, costs, law, ignore_cost=False):
    n, m, gamma = law
    hstar = np.atleast_2d(hstar)
    before = np.sum(np.abs(hstar-h)**n, -1)**(1/m)
    after = np.sum(np.abs(hstar[:, None, :]-h-effects[None, :, :])**n, -1)**(1/m)
    return normalize(gamma*(before[:, None]-after-(0 if ignore_cost else costs)))


def sample(namespace, world, index):
    r = rng('B1', namespace, world, index)
    d = 2+(index % 3)
    law = int(rng('B1-law', namespace, world).integers(4))
    ss = states(d)
    hs = ss[int(r.integers(len(ss)))]
    k = (2, 8, 32)[(index//3) % 3]
    shift = bool((index//9) % 2) if namespace == 'development' else False
    condition = ('native', 'zero-cost', 'independent-effort', 'alias')[(index//18) % 4]
    records = []
    for step in range(k+1):
        scale = 3. if shift and step == k else 1.
        h = r.uniform(-scale, scale, d)
        effects = r.choice((-.75, 0., .75), (3, d))
        if condition == 'alias':
            effects[:] = 0
        costs = np.sum(np.abs(effects), axis=1)*.35
        if condition == 'zero-cost':
            costs[:] = 0
        if condition == 'independent-effort':
            costs = r.uniform(0, 1, 3)
        prob = choice(hs, h, effects, costs, LAWS[law])[0]
        picked = int(r.choice(3, p=prob))
        records.append(dict(h=h.tolist(), effects=effects.tolist(), costs=costs.tolist(), picked=picked))
    current = dict(records.pop())
    y = current.pop('picked')
    return dict(d=d, k=k, history=records, current=current, shift=shift, condition=condition), dict(y=[y], setpoint=hs.tolist(), law=law)


def features(p):
    out = np.zeros((33, 25))
    for i, row in enumerate(p['history']+[p['current']]):
        j = i if i < p['k'] else 32
        d = p['d']
        out[j, :d] = row['h']
        effects = np.zeros((3, 4))
        effects[:, :d] = row['effects']
        out[j, 4:16] = effects.ravel()
        out[j, 16:19] = row['costs']
        if 'picked' in row:
            out[j, 19+row['picked']] = 1
        out[j, 22] = 1
        out[j, 23] = d/4
    return out.ravel()


def posterior(p, weighting='unit', known_law=None):
    ss = states(p['d'])
    laws = range(4) if known_law is None else [known_law]
    logp = np.zeros((len(laws), len(ss)))
    previous = []
    for record in p['history']:
        action = record['picked']
        habit = max(range(3), key=previous.count) if previous else action
        # Only costly departures get extra weight, without hidden truth access.
        weight = 1+record['costs'][action]*(action != habit) if weighting == 'effort' else 1.
        previous.append(action)
        for j, law in enumerate(laws):
            prob = choice(ss, np.asarray(record['h']), np.asarray(record['effects']), np.asarray(record['costs']),
                          LAWS[law], weighting == 'cost-ignorant')
            logp[j] += weight*np.log(prob[:, action])
    post = np.exp(logp-logsumexp(logp))
    cur = p['current']
    pred = np.zeros(3)
    for j, law in enumerate(laws):
        prob = choice(ss, np.asarray(cur['h']), np.asarray(cur['effects']), np.asarray(cur['costs']),
                      LAWS[law], weighting == 'cost-ignorant')
        pred += post[j]@prob
    return pred, post.sum(0)


def intervals(p, post, hs):
    ss = states(p['d'])
    bounds = []
    for dim in range(p['d']):
        mass = np.array([post[ss[:, dim] == v].sum() for v in (-1, 0, 1)])
        cdf = np.cumsum(mass)
        lo, hi = np.searchsorted(cdf, (.05, .95)).clip(0, 2)-1
        bounds.append([int(lo), int(hi)])
    return dict(bounds=bounds, coverage=float(np.mean([lo <= val <= hi for (lo, hi), val in zip(bounds, hs)])),
                width=float(np.mean([hi-lo for lo, hi in bounds])))


def evaluate(p, truth):
    readers, recovery = {}, {}
    for name, weight in (('law-marginal', 'unit'), ('effort-weighted', 'effort'), ('cost-ignorant', 'cost-ignorant')):
        prob, post = posterior(p, weight)
        readers[name] = prob[None, :]
        recovery[name] = intervals(p, post, truth['setpoint'])
    readers['oracle-law'] = posterior(p, known_law=truth['law'])[0][None, :]
    cur = p['current']
    readers['oracle-setpoint-law'] = choice(truth['setpoint'], np.array(cur['h']), np.array(cur['effects']),
                                          np.array(cur['costs']), LAWS[truth['law']])
    counts = np.bincount([row['picked'] for row in p['history']], minlength=3)+.5
    readers['frequency'] = (counts/counts.sum())[None, :]
    dist = float(np.linalg.norm(np.array(truth['setpoint'])-cur['h']))
    consistent = float(readers['oracle-setpoint-law'].max())
    return readers, dict(d=p['d'], k=p['k'], shift=p['shift'], condition=p['condition'],
                         interval_recovery=recovery, distance=dist, choice_consistency=consistent)
