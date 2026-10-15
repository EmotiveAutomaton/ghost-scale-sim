"""N3: noisy preference corrections, contextual fixes and final choices."""
from functools import lru_cache
from itertools import product
import numpy as np
from scipy.special import logsumexp
from .common import rng, normalize


@lru_cache(None)
def directions(d):
    raw = np.array([x for x in product((-1., 0., 1.), repeat=d) if any(x)])
    return raw/np.linalg.norm(raw, axis=1, keepdims=True)


def sample(v, phase, world, case, law=None):
    s = v['settings']; r = rng(v['id'], phase, world, case)
    law = world % 16 if law is None else law
    # Sixteen nuisance edit-fee laws, observed through the offered menu costs.
    fee = np.linspace(.02, .3, 16)[law]
    ws = directions(s['dimensions']); index = int(r.integers(len(ws))); w = ws[index]
    history = []
    local_flags = []
    for _ in range(s['records']):
        x0 = r.integers(-1, 2, s['dimensions'])
        edits = np.zeros((s['edits']+1, s['dimensions']))
        for j in range(1, len(edits)):
            dims = r.choice(s['dimensions'], size=int(r.integers(1, 3)), replace=False)
            edits[j, dims] = r.choice([-1, 1], len(dims))
        costs = fee*np.abs(edits).sum(1)
        context = ws[r.integers(len(ws))]
        local = bool(r.random() < s['local_share']); local_flags.append(local)
        preference = context if local else w
        picked = int(r.choice(len(edits), p=normalize(s['beta']*(edits@preference-costs))))
        menu = r.integers(-1, 2, (4, s['dimensions']))
        choice = int(r.choice(4, p=normalize(s['beta']*(menu@w))))
        history.append(dict(x0=x0.tolist(), edits=edits.tolist(), costs=costs.tolist(),
                            context=context.tolist(), picked=picked, menu=menu.tolist(), choice=choice))
    query = r.integers(-1, 2, (4, s['dimensions']))
    y = int(r.choice(4, p=normalize(s['beta']*(query@w))))
    return dict(history=history, query=query.tolist()), dict(y=[y], direction=index, w=w.tolist(),
                                                           local=local_flags, law=int(law))


def posterior(v, p, reader='law-marginal'):
    s = v['settings']; ws = directions(s['dimensions'])
    # Sweep metadata supplies precision and local-fix prevalence to every
    # matched reader. The actual direction and per-edit class remain hidden.
    betas = (s['beta'],)
    shares = (s['local_share'],)
    ll = np.zeros((len(betas), len(shares), len(ws)))
    summed = np.zeros(s['dimensions'])
    for row in p['history']:
        edits, menu = np.array(row['edits']), np.array(row['menu'])
        costs = np.array(row['costs']); picked = row['picked']
        summed += edits[picked]
        for b, beta in enumerate(betas):
            preference = normalize(beta*(ws@edits.T-costs))[:, picked]
            local = normalize(beta*(edits@np.array(row['context'])-costs))[picked]
            choices = normalize(beta*(ws@menu.T))[:, row['choice']]
            for j, share in enumerate(shares):
                if reader in ('law-marginal', 'corrections-only'):
                    ll[b, j] += np.log((1-share)*preference+share*local)
                elif reader in ('half-space', 'soft-constraint-supplement'):
                    if picked:
                        positive = ws@edits[picked] > 0
                        if reader == 'half-space': ll[b, j] = np.where(positive, ll[b, j], -np.inf)
                        else: ll[b, j] += np.log(np.where(positive, .9, .1))
                if reader in ('law-marginal', 'final-choice-only', 'half-space', 'soft-constraint-supplement'):
                    ll[b, j] += np.log(choices)
    if reader == 'coactive':
        length = np.linalg.norm(summed)
        estimate = summed/length if length else summed
        return normalize(s['beta']*(np.array(p['query'])@estimate))[None, :], dict(direction=estimate.tolist(), supplied_precision=s['beta'])
    if not np.isfinite(ll).any():
        return None, dict(state='EMPTY_HARD_CONSTRAINT_SUPPORT', prediction_available=False,
                         interpretation='misspecified half-space reader contradicted by noisy edits; no smoothing, fallback or case deletion')
    post = normalize(ll.ravel()).reshape(ll.shape)
    probability = sum(post[b].sum(0)@normalize(beta*(ws@np.array(p['query']).T))
                      for b, beta in enumerate(betas))
    return probability[None, :], dict(direction_posterior=post.sum((0, 1)).tolist(),
                                       nuisance_posterior=post.sum(2).tolist())


def evaluate(v, p, truth=None):
    readers, detail = {}, {}
    for name in ('law-marginal', 'corrections-only', 'half-space', 'soft-constraint-supplement', 'coactive', 'final-choice-only'):
        readers[name], detail[name] = posterior(v, p, name)
    readers['corrections-plus-choices'] = readers['law-marginal']
    if truth is not None:
        w = np.array(truth['w'])
        detail['diagnostics'] = dict(nonpositive_preference_edits=sum(
            np.array(row['edits'][row['picked']])@w <= 0 for row in p['history']))
    return readers, detail


def features(v, p):
    # Full offered edits and context, padded only; no sidecar or inferred weights.
    x = np.zeros((32, 17*5+17+5+5+20+2))
    for i, row in enumerate(p['history']):
        edits = np.zeros((17, 5)); edits[:len(row['edits']), :v['settings']['dimensions']] = row['edits']
        costs = np.zeros(17); costs[:len(row['costs'])] = row['costs']
        context = np.pad(row['context'], (0, 5-len(row['context'])))
        origin = np.pad(row['x0'], (0, 5-len(row['x0'])))
        menu = np.zeros((4, 5)); menu[:, :v['settings']['dimensions']] = row['menu']
        x[i] = np.r_[edits.ravel(), costs, context, origin, menu.ravel(), row['picked']/16, row['choice']/3]
    query = np.zeros((4, 5)); query[:, :v['settings']['dimensions']] = p['query']
    return np.r_[x.ravel(), query.ravel()]
