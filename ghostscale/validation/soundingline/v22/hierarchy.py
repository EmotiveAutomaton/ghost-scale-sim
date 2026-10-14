"""C1: exact likelihood-weighted law mixture over binary trees."""
import numpy as np
from scipy.special import logsumexp
from .common import rng, normalize, bern

RELIABILITY = (.65, .8, .95)


def tree(branching, depth=4):
    parents = [-1]
    levels = [0]
    cursor = 0
    while cursor < len(parents):
        if levels[cursor] < depth:
            parents.extend([cursor]*branching)
            levels.extend([levels[cursor]+1]*branching)
        cursor += 1
    return np.asarray(parents), np.asarray(levels)


def sample(namespace, world, index):
    r = rng('C1', namespace, world, index)
    branching = 2+index % 2
    parents, levels = tree(branching)
    law = int(rng('C1-law', namespace, world).integers(3))
    reliability = RELIABILITY[law]
    state = np.zeros(len(parents), int)
    state[0] = r.integers(2)
    for node in range(1, len(parents)):
        state[node] = state[parents[node]] ^ int(r.random() > reliability)
    coverage = (.25, .5, .75)[(index//2) % 3]
    # Library coverage exposes operation evidence, not hidden internal labels.
    observed = (levels == 4) & (r.random(len(parents)) < coverage)
    evidence = np.where(observed, state ^ (r.random(len(parents)) > .9), -1)
    y = np.zeros(121, int)
    y[:len(state)] = state
    return dict(branching=branching, coverage=coverage, evidence=evidence.tolist()), dict(y=y.tolist(), law=law)


def features(p):
    x = np.zeros((121, 2))
    evidence = np.asarray(p['evidence'])
    x[:len(evidence), 0] = np.where(evidence < 0, 0, evidence*2-1)
    x[:len(evidence), 1] = evidence >= 0
    return np.r_[x.ravel(), p['branching']/3, p['coverage']]


def belief_propagation(parents, evidence, reliability, mode='joint'):
    n = len(parents)
    children = [np.flatnonzero(parents == i) for i in range(n)]
    transition = np.array([[reliability, 1-reliability], [1-reliability, reliability]])
    logt = np.log(transition)
    emission = np.zeros((n, 2))
    for i, obs in enumerate(evidence):
        if obs >= 0:
            emission[i] = np.log([.9 if obs == 0 else .1, .9 if obs == 1 else .1])
    below = emission.copy()
    messages = np.zeros((n, 2))
    for i in range(n-1, 0, -1):
        messages[i] = logsumexp(logt+below[i][None, :], axis=1)
        below[parents[i]] += messages[i]
    log_evidence = float(logsumexp(below[0])-np.log(2))
    above = np.zeros((n, 2))-np.log(2)
    if mode == 'joint':
        for i in range(n):
            for child in children[i]:
                outside = above[i]+below[i]-messages[child]
                above[child] = logsumexp(outside[:, None]+logt, axis=0)
        prob = normalize(above+below)
    elif mode == 'bottom-up':
        prob = normalize(below)
    elif mode == 'top-down':
        # One forward pass cannot send observed leaves back to ancestors.
        prob = np.zeros((n, 2))
        prob[0] = normalize(emission[0]-np.log(2))
        for i in range(1, n):
            prob[i] = normalize(np.log(prob[parents[i]]@transition)+emission[i])
    else:
        raise ValueError(mode)
    return prob, log_evidence


def infer(p, mode='joint', known_law=None):
    parents, _ = tree(p['branching'])
    laws = range(3) if known_law is None else [known_law]
    probs, likelihoods = zip(*(belief_propagation(parents, p['evidence'], RELIABILITY[i], mode) for i in laws))
    # Full evidence updates the unknown law in every ablation. Explicit privilege.
    weights = normalize(likelihoods)
    mixture = np.tensordot(weights, np.array(probs), axes=1)
    out = np.full((121, 2), .5)
    out[:len(parents)] = mixture
    return out


def evaluate(p, truth):
    parents, levels = tree(p['branching'])
    readers = {'law-marginal': infer(p), 'bottom-up': infer(p, 'bottom-up'),
               'top-down': infer(p, 'top-down'), 'oracle-law': infer(p, known_law=truth['law'])}
    ablations = {}
    # Internal focus at depth three has observed immediate children and remote leaves.
    focus = int(np.flatnonzero(levels == 3)[0])
    for name, remove in [('adjacent', parents == focus), ('distant', (levels == 4) & (parents != focus))]:
        evidence = np.array(p['evidence'])
        evidence[remove] = -1
        ablations[name] = infer(dict(p, evidence=evidence.tolist()))[focus].tolist()
    return readers, dict(branching=p['branching'], coverage=p['coverage'], levels=levels.tolist(),
                         focus=focus, ablations=ablations, actual_observed=int(sum(v >= 0 for v in p['evidence'])))
