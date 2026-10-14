"""A1: finite-q common/independent-cause source inference, unknown noise law."""
import itertools
import numpy as np
from scipy.special import logsumexp
from .common import rng, softmax, normalize, lognormal

Q = np.linspace(-3, 3, 9)
QP = softmax(-Q*Q/2)
# goal: zero informs, one steers; competence and law are genuinely hidden.
TYPES = np.array(list(itertools.product((0, 1), (.35, 1.2), (-1.5, 1.5), (.8, 1.2))))


def sample(namespace, world, index):
    r = rng('A1', namespace, world, index)
    law = int(rng('A1-law', namespace, world).integers(2))
    z = int(r.integers(len(TYPES)//2))*2+law
    goal, sigma, agenda, multiplier = TYPES[z]
    k = (0, 2, 8, 32)[index % 4]
    sr = (.4, 1.2)[(index//4) % 2]
    qh = r.choice(Q, 32, p=QP)
    ch = np.where(goal == 0, qh, agenda)+r.normal(0, sigma*multiplier, 32)
    # Duplicate identity is public; never silently count a repeated row as fresh.
    roots = np.arange(32)
    control = ('native', 'unrelated', 'duplicated')[(index//8) % 3]
    if control == 'unrelated':
        other = TYPES[int(r.integers(len(TYPES)))]
        ch = (qh if other[0] == 0 else other[2])+r.normal(0, other[1]*other[3], 32)
    if control == 'duplicated':
        roots = np.arange(32)//2
        qh = qh[roots]
        ch = ch[roots]
    y = int(r.choice(len(Q), p=QP))
    own = Q[y]+r.normal(0, sr)
    claim = (Q[y] if goal == 0 else agenda)+r.normal(0, sigma*multiplier)
    p = dict(k=k, sr=sr, qh=qh[:k].tolist(), ch=ch[:k].tolist(), roots=roots[:k].tolist(),
             own=float(own), claim=float(claim), label=float((.2, .54, .9)[(index//24) % 3]),
             condition=control)
    return p, dict(y=[y], source=z, law=law)


def features(p):
    out = np.zeros((32, 4))
    for i in range(p['k']):
        out[i] = (p['qh'][i]/3, p['ch'][i]/3, 1, p['roots'][i]/31)
    return np.r_[out.ravel(), p['own']/3, p['claim']/3, p['sr'], p['label'], p['condition'] == 'unrelated']


def posterior(p, unique=True, supplied=None, allow_unrelated=True):
    types = TYPES if supplied is None else TYPES[np.asarray(supplied)]
    history = np.zeros(len(types))
    seen = set()
    for q, c, root in zip(p['qh'], p['ch'], p['roots']):
        if allow_unrelated and p['condition'] == 'unrelated':
            continue
        if unique and root in seen:
            continue
        seen.add(root)
        mu = np.where(types[:, 0] == 0, q, types[:, 2])
        history += lognormal(c, mu, types[:, 1]*types[:, 3])
    own = lognormal(p['own'], Q, p['sr'])+np.log(QP)
    mu = np.where(types[:, 0, None] == 0, Q[None, :], types[:, 2, None])
    joint = history[:, None]+own[None, :]+lognormal(p['claim'], mu, types[:, 1, None]*types[:, 3, None])-np.log(len(types))
    norm = logsumexp(joint)
    return np.exp(logsumexp(joint, axis=0)-norm), np.exp(logsumexp(joint, axis=1)-norm)


def evaluate(p, truth):
    exact, source = posterior(p)
    own = normalize(lognormal(p['own'], Q, p['sr'])+np.log(QP))
    informed = normalize(np.log(own)+lognormal(p['claim'], Q, .8))
    readers = {'law-marginal': exact[None, :], 'own-only': own[None, :],
               'duplicated-product': posterior(p, unique=False)[0][None, :],
               'rigid-same-source': posterior(p, allow_unrelated=False)[0][None, :],
               'oracle-source': posterior(p, supplied=[truth['source']])[0][None, :]}
    for trust in (.2, .54, .9):
        readers[f'fixed-{trust}'] = ((1-trust)*own+trust*informed)[None, :]
    readers['label-only'] = ((1-p['label'])*own+p['label']*informed)[None, :]
    curves = []
    for discrepancy in np.linspace(0, 6, 13):
        probe = dict(p, own=0., claim=float(discrepancy))
        prob, _ = posterior(probe)
        curves.append(float(prob@Q))
    expected = float(exact@Q)
    return readers, dict(k=p['k'], expertise=p['sr'], source_goal=int(TYPES[truth['source'], 0]),
                         competence=float(TYPES[truth['source'], 1]), condition=p['condition'],
                         belief_shift=expected-float(own@Q), signed_error=expected-float(Q[truth['y'][0]]),
                         steering_probability=float(source[TYPES[:, 0] == 1].sum()),
                         fixed_discrepancy_grid=list(np.linspace(0, 6, 13)), shift_curve=curves)
