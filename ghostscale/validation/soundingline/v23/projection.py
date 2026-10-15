"""N5: continuous Dirichlet makers and explicitly approximate quadrature reader.

The projection/calibration contrast is exact. Quality-tilted Dirichlet inference
has no claimed finite-support exact ceiling: retain quadrature convergence and
exclude this approximation from all learned-headroom gates.
"""
from functools import lru_cache
import numpy as np
from scipy.special import gammaincinv, logsumexp
from scipy.stats import qmc
from .common import rng, seed, normalize

BASE = np.array([.50, .22, .23, .05])
QUALITY = np.array([1., .25, -1., -.25])


def offsets(v):
    s = v['settings']
    direction = np.array([0., 1., -1., 0.]) if s['shape'] == 'anchor' else rng('projection-direction').normal(size=4)
    direction -= direction.mean(); direction /= np.max(np.abs(direction))
    return s['offset']*direction


def sample(v, phase, world, case, law=None):
    s = v['settings']; r = rng(v['id'], phase, world, case)
    law = world % 16 if law is None else law
    # A public quality scale supplies sixteen nuisance contexts, not latent labels.
    quality_scale = np.linspace(.5, 1.5, 16)[law]
    propensity = r.dirichlet(BASE*s['concentration'])
    rows = []
    for _ in range(s['records']+1):
        quality = float(r.uniform(-1, 1)*quality_scale)
        probability = normalize(np.log(propensity)+quality*QUALITY)
        rows.append(dict(quality=quality, choice=int(r.choice(4, p=probability))))
    query = rows.pop(); y = query.pop('choice')
    return dict(history=rows, query=query), dict(y=[y], propensity=propensity.tolist(), law=int(law))


@lru_cache(None)
def prior_nodes(concentration, power):
    u = qmc.Sobol(4, scramble=True, seed=seed('N5-prior-quadrature', concentration) % 2**32).random_base2(power)
    gamma = gammaincinv(BASE*concentration, u)
    return gamma/gamma.sum(1, keepdims=True)


def quadrature(v, p, power):
    nodes = prior_nodes(v['settings']['concentration'], power)
    logpi = np.log(nodes); ll = np.zeros(len(nodes))
    for row in p['history']:
        logits = logpi+row['quality']*QUALITY
        ll += logits[:, row['choice']]-logsumexp(logits, axis=1)
    post = normalize(ll)
    return post@normalize(logpi+p['query']['quality']*QUALITY)


def evaluate(v, p, truth=None):
    s = v['settings']; quality = p['query']['quality']
    own = normalize(np.log(BASE)+offsets(v))
    quality_effect = quality*QUALITY if s['quality'] == 'correct' else np.zeros(4)
    projection = normalize(np.log(own)+quality_effect)
    calibrated = normalize(np.log(projection)+np.log(BASE/own))
    counts = np.bincount([row['choice'] for row in p['history']], minlength=4)
    empirical = (counts+.5)/(len(p['history'])+2)
    pooled = (counts+s['concentration']*BASE)/(len(p['history'])+s['concentration'])
    shrink = len(p['history'])/(len(p['history'])+s['concentration'])
    adjusted = normalize(np.log(own)+shrink*(np.log(empirical)-np.log(own))+quality_effect)
    low, high = quadrature(v, p, 8), quadrature(v, p, 10)
    error = float(np.max(np.abs(low-high)))
    readers = {'projection': projection, 'calibrated-projection': calibrated,
               'projection-plus-frequency': normalize(np.log(own)+np.log(empirical/BASE)+quality_effect),
               'partial-pooling': normalize(np.log(pooled)+quality*QUALITY),
               'adjusted-projection': adjusted, 'base-rates': normalize(np.log(BASE)+quality*QUALITY),
               'approximate-hierarchical': high}
    if truth is not None:
        readers['oracle-propensity'] = normalize(np.log(truth['propensity'])+quality*QUALITY)
    return {k: x[None, :] for k, x in readers.items()}, dict(
        quadrature_max_difference=error, approximate_ceiling=True, headroom_eligible=False,
        continuous_dirichlet=True, specified_base_rates=BASE.tolist(),
        realism_status='illustrative CoAuthor-like anchor; empirical summary pin required for twin')


def features(v, p):
    x = np.zeros((32, 6))
    for i, row in enumerate(p['history']):
        x[i, 0] = row['quality']; x[i, 1+row['choice']] = 1; x[i, 5] = 1
    return np.r_[x.ravel(), p['query']['quality'], offsets(v)]
