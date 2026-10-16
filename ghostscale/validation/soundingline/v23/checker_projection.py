"""Independent N5 probability arithmetic and nested quadrature diagnostics.

The approximate hierarchical reader is checked against the supplied integration
nodes, not promoted to an exact ceiling. No production posterior routine is used.
"""
import numpy as np
from .checker_new import check


def probability(x):
    x = np.asarray(x, float)
    e = np.exp(x-x.max(axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)


def reconstruct(v, p, offset, nodes):
    base = np.array([.50, .22, .23, .05]); direction = np.array([1., .25, -1., -.25])
    s = v['settings']; own = base*np.exp(offset); own /= own.sum()
    feature = p['query']['quality']*direction if s['quality'] == 'correct' else np.zeros(4)
    projected = own*np.exp(feature); projected /= projected.sum()
    calibrated = projected*base/own; calibrated /= calibrated.sum()
    counts = np.array([sum(row['choice'] == k for row in p['history']) for k in range(4)])
    n = len(p['history']); frequency = (counts+.5)/(n+2)
    pool = (counts+s['concentration']*base)/(n+s['concentration'])
    adjusted = own**(1-n/(n+s['concentration']))*frequency**(n/(n+s['concentration']))*np.exp(feature)
    frequency_projection = own*frequency/base*np.exp(feature)
    likelihood = np.ones(len(nodes))
    for row in p['history']:
        distribution = nodes*np.exp(row['quality']*direction)
        distribution /= distribution.sum(1, keepdims=True)
        likelihood *= distribution[:, row['choice']]
        likelihood /= likelihood.sum()
    likelihood /= likelihood.sum()
    query = nodes*np.exp(p['query']['quality']*direction)
    query /= query.sum(1, keepdims=True)
    answer = {'projection': projected, 'calibrated-projection': calibrated,
              'projection-plus-frequency': frequency_projection/frequency_projection.sum(),
              'partial-pooling': probability(np.log(pool)+p['query']['quality']*direction),
              'adjusted-projection': adjusted/adjusted.sum(),
              'base-rates': probability(np.log(base)+p['query']['quality']*direction),
              'approximate-hierarchical': likelihood@query}
    return {name: np.array(value)[None, :] for name, value in answer.items()}


def verify(v, p, truth, prediction, detail, offset, nodes):
    expected = reconstruct(v, p, offset, nodes)
    if 'oracle-propensity' in prediction:
        x = np.array(truth['propensity'])*np.exp(p['query']['quality']*np.array([1., .25, -1., -.25]))
        expected['oracle-propensity'] = (x/x.sum())[None, :]
    if not detail['approximate_ceiling'] or detail['headroom_eligible']:
        raise ValueError('approximation was presented as exact')
    return check(prediction, expected)
