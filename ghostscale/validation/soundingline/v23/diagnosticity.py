"""N2: latent mark stories, response-time calibration and matched tempering."""
from functools import lru_cache
import numpy as np
from scipy.special import expit, logsumexp
from .common import rng, normalize, bern

CLASSES = ('preference', 'near-tie', 'mixed-strategy', 'spontaneous', 'prime', 'context')


def class_prior(v):
    s = v['settings']; prior = np.zeros(6)
    prior[0] = 1-s['nondiagnostic']
    prior[CLASSES.index(s['mark_class'])] += s['nondiagnostic']
    return prior


def probabilities(v, opponent, previous, context):
    beta = v['settings']['beta']; signs = np.array([-1., 1.])
    return np.array([expit(beta*signs), expit(.05*beta*signs),
                     np.repeat(opponent, 2), np.repeat(.5, 2),
                     .6*previous+.4*expit(beta*signs), expit(beta*(signs+2*context))])


def rt_prob(v, probabilities_by_class):
    # Timing depends on the absolute preference likelihood separation, not the
    # maker's hidden realized type. Class changes its distribution.
    p = probabilities_by_class
    separation = np.abs(np.log(p[:, 1])-np.log(p[:, 0]))
    slope = v['settings']['rt_slope']
    return normalize(-slope*separation[:, None]*np.arange(5)[None, :])


def mark(v, r, preference, law, previous):
    prior = class_prior(v); kind = int(r.choice(6, p=prior))
    opponent = np.linspace(.15, .85, 16)[law]
    context = int(r.choice([-1, 1]))
    probability = probabilities(v, opponent, previous, context)
    chosen = int(r.random() < probability[kind, preference])
    latency = int(r.choice(5, p=rt_prob(v, probability)[kind]))
    # A noisy public context cue, never the true class annotation.
    observed = kind if r.random() < .7 else int(r.choice([k for k in range(6) if k != kind]))
    visible_opponent = bool(r.integers(2))
    d = abs(np.log(probability[kind, 1] if chosen else 1-probability[kind, 1])-
            np.log(probability[kind, 0] if chosen else 1-probability[kind, 0]))
    return dict(choice=chosen, latency_bin=latency, context=context, class_cue=observed,
                opponent=opponent if visible_opponent else None, previous=previous), dict(kind=kind, diagnosticity=float(d))


def sample(v, phase, world, case, law=None):
    r = rng(v['id'], phase, world, case); law = world % 16 if law is None else law
    preference = int(r.integers(2)); history, truth = [], []; previous = .5
    for _ in range(v['settings']['records']):
        row, hidden = mark(v, r, preference, law, previous)
        history.append(row); truth.append(hidden); previous = row['choice']
    return dict(history=history), dict(y=[preference], marks=truth, law=int(law))


@lru_cache(None)
def trained_weights(family_id, kind, share, slope, beta):
    v = dict(id=family_id, settings=dict(mark_class=kind, nondiagnostic=share, rt_slope=slope, beta=beta))
    r = rng('diagnosticity-rt-training', family_id)
    total, count = np.zeros(5), np.zeros(5)
    for i in range(4096):
        row, truth = mark(v, r, int(r.integers(2)), i % 16, float(r.integers(2)))
        total[row['latency_bin']] += truth['diagnosticity']; count[row['latency_bin']] += 1
    mean = total/np.maximum(count, 1)
    # Absolute diagnosticity relative to a clean preference mark. Weight clipping
    # is a declared heuristic feature, not score clipping or an e-value operation.
    weights = np.clip(mean/beta, 0, 1)
    return weights, count


def evaluate(v, p, truth=None):
    s = v['settings']; prior = class_prior(v)
    weights, train_counts = trained_weights(v['id'], s['mark_class'], s['nondiagnostic'], s['rt_slope'], s['beta'])
    ll = {name: np.zeros(2) for name in ('law-marginal', 'class-aware-no-rt',
          'everything-is-preference', 'response-time-weighted', 'matched-tempering', 'exact-plus-weights')}
    oracle_ll = np.zeros(2); estimated_d = []
    exact_law, no_rt_law, weighted_law = [np.zeros((16, 2)) for _ in range(3)]
    average_weight = float(np.average(weights, weights=train_counts))
    for i, row in enumerate(p['history']):
        likelihood, no_rt = np.zeros((16, 2)), np.zeros((16, 2))
        for law, opponent in enumerate(np.linspace(.15, .85, 16)):
            if row['opponent'] is not None and not np.isclose(opponent, row['opponent'], atol=1e-12):
                continue
            prob = probabilities(v, opponent, row['previous'], row['context'])
            choice_p = prob if row['choice'] else 1-prob
            cue = np.where(np.arange(6) == row['class_cue'], .7, .3/5)
            latent = prior[:, None]*cue[:, None]*choice_p
            no_rt[law] = latent.sum(0)
            likelihood[law] = (latent*rt_prob(v, prob)[:, row['latency_bin'], None]).sum(0)
        naive = np.log(np.where(row['choice'], expit(s['beta']*np.array([-1, 1])), 1-expit(s['beta']*np.array([-1, 1]))))
        weight = weights[row['latency_bin']]; estimated_d.append(float(weight*s['beta']))
        with np.errstate(divide='ignore'):
            exact_law += np.log(likelihood)
            no_rt_law += np.log(no_rt)
            # A zero heuristic weight ignores stochastic evidence, not an
            # observed-law support constraint. Avoid 0 * -inf producing NaNs.
            weighted_law += (weight*np.log(np.where(likelihood > 0, likelihood, 1))
                             + np.where(likelihood > 0, 0, -np.inf))
        ll['everything-is-preference'] += naive
        ll['response-time-weighted'] += weight*naive
        ll['matched-tempering'] += average_weight*naive
        if truth is not None:
            oracle_ll += truth['marks'][i]['diagnosticity']/s['beta']*naive
    pred = {name: normalize(value)[None, :] for name, value in ll.items()}
    for name, value in [('law-marginal', exact_law), ('class-aware-no-rt', no_rt_law),
                        ('exact-plus-weights', weighted_law)]:
        pred[name] = normalize(value.ravel()).reshape(16, 2).sum(0)[None, :]
    if truth is not None:
        pred['oracle-weighted'] = normalize(oracle_ll)[None, :]
    return pred, dict(rt_weights=weights.tolist(), training_bin_counts=train_counts.tolist(),
                      estimated_diagnosticity=estimated_d, heldout_training=True)


def features(v, p):
    x = np.zeros((128, 8))
    for i, row in enumerate(p['history']):
        x[i] = [row['choice'], row['latency_bin']/4, row['context'], row['class_cue']/5,
                .5 if row['opponent'] is None else row['opponent'], row['opponent'] is not None,
                row['previous'], 1]
    return x.ravel()
