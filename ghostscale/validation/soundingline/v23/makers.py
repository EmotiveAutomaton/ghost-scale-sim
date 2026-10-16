"""Finite recipe dictionaries and exact hierarchical population inference."""
from functools import lru_cache
from itertools import combinations, product
import numpy as np
from scipy.special import logsumexp
from .common import rng, normalize, bern


@lru_cache(None)
def recipe_library(drives=4, maximum=2, seeking='dimension', exactly=None):
    # The final drive is SEEKING; it is excluded from values in vigor models.
    allowed = list(range(drives if seeking == 'dimension' else drives-1))
    recipes = [(i,) for i in allowed]+list(combinations(allowed, 2))
    holdings, weights = [], []
    for size in ([exactly] if exactly is not None else range(1, maximum+1)):
        for held in combinations(range(len(recipes)), size):
            for w in product((1., 2.), repeat=size):
                vector = np.zeros(len(recipes)); vector[list(held)] = w
                weights.append(vector); holdings.append(held)
    return tuple(recipes), np.asarray(weights), tuple(holdings)


def satisfaction(options, recipes, meaning='conjunctive', need_scale=1.):
    options = np.asarray(options)
    values = []
    for recipe in recipes:
        part = options[..., list(recipe)]
        sat = part.prod(-1) if meaning == 'conjunctive' else part.mean(-1)
        if 0 in recipe:
            sat = sat*need_scale
        values.append(sat)
    return np.stack(values, axis=-1)


def probabilities(options, weights, recipes, beta, meaning='conjunctive', need_scale=1.):
    sat = satisfaction(options, recipes, meaning, need_scale)
    return normalize(np.asarray(beta)*np.einsum('...ar,tr->...ta', sat, weights))


@lru_cache(None)
def cultures(concentration):
    return rng('culture-priors', concentration).dirichlet(np.full(200, concentration), size=32)


def record_loglik(menus, choices, beta, recipes, weights):
    standard=recipe_library()
    if recipes!=standard[0] or not np.array_equal(weights,standard[1]):
        raise ValueError('population likelihood requires its declared recipe library')
    logp = np.zeros(len(weights))
    for menu, choice in zip(menus, choices):
        logp += population_logprob(menu, beta)[:, choice]
    return logp


@lru_cache(12288)
def _population_logprob(key, beta):
    """The declared binary three-option menu has only 4,096 possible values.

    Cache likelihood arithmetic, never observations, posteriors or population
    evidence. The menu and precision fully determine this finite likelihood.
    """
    menu = np.array(key).reshape(3, 4)
    recipes, weights, _ = recipe_library()
    logits = beta*(satisfaction(menu, recipes)@weights.T).T
    result = logits-logsumexp(logits, axis=1, keepdims=True)
    result.flags.writeable = False
    return result


def population_logprob(menu, beta):
    menu = np.asarray(menu)
    if menu.shape != (3, 4) or not np.isin(menu, (0, 1)).all():
        raise ValueError('population cache requires the declared binary menu')
    return _population_logprob(tuple(map(int, menu.ravel())), float(beta))


def type_filter(menus, choices, beta, recipes, weights, priors, drift):
    """Exact forward recursion; rank-one type resets, no quadratic transition."""
    standard=recipe_library()
    if recipes!=standard[0] or not np.array_equal(weights,standard[1]):
        raise ValueError('population likelihood requires its declared recipe library')
    priors = np.atleast_2d(priors)
    posterior, evidence = priors.copy(), np.zeros(len(priors))
    for step, (menu, choice) in enumerate(zip(menus, choices)):
        if step:
            posterior = (1-drift)*posterior+drift*priors
        likelihood = np.exp(population_logprob(menu, beta)[:, choice])
        updated = posterior*likelihood
        normalizer = updated.sum(1)
        if np.any(normalizer <= 0):
            raise ValueError('empty HMM support')
        evidence += np.log(normalizer); posterior = updated/normalizer[:, None]
    return posterior, evidence


def n1_world(v, phase, world, law=None, pulse=lambda **kw: None):
    s = v['settings']; r = rng(v['id'], phase, world, 'population')
    # Each independent 16-world block samples 16 distinct cultures from all 32.
    # Repeatedly using the first half would change the target distribution.
    culture = int(rng(v['id'], phase, 'law-block', world//16).permutation(32)[world % 16]) if law is None else law
    pi = cultures(s['concentration'])[culture]
    recipes, weights, _ = recipe_library()
    n, length = s['makers'], s['records']
    drift = float(s['structure'].split('-')[1]) if s['structure'].startswith('drift') else 0.
    menus = r.integers(2, size=(n, length, 3, 4), dtype=np.int8)
    own = r.integers(2, size=(n, 3, 4), dtype=np.int8)
    other = r.integers(2, size=(n, 3, 4), dtype=np.int8)
    pairs = np.array(list(combinations(range(4), 2)))
    alias_pairs = pairs[r.integers(len(pairs), size=n)]
    if s['structure'] == 'alias':
        for i, (a, b) in enumerate(alias_pairs):
            menus[i, :, :, b] = menus[i, :, :, a]; own[i, :, b] = own[i, :, a]
            other[i, 0, a], other[i, 0, b] = 1, 0
            other[i, 1, a], other[i, 1, b] = 0, 1
    types = r.choice(200, size=n, p=pi); beta = r.choice([1., 2., 4.], size=n)
    initial_types = types.copy(); decisions = np.empty((n, length), np.int8)
    for step in range(length):
        if step and drift:
            reset = r.random(n) < drift; types[reset] = r.choice(200, size=int(reset.sum()), p=pi)
        utility = satisfaction(menus[:, step], recipes)
        probs = normalize(beta[:, None]*np.einsum('nar,nr->na', utility, weights[types]))
        decisions[:, step] = (r.random(n)[:, None] > np.cumsum(probs, axis=1)).sum(1).clip(0, 2)
        if step % 16 == 0:
            pulse(phase='population-generation', world=world, step=step)
    answers = []
    for query in (own, other):
        probs = normalize(beta[:, None]*np.einsum('nar,nr->na', satisfaction(query, recipes), weights[types]))
        answers.append((r.random(n)[:, None] > np.cumsum(probs, axis=1)).sum(1).clip(0, 2))
    public = dict(menus=menus, choices=decisions, own=own, other=other,
                  alias_pairs=alias_pairs if s['structure'] == 'alias' else None)
    truth = dict(y=answers[1], in_context=answers[0], type=types, initial_type=initial_types,
                 presence=(weights[types] > 0).astype(np.int8), beta=beta, law=int(culture))
    return public, truth


def n1_infer(v, p, pulse=lambda **kw: None):
    s = v['settings']; n = len(p['choices'])
    recipes, weights, _ = recipe_library()
    priors = cultures(s['concentration'])
    drift = float(s['structure'].split('-')[1]) if s['structure'].startswith('drift') else 0.
    # Per-maker conditional posteriors: 32 cultures x 3 precisions x 200 types.
    post = np.empty((n, 32, 3, 200)); log_evidence = np.empty((n, 32, 3))
    individual = np.empty((n, 3, 200)); individual_evidence = np.empty((n, 3))
    for i, (menus, choices) in enumerate(zip(p['menus'], p['choices'])):
        for b, beta in enumerate((1., 2., 4.)):
            if drift:
                post[i, :, b], log_evidence[i, :, b] = type_filter(menus, choices, beta, recipes, weights, priors, drift)
                ip, ie = type_filter(menus, choices, beta, recipes, weights, np.full(200, 1/200), drift)
                individual[i, b], individual_evidence[i, b] = ip[0], ie[0]
            else:
                ll = record_loglik(menus, choices, beta, recipes, weights)
                with np.errstate(divide='ignore'):
                    joint = np.log(priors)+ll  # Stored finite menu may contain exact zeros.
                log_evidence[i, :, b] = logsumexp(joint, axis=1)
                post[i, :, b] = normalize(joint)
                individual[i, b] = normalize(ll)
                individual_evidence[i, b] = logsumexp(ll)-np.log(200)
        if i % 10 == 0:
            pulse(phase='population-inference', maker=i, makers=n)
    maker_evidence = logsumexp(log_evidence, axis=2)-np.log(3)
    culture_post = normalize(maker_evidence.sum(0))
    per_culture_beta = normalize(log_evidence, axis=2)
    # Conditional personal evidence has already entered culture_post once.
    exact = np.einsum('c,ncb,ncbt->nbt', culture_post, per_culture_beta, post)
    individual *= normalize(individual_evidence)[:, :, None]
    leave_out = normalize(maker_evidence.sum(0)[None, :]-maker_evidence)
    population_types = leave_out@priors
    population = np.repeat((population_types/3)[:, None, :], 3, axis=1)
    # Empirical Bayes: plug in the leave-one-out posterior mean population.
    pooled = np.empty_like(individual)
    for i in range(n):
        evidence = []
        for b, beta in enumerate((1., 2., 4.)):
            pp, ee = type_filter(p['menus'][i], p['choices'][i], beta, recipes, weights, population_types[i], drift)
            pooled[i, b] = pp[0]; evidence.append(ee[0])
        pooled[i] *= normalize(evidence)[:, None]
    distributions = {'law-marginal': exact, 'individual-only': individual,
                     'population-only': population, 'partial-pooling': pooled}
    # Deterministic prototype quantization of the learned population distribution.
    # Every type maps to a representative using public recipe-feature distances.
    for count in (4, 16):
        centers = np.argsort(-population_types.mean(0), kind='stable')[:count]
        distance = ((weights[:, None, :]-weights[centers][None, :, :])**2).sum(-1)
        assignment = distance.argmin(1)
        prototype = np.zeros_like(pooled)
        for cluster, center in enumerate(centers):
            prototype[:, :, center] = pooled[:, :, assignment == cluster].sum(2)
        distributions['prototypes-'+str(count)] = prototype
    predictions, details = {}, {}
    for name, dist in distributions.items():
        out, inside = [], []
        for i in range(n):
            for target, menu in ((out, p['other'][i]), (inside, p['own'][i])):
                target.append(sum(dist[i, b]@probabilities(menu, weights, recipes, beta)
                                  for b, beta in enumerate((1., 2., 4.))))
        predictions[name] = np.asarray(out)
        presence = dist.sum(1)@(weights > 0)
        details[name] = dict(in_context=np.asarray(inside), recipe_presence=presence,
                             top_type=dist.sum(1).max(1))
    return predictions, dict(readers=details, culture_posterior=culture_post,
                             population_types=population_types)


def n1_features(v, p):
    # Raw sufficient summaries of choices and affordances; no latent/culture labels.
    x = []
    selected = np.take_along_axis(p['menus'], p['choices'][..., None, None], axis=2)[:, :, 0]
    own_mean = selected.mean(1); population_mean = own_mean.mean(0)
    for i in range(len(p['choices'])):
        counts = np.bincount(p['choices'][i], minlength=3)/len(p['choices'][i])
        x.append(np.r_[own_mean[i], p['menus'][i].mean((0, 1)), counts,
                       population_mean, p['other'][i].ravel(), p['own'][i].ravel()])
    return np.asarray(x)
