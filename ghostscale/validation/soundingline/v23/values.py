"""N4: finite drive recipes, linear ablation, and pursuit vigor."""
import numpy as np
from scipy.special import expit, logsumexp
from .common import rng, normalize
from .makers import recipe_library, satisfaction

BETAS = np.linspace(.75, 3., 16)
VIGOR = (.5, 1., 2.)


def menu(r, drives, mode):
    result = r.integers(2, size=(4, drives))
    if mode == 'separating':
        pair = r.choice(drives, 2, replace=False)
        result[:4, pair[0]] = [0, 0, 1, 1]
        result[:4, pair[1]] = [0, 1, 0, 1]
    return result


def sample(v, phase, world, case, law=None):
    s = v['settings']; r = rng(v['id'], phase, world, case)
    law = world % 16 if law is None else law
    if s['extension']:
        return sample_extension(v, phase, world, case, law)
    recipes, weights, _ = recipe_library(s['drives'], exactly=s['recipes'], seeking=s['seeking'])
    index = int(r.integers(len(weights))); vigor = int(r.integers(3)) if s['seeking'] == 'vigor' else 1
    rows = []
    for i in range(s['records']+1):
        options = menu(r, s['drives'], s['menus'] if i < s['records'] else 'separating')
        need = float(r.choice([.5, 1., 2.]))
        actp = float(expit(-.5+VIGOR[vigor])) if s['seeking'] == 'vigor' else .75
        beta = BETAS[law]*(VIGOR[vigor] if s['seeking'] == 'vigor' else 1)
        probs = normalize(beta*(satisfaction(options, recipes, s['meaning'], need)@weights[index]))
        acted = bool(r.random() < actp); selected = int(r.choice(4, p=probs))
        rows.append(dict(options=options.tolist(), need=need, acted=acted,
                         choice=selected if acted or i == s['records'] else None))
    query = rows.pop(); choice = query.pop('choice'); acted = query.pop('acted')
    return dict(history=rows, query=query), dict(y=[choice], acted=acted, type=index,
        presence=(weights[index] > 0).astype(int).tolist(), vigor=vigor, law=int(law))


def infer(v, p, linear=False, seeking=None, known_law=None):
    s = v['settings']; mode = seeking or s['seeking']
    recipes, weights, _ = recipe_library(s['drives'], exactly=s['recipes'], seeking=mode)
    meaning = 'additive' if linear else s['meaning']
    vigors = range(3) if mode == 'vigor' else [1]
    laws = range(16) if known_law is None else [known_law]
    chunks, lp, act = [], [], []
    for law in laws:
        for level in vigors:
            beta = BETAS[law]*(VIGOR[level] if mode == 'vigor' else 1)
            actp = float(expit(-.5+VIGOR[level])) if mode == 'vigor' else .75
            ll = np.zeros(len(weights))
            for row in p['history']:
                # A recorded selected option is available only when the maker acts.
                # Counterfactual choice stays out of public records for abstentions.
                ll += np.log(actp if row['acted'] else 1-actp)
                if row['acted']:
                    utility = weights@satisfaction(row['options'], recipes, meaning, row['need']).T
                    logits = beta*utility
                    ll += logits[:, row['choice']]-logsumexp(logits, axis=1)
            logits = beta*(weights@satisfaction(p['query']['options'], recipes, meaning, p['query']['need']).T)
            chunks.append(normalize(logits)); lp.append(ll); act.append(actp)
    stacked = np.array(lp); posterior = normalize(stacked.ravel()).reshape(stacked.shape)
    prediction = np.einsum('nt,nta->a', posterior, np.array(chunks))
    return prediction[None, :], dict(recipe_presence=(posterior.sum(0)@(weights > 0)).tolist(),
        act_probability=float(posterior.sum(1)@np.array(act)), support_states=int(stacked.size),
        seeking_model=mode, meaning=meaning, exact_finite=True)


def evaluate(v, p, truth=None):
    if v['settings']['extension']:
        return evaluate_extension(v, p, truth)
    readers, details = {}, {}
    for name, kwargs in [('law-marginal', {}), ('drive-weights-only', dict(linear=True)),
                         ('seeking-dimension', dict(seeking='dimension')), ('seeking-vigor', dict(seeking='vigor'))]:
        readers[name], details[name] = infer(v, p, **kwargs)
    if truth is not None:
        readers['oracle-law'], details['oracle-law'] = infer(v, p, known_law=truth['law'])
    return readers, details


def features(v, p):
    if v['settings']['extension']:
        return np.r_[np.asarray(p['history']).ravel(), p['changed'], np.zeros(512*44+37-2*len(p['history'])-1)]
    x = np.zeros((512, 44))
    for i, row in enumerate(p['history']):
        options = np.zeros((4, 9)); options[:, :v['settings']['drives']] = row['options']
        x[i, :36] = options.ravel(); x[i, 36] = row['need']; x[i, 37] = row['acted']
        if row['acted']:
            x[i, 38+row['choice']] = 1
        x[i, 42] = 1
    query = np.zeros((4, 9)); query[:, :v['settings']['drives']] = p['query']['options']
    return np.r_[x.ravel(), query.ravel(), p['query']['need']]


def sample_extension(v, phase, world, case, law):
    s = v['settings']; r = rng(v['id'], phase, world, case)
    rel = np.linspace(.6, .95, 16)[law]
    mechanism = ('transition', 'habit', 'intrinsic').index(s['mechanism'])
    history = [[int(r.random() < rel), 0] for _ in range(s['records'])]
    # The probe changes real contingencies. A distorted transition map, a habit,
    # and intrinsic preference differ in their response, not their history.
    current_p = (1-rel, .5, rel)[mechanism] if s['changed'] else rel
    return dict(history=history, changed=s['changed']), dict(y=[int(r.random() < current_p)],
                                                              mechanism=mechanism, law=int(law))


def evaluate_extension(v, p, truth=None):
    rels = np.linspace(.6, .95, 16); count = sum(row[0] for row in p['history'])
    posterior_law = normalize(count*np.log(rels)+(len(p['history'])-count)*np.log1p(-rels))
    predictions = np.array([1-rels, np.full(16, .5), rels]) if p['changed'] else np.repeat(rels[None, :], 3, axis=0)
    marginal = float(predictions.mean(0)@posterior_law)
    readers = {'law-marginal': np.array([[1-marginal, marginal]])}
    for i, name in enumerate(('transition', 'habit', 'intrinsic')):
        q = float(predictions[i]@posterior_law); readers[name] = np.array([[1-q, q]])
    return readers, dict(mechanism_posterior=[1/3]*3,
                        history_alias_exact=True, probe_not_observed_before_prediction=True)
