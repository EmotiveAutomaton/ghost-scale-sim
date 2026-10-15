"""N6: warmth/competence inference separated from uptake and payoff."""
from itertools import product
import numpy as np
from .common import rng, normalize, binary_ll

STATES = np.array(list(product(range(2), range(2), range(16))))
RELIABILITY = np.linspace(.6, .95, 16)


def prior(v):
    s = v['settings']; low = s['lowlow_share']
    return np.where((STATES[:, 0] == 0) & (STATES[:, 1] == 0), low, (1-low)/3)/16


def action_prob(warmth, competence, reliability):
    # Four actions combine served goal and success; dependence is explicit.
    pw = np.where(warmth, reliability, 1-reliability)
    pc = np.where(competence, reliability, 1-reliability)
    return np.stack([(1-pw)*(1-pc), (1-pw)*pc, pw*(1-pc), pw*pc], axis=-1)


def sample(v, phase, world, case, law=None):
    s = v['settings']; r = rng(v['id'], phase, world, case)
    law = world % 16 if law is None else law
    pair = int(r.choice(4, p=[s['lowlow_share']]+[(1-s['lowlow_share'])/3]*3))
    warm, competent = pair//2, pair % 2
    rel = RELIABILITY[law]
    history = [dict(claim=int(r.random() < (rel if warm else 1-rel)),
                    procedure=int(r.random() < (rel if competent else 1-rel))) for _ in range(s['records'])]
    y = int(r.choice(4, p=action_prob(warm, competent, rel)))
    return dict(history=history), dict(y=[y], warmth=warm, competence=competent, law=int(law))


def infer(v, p, withdraw=False, law=None):
    s = v['settings']; warm, competent, l = STATES.T
    ll = np.log(prior(v))
    if law is not None:
        ll = np.where(l == law, ll, -np.inf)
    stopped = False; uptake = technique = compute = 0.; path = []
    for row in p['history']:
        before = normalize(ll)
        lowlow = float(before@((warm == 0) & (competent == 0)))
        stopped |= withdraw and lowlow >= s['threshold']
        if not stopped:
            ll += binary_ll(row['claim'], np.where(warm, RELIABILITY[l], 1-RELIABILITY[l]))
            ll += binary_ll(row['procedure'], np.where(competent, RELIABILITY[l], 1-RELIABILITY[l]))
            compute += 1
        post = normalize(ll); p_warm = float(post@warm); p_comp = float(post@competent)
        accept_technique = not stopped and p_comp >= s['threshold']
        accept_purpose = not stopped and p_warm >= s['threshold']
        technique += accept_technique
        uptake += accept_purpose or (s['coupled'] and accept_technique)
        path.append(dict(stopped=bool(stopped), warmth=p_warm, competence=p_comp,
                         purpose_accepted=bool(accept_purpose), technique_accepted=bool(accept_technique)))
    return (normalize(ll)@action_prob(warm, competent, RELIABILITY[l]))[None, :], dict(
        path=path, updates=compute, adopted_purposes=uptake, adopted_techniques=technique)


def evaluate(v, p, truth=None):
    model, md = infer(v, p); withdraw, wd = infer(v, p, withdraw=True)
    readers = {'law-marginal': model, 'model-and-gate': model, 'withdraw': withdraw,
               'absorb-everything': model}
    detail = dict(model=md, withdraw=wd, primary_stratum='warmth=0 and competence=0',
                  compute_fee=.001, stakes=v['settings']['stakes'])
    if truth is not None:
        readers['oracle-law'] = infer(v, p, law=truth['law'])[0]
        for d in (md, wd):
            d['unwanted_uptake'] = d['adopted_purposes']*(1-truth['warmth'])
            d['useful_technique'] = d['adopted_techniques']*truth['competence']
            d['failed_technique'] = d['adopted_techniques']*(1-truth['competence'])
            d['nonprediction_payoff'] = d['useful_technique']-d['unwanted_uptake']-d['failed_technique']-.001*d['updates']
        detail['primary_eligible'] = truth['warmth'] == truth['competence'] == 0
    return readers, detail


def features(v, p):
    x = np.zeros((32, 3))
    for i, row in enumerate(p['history']):
        x[i] = [row['claim'], row['procedure'], 1]
    return x.ravel()
