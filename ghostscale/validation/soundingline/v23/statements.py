"""N7: noisy stated recipes, population shift correction and aspiration."""
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp
from .common import rng, normalize
from .makers import recipe_library, satisfaction

BETAS = np.linspace(.75, 3., 16)


def shift_direction():
    recipes, _, _ = recipe_library()
    # Four-drive core: need, CARE, RAGE, SEEKING. Statements are recipe scores.
    return np.array([(1 in recipe)-(2 in recipe) for recipe in recipes], float)


def sample(v, phase, world, case, law=None):
    s = v['settings']; r = rng(v['id'], phase, world, case)
    law = world % 16 if law is None else law
    recipes, weights, _ = recipe_library(); kind = int(r.integers(len(weights)))
    sign = float(r.choice([-1, 1])) if s['shift_kind'] == 'individual' else 1.
    shift = sign*s['shift']*shift_direction()
    aspirational = bool(r.random() < s['aspiration'])
    statement = weights[kind]+shift+r.normal(0, s['statement_sd'], len(recipes))
    history = []
    for step in range(s['records']):
        options = r.integers(2, size=(3, 4))
        progression = step/max(1, s['records'])
        personal = weights[kind]+aspirational*progression*shift
        probability = normalize(BETAS[law]*(satisfaction(options, recipes)@personal))
        history.append(dict(options=options.tolist(), choice=int(r.choice(3, p=probability))))
    query = r.integers(2, size=(3, 4)); future = weights[kind]+aspirational*shift
    probability = normalize(BETAS[law]*(satisfaction(query, recipes)@future))
    return dict(history=history, statement=statement.tolist(), query=query.tolist()), dict(
        y=[int(r.choice(3, p=probability))], type=kind, presence=(weights[kind] > 0).astype(int).tolist(),
        law=int(law), aspiration=aspirational, shift=shift.tolist())


def infer_scalar(v, p, mode='law-marginal', shift_estimate=None):
    s = v['settings']; recipes, weights, _ = recipe_library()
    signs = (-1., 1.) if s['shift_kind'] == 'individual' else (1.,)
    aspiration_flags = (False, True) if s['aspiration'] else (False,)
    params, likelihoods, predictions = [], [], []
    for sign in signs:
        actual_shift = sign*s['shift']*shift_direction()
        statement_shift = actual_shift if mode == 'law-marginal' else np.zeros(len(recipes))
        if mode == 'pooled-correction':
            statement_shift = np.asarray(shift_estimate)
        for aspiring in aspiration_flags:
            logprior = np.log(s['aspiration'] if aspiring else 1-s['aspiration']) if s['aspiration'] else 0.
            for beta in BETAS:
                ll = np.full(len(weights), logprior)
                if mode not in ('choices-only', 'training-choices'):
                    residual = np.array(p['statement'])[None, :]-weights-statement_shift
                    ll -= .5*np.sum((residual/s['statement_sd'])**2, axis=1)
                if mode != 'statements-only':
                    for step, row in enumerate(p['history']):
                        current = weights+aspiring*(step/max(1, s['records']))*actual_shift
                        logits = beta*(current@satisfaction(row['options'], recipes).T)
                        ll += logits[:, row['choice']]-logsumexp(logits, axis=1)
                future = weights+aspiring*actual_shift
                predictions.append(normalize(beta*(future@satisfaction(p['query'], recipes).T)))
                likelihoods.append(ll); params.append((sign, aspiring, beta))
    ll = np.array(likelihoods); posterior = normalize(ll.ravel()).reshape(ll.shape)
    prediction = np.einsum('nt,nta->a', posterior, np.array(predictions))
    return prediction[None, :], dict(recipe_presence=(posterior.sum(0)@(weights > 0)).tolist(),
        expected_weights=(posterior.sum(0)@weights).tolist(), support_states=int(ll.size))


def infer(v, p, mode='law-marginal', shift_estimate=None):
    """Same finite model with law/type updates broadcast across nuisance states."""
    from itertools import product
    s=v['settings'];recipes,weights,_=recipe_library()
    states=list(product((-1.,1.) if s['shift_kind']=='individual' else (1.,),
        (False,True) if s['aspiration'] else (False,),BETAS))
    sign=np.array([row[0] for row in states]);aspiring=np.array([row[1] for row in states]);beta=np.array([row[2] for row in states])
    shifts=sign[:,None]*s['shift']*shift_direction()[None,:]
    prior=np.where(aspiring,s['aspiration'],1-s['aspiration']) if s['aspiration'] else np.ones(len(states))
    ll=np.broadcast_to(np.log(prior)[:,None],(len(states),len(weights))).copy()
    if mode not in ('choices-only','training-choices'):
        observed_shift=shifts if mode=='law-marginal' else np.zeros_like(shifts)
        if mode=='pooled-correction':observed_shift=np.broadcast_to(shift_estimate,shifts.shape)
        residual=np.array(p['statement'])[None,None,:]-weights[None,:,:]-observed_shift[:,None,:]
        ll-=.5*np.sum((residual/s['statement_sd'])**2,axis=2)
    if mode!='statements-only':
        for step,row in enumerate(p['history']):
            sat=satisfaction(row['options'],recipes)
            utility=(weights@sat.T)[None,:,:]+aspiring[:,None,None]*step/max(1,s['records'])*(shifts@sat.T)[:,None,:]
            logits=beta[:,None,None]*utility
            ll+=logits[:,:,row['choice']]-logsumexp(logits,axis=2)
    sat=satisfaction(p['query'],recipes)
    utility=(weights@sat.T)[None,:,:]+aspiring[:,None,None]*(shifts@sat.T)[:,None,:]
    post=normalize(ll.ravel()).reshape(ll.shape)
    predicted=np.einsum('nt,nta->a',post,normalize(beta[:,None,None]*utility))
    return predicted[None,:],dict(recipe_presence=(post.sum(0)@(weights>0)).tolist(),
        expected_weights=(post.sum(0)@weights).tolist(),support_states=int(ll.size))


@lru_cache(None)
def shift_training(identity, sd, shift, shift_kind, aspiration):
    v = dict(id=identity, settings=dict(statement_sd=sd, shift=shift, shift_kind=shift_kind,
                                      aspiration=aspiration, records=128))
    residuals = []; retained = []
    # Separate training makers. Their hidden type labels are never used to fit
    # the shift; the choice-only posterior supplies the latent value estimate.
    for i in range(32):
        p, _ = sample(v, 'shift-training', i % 16, i)
        _, detail = infer(v, p, 'training-choices')
        residuals.append(np.array(p['statement'])-detail['expected_weights'])
        retained.append(dict(identity=['shift-training',identity,i],public=p,
                             choice_only_expected_weights=detail['expected_weights']))
    return np.mean(residuals, axis=0), retained


def trained_shift(identity, sd, shift, shift_kind, aspiration):
    return shift_training(identity, sd, shift, shift_kind, aspiration)[0]


def shift_training_identity(s):
    # The same population correction is evaluated at all four personal-history
    # depths. These independent training makers always have 128 records.
    return 'N7-shared-shift-'+str([s[k] for k in ('statement_sd','shift','shift_kind','aspiration')])


def evaluate(v, p, truth=None):
    s = v['settings']
    shift = trained_shift(shift_training_identity(s), s['statement_sd'], s['shift'], s['shift_kind'], s['aspiration'])
    pred, detail = {}, {}
    for name in ('law-marginal', 'choices-only', 'statements-only', 'both-naive', 'pooled-correction'):
        pred[name], detail[name] = infer(v, p, name, shift)
    detail['training'] = dict(makers=32, records=128, shift=shift.tolist(), hidden_labels_used=False)
    return pred, detail


def features(v, p):
    x = np.zeros((32, 16))
    for i, row in enumerate(p['history']):
        x[i, :12] = np.asarray(row['options']).ravel(); x[i, 12+row['choice']] = 1; x[i, 15] = 1
    return np.r_[x.ravel(), p['statement'], np.asarray(p['query']).ravel()]
