"""Frozen finite Bayesian family policy; calibration is a separate instrument."""
import math
import numpy as np
from scipy.special import ndtr, logsumexp
from .common import rng

RUNGS = (16, 16, 48, 144, 144, 144, 144)
PRIOR_GRID = ((.03, .02), (.05, .03), (.08, .05))
KILL_GRID = (.05, .10, .15)


def posterior(means, ses, mu_scale=.05, tau_scale=.03, margin=.02, nodes=121):
    """Integrate mu analytically and tau by checked trapezoid quadrature.

    Inputs are inverse-variance aggregates of independent variant batches,
    conditional on their frozen fits. This does not pretend to integrate
    training-run uncertainty. Different variants use disjoint world seeds.
    """
    means, ses = np.asarray(means, float), np.asarray(ses, float)
    if means.ndim != 1 or means.shape != ses.shape or len(means) == 0:
        raise ValueError('one estimate and standard error per variant required')
    if not np.isfinite(means).all() or not np.isfinite(ses).all() or np.any(ses <= 0):
        raise ValueError('nonfinite/zero-error estimate cannot enter Gaussian approximation')
    tau = np.linspace(0, max(8*tau_scale, .16), nodes)
    variances = tau[:, None]**2+ses[None, :]**2
    precision = 1/variances
    mu_var = 1/(1/mu_scale**2+precision.sum(1))
    mu_mean = mu_var*(precision*means).sum(1)
    logw = -.5*(np.log(variances).sum(1)+np.log(mu_scale**2/mu_var)
                 +(precision*means**2).sum(1)-mu_mean**2/mu_var)
    logw -= .5*(tau/tau_scale)**2
    logw[[0, -1]] += math.log(.5)
    weights = np.exp(logw-logsumexp(logw))
    shrink = tau[:, None]**2/variances
    conditional_mean = shrink*means+(1-shrink)*mu_mean[:, None]
    conditional_var = shrink*ses**2+(1-shrink)**2*mu_var[:, None]
    marginal_mean = weights@conditional_mean
    tail = weights@ndtr((conditional_mean-margin)/np.sqrt(conditional_var))
    best = int(np.argmax(marginal_mean))  # Stable frozen index tie-break.
    return dict(mean=marginal_mean.tolist(), probability_above_margin=tail.tolist(),
                best=best, q=float(tail[best]), nodes=nodes,
                posterior='conditional_on_frozen_training_and_gaussian_batch_model')


def checked_posterior(means, ses, **kwargs):
    a = posterior(means, ses, nodes=121, **kwargs)
    for nodes in (241, 481, 961, 1921):
        b = posterior(means, ses, nodes=nodes, **kwargs)
        c = max(float(np.max(np.abs(np.array(a['probability_above_margin'])-b['probability_above_margin']))),
                float(np.max(np.abs(np.array(a['mean'])-b['mean']))))
        if c <= .002:
            return dict(b, quadrature_max_difference=c, quadrature_passed=True)
        a = b
    raise ValueError('posterior quadrature unresolved at 1921 nodes; no decision')


def combine(estimates):
    means, ses = [], []
    for rows in estimates:
        if not rows:
            raise ValueError('every variant requires burn-in evidence')
        variances = np.array([r['se']**2 for r in rows])
        if not np.isfinite(variances).all() or np.any(variances <= 0):
            raise ValueError('invalid batch uncertainty')
        precision = 1/variances
        means.append(float(np.dot(precision, [r['mean'] for r in rows])/precision.sum()))
        ses.append(float(1/np.sqrt(precision.sum())))
    return means, ses


def decide(batch, estimates, active, late_effect, kill=.1, mu_scale=.05, tau_scale=.03,
           eligible=None, checked=True):
    means, ses = combine(estimates)
    inference = (checked_posterior if checked else posterior)(means, ses, mu_scale=mu_scale, tau_scale=tau_scale)
    eligible = list(range(len(estimates))) if eligible is None else list(eligible)
    if not eligible:
        return dict(state='MAP_ONLY', active=list(active), posterior=inference)
    best = max(eligible, key=lambda i: (inference['mean'][i], -i))
    q = inference['probability_above_margin'][best]
    state = ('BURN_IN' if batch < 2 else 'STOPPED_BELOW_MARGIN' if q < kill else
             'NOMINATED' if q >= .9 else 'INCONCLUSIVE_AT_CAP' if batch >= 7 else 'CONTINUE')
    next_active = list(active)
    if state == 'CONTINUE':
        count = max(1, math.ceil(len(active)/3))
        keep = sorted(active, key=lambda i: (-inference['mean'][i], i))[:count]
        # Batch 4 is still protected. Halving for batch 5 may park these variants.
        protected = [i for i in active if i in late_effect] if batch+1 <= 4 else []
        next_active = sorted(set(keep+protected))
    return dict(state=state, batch=batch, best=best, q=float(q), active=next_active,
                posterior=inference, margin=.02, kill=kill)


def wilson(success, total, z=1.959963984540054):
    if total == 0:
        return dict(rate=None, lower=None, upper=None, trials=0)
    p = success/total
    den = 1+z*z/total
    mid = (p+z*z/(2*total))/den
    radius = z*np.sqrt(p*(1-p)/total+z*z/(4*total**2))/den
    return dict(rate=p, lower=max(0., mid-radius), upper=min(1., mid+radius), trials=total)


def calibration_trial(key, n, mu, tau, se, slow, kill, prior):
    r = rng('calibration', key)
    theta = r.normal(mu, tau, n)
    active, estimates = list(range(n)), [[] for _ in range(n)]
    late = set(range(n//3))
    trajectory = []
    consecutive, counting_killed = 0, {k: False for k in (2, 5, 7)}
    observations = []
    for batch, worlds in enumerate(RUNGS, 1):
        err = se*np.sqrt(16/worlds)
        # In a slow starter, future capacity improves. The target is the asymptote;
        # mismatch with a constant-effect model is intentionally stress-tested.
        gain = theta*(1-.5**(1+max(0, round(math.log(worlds/16, 3))))) if slow else theta
        observed = gain+r.normal(0, err, n)
        observations.append((observed, err))
        # Counting comparators see all variants each batch; their resource
        # difference is recorded instead of pretending identical allocation.
        clear = bool(np.max(observed-1.96*err) > .02)
        consecutive = 0 if clear else consecutive+1
        for k in counting_killed:
            counting_killed[k] |= consecutive >= k
    for batch, (observed, err) in enumerate(observations, 1):
        for i in active:
            estimates[i].append(dict(mean=float(observed[i]), se=float(err)))
        result = decide(batch, estimates, active, late, kill, *prior, checked=True)
        trajectory.append(dict(batch=batch, q=result['q'], state=result['state']))
        active = result['active']
        if result['state'] not in ('CONTINUE', 'BURN_IN'):
            break
    return dict(strong_alternative=bool(theta.max() > .03),
                null=bool(theta.max() <= .02), state=result['state'],
                counting_killed=counting_killed, trajectory=trajectory)
