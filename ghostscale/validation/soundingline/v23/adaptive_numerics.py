"""Algebraically equivalent arithmetic for the frozen Gaussian family model.

No epsilon, variance floor, prior change, node change, or margin change. This
implementation requires a complete calibration-equivalence receipt before use.
The original calibrated module and all its accepted capsules remain preserved.
"""
import math
import numpy as np
from scipy.special import ndtr, logsumexp


def combine(estimates):
    means, ses = [], []
    for rows in estimates:
        if not rows:
            raise ValueError('every variant requires burn-in evidence')
        errors = np.array([row['se'] for row in rows], float)
        values = np.array([row['mean'] for row in rows], float)
        if not np.isfinite(errors).all() or np.any(errors <= 0) or not np.isfinite(values).all():
            raise ValueError('invalid batch uncertainty')
        scale = errors.min()
        weights = (scale/errors)**2
        means.append(float(np.dot(weights, values)/weights.sum()))
        ses.append(float(scale/np.sqrt(weights.sum())))
    return means, ses


def posterior(means, ses, mu_scale=.05, tau_scale=.03, margin=.02, nodes=121):
    means, ses = np.asarray(means, float), np.asarray(ses, float)
    if means.ndim != 1 or means.shape != ses.shape or len(means) == 0:
        raise ValueError('one estimate and standard error per variant required')
    if not np.isfinite(means).all() or not np.isfinite(ses).all() or np.any(ses <= 0):
        raise ValueError('nonfinite/zero-error estimate cannot enter Gaussian approximation')
    tau = np.linspace(0, max(8*tau_scale, .16), nodes)
    sd = np.hypot(tau[:, None], ses[None, :])
    scale = np.minimum(sd.min(1), mu_scale)
    precision_scaled = (scale[:, None]/sd)**2
    denominator = (scale/mu_scale)**2+precision_scaled.sum(1)
    mu_sd = scale/np.sqrt(denominator)
    mu_mean = (precision_scaled*means).sum(1)/denominator
    # Centered residuals avoid cancellation of two enormous quadratic forms.
    quadratic = np.sum(((means-mu_mean[:, None])/sd)**2, axis=1)+(mu_mean/mu_scale)**2
    logw = -.5*(2*np.log(sd).sum(1)+2*math.log(mu_scale)-2*np.log(mu_sd)+quadratic)
    logw -= .5*(tau/tau_scale)**2
    logw[[0, -1]] += math.log(.5)
    weights = np.exp(logw-logsumexp(logw))
    shrink = (tau[:, None]/sd)**2
    conditional_mean = shrink*means+(1-shrink)*mu_mean[:, None]
    conditional_sd = np.hypot(np.sqrt(shrink)*ses, (1-shrink)*mu_sd[:, None])
    marginal_mean = weights@conditional_mean
    tail = weights@ndtr((conditional_mean-margin)/conditional_sd)
    best = int(np.argmax(marginal_mean))
    return dict(mean=marginal_mean.tolist(), probability_above_margin=tail.tolist(),
        best=best, q=float(tail[best]), nodes=nodes,
        posterior='conditional_on_frozen_training_and_gaussian_batch_model')


def stable_log_loss(probabilities, truth):
    """Normalize a rounded simplex while retaining its small complementary mass."""
    p = np.asarray(probabilities, float)
    y = np.asarray(truth, int)
    if (p.shape[:-1] != y.shape or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(-1), 1, atol=1e-10, rtol=0)):
        raise ValueError('invalid predictive distribution')
    chosen = np.take_along_axis(p, y[..., None], axis=-1)[..., 0]
    mask = np.arange(p.shape[-1]) != y[..., None]
    other = np.sum(np.where(mask, p, 0.), axis=-1)
    result = np.empty_like(chosen)
    high = chosen > .5
    # log((p_y + other)/p_y), evaluated without rounding p_y + other to p_y.
    result[high] = np.log1p(other[high]/chosen[high])
    with np.errstate(divide='ignore'):
        result[~high] = -np.log(chosen[~high])+np.log(chosen[~high]+other[~high])
    return result


def independent_log_loss(probabilities, truth):
    """Scalar positive-mass ratio, separate from the vector score consumer."""
    result = []
    for row, y in zip(probabilities, truth, strict=True):
        p = list(map(float, row))
        if abs(math.fsum(p)-1) > 1e-10 or min(p) < 0 or not all(math.isfinite(x) for x in p):
            raise ValueError('invalid checked probability')
        other = math.fsum(value for index, value in enumerate(p) if index != y)
        if not p[y]:
            result.append(math.inf)
        elif p[y] > .5:
            result.append(math.log1p(other/p[y]))
        else:
            result.append(math.log(math.fsum(p))-math.log(p[y]))
    return np.array(result)


def interval(values,identity,draws=4096):
    from .common import interval as original_interval,rng
    result=original_interval(values,identity,draws)
    if not result['finite']:return result
    values=np.asarray(values,float);scale=float(np.max(np.abs(values)))
    if scale and scale<1e-100:
        indices=rng('bootstrap',identity).integers(len(values),size=(draws,len(values)))
        # Same bootstrap draws and estimator; scaling only protects variance
        # arithmetic from squaring small, still-representable observations.
        result['se']=float((values[indices]/scale).mean(1).std(ddof=1)*scale)
    return result


def install():
    """Only called in an isolated repaired packet after validation/pin checks."""
    from . import adaptive
    adaptive.combine = combine
    adaptive.posterior = posterior


def decide(*args,**kwargs):
    from . import adaptive
    previous=(adaptive.combine,adaptive.posterior)
    try:
        install()
        return adaptive.decide(*args,**kwargs)
    finally:
        adaptive.combine,adaptive.posterior=previous


def validate(plan):
    from pathlib import Path
    from .common import digest,read
    from .family import verify_packet
    root=Path(plan['numerical_validation_packet'])
    pp,_=verify_packet(root)
    if (digest(root/'COMPLETE.json')!=plan['numerical_validation_complete_sha256'] or
            read(root/'original/EQUIVALENCE.json')['state']!='CALIBRATION_EQUIVALENT'):
        raise ValueError('numerical calibration equivalence unresolved')
    source='ghostscale/validation/soundingline/v23/adaptive_numerics.py'
    # The accepted validator freezes the repaired combine/posterior/score
    # function ASTs. Admission binds those exact functions separately from
    # orchestration added after calibration; no changed arithmetic is accepted.
    import ast,hashlib
    def functions(path):
        tree=ast.parse(Path(path).read_text(encoding='utf-8'))
        names={'combine','posterior','stable_log_loss','independent_log_loss'}
        return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
                for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
    frozen=Path(pp['source_archive']).with_suffix('')/source
    if functions(frozen)!=functions(__file__):raise ValueError('validated numerical arithmetic changed')
