"""Numerical primitives, public-data direct reader and marginal proper scores."""
import hashlib
import time
import numpy as np
from scipy.special import logsumexp


def seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(('v22', parts)).encode()).digest()[:8], 'little')


def rng(*parts):
    return np.random.default_rng(seed(*parts))


def normalize(logp, axis=-1):
    return np.exp(logp - logsumexp(logp, axis=axis, keepdims=True))


def softmax(value):
    return normalize(np.asarray(value, float))


def bern(p):
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or (p < -1e-12).any() or (p > 1+1e-12).any():
        raise ValueError('invalid binary marginal')
    # Dot products of a normalized finite distribution may exceed one by an ulp.
    # Preserve exact zero/one; this is roundoff repair, never a probability floor.
    p = np.clip(p, 0., 1.)
    return np.stack((1-p, p), axis=-1)


def lognormal(x, mean, sd):
    return -.5*((np.asarray(x)-mean)/sd)**2-np.log(sd)-.5*np.log(2*np.pi)


def score(p, y, mask=None):
    p = np.asarray(p, float)
    if p.ndim == 2:
        p = p[:, None, :]
    y = np.asarray(y, int).reshape(p.shape[:2])
    mask = np.ones_like(y, float) if mask is None else np.asarray(mask, float)
    if not np.allclose(p.sum(-1), 1, atol=1e-8) or (p < 0).any() or not np.isfinite(p).all():
        raise ValueError('invalid probability')
    q = np.take_along_axis(p, y[..., None], -1)[..., 0]
    with np.errstate(divide='ignore'):
        loss = -np.log(q)
    mode = p.argmax(-1)
    return dict(log_loss=np.where(mask, loss, 0).sum(1)/mask.sum(1),
                brier=(((p*p).sum(-1)-2*q+1)*mask).sum(1)/mask.sum(1),
                accuracy=((mode == y)*mask).sum(1)/mask.sum(1),
                confident_error=(((p.max(-1) >= .9) & (mode != y))*mask).sum(1)/mask.sum(1))


def bootstrap(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        return dict(mean=None, lower=None, upper=None, finite=False, worlds=len(values))
    draws = rng('paired-bootstrap').integers(len(values), size=(4096, len(values)))
    means = values[draws].mean(1)
    return dict(mean=float(values.mean()), lower=float(np.quantile(means, .025)),
                upper=float(np.quantile(means, .975)), finite=True, worlds=len(values))


class Direct:
    """Same public features, multi-target marginal cross entropy, saved validation stop."""
    def __init__(self, x, y, vx, vy, mask, vmask, classes, fit=0, max_epochs=120,
                 pulse=lambda **kw: None):
        start = time.process_time()
        r = rng('direct', fit)
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), .1)
        x = ((x-self.mean)/self.scale).astype(np.float32)
        vx = ((vx-self.mean)/self.scale).astype(np.float32)
        self.heads = y.shape[1]
        self.classes = classes
        self.params = [r.normal(0, 1/np.sqrt(x.shape[1]), (x.shape[1], 64)).astype(np.float32),
                       np.zeros(64, np.float32),
                       r.normal(0, .03, (64, self.heads*classes)).astype(np.float32),
                       np.zeros(self.heads*classes, np.float32)]
        first = [np.zeros_like(p) for p in self.params]
        second = [np.zeros_like(p) for p in self.params]
        best = np.inf
        best_params = None
        stale = steps = 0
        curve = []
        for epoch in range(max_epochs):
            order = r.permutation(len(y))
            for start_i in range(0, len(y), 256):
                ids = order[start_i:start_i+256]
                a, b, c, d = self.params
                h = np.tanh(x[ids]@a+b)
                p = normalize((h@c+d).reshape(-1, self.heads, classes).astype(float)).astype(np.float32)
                p[np.arange(len(ids))[:, None], np.arange(self.heads)[None, :], y[ids]] -= 1
                p *= (mask[ids]/mask[ids].sum(1, keepdims=True))[..., None]/len(ids)
                g = p.reshape(len(ids), -1)
                dh = (g@c.T)*(1-h*h)
                gradients = [x[ids].T@dh+1e-4*a, dh.sum(0), h.T@g+1e-4*c, g.sum(0)]
                steps += 1
                for i, (param, grad) in enumerate(zip(self.params, gradients)):
                    first[i] = .9*first[i]+.1*grad
                    second[i] = .999*second[i]+.001*grad*grad
                    param -= .003*(first[i]/(1-.9**steps))/(np.sqrt(second[i]/(1-.999**steps))+1e-8)
            loss = float(score(self._predict(vx), vy, vmask)['log_loss'].mean())
            curve.append(dict(epoch=epoch+1, validation_log_loss=loss))
            pulse(phase='direct-fit', epoch=epoch+1)
            if loss < best-1e-4:
                best = loss
                best_params = [p.copy() for p in self.params]
                stale = 0
            else:
                stale += 1
            if stale >= 10:
                break
        self.params = best_params
        self.metadata = dict(fit=fit, labels=len(y), validation_labels=len(vy),
                             early_stopped=stale >= 10, epochs=len(curve), curve=curve,
                             validation_log_loss=best, cpu_seconds=time.process_time()-start,
                             output='separate marginal heads; not a joint score')

    def _predict(self, x):
        a, b, c, d = self.params
        return normalize((np.tanh(x@a+b)@c+d).reshape(-1, self.heads, self.classes).astype(float))

    def predict(self, x):
        return self._predict(((x-self.mean)/self.scale).astype(np.float32))

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, scale=self.scale,
                            heads=self.heads, classes=self.classes,
                            **{f'p{i}': p for i, p in enumerate(self.params)})
