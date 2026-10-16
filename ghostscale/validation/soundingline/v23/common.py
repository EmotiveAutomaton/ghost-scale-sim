"""Version-specific deterministic primitives; no closed campaign entry points."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
from scipy.special import logsumexp


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()).encode('utf-8')


def seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical(['v23', *parts])).digest()[:8], 'little')


def rng(*parts):
    return np.random.default_rng(seed(*parts))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def retry_io(operation):
    """Bound transient Windows sharing failures; never ignore a failed write."""
    for attempt in range(40):
        try: return operation()
        except PermissionError:
            if attempt == 39: raise
            time.sleep(min(.005*2**min(attempt,4), .05))


def write(path, value, *, immutable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + b'\n'
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('xb') as target:
        target.write(data); target.flush(); os.fsync(target.fileno())
    try:
        if immutable: retry_io(lambda: os.link(tmp, path))
        else: retry_io(lambda: os.replace(tmp, path))
    finally:
        if tmp.exists(): retry_io(tmp.unlink)


def read(path):
    return json.loads(retry_io(lambda: Path(path).read_bytes()).decode('utf-8-sig'))


def records(path, rows):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as target:
            for row in rows:
                target.write(canonical(row) + b'\n')


def log_normalize(logp, axis=-1):
    values = np.asarray(logp, float)
    normalizer = logsumexp(values, axis=axis, keepdims=True)
    if not np.isfinite(normalizer).all():
        raise ValueError('empty or nonfinite posterior support')
    return values - normalizer


def normalize(logp, axis=-1):
    return np.exp(log_normalize(logp, axis))


def bern(p):
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or np.any((p < -1e-12) | (p > 1 + 1e-12)):
        raise ValueError('invalid Bernoulli probability')
    p = np.clip(p, 0, 1)  # Roundoff only; exact zeros and infinities remain.
    return np.stack([1-p, p], axis=-1)


def binary_ll(obs, probability):
    with np.errstate(divide='ignore'):
        return np.log(np.where(np.asarray(obs), probability, 1-np.asarray(probability)))


def losses(p, y):
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(-1), 1, atol=1e-10):
        raise ValueError('invalid predictive distribution')
    q = np.take_along_axis(p, y[..., None], axis=-1)[..., 0]
    with np.errstate(divide='ignore'):
        loss = -np.log(q)
    return dict(log_loss=loss, half_brier=((p*p).sum(-1)-2*q+1)/2,
                accuracy=(p.argmax(-1) == y).astype(float))


def interval(values, identity, draws=4096):
    values = np.asarray(values, float)
    if len(values) < 2 or not np.isfinite(values).all():
        return dict(mean=None, lower=None, upper=None, se=None, finite=False,
                    worlds=len(values), infinite=int(np.isinf(values).sum()))
    sampled = rng('bootstrap', identity).integers(len(values), size=(draws, len(values)))
    means = values[sampled].mean(1)
    return dict(mean=float(values.mean()), lower=float(np.quantile(means, .025)),
                upper=float(np.quantile(means, .975)), se=float(means.std(ddof=1)),
                finite=True, worlds=len(values))


class Direct:
    """V22's fixed network/Adam/validation stop, with V23 family-specific seeds.

    A fresh-law provider may replace each training minibatch, while the fixed
    validation population and stopping rule remain unchanged. No test outcomes
    select weights, epochs or normalization.
    """
    def __init__(self, x, y, vx, vy, classes, family, fit, max_epochs=120,
                 fresh=None, pulse=lambda **kw: None):
        start = time.process_time()
        r = rng('direct', family, fit, 'fresh' if fresh else 'fixed')
        self.mean, self.scale = x.mean(0), np.maximum(x.std(0), .1)
        x = ((x-self.mean)/self.scale).astype(np.float32)
        vx = ((vx-self.mean)/self.scale).astype(np.float32)
        y, vy = np.asarray(y, int), np.asarray(vy, int)
        if y.ndim == 1:
            y, vy = y[:, None], vy[:, None]
        self.heads, self.classes = y.shape[1], classes
        self.params = [r.normal(0, 1/np.sqrt(x.shape[1]), (x.shape[1], 64)).astype(np.float32),
                       np.zeros(64, np.float32),
                       r.normal(0, .03, (64, self.heads*classes)).astype(np.float32),
                       np.zeros(self.heads*classes, np.float32)]
        first, second = [[np.zeros_like(p) for p in self.params] for _ in range(2)]
        best, stale, steps = np.inf, 0, 0
        best_params, curve = None, []
        for epoch in range(max_epochs):
            order = r.permutation(len(y))
            for start_i in range(0, len(y), 256):
                ids = order[start_i:start_i+256]
                xx, yy = x[ids], y[ids]
                if fresh is not None:
                    raw_x, yy = fresh(epoch, start_i, len(ids))
                    xx = ((raw_x-self.mean)/self.scale).astype(np.float32)
                    yy = np.asarray(yy, int).reshape(len(ids), self.heads)
                a, b, c, d = self.params
                h = np.tanh(xx@a+b)
                p = normalize((h@c+d).reshape(-1, self.heads, classes).astype(float)).astype(np.float32)
                p[np.arange(len(ids))[:, None], np.arange(self.heads)[None, :], yy] -= 1
                g = (p/(len(ids)*self.heads)).reshape(len(ids), -1)
                dh = (g@c.T)*(1-h*h)
                gradients = [xx.T@dh+1e-4*a, dh.sum(0), h.T@g+1e-4*c, g.sum(0)]
                steps += 1
                for i, (param, grad) in enumerate(zip(self.params, gradients)):
                    first[i] = .9*first[i]+.1*grad
                    second[i] = .999*second[i]+.001*grad*grad
                    param -= .003*(first[i]/(1-.9**steps))/(np.sqrt(second[i]/(1-.999**steps))+1e-8)
            loss = float(losses(self._predict(vx), vy)['log_loss'].mean())
            curve.append(dict(epoch=epoch+1, validation_log_loss=loss))
            pulse(phase='direct-fit', family=family, fit=fit, epoch=epoch+1)
            if loss < best-1e-4:
                best, best_params, stale = loss, [p.copy() for p in self.params], 0
            else:
                stale += 1
            if stale >= 10:
                break
        self.params = best_params
        self.metadata = dict(fit=fit, labels=len(y), validation_labels=len(vy), family=family,
                             early_stopped=stale >= 10, epochs=len(curve), curve=curve,
                             validation_log_loss=best, fresh_law_minibatches=fresh is not None,
                             generated_training_labels=len(y)*(1+len(curve)) if fresh else len(y),
                             optimizer_label_presentations=len(y)*len(curve),
                             matched_label_budget=fresh is None,
                             fit_seed=seed('direct', family, fit, 'fresh' if fresh else 'fixed'))
        self.cpu_seconds = time.process_time()-start

    def _predict(self, x):
        a, b, c, d = self.params
        return normalize((np.tanh(x@a+b)@c+d).reshape(-1, self.heads, self.classes).astype(float))

    def predict(self, x):
        return self._predict(((np.asarray(x)-self.mean)/self.scale).astype(np.float32))

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, scale=self.scale, heads=self.heads,
                            classes=self.classes, **{f'p{i}': p for i, p in enumerate(self.params)})

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with np.load(path) as source:
            obj.mean, obj.scale = source['mean'], source['scale']
            obj.heads, obj.classes = int(source['heads']), int(source['classes'])
            obj.params = [source[f'p{i}'] for i in range(4)]
        return obj
