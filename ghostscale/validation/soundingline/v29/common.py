"""Prospective V29 seeds, block uncertainty, and class-evidence conventions."""
import json
import zlib
import numpy as np
from scipy.special import logsumexp, expit
from scipy.stats import t as student
from ..v27.common import read, write, digest


def seed(*parts):
    return zlib.crc32(json.dumps(['V29', *parts], sort_keys=True, separators=(',', ':')).encode())


def rng(*parts):
    return np.random.default_rng(seed(*parts))


def logistic(x, floor=.02):
    return np.clip(expit(x), floor, 1-floor)


def class_evidence(ll, classes):
    return np.stack([logsumexp(ll[..., classes == k], axis=-1)-np.log(np.sum(classes == k))
                     for k in np.unique(classes)], axis=-1)


def sustained(odds, bar=3.):
    """One-based first observation at/above bar that stays there; T+1 is censored."""
    steps = odds.shape[-1]
    last = np.max(np.where(odds < bar, np.arange(steps), -1), axis=-1)
    return np.where(last == steps-1, steps+1, last+2)


def mean_blocks(values, blocks=8):
    x = np.asarray(values, float)
    if len(x) % blocks or not np.isfinite(x).all():
        raise ValueError('finite equal-size independent blocks required')
    m = x.reshape(blocks, -1).mean(axis=1)
    se = float(m.std(ddof=1)/np.sqrt(blocks)); estimate = float(m.mean())
    width = float(student.ppf(.975, blocks-1))*se
    return dict(estimate=estimate, interval95=[estimate-width, estimate+width],
                standard_error=se, blocks=blocks, worlds=len(x), block_means=m.tolist())


def censored_blocks(values, horizon, blocks=8):
    """Whole-block percentile bootstrap; never substitute horizon for censoring."""
    x = np.asarray(values, float).reshape(blocks, -1)
    x = np.where(x > horizon, np.inf, x)
    draw = rng('fixed-block-bootstrap', blocks).integers(blocks, size=(4000, blocks))
    medians = np.median(x[draw].reshape(4000, -1), axis=1)
    ordered = np.sort(medians)
    encode = lambda v: 'unresolved' if np.isinf(v) else float(v)
    return dict(estimate=encode(np.median(x)), interval95=[encode(ordered[100]), encode(ordered[3899])],
                unresolved=mean_blocks((x.ravel() > horizon).astype(float), blocks),
                horizon=horizon, blocks=blocks, worlds=x.size,
                interval_method='4000 fixed-seed whole-block bootstrap replicates; censoring retained')


def audit_controls(checks):
    return dict(passed=all(row['passed'] for row in checks), checks=checks,
                no_registered_claim=True, miniature='architecture untested')


def conditional_blocks(values,mask,blocks=8):
    """Ratio of event sums, bootstrapping whole original blocks; empty stays absent."""
    values=np.asarray(values,float);mask=np.asarray(mask,bool)
    sums=np.where(mask,values,0).reshape(blocks,-1).sum(1);counts=mask.reshape(blocks,-1).sum(1)
    if counts.sum()==0:return dict(estimate=None,interval95=[None,None],events=0,block_counts=counts.tolist(),blocks=blocks)
    draws=rng('conditional-block-bootstrap',blocks).integers(blocks,size=(4000,blocks))
    den=counts[draws].sum(1);num=sums[draws].sum(1);sample=num[den>0]/den[den>0]
    return dict(estimate=float(sums.sum()/counts.sum()),interval95=np.quantile(sample,[.025,.975]).tolist(),events=int(counts.sum()),block_counts=counts.tolist(),blocks=blocks)
