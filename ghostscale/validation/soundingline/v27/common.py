"""Version-specific seeds with the validated whole-world interval contract."""
import hashlib
import numpy as np
from ..v26.common import canonical, read, write, digest, close
from ..v26.day_common import (posteriors, margin, sustained, scores, measure,
    summarize, bar_verdict, choices, draw_menu, class_prior, marginals)


def rng(*parts):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(canonical(['v27',*parts])).digest()[:8],'little'))


def bernoulli(p, observed):
    p=np.asarray(p,dtype=float)
    with np.errstate(divide='ignore'):
        return np.log(p if observed else 1-p)


def true_logscore(p, truth):
    return float(np.log(p[truth]))


def verify_complete(packet):
    complete=read(packet/'COMPLETE.json')
    for key,name in [('plan_sha256','PLAN.json'),('summary_sha256','original/SUMMARY.json'),
                     ('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json')]:
        if digest(packet/name)!=complete[key]:raise ValueError('completed packet changed: '+name)
    if not read(packet/'INDEPENDENT_REVIEW.json')['state']=='INDEPENDENT_RECONSTRUCTION_PASSED':raise ValueError('review incomplete')
