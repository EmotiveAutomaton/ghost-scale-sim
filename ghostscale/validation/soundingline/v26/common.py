"""Separate seed domain; inherited validated scoring primitives."""
import hashlib
import numpy as np
from ..v25.common import canonical,read,write,digest,softmax,posterior,scores,information,close,record,simplex

def rng(*parts):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(canonical(['v26',*parts])).digest()[:8],'little'))

def marginal(p,ids,n):return np.bincount(ids,weights=p,minlength=n)

def logscore(p,truth,prior=None):return scores(p,truth,prior)['log_gain']
