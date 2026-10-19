"""SBC rank instrument; never declares identification or calibration by itself.

The caller must draw truth from the declared prior, data conditionally from that
truth, then 100 independent posterior draws. Fixed-truth sweeps do not satisfy
this contract. Discrete ties use an independent recorded uniform random draw.
"""
import math
import numpy as np


def rank(truth,posterior_draws,tie_uniform):
    draws=np.asarray(posterior_draws,float)
    if draws.shape!=(100,) or not np.isfinite(draws).all() or not math.isfinite(truth):
        raise ValueError('one finite truth and exactly 100 finite posterior draws required')
    if not math.isfinite(tie_uniform) or not 0<=tie_uniform<1:raise ValueError('independent tie uniform must be in [0,1)')
    lower=int(np.count_nonzero(draws<truth));ties=int(np.count_nonzero(draws==truth))
    return lower+int(tie_uniform*(ties+1))


def histogram(ranks):
    values=list(ranks)
    if not values or any(type(r) is not int or not 0<=r<=100 for r in values):
        raise ValueError('complete integer ranks in [0,100] required')
    return dict(state='DIAGNOSTIC_ONLY',independent_replicates=len(values),
        posterior_draws_per_replicate=100,counts=np.bincount(values,minlength=101).tolist(),
        expected_count_per_rank=len(values)/101,calibrated=None,
        limitation='Uniform ranks are a necessary diagnostic, not sufficient proof of a correct informative posterior.')
