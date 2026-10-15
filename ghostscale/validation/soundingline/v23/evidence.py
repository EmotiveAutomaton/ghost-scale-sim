"""Uncapped, support-certified betting and fixed-order e-LOND.

Independent balanced blocks, not individual stratified worlds, are the betting
units. Conditional on frozen discovery/training, each block averages all sixteen
laws with new independent randomness. Thus its mean is the frozen mixture mean.
Under H0 E[X_block | past] <= margin and X_block >= L, every nonnegative
predictable factor has conditional expectation <= 1. No lower truncation occurs.
"""
import math
import numpy as np


def betting_evalue(world_gains, lower_bound, margin=.02, laws_per_block=16,
                  independent_balanced_blocks=False):
    if not independent_balanced_blocks:
        raise ValueError('an audited independent balanced-block sampler is required')
    values = np.asarray(world_gains, float)
    if values.ndim != 1 or len(values) % laws_per_block or not len(values):
        raise ValueError('complete balanced law blocks required')
    if not np.isfinite(lower_bound) or lower_bound >= margin:
        raise ValueError('finite certified lower bound below the margin required')
    if not np.isfinite(values).all():
        return dict(state='CONFIRMATION_UNAVAILABLE', reason='genuine nonfinite raw gain',
                    log_e=None, worlds=len(values))
    if np.any(values < lower_bound-1e-12):
        raise ValueError('certified support violation; preserve raw output and halt')
    blocks = values.reshape(-1, laws_per_block).mean(1)
    # Constant bets are predictable. Never estimate a support bound from gains.
    fraction = .5/(margin-lower_bound)
    factors = 1+fraction*(blocks-margin)
    if np.any(factors <= 0):
        raise ValueError('invalid betting factor')
    log_e = float(np.log(factors).sum())
    return dict(state='VALID_EVALUE', log_e=log_e,
                e_value=math.exp(log_e) if log_e < 700 else None,
                log_representation_authoritative=True, margin=margin,
                certified_lower_bound=float(lower_bound), betting_fraction=fraction,
                worlds=len(values), independent_blocks=len(blocks),
                block_means=blocks.tolist(), factors=factors.tolist(), truncated=0)


def elond(log_evalues, alpha=.1):
    if not 0 < alpha < 1:
        raise ValueError('invalid false discovery target')
    discoveries, out = 0, []
    for j, log_e in enumerate(log_evalues, 1):
        gamma = 1/(j*(j+1))
        threshold = -math.log(alpha*gamma*(discoveries+1))
        if log_e is not None and not math.isfinite(log_e):
            raise ValueError('finite log e-value or unavailable required')
        accepted = log_e is not None and log_e >= threshold
        out.append(dict(order=j, earlier_discoveries=discoveries, gamma=gamma,
                        log_threshold=threshold, log_e=log_e, confirmed=accepted))
        discoveries += int(accepted)
    return out


def certified_log_gain_bound(minimum_mechanism_probability):
    """log(p_mechanism/p_rival) >= log(p_mechanism_min), since p_rival <= 1.

    The caller must supply a proof-backed bound over every allowed input, not a
    minimum measured on development data. This helper only checks arithmetic.
    """
    p = float(minimum_mechanism_probability)
    if not 0 < p <= 1:
        raise ValueError('strictly positive certified probability required')
    return math.log(p)
