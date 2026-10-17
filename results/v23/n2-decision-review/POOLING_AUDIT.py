"""Reproduce the retained-data N2 pooling diagnostic; no sampling or card changes.

Run with the repository scientific interpreter. Inputs are adjacent reviewed
diagnostics. This integrates the frozen Gaussian model/prior pair on nonuniform
tau grids, with trapezoid widths. It does not define a replacement stopping rule.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.special import logsumexp, ndtr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from ghostscale.validation.soundingline.v23.adaptive_numerics import combine


def audit(directory):
    diagnostics = json.loads((directory / 'DIAGNOSTICS.json').read_text(encoding='utf-8'))
    estimates = [[row['summary']['primary'] for row in batches]
                 for identity, batches in sorted(diagnostics.items())
                 if not batches[0]['summary']['structural_zero']]
    means, ses = map(np.asarray, combine(estimates))
    # Fixed chosen policy: CHOSEN_BEFORE_VALIDATION.json, documented calibration.
    mu_scale, tau_scale = .08, .05

    def integrate(nodes):
        tau = np.r_[0., np.geomspace(float(ses.min()) * 1e-6, max(8 * tau_scale, .16), nodes)]
        sd = np.hypot(tau[:, None], ses[None, :])
        scale = np.minimum(sd.min(1), mu_scale)
        precision = (scale[:, None] / sd) ** 2
        denominator = (scale / mu_scale) ** 2 + precision.sum(1)
        mu_sd = scale / np.sqrt(denominator)
        mu = (precision * means).sum(1) / denominator
        quadratic = (((means - mu[:, None]) / sd) ** 2).sum(1) + (mu / mu_scale) ** 2
        log_density = -(np.log(sd).sum(1) + math.log(mu_scale) - np.log(mu_sd)
                        + .5 * quadratic + .5 * (tau / tau_scale) ** 2)
        widths = np.r_[(tau[1] - tau[0]) / 2, (tau[2:] - tau[:-2]) / 2,
                       (tau[-1] - tau[-2]) / 2]
        log_weights = log_density + np.log(widths)
        weights = np.exp(log_weights - logsumexp(log_weights))
        shrink = (tau[:, None] / sd) ** 2
        conditional_mean = shrink * means + (1 - shrink) * mu[:, None]
        conditional_sd = np.hypot(np.sqrt(shrink) * ses, (1 - shrink) * mu_sd[:, None])
        tails = weights @ ndtr((conditional_mean - .02) / conditional_sd)
        return dict(nodes=len(tau), max_probability_above_margin=float(tails.max()),
                    max_posterior_mean=float((weights @ conditional_mean).max()),
                    tau_mass_below_1e_100=float(weights[tau < 1e-100].sum()),
                    zero_endpoint_mass=float(weights[0]))

    result = [integrate(n) for n in (4097, 8193)]
    recorded = json.loads((directory / 'N2_STOPPING_DIAGNOSTIC.json').read_text(encoding='utf-8'))
    for actual, expected in zip(result, recorded['same_model_nonuniform_quadrature'], strict=True):
        assert actual['nodes'] == expected['nodes']
        for key in actual:
            if key != 'nodes':
                assert math.isclose(actual[key], expected[key], rel_tol=1e-9, abs_tol=0), key
    return result


if __name__ == '__main__':
    print(json.dumps(audit(Path(__file__).resolve().parent), indent=2))
