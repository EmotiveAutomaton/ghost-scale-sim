from decimal import Decimal, localcontext
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import adaptive, adaptive_numerics as repair


def test_scaled_precision_matches_normal_domain_and_retains_tiny_variance():
    rows = [[dict(mean=.1, se=.02), dict(mean=.2, se=.03)]]
    np.testing.assert_allclose(repair.combine(rows), adaptive.combine(rows), atol=1e-15, rtol=0)
    mean, se = repair.combine([[dict(mean=2e-158, se=1e-158), dict(mean=4e-158, se=2e-158)]])
    assert mean[0]/1e-158 == pytest.approx(2.4)
    assert se[0]/1e-158 == pytest.approx(1/np.sqrt(1.25))
    with pytest.raises(ValueError):repair.combine([[dict(mean=0, se=0)]])


def test_centered_posterior_preserves_frozen_model_and_extreme_scale():
    rng = np.random.default_rng(120)
    for count in (1, 27, 81):
        means, ses = rng.normal(.02, .03, count), rng.uniform(.005, .03, count)
        for nodes in (121, 241):
            a = adaptive.posterior(means, ses, nodes=nodes)
            b = repair.posterior(means, ses, nodes=nodes)
            np.testing.assert_allclose(a['mean'], b['mean'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(a['probability_above_margin'], b['probability_above_margin'], atol=1e-12, rtol=0)
    actual = repair.posterior([1e-158, .1], [1e-158, .01])
    assert np.isfinite(actual['mean']).all() and np.isfinite(actual['probability_above_margin']).all()


def test_raw_score_matches_high_precision_without_flooring():
    p = [[1., 1e-160], [.75, .25], [0., 1.]]
    with localcontext() as context:
        context.prec = 200
        expected = float((Decimal(1)+Decimal('1e-160')).ln())
    scores = repair.stable_log_loss(p, [0, 0, 0])
    assert scores[0] == pytest.approx(expected, rel=1e-14, abs=0)
    assert scores[1] == pytest.approx(-np.log(.75)) and np.isinf(scores[2])
    np.testing.assert_array_equal(scores, repair.independent_log_loss(p, [0, 0, 0]))


def test_bootstrap_variance_preserves_resolvable_subnormal_squared_scale():
    ordinary=repair.interval([1.,2.,4.],'scale-control')
    tiny=repair.interval([1e-170,2e-170,4e-170],'scale-control')
    assert tiny['se']>0 and tiny['se']/1e-170==pytest.approx(ordinary['se'],rel=1e-14)
    assert repair.interval([1e-170]*3,'constant')['se']==0
