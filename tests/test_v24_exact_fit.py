"""Known-answer arithmetic tests; engineering fixtures, no scientific worlds."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp, softmax

from ghostscale.validation.soundingline.v24.exact_fit import grouped_objective
from runners.check_v23_backbone_stationarity import decimal_objective


def scalar(rows, center, offset, penalty):
    loss = penalty * (offset @ offset)
    g = 2 * penalty * offset.copy()
    h = 2 * penalty * np.eye(len(center))
    for row in rows:
        x = row.features
        logits = x @ (center + offset)
        p = softmax(logits)
        mean = p @ x
        loss += row.weight * (logsumexp(logits) - logits[row.taken])
        g += row.weight * (mean - x[row.taken])
        centered = x - mean
        h += row.weight * (centered.T * p) @ centered
    return loss, g, h


@pytest.mark.parametrize("dimension", [2, 9, 164])
def test_grouping_preserves_ragged_choices_weights_and_derivatives(dimension):
    random = np.random.default_rng(2420 + dimension)
    rows = [SimpleNamespace(features=random.normal(size=(k, dimension)),
                            taken=i % k, weight=(i % 4) / 3)
            for i, k in enumerate([1, 2, 4, 3, 4, 2, 1, 3])]
    center = random.normal(size=dimension)
    offset = random.normal(size=dimension)
    objective, hessian = grouped_objective(rows, center, .1)
    expected, g, h = scalar(rows, center, offset, .1)
    loss, gradient = objective(offset)
    assert loss == pytest.approx(expected, abs=2e-12, rel=0)
    np.testing.assert_allclose(gradient, g, rtol=0, atol=2e-12)
    np.testing.assert_allclose(hessian(offset), h, rtol=0, atol=2e-12)


def test_decimal_known_answer_and_rank_deficiency():
    rows = [SimpleNamespace(features=np.array([[0., 0.], [1., 1.]]),
                            taken=y, weight=1.) for y in [0, 1]]
    objective, hessian = grouped_objective(rows, np.zeros(2), .5)
    loss, gradient = objective(np.zeros(2))
    exact = decimal_objective([r.features for r in rows], [0, 1], [1., 1.],
                              [0., 0.], [0., 0.], .5)
    assert loss == pytest.approx(float(exact[0]), abs=1e-14)
    np.testing.assert_array_equal(gradient, np.zeros(2))
    np.testing.assert_allclose(hessian(np.zeros(2)), np.array(exact[2], dtype=float), atol=1e-14)
    assert np.linalg.eigvalsh(hessian(np.zeros(2))).min() >= 1.


def test_zero_weights_extreme_logits_and_option_permutation():
    x = np.array([[1000., -1000.], [-1000., 1000.], [2., 3.]])
    rows = [SimpleNamespace(features=x, taken=1, weight=0.),
            SimpleNamespace(features=x, taken=2, weight=1.)]
    center = np.array([1., 0.])
    offset = np.array([.1, -.2])
    a, ah = grouped_objective(rows, center, 2.)
    perm = [2, 0, 1]
    reordered = [SimpleNamespace(features=r.features[perm], taken=perm.index(r.taken),
                                 weight=r.weight) for r in rows]
    b, bh = grouped_objective(reordered, center, 2.)
    assert np.isfinite(a(offset)[0])
    np.testing.assert_allclose(a(offset)[1], b(offset)[1], atol=1e-12)
    np.testing.assert_allclose(ah(offset), bh(offset), atol=1e-12)
    assert a(offset)[0] == b(offset)[0]


def test_invalid_inputs_fail_and_cost_guard_is_called():
    calls = []
    f, h = grouped_objective([], np.zeros(2), 1., lambda: calls.append(1))
    assert f(np.ones(2))[0] == 2.
    h(np.zeros(2))
    assert len(calls) == 2
    with pytest.raises(ValueError):
        grouped_objective([], np.zeros(2), 0)
    with pytest.raises(ValueError):
        grouped_objective([SimpleNamespace(features=np.ones((2, 3)), taken=0, weight=1)], np.zeros(2), 1)
