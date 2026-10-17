"""Known-answer checks for the independent diagnosis, not solver acceptance."""
from decimal import Decimal
import numpy as np
from runners.check_v23_backbone_stationarity import decimal_objective


def test_balanced_binary_choices_have_known_stationary_curvature():
    rows = [np.array([[0.], [1.]]) for _ in range(2)]
    loss, gradient, hessian = decimal_objective(rows, [0, 1], [1., 1.], [0.], [0.], .5)
    assert abs(float(loss) - 2 * np.log(2)) < 1e-14
    assert gradient == [Decimal(0)]
    assert hessian == [[Decimal('1.5')]]


def test_zero_weight_rows_leave_only_ridge_with_rank_deficient_features():
    rows = [np.array([[1., 1.], [1., 1.]])]
    loss, gradient, hessian = decimal_objective(rows, [1], [0.], [3., 3.], [2., -1.], 2.)
    assert loss == 10
    assert gradient == [8, -4]
    assert hessian == [[4, 0], [0, 4]]


def test_gradient_and_curvature_match_a_nonstationary_binary_known_answer():
    rows = [np.array([[0.], [1.]])]
    loss, gradient, hessian = decimal_objective(rows, [1], [.5], [0.], [0.], 1.)
    assert abs(float(loss) - .5 * np.log(2)) < 1e-14
    assert gradient == [Decimal('-.25')]
    assert hessian == [[Decimal('2.125')]]
