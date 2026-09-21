import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import portfolio_review as R


def test_independent_known_answers_and_nulls():
    assert all(R.controls().values())


def test_centered_ridge_satisfies_penalized_normal_equations():
    x=np.array([[0.,0.],[1.,1.],[1.,0.],[0.,1.]])
    y=np.array([[.3,.7],[.2,.8],[.8,.2],[.4,.6]])
    h=R.ridge(x,y,.01);design=np.column_stack((np.ones(4),x))
    penalty=np.diag([0.,.01,.01])
    np.testing.assert_allclose(design.T@(design@h-y)+penalty@h,0,atol=1e-13)


def test_invalid_or_corrupted_verification_values_fail():
    with pytest.raises(ValueError):R.close([0.],[float('nan')])
    with pytest.raises(ValueError):R.close([[0.]], [0.])
    with pytest.raises(ValueError):R.close([.5,.5],[.6,.4])
