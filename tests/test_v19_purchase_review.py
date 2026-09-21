"""Known-answer checks also run inside the sole queued verification worker."""
import pytest
from ghostscale.validation.soundingline.v19 import purchase_review as R


def test_independent_controls():
    assert all(R.controls().values())


def test_corrupt_forecast_rejected():
    with pytest.raises(ValueError, match='differs'):
        R.close([.3, .7], [.5, .5])


def test_finite_threshold_semantics():
    assert R.crossing([2.] * 24, 2.) == 33
    assert R.crossing([2.] * 23 + [3.], 2.) == 31
