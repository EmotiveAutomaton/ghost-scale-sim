"""Substantive known-answer checks, not snapshots of implementation outputs."""
import pytest
from ghostscale.validation.soundingline.v27 import study

@pytest.mark.parametrize('family',['K1','K2','K3','K4','K5','K6','K7'])
def test_algebraic_and_independent_controls(family):
    assert study.module(family).controls()['passed']
