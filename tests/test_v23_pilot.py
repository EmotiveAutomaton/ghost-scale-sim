import pytest
from ghostscale.validation.soundingline.v23.pilot import known_answers,largest


@pytest.mark.parametrize('family',['R1','R2','R3','R4'])
def test_declared_repair_known_answers(family):
    assert known_answers(family)['passed']
    assert largest(family)['family']==family
