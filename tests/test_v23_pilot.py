import pytest
from ghostscale.validation.soundingline.v23.pilot import known_answers,largest


@pytest.mark.parametrize('family',['R1','R2','R3','R4'])
def test_declared_repair_known_answers(family):
    assert known_answers(family)['passed']
    assert largest(family)['family']==family


def test_r2_exact_gate_ties_agree_across_arithmetic_representations():
    from ghostscale.validation.soundingline.v23 import repairs,checker
    v=largest('R2')
    for world in range(16):
        public,truth=repairs.sample(v,'cost-pilot',world,0)
        predictions,details=repairs.evaluate(v,public,truth)
        assert checker.verify_r2(v,public,truth,predictions,details)<1e-12
