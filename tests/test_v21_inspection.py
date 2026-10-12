import numpy as np
from ghostscale.validation.soundingline.v21 import inspection as I, world as W
from ghostscale.validation.soundingline.v21.inspection_check import reference


def test_inspection_known_answer_controls():
    assert all(I.controls().values())


def test_all_channel_boundaries_against_row_enumeration():
    rows=W.enumerate_world(71000811);mass=I.joint(rows)
    for mechanism in ('independent','inspection-dependent'):
        for rate,false in ((0.,0.),(.5,.2),(.75,.05)):
            for reader in ('endpoint','naive-negative','missingness-ignorant','observation-aware','matched-direct','full-record-oracle'):
                actual,_=I.table(mass,rate,false,mechanism,reader)
                assert np.allclose(actual,reference(rows,rate,false,mechanism,reader),atol=1e-12)
