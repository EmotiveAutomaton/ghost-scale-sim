import itertools
import numpy as np
from ghostscale.validation.soundingline.v21.selection import controls,posterior,execute,effects
from ghostscale.validation.soundingline.v21.selection_check import posterior as reference,effects as effect_reference

def test_known_answers():
    assert all(controls().values())

def test_all_two_unit_plans_and_selection_records():
    for bits in itertools.product((0,1),repeat=4):
        plans=np.array(bits).reshape(2,2)
        for c,edits in itertools.product((0,1),(0,1,4,16)):
            a,_=execute(plans,c,edits)
            assert effects(plans,c,edits).tolist()==effect_reference(plans,c,edits)
            for record in (None,c):assert np.allclose(posterior(plans,a,.3,record),reference(plans,a,.3,record),atol=1e-12)

def test_edits_do_not_identify_an_aliased_choice():
    plans=np.ones((2,8));a,_=execute(plans,1,16)
    assert np.array_equal(posterior(plans,a,.25,None),[.75,.25])
    assert np.array_equal(posterior(plans,a,.25,1),[0,1])
    assert not effects(plans,1,16).any()
