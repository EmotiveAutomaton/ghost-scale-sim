from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import randomized_source_storage as P
from ghostscale.validation.soundingline.v19 import randomized_storage_review as R


def fixture(priors):
    return P.fixture([1,2,3],[1,2,3],3,{'a':[1],'b':[2],'c':[3],'d':[1,2]},priors)


def test_controls():
    assert all(R.controls().values())


@pytest.mark.parametrize('priors',[
    [[F(1,2),F(1,3),F(1,6)]],
    [[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)]],
    [[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)],[F(0),F(1),F(0)]],
    [[F(1,3)]*3]*3,
    [[F(1),F(0),F(0)],[F(0),F(1),F(0)],[F(0),F(0),F(1)]],
])
def test_independent_complete_vertices(priors):
    s=fixture(priors)
    assert R.lottery(s)==P.randomize(s)


def test_masks_bytes_and_baseline_cannot_be_corrupted():
    s=fixture([[F(1,3)]*3])
    for key,value in [('used_bytes',99)]:
        t=deepcopy(s);t['candidates'][0][key]=value
        with pytest.raises(AssertionError):R.lottery(t)
    t=deepcopy(s);t['minimum_mass']=[0,1]
    with pytest.raises(AssertionError):R.lottery(t)


def test_tied_vertices_do_not_claim_all_mixtures():
    value,ties,count=R.vertices([[F(1,2),F(1,2)]]*3)
    assert value==F(1,2) and len(ties)==count==3
    assert (F(1,3),)*3 not in ties


def test_prior_permutation():
    masses=[[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)],[F(0),F(1),F(0)]]
    assert R.vertices(masses)==R.vertices([list(reversed(row)) for row in masses])


def test_empty_and_full():
    for mass in (F(0),F(1)):
        assert R.vertices([[mass]*3])==(mass,[(F(1),)],1)
