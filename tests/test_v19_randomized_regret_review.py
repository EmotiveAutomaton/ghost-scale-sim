from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import randomized_source_regret as P
from ghostscale.validation.soundingline.v19 import randomized_regret_review as R


def test_controls():
    assert all(R.controls().values())


@pytest.mark.parametrize('priors',[
    [[F(1,2),F(1,3),F(1,6)]],
    [[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)]],
    [[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)],[F(0),F(1),F(0)]],
    [[F(1,3)]*3]*3,
    [[F(1),F(0),F(0)],[F(0),F(1),F(0)],[F(0),F(0),F(1)]],
])
def test_complete_independent_lottery(priors):
    s=P.fixture([1,2,3],[1,2,3],3,{'a':[1],'b':[2],'c':[3],'d':[1,2]},priors)
    assert R.lottery(s)==P.minimize(s)


@pytest.mark.parametrize('kind',['optimum','regret','bytes','baseline'])
def test_corrupt_parent_rejected(kind):
    s=deepcopy(P.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,3),F(2,3)]]))
    if kind=='optimum':s['best_prior_mass'][0]=[0,1]
    if kind=='regret':s['candidates'][0]['prior_regrets'][0]=[99,1]
    if kind=='bytes':s['candidates'][0]['used_bytes']=99
    if kind=='baseline':s['minimum_mass']=[0,1]
    with pytest.raises(AssertionError):R.lottery(s)


def test_empty_full_and_all_ties():
    for capacity,selected in [(0,[]),(2,[1,2])]:
        s=P.fixture([1,2],[1,1],capacity,{'only':selected},[[F(1,2)]*2]*3)
        assert R.lottery(s)==P.minimize(s)
    s=P.fixture([1,2,3],[1,1,1],1,{'a':[1],'b':[2],'c':[3]},[[F(1,3)]*3]*3)
    assert R.lottery(s)==P.minimize(s)
    assert R.lottery(s)['optimal_basic_count']==3
