from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import realized_risk_storage as P
from ghostscale.validation.soundingline.v19 import realized_risk_review as R


def test_controls():
    assert all(R.controls().values())


@pytest.mark.parametrize('priors', [
    [[F(1,2),F(1,3),F(1,6)]],
    [[F(7,10),F(1,10),F(1,5)],[F(1,10),F(7,10),F(1,5)]],
    [[F(1,3)]*3]*3,
    [[F(i==j) for j in range(3)] for i in range(3)],
])
def test_independent_complete_frontier(priors):
    s=P.fixture([1,2,3],[1,1,1],1,{'a':[1],'b':[2],'c':[3]},priors)
    for f in (F(0),F(1,2),F(1)):
        assert R.risk(s,f)==P.constrain(s,f)


@pytest.mark.parametrize('kind',['bytes','baseline','floor'])
def test_corruption_rejected(kind):
    s=deepcopy(P.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,2)]*2]))
    f=F(1)
    if kind=='bytes':s['candidates'][0]['used_bytes']=99
    if kind=='baseline':s['minimum_mass']=[0,1]
    if kind=='floor':f=F(1,3)
    with pytest.raises(AssertionError):R.risk(s,f)


def test_empty_full_identity():
    for capacity,selected in [(0,[]),(2,[1,2])]:
        s=P.fixture([1,2],[1,1],capacity,{'only':selected},[[F(1,2)]*2]*3)
        for f in (F(0),F(1,2),F(1)):
            assert R.risk(s,f)==P.constrain(s,f)
