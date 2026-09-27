from copy import deepcopy
from itertools import permutations
import pytest
from ghostscale.validation.soundingline.v19 import asymmetric_channel as P
from ghostscale.validation.soundingline.v19 import asymmetric_channel_review as R
from fractions import Fraction as F

def fixture():
    return P.fixture([1,2],[1,1],1,{"a":[1],"b":[2]},[[F(3,4),F(1,4)],[F(3,8),F(5,8)],[F(1,2)]*2])

@pytest.mark.parametrize('counts',R.MIXTURES)
def test_independent_full_record(counts):
    assert R.assess(fixture(),counts)==P.solve(fixture(),counts)

def test_known_controls():assert all(R.controls().values())

def test_corrupt_capacity_rejected():
    s=fixture();s['candidates'][0]['used_bytes']=2
    with pytest.raises(AssertionError):R.assess(s,(3,1,0))

def test_relabel_complete_tie_set():
    s=fixture();counts=(3,1,0);base=R.assess(s,counts)
    for order in permutations(range(3)):
        t=deepcopy(s)
        for row in t['candidates']:row['masses']=[row['masses'][j] for j in order]
        a=R.assess(t,tuple(counts[j] for j in order))
        assert a['worst_regret']==base['worst_regret']
        assert sorted(a['optimal_policies'])==sorted([p[j] for j in order] for p in base['optimal_policies'])
