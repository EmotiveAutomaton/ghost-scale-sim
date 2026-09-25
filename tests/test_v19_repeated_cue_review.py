from fractions import Fraction as F
from itertools import product
import pytest
from ghostscale.validation.soundingline.v19 import repeated_cue_review as R, repeated_cue as P

@pytest.fixture
def library():
    return P.fixture([1,2],[1,2],2,{'a':[1],'b':[2]},
                     [[F(1),F(0)],[F(0),F(1)],[F(1,2)]*2])

@pytest.mark.parametrize('counts',[(2,2,0),(1,1,2),(4,0,0)])
@pytest.mark.parametrize('reliability',R.RELIABILITIES)
@pytest.mark.parametrize('dependence',R.DEPENDENCES)
def test_elementary_integration(library,counts,reliability,dependence):
    a=R.solve(library,counts,reliability,dependence)
    assert a==P.solve(library,counts,reliability,dependence)
    assert sum(F(*p['probability']) for p in a['pairs'])==1
    if dependence==(1,1): assert a['incremental_value']==[0,1]
    if dependence==(0,1): assert a['independent_regret']==[0,1]

def test_known_controls(): assert all(R.controls().values())

def test_corrupt_capacity(library):
    library['candidates'][0]['used_bytes']=4
    with pytest.raises(AssertionError):R.solve(library,(2,2,0),(2,3),(0,1))

def test_exhaustive_whole_policy(library):
    # Two masks yield 512 complete policies across all nine label pairs.
    library['candidates']=library['candidates'][:2]
    answer=R.solve(library,(1,1,2),(2,3),(1,2))
    scores=[{r['mask']:F(*r['mass']) for r in p['joint_mask_scores']} for p in answer['pairs']]
    exhaustive=max(sum(s[m] for s,m in zip(scores,policy)) for policy in product((1,2),repeat=9))
    assert exhaustive==F(*answer['retained_mass'])
