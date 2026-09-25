from fractions import Fraction as F
import itertools
import pytest
from ghostscale.validation.soundingline.v19.robust_source_mass import choose, from_parent, controls
from ghostscale.validation.soundingline.v19.source_mass_frontier import allocate, PRIORS


def test_controls(): assert all(controls().values())


@pytest.mark.parametrize('capacity', [0,1,3,5,8])
def test_exhaustive_library_and_convex_mixtures(capacity):
    times=[1,2,3,4];costs=[1,2,2,3]
    masks=[m for m in itertools.product((0,1),repeat=4) if sum(c*x for c,x in zip(costs,m))<=capacity]
    candidates={str(i):[t for t,x in zip(times,m) if x] for i,m in enumerate(masks)}
    priors=[[F(1,4)]*4,[F(1,10),F(2,10),F(3,10),F(4,10)],[F(4,10),F(3,10),F(2,10),F(1,10)]]
    a=choose(times,costs,capacity,candidates,priors)
    scores={}
    for selected in candidates.values():
        values=[sum((p[t-1] for t in selected),F(0)) for p in priors]
        for i,j in itertools.combinations(range(3),2):
            assert min((F(k,8)*values[i]+(1-F(k,8))*values[j] for k in range(9))) == min(values[i],values[j])
        scores[sum(1<<(t-1) for t in selected)]=min(values)
    expected=max(scores.values())
    assert F(*a['minimum_mass'])==expected
    assert a['optimal_masks']==sorted(m for m,s in scores.items() if s==expected)
    assert a['selected_mask']==max(a['optimal_masks'])


def test_prior_and_item_permutation():
    times=[1,2,3];costs=[1,2,2];ps=[[F(1,2),F(1,3),F(1,6)],[F(1,6),F(1,3),F(1,2)]]
    cs={'a':[1,2],'b':[1,3],'duplicate':[1,2]}
    a=choose(times,costs,3,cs,ps)
    b=choose(times[::-1],costs[::-1],3,cs,[p[::-1] for p in ps[::-1]])
    assert a['selected_mask']==b['selected_mask'] and a['minimum_mass']==b['minimum_mass']
    assert len(a['candidates'])==2 and sum(len(r['identities']) for r in a['candidates'])==3
    assert [r['masses'] for r in a['candidates']]==[r['masses'][::-1] for r in b['candidates']]


def test_equal_prior_recent_tie_and_all_fit():
    a=choose([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,2)]*2]*3)
    assert a['selected_mask']==2 and a['optimal_masks']==[1,2]
    assert choose([1,2],[1,1],2,{'all':[1,2]},[[F(1,2)]*2])['minimum_mass']==[1,1]


@pytest.mark.parametrize('bad', [[F(1,3)]*2,[F(-1),F(2)],[F(1)]])
def test_corrupt_prior(bad):
    with pytest.raises(ValueError):choose([1,2],[1,1],1,{'a':[1]},[bad])


def test_corrupt_bytes():
    with pytest.raises(ValueError):choose([1,2],[2,1],1,{'a':[1]},[[F(1,2)]*2])


def test_parent_library():
    rs=[dict(times=[1,2,3],costs=[1,2,3],capacity_bytes=3,prior=p,result=allocate((1,2,3),(1,2,3),3,p)) for p in PRIORS]
    a=from_parent(rs)
    assert sum(len(r['identities']) for r in a['candidates'])==12
    assert all(r['used_bytes']<=3 for r in a['candidates'])
    rs[0]['result']['optimal']['mass_numerator']+=1
    with pytest.raises(ValueError):from_parent(rs)
