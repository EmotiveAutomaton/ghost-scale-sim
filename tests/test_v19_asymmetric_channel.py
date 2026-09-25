from fractions import Fraction as F
from itertools import product, permutations
from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v19 import asymmetric_channel as M
from ghostscale.validation.soundingline.v19 import robust_channel_regret as S

def fixture():
    return M.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(3,4),F(1,4)],[F(3,8),F(5,8)],[F(1,2)]*2])

def expected(s,counts,policy,r,b):
    mass={x['mask']:[F(*v) for v in x['masses']] for x in s['candidates']}
    total=F(0)
    for j in range(3):
        total+=F(counts[j],4)*(r*mass[policy[j]][j]+(1-r)*b*mass[policy[(j+1)%3]][j]+(1-r)*(1-b)*mass[policy[(j+2)%3]][j])
    return total

@pytest.mark.parametrize('counts',M.MIXTURES)
def test_exact_rectangle_and_dense_interior(counts):
    s=fixture();a=M.solve(s,counts);policies=[tuple(p['policy']) for p in a['policies']]
    worst={p:F(0) for p in policies}
    for r,b in product((F(1,3),F(1,2),F(2,3),F(5,6),F(1)),(F(0),F(1,4),F(1,2),F(3,4),F(1))):
        values={p:expected(s,counts,p,r,b) for p in policies};opt=max(values.values())
        for p in policies:worst[p]=max(worst[p],opt-values[p])
    assert {tuple(x['policy']):F(*x['worst_regret']) for x in a['policies']}==worst
    assert a['optimal_policies']==[list(p) for p in policies if worst[p]==min(worst.values())]
    for row in a['policies']:
        for i,(r,b) in enumerate(M.VERTICES):assert F(*row['vertex_scores'][i])==expected(s,counts,row['policy'],F(*r),F(*b))

@pytest.mark.parametrize('counts',M.MIXTURES)
def test_symmetric_restriction(counts):
    s=fixture();a=M.solve(s,counts,(((1,3),(1,2)),((1,1),(1,2))))
    b=S.solve(s,counts)
    for name in ('selected_policy','optimal_policies','worst_regret'):assert a[name]==b[name]

def test_controls():assert all(M.controls().values())

def test_direction_reversal():
    s=fixture();p=(1,2,1)
    fixed=expected(s,(3,1,0),(1,1,1),F(1,3),F(0))
    assert expected(s,(3,1,0),p,F(1,3),F(0))>fixed
    assert expected(s,(3,1,0),p,F(1,3),F(1))<fixed

def test_corrupt_capacity():
    s=fixture();s['candidates'][0]['used_bytes']=2
    with pytest.raises(ValueError):M.solve(s,(3,1,0))

def test_all_source_label_permutations():
    s=fixture();counts=(3,1,0);base=M.solve(s,counts)
    for order in permutations(range(3)):
        t=deepcopy(s)
        for row in t['candidates']:row['masses']=[row['masses'][j] for j in order]
        a=M.solve(t,tuple(counts[j] for j in order))
        assert a['worst_regret']==base['worst_regret']
        expected_policies=sorted([p[j] for j in order] for p in base['optimal_policies'])
        assert sorted(a['optimal_policies'])==expected_policies
