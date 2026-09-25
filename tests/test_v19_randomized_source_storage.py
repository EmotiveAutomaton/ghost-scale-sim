from copy import deepcopy
from fractions import Fraction as F
from itertools import permutations, combinations
import pytest
from ghostscale.validation.soundingline.v19 import randomized_source_storage as M


def determinant(a):
    total=F()
    for p in permutations(range(len(a))):
        term=F(-1 if sum(p[i]>p[j] for i in range(len(a)) for j in range(i+1,len(a)))%2 else 1)
        for i,j in enumerate(p):term*=a[i][j]
        total+=term
    return total


def all_vertices(masses):
    # Enumerate all weight-zero/prior-active constraints in the full LP.
    n=len(masses);p=len(masses[0]);size=n+1
    constraints=[[int(i==j) for i in range(n)]+[0] for j in range(n)]
    constraints += [[row[j] for row in masses]+[-1] for j in range(p)]
    vertices={}
    for active in combinations(constraints,n):
        a=[[1]*n+[0]]+list(active);det=determinant(a)
        if not det:continue
        b=[1]+[0]*n;solution=[]
        for j in range(size):
            matrix=[list(row) for row in a]
            for i in range(size):matrix[i][j]=b[i]
            solution.append(determinant(matrix)/det)
        w,z=solution[:-1],solution[-1]
        if any(x<0 for x in w):continue
        if any(sum(w[i]*masses[i][j] for i in range(n))<z for j in range(p)):continue
        vertices[tuple(w)]=z
    best=max(vertices.values())
    return best, sorted(w for w,z in vertices.items() if z==best),len(vertices)


def example(priors=None):
    priors=priors or [[F(1),F(0),F(0)],[F(0),F(1),F(0)],[F(0),F(0),F(1)]]
    return M.fixture([1,2,3],[1,1,1],1,{'a':[1],'b':[2],'c':[3]},priors)


def test_controls(): assert all(M.controls().values())


@pytest.mark.parametrize('priors',[
    [[F(1,2),F(1,3),F(1,6)]],
    [[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)]],
    [[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)],[F(0),F(1),F(0)]],
    [[F(1,3)]*3]*3,
    [[F(1),F(0),F(0)],[F(0),F(1),F(0)],[F(0),F(0),F(1)]],
])
def test_full_constraint_enumeration(priors):
    s=example(priors);a=M.randomize(s)
    masses=[[F(*v) for v in r['masses']] for r in s['candidates']]
    best,ties,count=all_vertices(masses)
    encode=lambda w:[dict(mask=s['candidates'][i]['mask'],weight=[x.numerator,x.denominator]) for i,x in enumerate(w) if x]
    assert F(*a['minimum_expected_mass'])==best
    assert a['optimal_basic_lotteries']==[encode(w) for w in ties]
    assert a['selected_lottery']==encode(ties[-1])
    assert a['feasible_basic_count']==count


def test_three_mask_optimum():
    a=M.randomize(example())
    assert a['minimum_expected_mass']==[1,3] and len(a['selected_lottery'])==3
    assert a['worst_realized_mass']==[0,1]


def test_one_prior_and_identical_masks():
    s=M.fixture([1,2],[2,2],2,{'a':[1],'same':[1],'b':[2]},[[F(3,4),F(1,4)]])
    a=M.randomize(s)
    assert a['expected_gain']==[0,1] and a['selected_lottery']==[dict(mask=1,weight=[1,1])]


def test_prior_permutation():
    s=example([[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)],[F(0),F(1),F(0)]])
    a=M.randomize(s);t=deepcopy(s)
    for r in t['candidates']:r['masses'].reverse()
    b=M.randomize(t);b['expected_masses'].reverse()
    assert a==b


def test_item_and_identity_permutation():
    s=example();a=M.randomize(s);t=deepcopy(s)
    t['times'].reverse();t['costs'].reverse();t['candidates'].reverse()
    for row in t['candidates']:row['identities']=['arbitrary'];row['selected_times'].reverse()
    assert M.randomize(t)==a


def test_expected_and_realized_bytes_distinct():
    s=M.fixture([1,2],[1,3],3,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)]])
    a=M.randomize(s)
    assert a['expected_used_bytes']==[2,1] and a['maximum_realized_bytes']==3


@pytest.mark.parametrize('corruption',['cost','mask','duplicate','mass','baseline','item','empty'])
def test_corrupt_inputs_rejected(corruption):
    s=example()
    if corruption=='cost':s['candidates'][0]['used_bytes']=2
    if corruption=='mask':s['candidates'][0]['mask']=0
    if corruption=='duplicate':s['candidates'].append(deepcopy(s['candidates'][0]))
    if corruption=='mass':s['candidates'][0]['masses'][0]=[-1,1]
    if corruption=='baseline':s['minimum_mass']=[99,1]
    if corruption=='item':s['candidates'][0]['selected_times']=[99]
    if corruption=='empty':s['candidates']=[]
    with pytest.raises(ValueError):M.randomize(s)


def test_average_only_budget_not_allowed():
    s=M.fixture([1,2],[1,3],3,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)]])
    s['capacity_bytes']=2
    with pytest.raises(ValueError,match='realized byte'):M.randomize(s)
