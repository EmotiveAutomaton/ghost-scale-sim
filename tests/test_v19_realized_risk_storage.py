from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import realized_risk_storage as M
from ghostscale.validation.soundingline.v19.randomized_storage_review import lottery


def fixture(priors=None):
    return M.fixture([1,2,3],[1,2,3],3,{'a':[1],'b':[2],'c':[3],'d':[1,2]},priors or [[F(1,3)]*3]*3)


def test_controls():assert all(M.controls().values())


@pytest.mark.parametrize('priors',[
    [[F(1,2),F(1,3),F(1,6)]],
    [[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)]],
    [[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)],[F(0),F(1),F(0)]],
    [[F(1,3)]*3]*3,
    [[F(1),F(0),F(0)],[F(0),F(1),F(0)],[F(0),F(0),F(1)]],
])
def test_independent_risk_frontier(priors):
    s=fixture(priors);baseline=F(*s['minimum_mass']);sets=[];values=[]
    for f in (F(0),F(1,2),F(1)):
        expected=[r for r in s['candidates'] if all(F(*x)>=f*baseline for x in r['masses'])]
        result=M.constrain(s,f)
        assert result['lottery']==lottery(dict(s,candidates=expected))
        assert result['surviving_masks']==sorted(r['mask'] for r in expected)
        assert F(*result['lottery']['worst_realized_mass'])>=f*baseline
        assert F(*result['lottery']['minimum_expected_mass'])>=baseline
        sets.append(set(result['surviving_masks']));values.append(F(*result['lottery']['minimum_expected_mass']))
    assert sets[2]<=sets[1]<=sets[0] and values[2]<=values[1]<=values[0]
    assert M.constrain(s,F(0))['lottery']==M.randomize(s)


@pytest.mark.parametrize('bad',[-1,F(1,3),2])
def test_invalid_floor(bad):
    with pytest.raises(ValueError):M.constrain(fixture(),bad)


@pytest.mark.parametrize('kind',['cost','mask','budget','baseline'])
def test_corrupted_parent(kind):
    s=fixture()
    if kind=='cost':s['candidates'][0]['used_bytes']=99
    if kind=='mask':s['candidates'][0]['mask']=999
    if kind=='budget':s['capacity_bytes']=0
    if kind=='baseline':s['minimum_mass']=[0,1]
    with pytest.raises(ValueError):M.constrain(s,F(1))


def test_permutations_and_ties():
    s=fixture([[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)]])
    t=deepcopy(s);t['times'].reverse();t['costs'].reverse();t['candidates'].reverse()
    for f in (F(0),F(1,2),F(1)):assert M.constrain(s,f)==M.constrain(t,f)
    for row in t['candidates']:row['masses'].reverse()
    for f in (F(0),F(1,2),F(1)):
        a=M.constrain(s,f);b=M.constrain(t,f);b['lottery']['expected_masses'].reverse();assert a==b
    s=M.fixture([1,2,3],[1,1,1],1,{'a':[1],'b':[2],'c':[3]},[[F(1,3)]*3]*3)
    assert M.constrain(s,F(1))['lottery']['optimal_basic_count']==3


def test_empty_and_full():
    for capacity,selected in [(0,[]),(2,[1,2])]:
        s=M.fixture([1,2],[1,1],capacity,{'only':selected},[[F(1,2)]*2]*3)
        for f in (F(0),F(1,2),F(1)):
            result=M.constrain(s,f)
            assert result['expected_mass_cost']==result['lottery']['expected_gain']==[0,1]
