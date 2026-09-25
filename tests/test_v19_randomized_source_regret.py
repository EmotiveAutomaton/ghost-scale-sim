from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import randomized_source_regret as M
from ghostscale.validation.soundingline.v19.randomized_storage_review import vertices


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
def test_independent_full_constraint_enumeration(priors):
    s=fixture(priors);a=M.minimize(s);rows=sorted(s['candidates'],key=lambda r:r['mask'])
    masses=[[F(*x) for x in r['masses']] for r in rows];optima=[F(*x) for x in s['best_prior_mass']]
    value,ties,count=vertices(masses,optima)
    encode=lambda w:[dict(mask=rows[i]['mask'],weight=[x.numerator,x.denominator]) for i,x in enumerate(w) if x]
    assert F(*a['maximum_expected_regret'])==-value
    assert a['optimal_basic_lotteries']==[encode(w) for w in ties]
    assert a['selected_lottery']==encode(ties[-1]) and a['feasible_basic_count']==count
    w=ties[-1];rr=[[optima[j]-row[j] for j in range(len(optima))] for row in masses]
    assert F(*a['worst_realized_regret'])==max(max(rr[i]) for i,v in enumerate(w) if v)
    assert all(F(*a[k])>=0 for k in ('regret_improvement','gain_over_deterministic','minimum_mass_cost'))


def test_prior_permutation():
    s=fixture([[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)],[F(0),F(1),F(0)]])
    a=M.minimize(s);t=deepcopy(s);t['best_prior_mass'].reverse()
    for r in t['candidates']:r['masses'].reverse();r['prior_regrets'].reverse()
    b=M.minimize(t)
    for key in ('expected_masses','expected_prior_regrets'):b[key].reverse()
    assert a==b


def test_item_permutation():
    s=fixture();a=M.minimize(s);s['times'].reverse();s['costs'].reverse();s['candidates'].reverse()
    assert a==M.minimize(s)


@pytest.mark.parametrize('kind',['optimum','regret','cost','mask','budget'])
def test_corruption(kind):
    s=fixture()
    if kind=='optimum':s['best_prior_mass'][0]=[0,1]
    if kind=='regret':s['candidates'][0]['prior_regrets'][0]=[99,1]
    if kind=='cost':s['candidates'][0]['used_bytes']=99
    if kind=='mask':s['candidates'][0]['mask']=999
    if kind=='budget':s['capacity_bytes']=0
    with pytest.raises(ValueError):M.minimize(s)


def test_empty_full():
    for capacity,chosen in [(0,[]),(2,[1,2])]:
        a=M.minimize(M.fixture([1,2],[1,1],capacity,{'only':chosen},[[F(1,2)]*2]*3))
        assert a['maximum_expected_regret']==a['minimum_mass_cost']==[0,1]
