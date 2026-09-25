from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import pytest
from ghostscale.validation.soundingline.v19 import prior_information as M


def fixture():
    return M.fixture([1,2,3],[1,2,3],3,{'a':[1],'b':[2],'c':[3],'d':[1,2]},
        [[F(1,2),F(1,3),F(1,6)],[F(0),F(0),F(1)],[F(1,3)]*3])


def test_controls():assert all(M.controls().values())


@pytest.mark.parametrize('counts',M.MIXTURES)
def test_exhaustive_policies_and_fixed_lotteries(counts):
    s=fixture();r=M.value(s,counts);rows=s['candidates'];w=[F(x,4) for x in counts]
    masses=[[F(*x) for x in row['masses']] for row in rows]
    expected=[sum(w[j]*m[j] for j in range(3)) for m in masses]
    policies=list(product(range(len(rows)),repeat=3))
    scores=[sum(w[j]*masses[p[j]][j] for j in range(3)) for p in policies]
    assert F(*r['fixed_mass'])==max(expected)
    assert F(*r['revealed_mass'])==max(scores)
    assert r['fixed_optimal_masks']==[row['mask'] for row,x in zip(rows,expected) if x==max(expected)]
    assert r['selected_fixed_mask']==max(r['fixed_optimal_masks'])
    assert F(*r['information_value'])==max(scores)-max(expected)
    # Every denominator-four fixed lottery is a convex average of fixed scores.
    for units in product(range(5),repeat=len(rows)):
        if sum(units)==4:
            assert sum(F(u,4)*x for u,x in zip(units,expected))<=max(expected)
    for j,mask in enumerate(r['selected_revealed_masks']):
        row=next(x for x in rows if x['mask']==mask)
        assert row['used_bytes']==r['revealed_used_bytes'][j]<=s['capacity_bytes']


def test_prior_and_source_permutations():
    s=fixture();t=deepcopy(s);t['times'].reverse();t['costs'].reverse();t['candidates'].reverse()
    assert M.value(s,(1,2,1))==M.value(t,(1,2,1))
    for row in t['candidates']:row['masses'].reverse()
    a=M.value(s,(1,3,0));b=M.value(t,(0,3,1))
    for k in ('fixed_mass','revealed_mass','information_value','coverage_lottery_mass','fixed_over_coverage','expected_revealed_bytes'):
        assert a[k]==b[k]
    assert a['selected_revealed_masks']==b['selected_revealed_masks'][::-1]


@pytest.mark.parametrize('kind',['bytes','mask','capacity','baseline','mixture'])
def test_corrupt_input_rejected(kind):
    s=fixture();counts=(1,2,1)
    if kind=='bytes':s['candidates'][0]['used_bytes']=99
    if kind=='mask':s['candidates'][0]['mask']=999
    if kind=='capacity':s['capacity_bytes']=0
    if kind=='baseline':s['minimum_mass']=[99,1]
    if kind=='mixture':counts=(1,1,1)
    with pytest.raises(ValueError):M.value(s,counts)


def test_empty_full_ties():
    for cap,selected in [(0,[]),(2,[1,2])]:
        s=M.fixture([1,2],[1,1],cap,{'only':selected},[[F(1,2)]*2]*3)
        assert all(M.value(s,c)['information_value']==[0,1] for c in M.MIXTURES)
    s=M.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,2)]*2]*3)
    assert M.value(s,(1,1,2))['fixed_optimal_masks']==[1,2]
