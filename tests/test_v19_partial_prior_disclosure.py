from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import pytest
from ghostscale.validation.soundingline.v19 import partial_prior_disclosure as M
from tests.test_v19_prior_information import fixture


def test_known_controls():assert all(M.controls().values())


@pytest.mark.parametrize('counts',M.MIXTURES)
def test_exhaustive_disclosure_policies(counts):
    s=fixture();r=M.value(s,counts);rows=s['candidates'];weights=[F(x,4) for x in counts]
    masses=[[F(*x) for x in row['masses']] for row in rows]
    scores=[]
    for p,result in zip(M.PARTITIONS,r['disclosures']):
        policies=list(product(range(len(rows)),repeat=len(p)))
        objectives=[sum(weights[j]*masses[policy[i]][j] for i,cell in enumerate(p) for j in cell) for policy in policies]
        best=max(objectives);assert F(*result['retained_mass'])==best;scores.append(best)
        for cell,group in zip(result['cells'],p):
            prob=sum(weights[j] for j in group)
            assert F(*cell['probability'])==prob
            assert (cell['conditional_prior'] is None)==(prob==0)
            assert cell['selected_mask']==max(cell['optimal_masks'])
            assert cell['selected_used_bytes']<=s['capacity_bytes']
            if not prob:assert cell['optimal_masks']==sorted(row['mask'] for row in rows)
        assert result['maximum_realized_bytes']<=s['capacity_bytes']
    assert all(scores[0]<=x<=scores[-1] for x in scores)
    if 4 in counts:assert len(set(scores))==1


def test_prior_permutation():
    s=fixture();t=deepcopy(s)
    for row in t['candidates']:row['masses'].reverse()
    a=M.value(s,(1,3,0))['disclosures'];b=M.value(t,(0,3,1))['disclosures']
    for i,j in enumerate((0,3,2,1,4)):
        for k in ('retained_mass','information_value','full_disclosure_gap','expected_used_bytes'):
            assert a[i][k]==b[j][k]


@pytest.mark.parametrize('kind',['bytes','mask','capacity','mixture'])
def test_corrupt_input(kind):
    s=fixture();counts=(1,2,1)
    if kind=='bytes':s['candidates'][0]['used_bytes']=99
    if kind=='mask':s['candidates'][0]['mask']=999
    if kind=='capacity':s['capacity_bytes']=0
    if kind=='mixture':counts=(1,1,1)
    with pytest.raises(ValueError):M.value(s,counts)
