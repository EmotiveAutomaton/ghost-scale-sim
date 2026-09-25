from fractions import Fraction as F
from itertools import product, permutations
import pytest
from ghostscale.validation.soundingline.v19 import repeated_cue as M
from test_v19_randomized_channel_regret import fixture


def test_controls():assert all(M.controls().values())


@pytest.mark.parametrize('counts',M.MIXTURES)
@pytest.mark.parametrize('rate',M.RELIABILITIES)
@pytest.mark.parametrize('dependence',M.DEPENDENCES)
def test_independent_pair_integration(counts,rate,dependence):
    s=fixture();result=M.solve(s,counts,rate,dependence)
    r=F(*rate);d=F(*dependence);matrix=[[r if j==k else (1-r)/2 for k in range(3)] for j in range(3)]
    mass={a['mask']:[F(*v) for v in a['masses']] for a in s['candidates']}
    total=F(0);naive_total=F(0);probability_total=F(0)
    for pair in result['pairs']:
        a,b=pair['labels']
        joint=[F(counts[j],4)*matrix[j][a]*(d*(a==b)+(1-d)*matrix[j][b]) for j in range(3)]
        assert [F(*v) for v in pair['joint_prior']]==joint
        probability=sum(joint);probability_total+=probability
        assert F(*pair['probability'])==probability
        assert pair['conditional_prior']==([[ (v/probability).numerator,(v/probability).denominator] for v in joint] if probability else None)
        scores={m:sum(joint[j]*mass[m][j] for j in range(3)) for m in mass}
        assert {r['mask']:F(*r['mass']) for r in pair['joint_mask_scores']}==scores
        assert pair['optimal_masks']==[m for m in sorted(scores) if scores[m]==max(scores.values())]
        total+=max(scores.values());naive_total+=scores[pair['independent_selected_mask']]
        assert pair['used_bytes']<=s['capacity_bytes'] and pair['independent_used_bytes']<=s['capacity_bytes']
    assert probability_total==1 and F(*result['retained_mass'])==total
    assert F(*result['independent_assumption_mass'])==naive_total
    assert F(*result['incremental_value'])==total-F(*result['single_mass'])>=0
    for a in range(3):
        assert sum(F(*p['probability']) for p in result['pairs'] if p['labels'][0]==a)==F(*result['single_cue']['cues'][a]['probability'])
    if dependence==(1,1) or rate in ((1,3),(1,1)):assert result['incremental_value']==[0,1]
    if dependence==(0,1):assert result['independent_regret']==[0,1]


def test_wrong_independence_can_harm():
    s=M.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(3,4),F(1,4)],[F(3,8),F(5,8)],[F(1,2)]*2])
    result=M.solve(s,(3,1,0),(2,3),(1,1))
    assert result['independent_gain_over_single']==[-1,48]


def test_corrupt_bytes_and_dependence():
    s=fixture()
    with pytest.raises(ValueError):M.solve(s,(2,2,0),(2,3),(2,1))
    s['candidates'][0]['used_bytes']=s['capacity_bytes']+1
    with pytest.raises((ValueError,AssertionError)):M.solve(s,(2,2,0),(2,3),(1,2))


def test_source_label_permutation():
    prior=[[F(3,4),F(1,4)],[F(3,8),F(5,8)],[F(1,2)]*2];counts=(2,1,1)
    reference=None
    for order in permutations(range(3)):
        s=M.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[prior[i] for i in order])
        r=M.solve(s,tuple(counts[i] for i in order),(2,3),(1,2))
        values=tuple(r[k] for k in ('retained_mass','single_mass','independent_assumption_mass','expected_used_bytes'))
        if reference is None:reference=values
        assert values==reference
