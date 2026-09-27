from fractions import Fraction as F
from itertools import product
import pytest
from ghostscale.validation.soundingline.v19 import adaptive_second_cue as M

def fixture():
    return M.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)],[F(1,2)]*2])

def test_controls():assert all(M.controls().values())

@pytest.mark.parametrize('counts',M.MIXTURES)
@pytest.mark.parametrize('rate',M.RELIABILITIES)
@pytest.mark.parametrize('dep',M.DEPENDENCES)
def test_all_conditional_policy_choices(counts,rate,dep):
    s=fixture();result=M.solve(s,counts,rate,dep);r,d=F(*rate),F(*dep)
    mass={x['mask']:[F(*v) for v in x['masses']] for x in s['candidates']}
    cost={x['mask']:x['used_bytes'] for x in s['candidates']}
    def prob(j,k):return r if j==k else (1-r)/2
    for a in result['fees']:
        fee=F(*a['fee']);total=F(0);expected_fee=F(0);bytes_=F(0)
        for first,item in enumerate(a['conditional']):
            joint=[F(counts[j],4)*prob(j,first) for j in range(3)];p=sum(joint)
            pairs=[[joint[j]*(d*(first==second)+(1-d)*prob(j,second)) for j in range(3)] for second in range(3)]
            free={m:sum(joint[j]*mass[m][j] for j in range(3)) for m in mass}
            paid={policy:sum(pairs[k][j]*mass[policy[k]][j] for k in range(3) for j in range(3))-fee*p for policy in product(mass,repeat=3)}
            best=max(max(free.values()),max(paid.values()));total+=best
            assert F(*item['net_contribution'])==best
            expected_ties=[n for n,v in [('decline',max(free.values())),('buy',max(paid.values()))] if v==best]
            assert item['optimal_decisions']==expected_ties
            assert item['decline_optimal_masks']==[m for m,v in free.items() if v==max(free.values())]
            assert set(product(*item['buy_optimal_masks_by_second']))=={p for p,v in paid.items() if v==max(paid.values())}
            buy=item['selected_decision']=='buy'
            assert buy==(max(paid.values())>max(free.values()))
            expected_fee+=p*fee*buy
            bytes_+=sum(sum(pairs[k])*cost[item['buy_selected_policy'][k]] for k in range(3)) if buy else p*cost[item['decline_selected_mask']]
            if not p:assert item['conditional_break_even_fee'] is None and not buy
        assert F(*a['net_utility'])==total
        assert F(*a['expected_fee'])==expected_fee and F(*a['expected_used_bytes'])==bytes_
        assert F(*a['gross_retained_mass'])-expected_fee==total
        assert total>=max(F(*a['never_net']),F(*a['always_net']))
        if fee>0 and (dep==(1,1) or rate in ((1,3),(1,1))):assert a['buy_probability']==[0,1]

def test_corrupt_bytes_and_fee():
    s=fixture();s['candidates'][0]['used_bytes']=2
    with pytest.raises((ValueError,AssertionError)):M.solve(s,(2,2,0),(2,3),(0,1))
    with pytest.raises(ValueError):M.price({},(-1,1))
