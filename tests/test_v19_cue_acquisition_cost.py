from fractions import Fraction as F
from itertools import product
import pytest
from ghostscale.validation.soundingline.v19 import cue_acquisition_cost as M
from test_v19_randomized_channel_regret import fixture


def test_controls(): assert all(M.controls().values())


@pytest.mark.parametrize('counts',M.MIXTURES)
@pytest.mark.parametrize('reliability',M.RELIABILITIES)
def test_independent_finite_integration_and_net_optimum(counts,reliability):
    s=fixture();r=M.solve(s,counts,reliability)
    mass={a['mask']:[F(*v) for v in a['masses']] for a in s['candidates']}
    rate=F(*reliability);weights=[F(c,4) for c in counts]
    scores={p:sum(weights[j]*(rate if j==k else (1-rate)/2)*mass[p[k]][j] for j in range(3) for k in range(3)) for p in product(sorted(mass),repeat=3)}
    assert {tuple(a['policy']):F(*a['gross_mass']) for a in r['all_paid_policies']}==scores
    fixed={p:v for p,v in scores.items() if p[0]==p[1]==p[2]}
    assert F(*r['break_even_fee'])==max(scores.values())-max(fixed.values())
    last=True
    for level in r['fee_levels']:
        fee=F(*level['fee']);options={(False,p):v for p,v in fixed.items()}
        options.update({(True,p):v-fee for p,v in scores.items()});best=max(options.values())
        assert F(*level['net_utility'])==best
        assert {(t['acquire'],tuple(t['policy'])) for t in level['optimal_choices']}=={k for k,v in options.items() if v==best}
        assert last or not level['selected']['acquire'];last=level['selected']['acquire']
        assert level['maximum_realized_bytes']<=s['capacity_bytes']
    assert r['fee_levels'][0]['gross_mass']==r['free_access']['informed_mass']


def test_zero_support_and_null_and_corruption():
    s=M.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,2)]*2]*3)
    r=M.solve(s,(4,0,0),(1,1))
    assert all(not l['selected']['acquire'] and l['net_gain_over_no_access']==[0,1] for l in r['fee_levels'])
    assert r['free_access']['cues'][1]['conditional_prior'] is None
    with pytest.raises(ValueError):M.solve(s,(4,0,0),(1,1),fees=[[-1,1]])
    s['candidates'][0]['used_bytes']=s['capacity_bytes']+1
    with pytest.raises((ValueError,AssertionError)):M.solve(s,(4,0,0),(1,1))
