from fractions import Fraction as F
from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v19 import channel_realized_risk as M
from ghostscale.validation.soundingline.v19 import randomized_channel_regret as R
from test_v19_randomized_channel_regret import hull_value,fixture


def test_controls():assert all(M.controls().values())


@pytest.mark.parametrize('counts',M.MIXTURES)
def test_full_policies_lotteries_risk_and_unchanged_reference(counts):
    s=fixture();r=M.solve(s,counts);ref=R.solve(s,counts)
    assert r['deterministic_worst_regret']==ref['deterministic']['worst_regret']
    assert r['risk_levels'][-1]['worst_expected_regret']==ref['worst_expected_regret']
    table={tuple(p['policy']):p for p in ref['deterministic']['policies']}
    for i,p in enumerate(r['policy_masks']):
        a=table[tuple(p)];lo=F(*a['intercept']);slope=F(*a['slope'])
        assert [F(n,r['score_denominator']) for n in r['policy_score_numerators'][i]]==[lo+slope*F(*t) for t in M.RELIABILITIES]
    masses={v['mask']:[F(*x) for x in v['masses']] for v in s['candidates']}
    priorbest=[max(v[j] for v in masses.values()) for j in range(3)]
    last=set();lastregret=None
    for level in r['risk_levels']:
        q=r['lottery_sets'][level['lottery_set']];surv=set(q['surviving_policies'])
        assert last<=surv;last=surv
        assert set(q['excluded_policies'])==set(range(len(table)))-surv
        for i,p in enumerate(r['policy_masks']):
            risk=max(priorbest[j]-masses[p[k]][j] for j in range(3) if counts[j] for k in range(3))
            assert risk==F(r['policy_realized_loss_numerators'][i],r['mass_denominator'])
            assert (i in surv)==(risk<=F(*level['threshold']))
        points=[tuple(F(n,r['score_denominator']) for n in r['policy_endpoint_regret_numerators'][i]) for i in surv]
        assert F(*level['worst_expected_regret'])==hull_value(points)
        assert lastregret is None or F(*level['worst_expected_regret'])<=lastregret
        lastregret=F(*level['worst_expected_regret'])
        endpoints={tuple(r['policy_masks'][i]):tuple(F(n,r['score_denominator']) for n in r['policy_endpoint_regret_numerators'][i]) for i in surv}
        candidates,_,_,_=R.lotteries(endpoints)
        actual={}
        for i,j,n,d,a,b,den in q['candidates']:actual[(tuple(r['policy_masks'][i]),tuple(r['policy_masks'][j]),F(n,d))]=(F(a,den),F(b,den))
        assert actual==candidates
        assert all(d['maximum_realized_bytes']<=s['capacity_bytes'] for d in level['diagnostics'])


def test_null_zero_prior_support_and_bad_charge():
    s=fixture()
    for row in s['candidates']:row['masses']=[[1,2]]*3
    r=M.solve(s,(4,0,0))
    assert all(l['maximum_realized_coverage_loss']==[0,1] and l['worst_expected_regret']==[0,1] for l in r['risk_levels'])
    s['candidates'][0]['used_bytes']=s['capacity_bytes']+1
    with pytest.raises(ValueError):M.solve(s,(4,0,0))


def test_policy_order_and_prior_permutation():
    s=fixture();a=M.solve(s,(1,1,2));s['candidates'].reverse();assert M.solve(s,(1,1,2))==a
    for row in s['candidates']:row['masses']=[row['masses'][i] for i in (2,0,1)]
    b=M.solve(s,(2,1,1))
    assert [x['worst_expected_regret'] for x in a['risk_levels']]==[x['worst_expected_regret'] for x in b['risk_levels']]
