from fractions import Fraction as F
from itertools import product
from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v19 import prior_mixture_misspecification as M
from ghostscale.validation.soundingline.v19 import noisy_prior_review as R


def fixture():
    return M.fixture([1,2,3],[1,2,1],2,{'a':[1,3],'b':[2],'c':[3]},
        [[F(1,2),F(1,3),F(1,6)],[F(1,6),F(2,3),F(1,6)],[F(0),F(0),F(1)]])


def test_known_answers():
    assert all(M.controls().values())


@pytest.mark.parametrize('reliability',M.RELIABILITIES)
def test_all225_pairs_against_independent_whole_policy_scores(reliability):
    s=fixture(); mass={r['mask']:[F(*x) for x in r['masses']] for r in s['candidates']}
    ch=M.channel(reliability)
    policies=[R.value(s,c,reliability) for c in M.MIXTURES]
    for nominal,actual in product(policies,repeat=2):
        got=M.evaluate(s,nominal,actual)
        def score(policy):
            return sum(F(actual['mixture_counts'][p],4)*ch[p][label]*mass[policy[label]][p]
                       for p in range(3) for label in range(3))
        choice=tuple(c['selected_mask'] for c in nominal['cues'])
        best=max(score(p) for p in product(mass,repeat=3))
        assert F(*got['retained_mass'])==score(choice)
        assert F(*got['regret'])==best-score(choice)
        assert F(*got['gain_over_fixed'])==score(choice)-max(score((m,m,m)) for m in mass)
        assert got['maximum_realized_bytes']<=s['capacity_bytes']
        if nominal['mixture_counts']==actual['mixture_counts']:assert got['regret']==[0,1]


def test_identical_prior_null():
    s=M.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,2)]*2]*3)
    assert all(r['regret']==[0,1] for r in M.compare(s)['comparisons'])


def test_zero_nominal_positive_actual_keeps_frozen_tie_choice():
    s=fixture();n=M.value(s,(4,0,0),M.channel((1,1)));a=M.value(s,(0,4,0),M.channel((1,1)))
    got=M.evaluate(s,n,a)
    assert got['unexpected_label_probability']==[1,1]
    assert got['cues'][1]['nominal_zero'] and got['cues'][1]['actual_probability']==[1,1]
    assert got['cues'][1]['selected_mask']==max(r['mask'] for r in s['candidates'])


def test_corrupt_charge_rejected():
    s=fixture();n=M.value(s,(1,1,2),M.channel((2,3)));s['candidates'][0]['used_bytes']=3
    with pytest.raises(AssertionError):M.evaluate(s,n,n)


def test_joint_prior_and_label_permutation():
    s=fixture();order=[2,0,1];q=deepcopy(s)
    for row in q['candidates']:row['masses']=[row['masses'][i] for i in order]
    for p in M.RELIABILITIES:
        a=M.evaluate(s,M.value(s,(1,1,2),M.channel(p)),M.value(s,(2,1,1),M.channel(p)))
        b=M.evaluate(q,M.value(q,(2,1,1),M.channel(p)),M.value(q,(1,2,1),M.channel(p)))
        for k in ('retained_mass','regret','gain_over_fixed','unexpected_label_probability'):assert a[k]==b[k]
