from fractions import Fraction as F
from itertools import product
from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v19 import channel_misspecification as M
from ghostscale.validation.soundingline.v19 import noisy_prior_review as R


def fixture():
    return M.fixture([1,2,3],[1,2,1],2,{'a':[1,3],'b':[2],'c':[3]},
        [[F(1,2),F(1,3),F(1,6)],[F(1,6),F(2,3),F(1,6)],[F(0),F(0),F(1)]])


def test_known_answers():
    assert all(M.controls().values())


@pytest.mark.parametrize('counts',M.MIXTURES)
def test_all_channel_pairs_against_whole_policy_enumeration(counts):
    s=fixture(); mass={r['mask']:[F(*x) for x in r['masses']] for r in s['candidates']}
    policies=[R.value(s,counts,p) for p in M.RELIABILITIES]
    for i,j in product(range(3),repeat=2):
        n,a=policies[i],policies[j]; got=M.evaluate(s,n,a); ch=M.channel(M.RELIABILITIES[j])
        def score(policy):
            return sum(F(counts[p],4)*ch[p][label]*mass[policy[label]][p]
                       for p in range(3) for label in range(3))
        chosen=tuple(c['selected_mask'] for c in n['cues'])
        best=max(score(p) for p in product(mass,repeat=3))
        assert F(*got['retained_mass'])==score(chosen)
        assert F(*got['regret'])==best-score(chosen)
        assert F(*got['gain_over_fixed'])==score(chosen)-max(score((m,m,m)) for m in mass)
        assert got['maximum_realized_bytes']<=s['capacity_bytes']
        if i==j: assert got['regret']==[0,1]
        if j==0: assert F(*got['gain_over_fixed'])<=0


def test_identical_prior_null():
    s=M.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,2)]*2]*3)
    assert all(r['regret']==[0,1] for r in M.compare(s)['comparisons'])


def test_impossible_nominal_label_uses_frozen_choice():
    s=fixture();n=M.value(s,(4,0,0),M.channel((1,1)));a=M.value(s,(4,0,0),M.channel((1,3)))
    got=M.evaluate(s,n,a)
    assert got['unexpected_label_probability']==[2,3]
    assert got['cues'][1]['selected_mask']==max(r['mask'] for r in s['candidates'])


def test_mixture_change_rejected():
    s=fixture();n=M.value(s,(4,0,0),M.channel((1,1)));a=M.value(s,(0,4,0),M.channel((1,3)))
    with pytest.raises(AssertionError):M.evaluate(s,n,a)


def test_corrupted_charge_rejected():
    s=fixture();n=M.value(s,(1,1,2),M.channel((2,3)));s['candidates'][0]['used_bytes']=3
    with pytest.raises(AssertionError):M.evaluate(s,n,n)


def test_joint_label_prior_permutation():
    s=fixture();q=deepcopy(s);order=[2,0,1]
    for row in q['candidates']:row['masses']=[row['masses'][i] for i in order]
    for n,a in product(M.RELIABILITIES,repeat=2):
        one=M.evaluate(s,M.value(s,(1,1,2),M.channel(n)),M.value(s,(1,1,2),M.channel(a)))
        two=M.evaluate(q,M.value(q,(2,1,1),M.channel(n)),M.value(q,(2,1,1),M.channel(a)))
        for k in ('retained_mass','regret','gain_over_fixed','unexpected_label_probability'):assert one[k]==two[k]
