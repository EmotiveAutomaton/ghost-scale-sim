from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import prior_mixture_review as R
from ghostscale.validation.soundingline.v19 import prior_mixture_misspecification as P


def library():
    return P.fixture([1,2,3],[1,2,1],2,{'a':[1,3],'b':[2],'c':[3]},
        [[F(1,2),F(1,3),F(1,6)],[F(1,6),F(2,3),F(1,6)],[F(0),F(0),F(1)]])


def test_known_answers():
    assert all(R.controls().values())


@pytest.mark.parametrize('reliability',R.RELIABILITIES)
def test_all_mixtures_independent_complete_policy_enumeration(reliability):
    s=library()
    policies=[P.value(s,c,P.channel(reliability)) for c in R.MIXTURES]
    for i,n in enumerate(R.MIXTURES):
        for j,a in enumerate(R.MIXTURES):
            assert R.assess(s,n,a,reliability)==P.evaluate(s,policies[i],policies[j])


def test_wrong_actual_mixture_does_not_change_nominal_choices():
    s=library()
    choices=[tuple(c['selected_mask'] for c in R.assess(s,(1,1,2),a,(2,3))['cues'])
             for a in R.MIXTURES]
    assert len(set(choices))==1


def test_nominal_impossibility_preserves_largest_mask():
    s=library();r=R.assess(s,(4,0,0),(0,4,0),(1,1))
    assert r['unexpected_label_probability']==[1,1]
    assert r['cues'][1]['selected_mask']==max(x['mask'] for x in s['candidates'])


def test_charge_corruption_is_rejected():
    s=deepcopy(library());s['candidates'][0]['used_bytes']=3
    with pytest.raises(AssertionError):R.assess(s,(1,1,2),(2,1,1),(2,3))


def test_label_permutation_preserves_values():
    s=library();q=deepcopy(s)
    for row in q['candidates']: row['masses']=[row['masses'][j] for j in (2,0,1)]
    for p in R.RELIABILITIES:
        a=R.assess(s,(1,1,2),(2,1,1),p);b=R.assess(q,(2,1,1),(1,2,1),p)
        for k in R.METRICS: assert a[k]==b[k]
