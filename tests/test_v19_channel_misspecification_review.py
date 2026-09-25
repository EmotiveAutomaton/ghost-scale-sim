from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import channel_misspecification_review as R
from ghostscale.validation.soundingline.v19 import channel_misspecification as P


def library():
    return P.fixture([1,2,3],[1,2,1],2,{'a':[1,3],'b':[2],'c':[3]},
        [[F(1,2),F(1,3),F(1,6)],[F(1,6),F(2,3),F(1,6)],[F(0),F(0),F(1)]])


def test_known_answers():
    assert all(R.controls().values())


@pytest.mark.parametrize('counts', R.MIXTURES)
def test_independent_full_policy_enumeration(counts):
    s=library()
    for n in R.RELIABILITIES:
        for a in R.RELIABILITIES:
            assert R.assess(s,counts,n,a)==P.evaluate(s,P.value(s,counts,P.channel(n)),P.value(s,counts,P.channel(a)))


def test_actual_channel_cannot_reselect_nominal_policy():
    s=library()
    choices=[tuple(c['selected_mask'] for c in R.assess(s,(1,1,2),(2,3),a)['cues']) for a in R.RELIABILITIES]
    assert len(set(choices))==1


def test_unsupported_label_keeps_fallback():
    s=library();r=R.assess(s,(4,0,0),(1,1),(1,3))
    assert r['unexpected_label_probability']==[2,3]
    assert r['cues'][1]['selected_mask']==max(x['mask'] for x in s['candidates'])


def test_charge_corruption_rejected():
    s=deepcopy(library());s['candidates'][0]['used_bytes']=3
    with pytest.raises(AssertionError):R.assess(s,(1,1,2),(2,3),(1,1))


def test_joint_label_permutation():
    s=library();q=deepcopy(s)
    for r in q['candidates']:r['masses']=[r['masses'][j] for j in (2,0,1)]
    for n in R.RELIABILITIES:
        for a in R.RELIABILITIES:
            one=R.assess(s,(1,1,2),n,a);two=R.assess(q,(2,1,1),n,a)
            assert all(one[k]==two[k] for k in R.METRICS)
