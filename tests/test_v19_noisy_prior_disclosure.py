from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import pytest
from ghostscale.validation.soundingline.v19 import noisy_prior_disclosure as N
from ghostscale.validation.soundingline.v19 import partial_prior_review as P


def fixture():
    return N.fixture([1,2,3], [1,2,1], 2, {'a':[1,3], 'b':[2], 'c':[3]},
        [[F(1,2),F(1,3),F(1,6)], [F(1,6),F(2,3),F(1,6)], [F(0),F(0),F(1)]])


def test_known_answers():
    assert all(N.controls().values())


@pytest.mark.parametrize('counts', N.MIXTURES)
def test_complete_policy_enumeration_and_endpoints(counts):
    s = fixture(); mass = {r['mask']:tuple(F(*x) for x in r['masses']) for r in s['candidates']}
    p = P.value(s,counts)['disclosures']
    for reliability in N.RELIABILITIES:
        channel = N.channel(reliability); r = N.value(s,counts,channel)
        scores = [sum(F(counts[j],4)*channel[j][label]*mass[policy[label]][j]
                      for label in range(3) for j in range(3)) for policy in product(mass,repeat=3)]
        assert F(*r['informed_mass']) == max(scores)
        if reliability == (1,3): assert r['informed_mass'] == p[0]['retained_mass']
        if reliability == (1,1): assert r['informed_mass'] == r['certainty_mass'] == p[-1]['retained_mass']
        assert all(c['selected_used_bytes'] <= s['capacity_bytes'] and c['certainty_used_bytes'] <= s['capacity_bytes'] for c in r['cues'])


def test_identical_prior_null():
    s = N.fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,2)]*2]*3)
    assert all(N.value(s,(1,1,2),N.channel(p))['information_value'] == [0,1] for p in N.RELIABILITIES)


def test_joint_prior_and_label_permutation():
    s = fixture(); order = [2,0,1]; permuted = deepcopy(s)
    for row in permuted['candidates']: row['masses'] = [row['masses'][i] for i in order]
    # The mass vectors are permuted for this algebraic control only.
    for p in N.RELIABILITIES:
        r = N.value(s,(1,1,2),N.channel(p)); q = N.value(permuted,(2,1,1),N.channel(p))
        for k in ('fixed_mass','informed_mass','certainty_mass','information_value','certainty_penalty'):
            assert r[k] == q[k]


@pytest.mark.parametrize('bad', [[[F(1)]], [[F(-1),F(1),F(1)]]*3, [[F(0)]*3]*3, [[.5,.25,.25]]*3])
def test_malformed_channel_rejected(bad):
    with pytest.raises(ValueError): N.value(fixture(),(1,1,2),bad)


def test_zero_probability_label():
    r = N.value(fixture(),(4,0,0),N.channel((1,1)))
    assert r['cues'][1]['probability'] == [0,1]
    assert r['cues'][1]['conditional_prior'] is None
    assert r['cues'][1]['conditional_retained_mass'] is None
