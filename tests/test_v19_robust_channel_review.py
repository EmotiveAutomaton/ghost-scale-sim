from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import robust_channel_review as R
from ghostscale.validation.soundingline.v19 import robust_channel_regret as P


def fixture():
    return P.fixture([1,2,3],[1,2,1],2,{'a':[1,3],'b':[2],'c':[3]},
        [[F(1,2),F(1,3),F(1,6)],[F(1,6),F(2,3),F(1,6)],[F(0),F(0),F(1)]])


def test_known_answers():
    assert all(R.controls().values())


@pytest.mark.parametrize('counts', R.MIXTURES)
def test_full_policy_and_diagnostic_agreement(counts):
    library = fixture()
    actual, proof = R.assess(library, counts)
    assert actual == P.solve(library, counts)
    assert proof['endpoint_reduction_verified']
    assert proof['policy_count'] == len(library['candidates'])**3
    assert proof['intersection_and_diagnostic_points'] >= 3
    assert F(*actual['improvement_over_fixed']) >= 0
    assert F(*actual['improvement_over_nominal']) >= 0


def test_point_channel_all_ties():
    library = fixture()
    actual, _ = R.assess(library, (1,1,2), ((2,3),(2,3)))
    assert actual == P.solve(library, (1,1,2), ((2,3),(2,3)))
    assert actual['worst_regret'] == [0,1]
    assert actual['selected_policy'] == max(actual['optimal_policies'])


def test_null_retains_all_ties():
    library = fixture()
    for row in library['candidates']:
        row['masses'] = [[1,2]]*3
    actual, _ = R.assess(library, (1,1,2))
    assert len(actual['optimal_policies']) == len(library['candidates'])**3
    assert actual['worst_regret'] == [0,1]


def test_permutation_and_widening():
    library = fixture(); changed = deepcopy(library)
    for r in changed['candidates']:
        r['masses'] = [r['masses'][i] for i in (2,0,1)]
    a, _ = R.assess(library, (1,1,2))
    b, _ = R.assess(changed, (2,1,1))
    assert a['worst_regret'] == b['worst_regret']
    assert {tuple(p[i] for i in (2,0,1)) for p in a['optimal_policies']} == {tuple(p) for p in b['optimal_policies']}
    narrow, _ = R.assess(library, (1,1,2), ((2,3),(1,1)))
    assert F(*narrow['worst_regret']) <= F(*a['worst_regret'])


def test_corrupt_charge_and_identity_rejected():
    library = fixture(); library['candidates'][0]['used_bytes'] = 3
    with pytest.raises(AssertionError):
        R.assess(library, (1,1,2))
    library = fixture(); library['candidates'].append(deepcopy(library['candidates'][0]))
    with pytest.raises(AssertionError):
        R.assess(library, (1,1,2))
