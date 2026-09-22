"""Independent review must reject omissions and numerical corruption."""
from itertools import combinations
import copy
import pytest
from ghostscale.validation.soundingline.v19 import marginal_transition as M, marginal_review as R


def test_partition_exhaustive_and_corruption():
    hs = [('none', 0, 0), ('purpose', 8, 0), ('skill', 8, 0), ('purpose', 9, 0), ('none', 0, 8)]
    for s in (8, 9, 10):
        for t in range(s, 12):
            saved = M.partitions(hs, s, t)
            actual = R.partition(hs, s, t, saved)
            pairs = [tuple(sorted((a, b))) for block in actual['blocks']
                     for a in actual['groups'][block['left']]['members']
                     for b in actual['groups'][block['right']]['members']]
            expected = {(a, b) for a, b in combinations(range(len(hs)), 2)
                        if R.state(hs[a], s) == R.state(hs[b], s) and R.state(hs[a], t) != R.state(hs[b], t)}
            assert len(pairs) == len(set(pairs)) and set(pairs) == expected
    row = M.partitions(hs, 8, 10); bad = copy.deepcopy(row); bad['blocks'].pop()
    with pytest.raises(ValueError): R.partition(hs, 8, 10, bad)
    bad = copy.deepcopy(row); bad['groups'][0]['members'].pop()
    with pytest.raises(ValueError): R.partition(hs, 8, 10, bad)


def test_law_scalar_known_answer_and_null():
    law = [[[1/8]*8 for _ in range(4)] for _ in range(16)]
    assert all(r['total_variation'] == 0 and r['identical_binary64'] for r in R.law_distances(law))
    law[0][0] = [1., 0., 0., 0., 0., 0., 0., 0.]
    law[1][0] = [0., 1., 0., 0., 0., 0., 0., 0.]
    result = R.law_distances(law)
    assert result[0]['total_variation'] == 1
    changed = copy.deepcopy(result); changed[0]['total_variation'] = .9
    with pytest.raises(ValueError): R.compare(changed, result)
    law[0][0][0] = float('nan')
    with pytest.raises(ValueError): R.law_distances(law)
