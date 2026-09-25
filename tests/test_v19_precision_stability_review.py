import json
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import precision_stability as P
from ghostscale.validation.soundingline.v19 import precision_stability_review as R


@pytest.mark.parametrize('width', [0., .125])
def test_independent_complete_pair_reconstruction(width):
    a = np.array([[.5, .25, .0], [.75, .125, .0], [.25, .125, .0], [.5, .375, .125]])
    choices = [[0], [0], [0]]
    actual = json.loads(json.dumps(P.evaluate(a, a+width, choices)))
    assert R.reconstruct(a, a+width, choices) == actual


def test_null_and_corrupt_choice():
    assert all(R.controls().values())
    a = np.zeros((4, 3))
    result = R.reconstruct(a, a, [[0]]*3)
    assert all(r['guaranteed_optima'] == r['feasible_assignments'] for r in result['summaries'])
    assert all(not any(r['recorded_regret_upper_numerators']) for r in result['summaries'])
    with pytest.raises(AssertionError):
        R.reconstruct(a, a, [[80]]*3)
