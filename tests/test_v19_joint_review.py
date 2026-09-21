"""Independent review controls reject corruption before scientific use."""
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import joint_review as R


def test_known_answers():
    assert all(R.controls().values())


def test_fixed_universe_score_by_explicit_vector():
    p=np.array([.8,32/33-.8]); alphabet=[0,217];truth={0:.2,217:.3,500:.5}
    full=np.full(R.UNIVERSE,1/33/(R.UNIVERSE-2));full[alphabet]=p
    target=np.zeros(R.UNIVERSE)
    for k,v in truth.items():target[k]=v
    score=R.score(p,alphabet,truth,32)
    assert abs(score['loss']+target@np.log(full))<1e-12
    assert abs(score['squared_error']-np.square(full-target).sum())<1e-12
    assert abs(score['compatible_mass']-full[[0,217,500]].sum())<1e-12


def test_score_corruption_rejected():
    with pytest.raises(ValueError):R.close([.1,.9],[.2,.8])


def test_all_joint_codes():
    for k in range(R.UNIVERSE):
        g,o=R.digits(k)
        r={'steps':[{'goal':R.GOALS[a],'operation':R.OPS[b]} for a,b in zip(g,o)]}
        assert R.label(r)==k
