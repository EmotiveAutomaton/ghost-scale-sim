from itertools import product
import numpy as np
from ghostscale.validation.soundingline.v21 import style as S
from ghostscale.validation.soundingline.v21.style_check import fit_reference,predict_reference


def test_known_mechanism_style_and_no_information_controls():
    assert all(S.controls().values())


def test_all_finite_selectors_independent_count_fit_and_rendering_invariance():
    rng=np.random.default_rng(3);y=rng.integers(2,size=127);u=rng.random(127)
    table,_=S.tables(y,u,.83);reference=fit_reference(y,u,.83)
    assert np.array_equal(table,reference)
    for local,decoder,m,s,x in product(S.LOCALITIES,('surface','decoded'),range(2),range(2),range(2)):
        actual=S.predict(table,np.array([m]),np.array([s]),np.array([x]),local,decoder)[0]
        assert np.array_equal(actual,predict_reference(reference,m,s,x,local,decoder))
        if decoder=='decoded':
            assert np.array_equal(actual,S.predict(table,np.array([m]),np.array([1-s]),np.array([1-x]),local,decoder)[0])
