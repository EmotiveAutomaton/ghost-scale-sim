import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import interchange_fixtures as F
from ghostscale.validation.soundingline.v19 import local_world as L
from ghostscale.validation.soundingline.v19.readout_data import local_cache


def test_known_controls():
    assert all(F.controls().values())


def test_dynamic_law_against_independent_native_enumeration():
    w=L.law(0);records=L.enumerate_world(w);observed=np.zeros((16,4,8))
    for r in records:
        a=r['final'];observed[r['maker_index'],r['context_index'],a[0]+2*a[1]+4*a[2]]+=64*r['probability']
    expected=np.array([[F.endpoint_law(w,m,c) for c in L.CONTEXTS] for m in L.MAKERS])
    assert np.max(abs(observed-expected))<1e-12
    assert np.max(abs(expected.reshape(16,32)-local_cache(records)))<1e-12
    assert np.allclose(expected.sum(-1),1)
    assert np.max(abs(expected[0]-expected[8]))>0.01


def test_all_change_stay_factors_with_nuisance():
    for maker in L.MAKERS:
        for factor in (0,2):
            for change in (False,True):
                donor=F.donor_for(maker,factor,change);target=F.hybrid(maker,donor,factor)
                assert donor[1]!=maker[1]
                assert all(target[i]==maker[i] for i in range(4) if i!=factor)
                assert (target[factor]!=maker[factor])==change
    with pytest.raises(ValueError):F.hybrid(L.MAKERS[0],L.MAKERS[1],1)


def test_generated_trajectory_replays_and_has_native_support():
    w=L.law(0)
    for maker in L.MAKERS:
        endpoint,events=F.sample_episode(w,maker,L.CONTEXTS[0],[0.1,0.53,0.99])
        for e in events:
            assert tuple(e['after'])==L.execute(e['before'],e['undo_buffer'],e['operation'],maker)
        assert F.endpoint_law(w,maker,L.CONTEXTS[0])[endpoint]>0


def test_proper_score_preserves_impossible_forecast_failure():
    assert np.isinf(F.loss(np.array([[1.,0.]]),np.array([[0.,1.]]))).all()
    assert F.loss(np.array([[1.,0.]]),np.array([[1.,0.]]))[0]==0


def test_decode_identity_and_forecast_normalization():
    a={'head.weight':np.arange(32*38).reshape(32,38)/10000,'head.bias':np.zeros(32)}
    h=np.array([np.zeros(38),np.ones(38)])
    p=F.decode(a,h)
    # Exact no-op uses the same batch; different BLAS batch kernels can round differently.
    assert np.array_equal(F.decode(a,h.copy()),p)
    assert np.allclose(F.decode(a,h[[1]]),p[[1]],atol=1e-14,rtol=0)
    assert np.allclose(p.sum(-1),1) and not np.allclose(p[0],p[1])


def test_corrupt_input_is_rejected(tmp_path):
    p=tmp_path/'input.json';p.write_text('{}')
    F.check_hash(p,F.file_digest(p))
    with pytest.raises(ValueError):F.check_hash(p,'0'*64)
