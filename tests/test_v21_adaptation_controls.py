import numpy as np
from ghostscale.validation.soundingline.v21.adaptation_controls import independent_count_forecast
from ghostscale.validation.soundingline.v21.audit import counts, forecast


def test_count_reference_matches_all_predictions_and_duplicate_removal():
    rng=np.random.default_rng(234)
    x=rng.integers(-1,2,(97,26));y=rng.integers(0,128,97)
    test=rng.integers(-1,2,(11,26));support=np.ones((11,128),bool);support[:,9]=False
    for drop in ((),(1,)):
        reference=independent_count_forecast(x,y,test,support,drop)
        assert np.allclose(reference,np.exp(forecast(counts(x,y),test,drop,support=support)),atol=1e-12)
        assert np.all(reference[:,9]==0)
        if drop:
            changed=test.copy();changed[:,[1,14]]=1
            assert np.array_equal(reference,independent_count_forecast(x,y,changed,support,drop))


def test_no_evidence_returns_smoothed_known_prior():
    x=np.empty((5,0),int);y=np.array([0,0,0,1,1]);test=np.empty((1,0),int)
    actual=independent_count_forecast(x,y,test,np.ones((1,128),bool))[0]
    assert np.isclose(actual[0],3.5/69) and np.isclose(actual[1],2.5/69)
    assert np.isclose(actual[127],.5/69)
