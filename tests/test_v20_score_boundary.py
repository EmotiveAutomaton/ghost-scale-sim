"""Scores must describe the saved forecast, including hard confidence cutoffs."""
import numpy as np
import pytest
from ghostscale.validation.soundingline.v20.readers import score
from ghostscale.validation.soundingline.v20.checker import reaggregate

@pytest.mark.parametrize('confidence',[np.nextafter(.9,0),.9,np.nextafter(.9,1)])
def test_saved_forecast_confidence_boundary(confidence):
    p=np.zeros((1,128));p[0,0]=confidence;p[0,1]=.1
    y=np.array([1]);before=p.copy();result=score(p,y)
    assert np.array_equal(p,before)
    assert result['confidence'][0]==confidence
    assert result['unsupported_attribution'][0]==float(confidence>=.9)
    assert reaggregate(p,y,result)['rows']==1

@pytest.mark.parametrize('values',[(.3,.3),(-.1,1.1),(np.nan,1)])
def test_score_refuses_unnormalized_or_invalid_forecast(values):
    p=np.zeros((1,128));p[0,:2]=values
    with pytest.raises(ValueError,match='normalized forecast'):score(p,np.array([0]))
