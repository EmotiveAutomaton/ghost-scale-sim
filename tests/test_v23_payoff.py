from copy import deepcopy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import registry,uptake,payoff,pipeline_new


@pytest.mark.parametrize('stakes,coupled',[(0,False),(1,False),(0,True),(1,True)])
def test_every_bellman_equation_and_observed_policy(stakes,coupled):
    result=payoff.verify_policy(.25,stakes,8,coupled)
    assert result['state']=='passed' and result['states']==sum((t+1)**2 for t in range(9))
    v=deepcopy(registry.variants('N6')[0]);v['settings'].update(records=8,stakes=stakes,coupled=coupled)
    p,t=uptake.sample(v,'engineering-payoff',15,0)
    pred,detail=uptake.evaluate(v,p,t)
    assert pipeline_new.verify_case(v,p,t,pred,detail)<1e-10
    bad=deepcopy(pred);bad['payoff-optimal']=np.zeros_like(pred['payoff-optimal'])
    with pytest.raises(ValueError,match='mismatch'):pipeline_new.verify_case(v,p,t,bad,detail)


def test_terminal_optimal_forecast_and_no_free_purpose_benefit():
    table=payoff.policy(.25,0,0,False)
    post=payoff.beliefs(.25,0,0,0);_,forecast,value=payoff.utilities(post,0,False)
    assert table['expected_initial_payoff']==pytest.approx(float(value))
    assert np.isfinite(forecast).all() and forecast.sum()==pytest.approx(1)
