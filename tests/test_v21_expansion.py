from itertools import product
import numpy as np
from ghostscale.validation.soundingline.v21 import expansion as E
from ghostscale.validation.soundingline.v21.expansion_check import reference

def test_known_answers_and_nulls():
    assert all(E.controls().values())

def test_exhaustive_observation_patterns_and_independent_policy():
    x=np.tile([0,1,2,3],(16,1));y=np.array(list(product((0,1),repeat=4)))
    for budget,noise in product((1,2,4),(.05,.45,.5)):
        result,_=E.analyze(x,y,.88,noise,budget)
        for i in range(len(x)):
            ref=reference(x[i],y[i],.88,noise,budget)
            for reader in E.READERS:
                assert np.allclose(result[reader]['probability_one'][i],ref[reader]['probability_one'],atol=1e-12)
                assert np.array_equal(result[reader]['selected'][i],ref[reader]['selected'])
                if reader=='expected-gain':assert result[reader]['likelihood_evaluations'][i]==4

def test_query_outcomes_absent_from_policy_and_alias_not_double_counted():
    x=np.array([[0,1,2,3]*2]);y=np.array([[0,1,1,0]*2]);r,a=E.analyze(x,y,.9,.05,4)
    assert np.allclose(E.subset_posterior(a['likelihood'],(1,)),E.subset_posterior(a['likelihood'],(1,4)))
    assert r['expected-gain']['selected'][0,3]==0
