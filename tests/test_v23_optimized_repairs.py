import numpy as np
from ghostscale.validation.soundingline.v23 import repairs,checker,pilot


def test_vectorized_r3_against_independent_scalar_posterior_at_largest_cell():
    v=pilot.largest('R3');p,t=repairs.sample(v,'optimization-control',15,3)
    for role in (None,0.,.5,1.):
        for weight in ('unit','effort','tempered'):
            a=repairs.r3_infer(v,p,role,weight);b=checker.r3(v,p,role,weight)
            for first,second in zip(a,b):np.testing.assert_allclose(first,second,atol=1e-12,rtol=0)


def test_vectorized_tree_against_scalar_probability_recursion():
    v=pilot.largest('R4');p,t=repairs.sample(v,'optimization-control',15,3)
    for settings in ({},{'cue':False},{'topdown':True},{'marginal_rho':True},{'fixed_trust':True},{'law':15}):
        a=checker.r4(v,p,**settings);b=checker.r4_grid(v,p,**settings)
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
    predictions,details=repairs.evaluate(v,p,t)
    assert checker.verify_r4(v,p,t,predictions,details)<1e-12
