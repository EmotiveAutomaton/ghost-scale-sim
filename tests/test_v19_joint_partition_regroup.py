from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import joint_forecast_partition as J
from ghostscale.validation.soundingline.v19 import joint_partition_regroup as R
from test_v19_joint_forecast_partition import scalar


@pytest.mark.parametrize('equal',[True,False])
@pytest.mark.parametrize('kind',['joint','marginal'])
def test_independent_cross_moments_against_all_scalar_resamples(equal,kind):
    rng=np.random.default_rng(349);q=rng.dirichlet(np.ones(27),6);q[2]=q[0];q[4]=q[1]
    p=q if kind=='joint' else np.stack([q@(J.D.GOALS[:,t,None]==np.arange(3)) for t in range(3)],axis=1)
    target=rng.dirichlet(np.ones(27),(3,6));weight=rng.dirichlet(np.ones(6),3)
    if equal:weight[:]=1/6
    weight[:,5]=0;weight/=weight.sum(1)[:,None]
    counts=np.array([n for n in product(range(4),repeat=3) if sum(n)==3]);s=J.sufficient(p,kind,target,weight)
    values=R.resample(s,counts,chunk=2)
    for n,v in zip(counts,values):assert np.allclose(v,scalar(p,kind,target,weight,n),atol=1e-14,rtol=0)
    assert np.allclose(values,J.resample(s,counts),atol=1e-14,rtol=0)
