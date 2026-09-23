"""Third centered-law regroup against full scalar enumeration."""
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.forecast_collision_review import sufficient
from ghostscale.validation.soundingline.v19.forecast_collision_regroup import resample
from test_v19_forecast_collision import scalar


@pytest.mark.parametrize('equal',[False,True])
def test_direct_centered_regroup(equal):
    rng=np.random.default_rng(929);p=rng.dirichlet([1,2,3],(6,3));p[1]=p[0];p[3]=p[2]
    t=rng.dirichlet([3,2,1],(3,6,3));w=rng.dirichlet(np.ones(6),3)
    if equal:w[:]=1/6
    w[:,5]=0;w/=w.sum(1)[:,None]
    counts=np.array([n for n in product(range(4),repeat=3) if sum(n)==3]);s=sufficient(p,t,w)
    values=resample(s,counts,chunk=2)
    for v,n in zip(values,counts):assert np.allclose(v,scalar(p,t,w,n),rtol=0,atol=1e-14)
    with pytest.raises(ValueError):resample(s,[[1,1,0]])


def test_direct_regroup_opposing_targets():
    p=np.tile([.5,.5,0.],(2,3,1));t=np.array([p.copy(),p.copy()]);t[0,:,:,:2]=[1,0];t[1,:,:,:2]=[0,1]
    v=resample(sufficient(p,t,[[.5,.5],[.5,.5]]),[[1,1],[2,0]])
    assert np.allclose(v[0,3,:,:2],.25) and np.max(abs(v[1,3]))<1e-14
    assert np.allclose(v[1,1,:,:2],.25) and np.max(abs(v[0,1]))<1e-14
