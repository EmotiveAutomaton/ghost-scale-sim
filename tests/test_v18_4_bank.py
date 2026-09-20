import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_4 import bank_data as B


def test_known_finite_bank_closure_and_observation_alias():
    # Two states have identical passive observations but different action responses.
    passive=np.array([[1.,.5,.5],[1.,.5,.5]])
    target=np.eye(2)
    mapping,check=B.linear_map(passive,target,1e-10)
    assert check['span_residual']==pytest.approx(.5)
    assert np.allclose(passive@mapping,[[.5,.5],[.5,.5]])
    controlled=np.column_stack((passive,np.eye(2)))
    mapping,check=B.linear_map(controlled,target,1e-10)
    for posterior in ([1.,0.],[0.,1.],[.3,.7]):
        assert np.allclose(np.array(posterior)@controlled@mapping,posterior,atol=1e-12)
    assert check['span_residual']<1e-12


def test_uniform_placebo_and_ill_conditioned_truncation():
    bank=np.array([[1.,.5,.5+1e-5],[1.,.5,.5-1e-5]])
    uniform=np.full((2,2),.5)
    for cutoff in (1e-10,1e-3):
        mapping,check=B.linear_map(bank,uniform,cutoff)
        assert np.allclose(bank@mapping,uniform,atol=1e-8)
    full,f=B.linear_map(bank,np.eye(2),1e-10)
    truncated,t=B.linear_map(bank,np.eye(2),1e-3)
    assert f['span_residual']<1e-8 and t['span_residual']>.49
    assert f['map_norm']>1e4 and t['rank']==1
    raw=(np.array([.5,.5])@bank+np.array([0.,0.,.1]))@full
    assert raw.min()<0
    repaired=B.repair(raw)
    assert np.all(repaired>0) and repaired.sum()==pytest.approx(1)
    assert not np.allclose(raw,repaired)
    with pytest.raises(ValueError):B.repair(np.array([np.nan,0.]))


def test_map_uses_only_the_declared_law_not_case_truth():
    from ghostscale.validation.soundingline.v18_4 import neural_data as D
    from ghostscale.validation.soundingline.v18_3 import world as W
    w=W.make_world(0,18040123)
    bank=np.concatenate([np.ones((len(W.STATES),1))]+[W.artifact_matrix(w,c) for c in D.TRAIN_QUERIES],axis=1)
    target=W.artifact_matrix(w,D.NEW_QUERIES[0])
    mapping,check=B.linear_map(bank,target,1e-10)
    posterior=np.arange(1,len(W.STATES)+1,dtype=float);posterior/=posterior.sum()
    assert np.allclose(posterior@bank@mapping,posterior@target,atol=1e-8)
    assert check['span_residual']<1e-8
