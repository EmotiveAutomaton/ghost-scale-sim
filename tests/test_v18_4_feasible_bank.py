import copy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_4 import feasible_bank as F
from ghostscale.validation.soundingline.v18_3 import world as W


def test_known_projection_and_placebo():
    assert all(F.controls().values())
    atoms=np.array([[0.,0.],[1.,0.],[0.,1.]])
    weights,c=F.project(atoms,[1.,1.])
    assert c['valid'] and np.allclose(weights@atoms,[.5,.5],atol=1e-6)
    assert c['residual']==pytest.approx(.5,abs=1e-8)
    bad=F.certificate(atoms,np.array([1.,1.]),np.array([1.,0.,0.]))
    assert bad['gap']>1e-7


def test_inconsistent_native_banks_require_certificate_not_solver_status():
    w=W.make_world(0,18049000);r=W.rng('v18.4-hull-timing',0)
    F.D.history(w,F.D.NEURAL_STATES[0],r,8)
    atoms=np.concatenate([W.artifact_matrix(w,c) for c in F.D.TRAIN_QUERIES],axis=1)
    for i in range(12):
        target=r.dirichlet([.5]*16,5).reshape(-1);weights,check=F.project(atoms,target)
        assert check['valid'] and check['gap']<=1e-7


def test_alias_weights_need_not_identify_future_and_coordinate_count():
    # Old predictions identical, future predictions differ: a solver cannot
    # identify which latent atom generated the bank from this input alone.
    atoms=np.full((2,80),1/16);weights,c=F.project(atoms,np.full(80,1/16))
    assert c['valid'] and np.allclose(weights@atoms,atoms[0])
    target=np.eye(2)
    assert not np.array_equal(np.array([1.,0.])@target,np.array([0.,1.])@target)
    assert atoms.shape[1]==80


def fixture():
    w=W.make_world(2,18049999);state=F.D.NEURAL_STATES[5]
    h=F.D.history(w,state,W.rng('v18.4-hull-control'),8)
    bank=np.stack([W.artifact_matrix(w,c)[state] for c in F.D.TRAIN_QUERIES])
    return dict(index=99,cell=2,support='fixture',scipy_version=F.scipy.__version__,
        payload=dict(world=w,history=h,state=state,banks={'flat-seed701':bank.tolist()}))


def test_native_scalar_certificate_scores_and_corruption():
    spec=fixture();u=F.unit(**spec);assert F.verify(u)
    broken=copy.deepcopy(u);broken['checks']['flat-seed701']['weights'][0]+=.1
    with pytest.raises(ValueError):F.verify(broken)
    broken=copy.deepcopy(u);broken['forecasts']['flat-seed701|hull'][0][0]+=.1
    with pytest.raises(ValueError):F.verify(broken)
    broken=copy.deepcopy(u);broken['rows'][0]['expected_loss']+=.1
    with pytest.raises(ValueError):F.verify(broken)


def test_evaluator_state_noninterference_and_exact_closure():
    spec=fixture();first=F.unit(**spec);spec['payload']['state']=F.D.NEURAL_STATES[6]
    second=F.unit(**spec)
    assert first['forecasts']==second['forecasts']
    assert first['rows']!=second['rows']
    assert first['checks']['flat-seed701']['residual']<1e-8
    w=spec['payload']['world'];p=np.ones(len(W.STATES))/len(W.STATES)
    p=W.posterior(W.packet(w,spec['payload']['history']))
    bank=np.concatenate([p@W.artifact_matrix(w,c) for c in F.D.TRAIN_QUERIES])
    atoms=np.concatenate([W.artifact_matrix(w,c) for c in F.D.TRAIN_QUERIES],axis=1)
    weights,check=F.project(atoms,bank)
    assert check['valid'] and np.max(abs(weights@atoms-bank))<1e-5
