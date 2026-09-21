import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import rollout_transfer as T,rollout as R,missing_tool as M,local_world as L
from ghostscale.validation.soundingline.v18_3.io import file_digest


def test_controls_and_zero_mass_scores():
    assert all(T.controls().values())
    assert T.score(np.eye(8)[2],3)['infinite_loss_mass']==1
    assert T.score(np.eye(8)[2],2)['finite_loss_contribution']==0
    with pytest.raises(ValueError):T.score(np.ones(8),2)


def test_retrieval_ties_are_symmetric():
    q=(1,0,2,4,4,4);a=(1,0,2,0,4,4);b=(1,0,2,4,0,4)
    pa=np.arange(1.,9.);pb=pa[::-1]
    assert np.array_equal(T.retrieval({a:pa,b:pb},q),np.ones(8)/8)
    assert np.array_equal(T.retrieval({b:pb,a:pa},q),np.ones(8)/8)


def test_changed_oracle_preserves_undo_and_no_tool():
    for rule in M.RULES:
        assert T.oracle((1,0,2,3,5,4),rule)==2
        assert T.oracle((1,0,2,4,4,4),rule)==2
    assert T.oracle((1,0,2,3,4,4),'original')!=T.oracle((1,0,2,3,4,4),'presentation-tool')


def test_frozen_models_full_fixture(tmp_path):
    inputs=tmp_path/'inputs/models';inputs.mkdir(parents=True)
    rr=M.enumerate_rule(L.law(190964),'original');counts,direct=R.fit(rr[:32]);keys=sorted(direct)
    model=inputs/'190301-32.npz'
    np.savez(model,transition_counts=counts,direct_keys=np.array(keys),direct_counts=np.array([direct[q] for q in keys]))
    before=file_digest(model)
    cfg=dict(arms=list(T.ARMS),rules=list(M.RULES),lineages=[190965],training_draws=[190301],budgets=[32],input_files={'models/190301-32.npz':before})
    result=T.run(tmp_path,{'design':cfg},lambda **kw:None)
    assert result['fits']==0 and result['paths']==27648 and file_digest(model)==before
    assert all(r['infinite_loss_mass']==0 and r['squared_error']==0 for r in result['cells'] if r['arm']=='known-rule-oracle')
    wrong=[r for r in result['cells'] if r['arm']=='oracle-exact' and r['rule']=='presentation-tool' and r['subset']=='changed']
    assert wrong and all(r['infinite_loss_mass']==1 for r in wrong)
