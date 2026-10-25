"""Production resource and completeness guards; no scientific draws."""
from datetime import datetime,timezone
import pytest
from ghostscale.validation.soundingline.v24 import production,n16
from ghostscale.validation.soundingline.v23.common import write

def test_complete_block_cannot_be_replaced_with_partial_or_duplicate_cells():
    plan=dict(block_index=7,kind='production',science_identity=production.IDENTITY)
    rows=[dict(identity=[7,l,v['id'],c],mixture_gain=.1) for l in range(16) for v in n16.model.variants() for c in range(5)]
    assert production.block_summary(rows,plan)['cases']==3840
    with pytest.raises(ValueError,match='incomplete'):production.block_summary(rows[:-1],plan)
    with pytest.raises(ValueError,match='duplicate'):production.block_summary(rows+[rows[0]],plan)

def test_production_requires_excluded_replay_and_shared_slot(tmp_path):
    now=datetime(2026,10,25,20,tzinfo=timezone.utc);write(tmp_path/'ACCEPTANCE.json',production.acceptance(now.isoformat()))
    (tmp_path/'accounting').mkdir()
    plan=dict(accounting_root=str(tmp_path),family='N16',kind='production',epsilon=.1,cases_per_cell=5,science_identity=production.IDENTITY,
        block_index=0,variant_ids=[v['id'] for v in n16.model.variants()],numerical_threads=1,priority='below-normal',cpu_cap_seconds=1000.,measured_wall_seconds=100.)
    with pytest.raises(ValueError,match='excluded'):production.admission(plan,now,0)
    write(tmp_path/'EXCLUDED_VERIFIED.json',{'fixture':True})
    assert production.admission(plan,now,1)
    with pytest.raises(ValueError,match='slot'):production.admission(plan,now,2)
    write(tmp_path/'accounting/used.json',{'cpu_seconds':64*3600-999})
    with pytest.raises(ValueError,match='ceiling'):production.admission(plan,now,0)

def test_confirmatory_phase_cannot_reuse_excluded_identity(tmp_path):
    write(tmp_path/'ACCEPTANCE.json',production.acceptance('2026-10-25T20:00:00+00:00'))
    plan=dict(accounting_root=str(tmp_path),family='N16',kind='production',epsilon=.1,cases_per_cell=5,science_identity=production.EXCLUDED,block_index=0)
    with pytest.raises(ValueError,match='identity'):production.validate(plan)

def test_exact_cache_does_not_alias_changed_likelihood_inputs():
    import numpy as np
    prior=np.ones((2,16))/32;law=np.full((2,2,16,2),.5)
    expected=n16.model.distributions(prior,law,[0,1],'correct')
    with production.repeated_distribution_cache():
        got=n16.model.distributions(prior,law,[0,1],'correct')
        np.testing.assert_array_equal(got[0],expected[0]);np.testing.assert_array_equal(got[1],expected[1])
        law[1,0,:,0]=.9;law[1,0,:,1]=.1;law[1,1,:,0]=.1;law[1,1,:,1]=.9
        changed=n16.model.distributions(prior,law,[0,1],'correct')
        assert changed[1][1]>0 and expected[1][1]==0
