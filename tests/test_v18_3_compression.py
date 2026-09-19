import numpy as np
from ghostscale.validation.soundingline.v18_3 import compression as C


def test_partition_certificate_and_independent_loss():
    assert len(C.partitions(4))==15
    assert len(C.partitions(8))==4140
    assert len(set(C.partitions(8)))==4140
    unit=C.unit(100000,cell=3,split='pilot')
    ph=np.array(unit['history_probabilities']);old=np.array(unit['old_predictions'])
    assert np.isclose(ph.sum(),1) and np.allclose(old.sum(axis=1),1)
    by={(r['method'],r['cardinality']):r for r in unit['rows']}
    for k in (1,2,4,8):
        flat=by['exhaustive-flat',k];product=by['observation-product',k]
        assert flat['old_loss']<=product['old_loss']+1e-12
        assert abs(C.reference_loss(flat['code'],ph,old)-flat['old_loss'])<1e-12
    assert by['exhaustive-flat',8]['old_loss']<=by['exhaustive-flat',1]['old_loss']+1e-12


def test_compression_null_and_known_positive():
    codes,masks,cardinality=C.codebooks();ph=np.ones(8)/8
    null=np.tile([.3,.7],(8,1));loss,_=C.code_losses(masks,ph,null)
    assert np.max(loss)-np.min(loss)<1e-12
    truth=np.eye(8);loss,_=C.code_losses(masks,ph,truth)
    assert np.allclose(loss[cardinality==1],np.log(8))
    assert np.allclose(loss[cardinality==8],0)


def test_shared_noise_changes_joint_without_changing_marginals():
    from ghostscale.validation.soundingline.v18_3 import world as W,verify
    w=W.make_world(0,99991)
    histories,p,_,_,_=C.finite_histories(w)
    _,shared,_,_,_=C.finite_histories(dict(w,shared=True))
    assert not np.allclose(p,shared)
    for axis in range(3):
        mask=np.array([h[axis]==1 for h in histories])
        assert abs(p[mask].sum()-shared[mask].sum())<1e-12
    assert verify.check_unit(C.unit(99991,cell=1,split='pilot'))['distributions']==17
