import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import interchange_alignment as A


def test_known_answer_and_placebo_controls():
    assert all(A.controls().values())


def test_probe_normal_equations_and_known_factor():
    x=np.array([[-1.,-2.],[-1.,2.],[1.,-2.],[1.,2.]])
    beta,bias,receipt=A.fit_probe(x,x[:,0])
    assert receipt['normal_residual']<1e-12 and abs(bias)<1e-12
    assert np.allclose(A.unit(beta),[1,0])


def test_shuffled_and_wrong_labels_do_not_recover_target():
    x=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    right=A.unit(A.fit_probe(x,x[:,0])[0]);wrong=A.unit(A.fit_probe(x,x[:,1])[0])
    assert abs(right@wrong)<1e-12
    assert np.allclose(A.swap(x[[0]],x[[3]],wrong),[[-1,1]])


def test_swap_preserves_orthogonal_nuisance_and_stay():
    recipient=np.array([[2.,5.],[2.,5.]]);donor=np.array([[7.,9.],[2.,9.]])
    assert np.array_equal(A.swap(recipient,donor,np.array([1.,0.])),[[7.,5.],[2.,5.]])
    assert np.array_equal(A.swap(recipient,recipient,np.array([1.,0.])),recipient)


def test_zero_direction_rejected():
    with pytest.raises(ValueError):A.unit(np.zeros(2))


def test_corrupt_inputs_rejected_before_fit(tmp_path):
    with pytest.raises(ValueError,match='corrupted'):
        p=tmp_path/'inputs'/'x';p.parent.mkdir();p.write_text('bad')
        A.run(tmp_path,{'design':{'input_files':{'x':'0'*64}}},lambda **kw:None)


def test_complete_synthetic_pipeline_and_selective_stay(tmp_path):
    from itertools import product
    from ghostscale.validation.soundingline.v18_3.io import write, file_digest
    from ghostscale.validation.soundingline.v19.readout_model import save_arrays
    source=tmp_path/'inputs';source.mkdir()
    write(source/'INDEPENDENT_REVIEW.json',dict(passed=True,per_fit_capability=[dict(passed=True)]))
    makers=np.array(list(product((0,1),repeat=4)));state=2.*makers-1
    weights=np.random.default_rng(42).normal(size=(32,4))*.1
    params={'head.weight':weights,'head.bias':np.zeros(32)}
    save_arrays(source/'inputs/models/10-20-transformer.npz',**params)
    for split,lineage in [('train',0),('development',1)]:
        save_arrays(source/'forecasts'/f'{split}-{lineage}-10-20-transformer.npz',hidden=np.repeat(state[:,None,:],32,axis=1))
        save_arrays(source/'evaluator'/f'{split}-{lineage}-10.npz',maker=makers,maker_indices=np.arange(16))
    save_arrays(source/'evaluator/development-1-laws.npz',laws=A.decode(params,state))
    cfg=dict(input_files={p.relative_to(source).as_posix():file_digest(p) for p in source.rglob('*') if p.is_file()},
        train_lineages=[0],development_lineages=[1],training_draws=[10],fit_seeds=[20],kinds=['transformer'],lengths=[8,32])
    result=A.run(tmp_path,{'design':cfg},lambda **kw:None)
    assert all(result['controls'].values()) and result['rows']==896 and result['probes']==2
    with np.load(tmp_path/'forecasts/1-10-20-transformer-32-0-0.npz') as z:
        assert np.allclose(z['probabilities'][A.ARMS.index('unchanged')],z['probabilities'][A.ARMS.index('learned')])
    with np.load(tmp_path/'forecasts/1-10-20-transformer-32-0-1.npz') as z:
        hybrid=makers.copy();hybrid[:,0]=1-hybrid[:,0]
        assert np.allclose(z['probabilities'][A.ARMS.index('learned')],A.decode(params,2.*hybrid-1))
