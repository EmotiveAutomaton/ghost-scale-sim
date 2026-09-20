import copy
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_4 import purpose_alignment as A, runtime as R
from ghostscale.validation.soundingline.v18_3 import compression as C
from ghostscale.validation.soundingline.v18_3.io import write,file_digest


def test_paired_rotations_preserve_supplemental_content_and_uniform_information():
    rng=np.random.default_rng(422);targets=[rng.dirichlet(np.ones(4),8) for _ in range(3)]
    banks=A.portfolios(targets);seen=[]
    entropy=lambda x: -np.sum(x*np.log(x),axis=-1)
    for shift in range(1,8):
        bank=banks[f'shift{shift}'];seen.append(np.roll(np.arange(8),shift))
        assert np.array_equal(bank[0],targets[0])
        assert np.array_equal(np.concatenate(bank[1:],axis=1),np.roll(np.concatenate(targets[1:],axis=1),shift,axis=0))
        for original,changed in zip(targets[1:],bank[1:]):
            assert np.allclose(original.mean(0),changed.mean(0),atol=1e-15)
            assert abs(entropy(original).mean()-entropy(changed).mean())<1e-14
    for i in range(8):assert sorted(np.asarray(seen)[:,i])==[j for j in range(8) if j!=i]
    for x in banks['marginal'][1:]:assert np.array_equal(x,np.repeat(x[:1],8,axis=0))


def test_live_known_binary_target_and_uniform_placebo():
    histories=list(product((0,1),repeat=3));ph=np.ones(8)/8
    targets=[np.eye(2)[np.array(histories)[:,0]]]*3
    row=next(r for r in A.select(histories,targets,0) if r['cardinality']==2 and r['method']=='portfolio-aligned')
    assert C.reference_loss(row['code'],ph,targets[0])==0
    uniform=[np.ones((8,4))/4]*3
    assert all(abs(r['training_loss']-np.log(4))<1e-12 for r in A.select(histories,uniform,0))


def test_future_blindness_extreme_cardinalities_and_corruption():
    unit=A.unit(422,cell=7);assert A.verify(unit)
    for k in (1,8):
        values=[r['new_loss'] for r in unit['rows'] if r['cardinality']==k]
        assert max(values)-min(values)<1e-12
    changed=copy.deepcopy(unit);changed['future_predictions'][0]=np.ones((8,16)).tolist()
    assert A.select(changed['histories'],changed['training_predictions'],422)==A.select(unit['histories'],unit['training_predictions'],422)
    with pytest.raises(ValueError):A.verify(changed)
    changed=copy.deepcopy(unit);changed['history_probabilities'][0]+=.01
    with pytest.raises(ValueError):A.verify(changed)
    changed=copy.deepcopy(unit);changed['rows'][0]['training_loss']+=.01
    with pytest.raises(ValueError):A.verify(changed)


def test_finite_aggregate_independent_means_and_exact_dispatch_replay(tmp_path):
    from runners.replay_v18_4 import finite
    spec=dict(family='P2',index=423,cell=0,rule='lexicographic')
    write(tmp_path/'PLAN.json',dict(design=dict(units=[spec],block_size=1)))
    unit=R.dispatch(spec);R.keep(tmp_path,'block-000000',[unit],0.,0.)
    summary=R.aggregate(tmp_path)
    write(tmp_path/'COMPLETE.json',dict(blocks=['block-000000'],summary_sha256=file_digest(tmp_path/'SUMMARY.json')))
    proof=finite(tmp_path)
    assert proof['independent_means']==len(summary['cells'])*len(A.METRICS)
    assert proof['whole_unit_replays']==[0]
    summary['cells'][next(iter(summary['cells']))]['new_loss']['mean']+=1
    write(tmp_path/'SUMMARY.json',summary,immutable=False)
    with pytest.raises(ValueError):finite(tmp_path)
