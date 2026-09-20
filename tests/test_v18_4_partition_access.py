import copy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_4 import partition_access as P, purpose_alignment as A, runtime as R
from ghostscale.validation.soundingline.v18_3.io import write,file_digest


def test_label_invariance_and_known_distinct_live_partitions():
    a=[0,0,0,0,1,1,1,1];relabeled=[9,9,9,9,4,4,4,4];b=[0,1,0,1,0,1,0,1]
    target=np.eye(2)[a]
    same=P.compare(a,relabeled,[target]);different=P.compare(a,b,[target])
    assert same['same_partition']==same['same_forecasts']==1
    assert same['pair_disagreement']==same['loss_difference']==0
    assert different['same_partition']==different['same_forecasts']==0
    assert different['forecast_max_difference']==.5
    assert different['aligned_loss']==0
    assert abs(different['comparison_loss']-np.log(2))<1e-12


def test_uniform_placebo_and_equal_score_without_equal_forecasts():
    a=[0,0,0,0,1,1,1,1];b=[0,1,0,1,0,1,0,1]
    uniform=P.compare(a,b,[np.ones((8,2))/2])
    assert uniform['same_partition']==0 and uniform['same_forecasts']==1
    assert uniform['loss_difference']==0
    target=(np.eye(2)[a]+np.eye(2)[b])/2
    tied=P.compare(a,b,[target])
    assert tied['same_forecasts']==0 and abs(tied['loss_difference'])<1e-12


def test_parent_reproduction_scalar_verification_and_corruption():
    parent=A.unit(471,cell=0);data=P.unit(471,0,None,parent)
    assert P.verify(data)
    bad=copy.deepcopy(parent);bad['rows'][0]['new_loss']+=.1
    # Choose a used portfolio row, not an irrelevant historical selector.
    used=next(r for r in bad['rows'] if r['method']=='portfolio-aligned');used['new_loss']+=.1
    with pytest.raises(ValueError,match='parent number'):P.unit(471,0,None,bad)
    bad=copy.deepcopy(data);bad['rows'][0]['same_partition']=0
    with pytest.raises(ValueError):P.verify(bad)
    bad=copy.deepcopy(parent);bad['future_predictions'][0][0][0]+=.1
    with pytest.raises(ValueError):P.unit(471,0,None,bad)


def test_finite_dispatch_aggregation_independent_replay_and_corruption(tmp_path):
    from runners.replay_v18_4 import finite
    spec=dict(family='P3',index=472,cell=3,rule=None,payload=A.unit(472,cell=3))
    write(tmp_path/'PLAN.json',dict(design=dict(units=[spec],block_size=1)))
    data=R.dispatch(spec);R.keep(tmp_path,'block-000000',[data],0.,0.)
    summary=R.aggregate(tmp_path)
    write(tmp_path/'COMPLETE.json',dict(blocks=['block-000000'],summary_sha256=file_digest(tmp_path/'SUMMARY.json')))
    proof=finite(tmp_path)
    assert proof['independent_means']==32*len(P.METRICS)
    assert proof['whole_unit_replays']==[0]
    summary['cells'][next(iter(summary['cells']))]['same_partition']['mean']+=1
    write(tmp_path/'SUMMARY.json',summary,immutable=False)
    with pytest.raises(ValueError,match='mean differs'):finite(tmp_path)
