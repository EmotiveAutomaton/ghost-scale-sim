"""Independent membership, mass, forecasting and corruption controls."""
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import quotient_review as R


def fixture():
    hs=[['none',0,0],['purpose',8,8],['purpose',9,0],['skill',9,0]]
    law=np.zeros((16,4,8));law[:,:,0]=1.;law[8,:,0]=0.;law[8,:,1]=1.;law[4,:,0]=0.;law[4,:,2]=1.
    W=np.array([[.25,.25,.5,0.]])
    # Lexicographic schedules: (0,0), (0,4), (0,8).
    weights=np.array([[.5,0.,.5]])
    forecasts=np.zeros((1,2,4,8));forecasts[:,0,:,0]=1.;forecasts[:,1,:,0]=.5;forecasts[:,1,:,1]=.5
    return hs,law,W,weights,forecasts,np.zeros((1,2))


def test_known_scalar_forecasts_and_zero_weight():
    hs,law,w,q,p,e=fixture()
    out=R.audit_arrays(hs,9,10,w,law,q,p,e)
    assert out['components']==3 and out['forecast_error']==out['weight_error']==0


@pytest.mark.parametrize('kind',['weight','forecast','error','shape'])
def test_corruption_fails(kind):
    hs,law,w,q,p,e=fixture()
    if kind=='weight':q[0,1]=.1
    elif kind=='forecast':p[0,1,0,0]+=.01
    elif kind=='error':e[0,0]=.1
    else:p=p[:,:1]
    with pytest.raises(ValueError):R.audit_arrays(hs,9,10,w,law,q,p,e)


def test_roster_stationary_and_transition_boundary():
    assert len(R.roster(32))==560 and len(R.roster(128))==3632
    assert R.schedule(['purpose',8,0],8,10)==(0,8,8)
    sig,mapping,groups=R.partition([['none',0,m] for m in range(16)],8,10)
    assert mapping==list(range(16)) and len(sig)==16 and groups==[[m] for m in range(16)]
    assert [len(R.partition(R.roster(32),t,32)[0]) for t in (8,16,17,20,32)]==[560,304,272,176,16]


def test_uniform_law_and_wrong_current_only_merge():
    hs,law,w,q,p,e=fixture()
    uniform=np.full_like(law,.125);p[:]=.125
    assert R.audit_arrays(hs,9,10,w,uniform,q,p,e)['forecast_error']==0
    wrong=q.copy();wrong[0]=[1.,0.,0.]
    with pytest.raises(ValueError,match='grouped weights'):R.audit_arrays(hs,9,10,w,uniform,wrong,p,e)


def test_complete_native_parent_and_mapping_corruption(tmp_path):
    import gzip
    import shutil
    from ghostscale.validation.soundingline.v19 import local_world as L, unknown_change as U, future_quotient as Q
    from ghostscale.validation.soundingline.v18_3.io import canonical, write, file_digest
    lineage=190966;parent=tmp_path/'parent'
    p=parent/'inputs'/f'lineage-{lineage}_points.json.gz';p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(canonical(L.enumerate_world(L.law(lineage))),mtime=0))
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    write(parent/'SUMMARY.json',U.run(parent,{'design':cfg},lambda **kw:None))
    root=tmp_path/'quotient'
    for evidence in ('aware','omitted'):
        for folder in ('raw','evaluator'):shutil.copytree(parent/folder,root/'inputs'/evidence/folder)
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32],checkpoints=[31,32],input_files={})
    write(root/'PLAN.json',{'design':cfg});write(root/'SUMMARY.json',Q.run(root,{'design':cfg},lambda **kw:None))
    result=R.review(root,tmp_path/'review')
    assert result['passed'] and result['rows']==256 and result['paired_source_rows']==128
    assert len(result['unavailable_checkpoints'])==2
    path=root/'evaluator/SCHEDULES.json';meta=R.read(path);meta['32-32']['membership'][0]=1;path.unlink();write(path,meta)
    with pytest.raises(ValueError,match='schedule membership'):R.review(root,tmp_path/'bad-review')
