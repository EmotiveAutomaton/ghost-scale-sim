"""Known-answer lossless grouping and intentionally destructive merging."""
import numpy as np
import pytest
import gzip
import shutil
from ghostscale.validation.soundingline.v19 import future_quotient as Q
from ghostscale.validation.soundingline.v19 import local_world as L, unknown_change as U
from ghostscale.validation.soundingline.v18_3.io import canonical, write, file_digest


def test_live_null_controls():
    assert all(Q.controls().values())


def test_future_schedule_is_not_current_state():
    hs=[('none',0,0),('purpose',8,0),('skill',9,0),('purpose',8,8)]
    for t in (8,9,10):
        s,i=Q.quotient(hs,t,10)
        assert np.array_equal(s[i],Q.independent_schedule(hs,t,10))
    s,i=Q.quotient(hs,8,10)
    assert len(s)==4
    bad=i.copy();bad[1]=bad[0]
    with pytest.raises(ValueError,match='schedule mapping'):
        Q.verify_forecasts(hs,8,10,[[.25]*4],s,bad,np.full((16,4,8),.125))


def test_scalar_full_hypothesis_reference_and_zero_mass():
    hs=[('none',0,0),('purpose',8,8),('purpose',9,0),('skill',9,0)]
    law=np.zeros((16,4,8));law[:,:,0]=1.;law[8,:,0]=0.;law[8,:,1]=1.;law[4,:,0]=0.;law[4,:,2]=1.
    s,i=Q.quotient(hs,9,10);w=np.array([.25,.25,.5,0.])
    grouped,p,e=Q.verify_forecasts(hs,9,10,[w],s,i,law)
    direct=np.array([sum((w[j]*law[Q.independent_schedule([h],t,t)[0,0]] for j,h in enumerate(hs)),np.zeros((4,8))) for t in (9,10)])
    assert np.array_equal(p[0],direct) and e.max()==0
    assert len(grouped[0])==3 and len(s)==3  # zero-weight schedule retained
    for bad in ([.1,.1,.1,.1],[-.1,.1,.5,.5],[float('nan'),0.,0.,1.]):
        with pytest.raises(ValueError):Q.weights_by_group(bad,i,len(s))
    with pytest.raises(ValueError):Q.quotient(hs,11,10)


def test_complete_native_parent_missing_checkpoint_and_corruption(tmp_path):
    lineage=190966;parent=tmp_path/'parent'
    p=parent/'inputs'/f'lineage-{lineage}_points.json.gz';p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(canonical(L.enumerate_world(L.law(lineage))),mtime=0))
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    write(parent/'SUMMARY.json',U.run(parent,{'design':cfg},lambda **kw:None))
    root=tmp_path/'quotient'
    for evidence in ('aware','omitted'):
        for folder in ('raw','evaluator'):shutil.copytree(parent/folder,root/'inputs'/evidence/folder)
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32],checkpoints=[31,32],input_files={})
    summary=Q.run(root,{'design':cfg},lambda **kw:None)
    assert all(summary['controls'].values()) and summary['rows']==256
    assert len(summary['unavailable_checkpoints'])==2 and summary['forecast_vectors']==1024
    assert all(r['components']==16 and r['original_components']==560 for r in summary['cells'])
    path=root/'inputs/aware/raw'/f'{lineage}-forecasts_points.json.gz'
    rows=__import__('json').loads(gzip.decompress(path.read_bytes()))
    next(r for r in rows if r['arm']=='unknown-time-type' and r['step']==32)['forecast'][0][0]+=.1
    path.write_bytes(gzip.compress(canonical(rows),mtime=0))
    corrupt=tmp_path/'corrupt';shutil.copytree(root/'inputs',corrupt/'inputs')
    with pytest.raises(ValueError,match='parent forecast identity'):
        Q.run(corrupt,{'design':cfg},lambda **kw:None)
