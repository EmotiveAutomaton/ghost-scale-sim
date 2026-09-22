"""Independent selection, score-floor, pairing and native corruption controls."""
from itertools import product
import gzip
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import compression_review as V
from ghostscale.validation.soundingline.v19 import mixture_compression as C
from ghostscale.validation.soundingline.v19 import unknown_change as U, source_omission as O, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_control_verdicts_are_native_json_booleans():
    verdicts=V.controls()
    assert all(type(v) is bool for v in verdicts.values())
    assert canonical(verdicts)


def test_scalar_projection_ties_and_zero_mass_floor():
    assert all(V.controls().values())
    hs=[('none',0,0),('purpose',16,0),('skill',16,1)]
    weights=[.5,.3,.2];ix,w,removed=V.selection(weights,'2')
    assert ix==[0,1] and removed==pytest.approx(.2)
    for step,expected in ((16,0),(17,8)):
        states=V.current_indices(hs,step);p=V.projection(states,ix,w)
        assert p[expected]>=.375
        assert p[0]==pytest.approx(1. if step==16 else .625)
    assert V.selection([.5,.5],'1')[0]==[0]
    assert V.projection([7,2],*V.selection([.5,.5],'1')[:2])[7]==1
    law=np.random.default_rng(190969).dirichlet(np.ones(8),size=(16,4))
    p=V.projection(V.current_indices(hs,17),ix,w);s,f=V.metrics(law,p,0)
    assert np.allclose(f,.625*law[0]+.375*law[8],atol=1e-15,rtol=0)
    assert s['endpoint_loss']==pytest.approx(-math.fsum(law[0,c,e]*math.log(f[c,e]) for c in range(4) for e in range(8))/4,abs=1e-14)
    for w,c in (([.4,.4],'1'),([-1,2],'1'),([float('nan'),1],'1'),([.5,.5],'0'),([.5,.5],'3')):
        with pytest.raises(ValueError): V.selection(w,c)


def test_complete_paired_draw_regroup():
    cells=[]
    for l,d,e,c in product((11,12),(101,102),V.EVIDENCE,V.COUNTS):
        value=l+d+V.COUNTS.index(c)+(2 if e=='omitted' else 0)
        cells.append(dict(lineage=l,draw=d,evidence=e,count=c,length=32,kind='purpose',switched=True,duplicates=True,step=32,**{m:value for m in V.MEASURES}))
    cfg=dict(bootstrap_seed=190501,bootstrap_resamples=100)
    got=V.regroup(cells,cfg)
    assert len(got['means'])==10 and len(got['contrasts'])==135
    for r in got['contrasts']:
        delta=V.COUNTS.index(r['count'])-V.COUNTS.index(r['baseline_count'])+(2 if r['evidence']=='omitted' else 0)-(2 if r['baseline_evidence']=='omitted' else 0)
        assert r['mean']==r['low']==r['high']==delta and r['draw_means']==[delta,delta]
    with pytest.raises(ValueError,match='duplicate'):V.regroup(cells+[cells[0]],cfg)
    with pytest.raises(KeyError):V.regroup(cells[:-1],cfg)


def test_complete_native_fixture_and_corruption(tmp_path):
    lineage=190968;records=L.enumerate_world(L.law(lineage))
    parent=tmp_path/'parent';p=parent/'inputs'/f'lineage-{lineage}_points.json.gz';p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32,128],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    write(parent/'SUMMARY.json',U.run(parent,{'design':cfg},lambda **kw:None))
    omitted=tmp_path/'omitted';(omitted/'inputs').mkdir(parents=True)
    shutil.copyfile(p,omitted/'inputs'/p.name);shutil.copytree(parent,omitted/'inputs/parent')
    cfg['input_files']={q.relative_to(omitted/'inputs').as_posix():file_digest(q) for q in (omitted/'inputs').rglob('*') if q.is_file()}
    write(omitted/'SUMMARY.json',O.run(omitted,{'design':cfg},lambda **kw:None))
    original=tmp_path/'review/inputs/original'
    for name,source in (('aware',parent),('omitted',omitted)):
        for folder in ('raw','evaluator'):shutil.copytree(source/folder,original/'inputs'/name/folder)
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32,128],retained_counts=list(C.COUNTS),input_files={})
    write(original/'PLAN.json',dict(design=cfg))
    write(original/'SUMMARY.json',C.run(original,{'design':cfg},lambda **kw:None))
    review=tmp_path/'review';plan=dict(design=dict(input_files={},target_plan_sha256=file_digest(original/'PLAN.json'),bootstrap_seed=190501,bootstrap_resamples=20))
    got=V.verify(review,plan,lambda **kw:None)
    assert got['rows']==14080 and got['cells']==880 and got['max_error']<1e-10
    assert got['parent_identity_rows']==got['current_marginal_controls']==2816
    assert got['independent_source_identity_rows']==3520
    grouped=read(review/'INDEPENDENT_REGROUP.json')
    assert len(grouped['means'])==880 and len(grouped['contrasts'])==11880
    # Change a retained index while preserving valid int32 storage: numerical verification must fail.
    path=original/'raw'/f'{lineage}-selection_points.npz'
    with np.load(path,allow_pickle=False) as z: arrays={n:z[n] for n in z.files}
    name=sorted(arrays)[0];arrays[name][0,0]=(arrays[name][0,0]+1)%560
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='selected indices'):V.verify(review,plan,lambda **kw:None)
