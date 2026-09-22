"""Finite ambiguity, mapping and retained-parent checks for checkpoint compression."""
import gzip
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import mixture_compression as C
from ghostscale.validation.soundingline.v19 import unknown_change as U, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_known_ambiguity_zero_mass_and_tie_controls():
    assert all(C.controls().values())
    hs=[('none',0,2),('none',0,7)]
    ix,w,_=C.truncate([.5,.5],1)
    assert C.project(hs,1,ix,w)[2]==1
    # Exact ties depend on roster identity, not truth or a hidden tie chooser.
    assert C.project(hs[::-1],1,ix,w)[7]==1
    assert C.truncate([.8,.2,0],2)[2]==0
    assert np.array_equal(C.truncate([.2,.8], 'all')[1],[.2,.8])


def test_scalar_projection_and_scores_across_change_boundary():
    hs=[('none',0,0),('purpose',16,0),('skill',16,1)]
    weights=np.array([.5,.3,.2]);ix,w,removed=C.truncate(weights,2)
    law=np.random.default_rng(190966).dirichlet(np.ones(8),size=(16,4))
    assert removed==pytest.approx(.2)
    for step in (16,17):
        expected=np.zeros(16)
        for j in ix:
            kind,t,m=hs[j]
            mm=int(U.FLIPS[kind][m]) if kind!='none' and step>t else m
            expected[mm]+=weights[j]/.8
        current=C.project(hs,step,ix,w)
        assert np.allclose(current,expected,rtol=0,atol=1e-15)
        values,forecast=C.T.score(law,current,0)
        scalar=np.array([[math.fsum(expected[m]*law[m,c,e] for m in range(16)) for e in range(8)] for c in range(4)])
        assert np.allclose(forecast,scalar,rtol=0,atol=1e-15)
        expected_loss=-math.fsum(law[0,c,e]*math.log(max(scalar[c,e],1e-300)) for c in range(4) for e in range(8))/4
        assert values['endpoint_loss']==pytest.approx(expected_loss,abs=1e-14)


@pytest.mark.parametrize('weights,count', [([.4,.4],1),([-1,2],1),([float('nan'),1],1),([.5,.5],0),([.5,.5],3)])
def test_bad_weights_and_capacity_rejected(weights,count):
    with pytest.raises(ValueError):C.truncate(weights,count)


def test_complete_native_parent_fixture_and_corruption(tmp_path):
    lineage=190966;records=L.enumerate_world(L.law(lineage))
    parent=tmp_path/'parent';p=parent/'inputs'/f'lineage-{lineage}_points.json.gz';p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32,128],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    write(parent/'SUMMARY.json',U.run(parent,{'design':cfg},lambda **kw:None))
    root=tmp_path/'compression'
    # Both roles deliberately share a native parent here; this must be an identity.
    for evidence in ('aware','omitted'):
        for folder in ('raw','evaluator'):shutil.copytree(parent/folder,root/'inputs'/evidence/folder)
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32,128],retained_counts=list(C.COUNTS),input_files={})
    summary=C.run(root,{'design':cfg},lambda **kw:None)
    assert all(summary['controls'].values()) and summary['rows']==14080 and summary['parent_identity_rows']==2816
    cells=summary['cells'];assert len(cells)==880 and all(r['makers']==16 for r in cells)
    key=lambda r:tuple(r[k] for k in ('draw','length','kind','switched','duplicates','step','count'))
    aware={key(r):r for r in cells if r['evidence']=='aware'}
    for r in cells:
        assert all(r[m]==aware[key(r)][m] for m in C.T.METRICS)
    path=root/'inputs/aware/raw'/f'{lineage}-forecasts_points.json.gz'
    rows=C.gz(path);next(r for r in rows if r['arm']==C.ARM)['posterior'][0]+=.1
    path.write_bytes(gzip.compress(canonical(rows),mtime=0))
    corrupt=tmp_path/'corrupt'
    shutil.copytree(root/'inputs',corrupt/'inputs')
    with pytest.raises(ValueError,match='parent reconstruction'):
        C.run(corrupt,{'design':cfg},lambda **kw:None)
