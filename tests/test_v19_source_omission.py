"""Known-answer source dependence, complete fixture and reader-role controls."""
import gzip
import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import source_omission as S, unknown_change as U, transient_filter as T, local_world as L, timing_review as R
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_known_answer_and_neutral_controls():
    assert all(S.controls().values())


def test_only_identity_changes_and_conflicts_rejected():
    a=dict(step=1,source_step=1,source_id='source-001',context=0,endpoint=0)
    original=[a,dict(a,step=2)]
    result=S.supplied_observations(original)
    assert original[1]['source_id']=='source-001'
    assert result[1]['source_id']=='source-002' and result[1]['source_step']==1
    with pytest.raises(ValueError,match='conflict'):
        S.supplied_observations([a,dict(a,step=2,endpoint=1)])
    with pytest.raises(ValueError,match='nonconsecutive'):
        S.supplied_observations([dict(a,step=2)])


def test_independent_products_and_all_filter_identities():
    law=np.random.default_rng(19).dirichlet(np.ones(8),size=(16,4))
    for kind in U.FLIPS:
        for copied in (False,True):
            original=U.make_stream(law,190969,190201,3,32,True,copied,kind)
            observed=S.supplied_observations(original)
            if not copied:assert observed==original
            for arm in U.ARMS:
                expected=R.product_checkpoints(law,observed,arm,32,kind,16,[8,16,17,20,32])
                actual=U.filter_checkpoints(law,observed,arm,32,kind,[8,16,17,20,32])
                for step in expected:
                    assert np.allclose(expected[step][0],actual[step][0],atol=1e-12,rtol=0)
                    assert np.allclose(expected[step][1],actual[step][1],atol=1e-12,rtol=0)
            assert len(T.unique_sources(observed))==32


def test_complete_native_fixture_and_parent_corruption(tmp_path):
    lineage=190969;records=L.enumerate_world(L.law(lineage))
    parent=tmp_path/'parent';p=parent/'inputs'/f'lineage-{lineage}_points.json.gz';p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    summary=U.run(parent,{'design':cfg},lambda **kw:None);write(parent/'SUMMARY.json',summary)
    root=tmp_path/'new';(root/'inputs').mkdir(parents=True);shutil.copyfile(p,root/'inputs'/p.name)
    target=root/'inputs/parent';target.mkdir()
    for n in ('raw','evaluator'):shutil.copytree(parent/n,target/n)
    shutil.copyfile(parent/'SUMMARY.json',target/'SUMMARY.json')
    cfg['input_files']={q.relative_to(root/'inputs').as_posix():file_digest(q) for q in (root/'inputs').rglob('*') if q.is_file()}
    result=S.run(root,{'design':cfg},lambda **kw:None)
    assert all(result['controls'].values()) and result['streams']==128 and result['rows']==3200
    assert len(result['cells'])==200 and all(x['makers']==16 for x in result['cells'])
    assert read(root/'PARENT_REPRODUCTION.json')['lineages'][0]['independent_identity_rows']==1600
    mapping=read(root/'evaluator'/f'{lineage}-joint-map.json')
    rows=json.loads(gzip.decompress((root/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
    with np.load(root/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as z:
        for row in rows:
            meta=mapping[row['joint_array']];joint=z[row['joint_array']][row['joint_row']]
            current=np.bincount(U.state_indices(meta['hypotheses'],row['step']),weights=joint,minlength=16)
            assert np.allclose(current,row['posterior'],atol=1e-14,rtol=0)
    for packet in read(root/'PUBLIC_PACKET.json')['cases']:
        assert set(packet)=={'input_sha256','inputs'}
        assert set(packet['inputs'])=={'contexts','observations'}
        assert all(set(r)=={'step','source_step','source_id','context','endpoint'} for r in packet['inputs']['observations'])
        assert len({r['source_id'] for r in packet['inputs']['observations']})==len(packet['inputs']['observations'])
    bad=read(target/'SUMMARY.json');bad['cells'][0]['endpoint_loss']+=.1
    (target/'SUMMARY.json').write_bytes(canonical(bad))
    with pytest.raises(ValueError,match='reconstruction'):
        S.C.reproduce_parent(target,lineage,T.endpoint_law(records))
