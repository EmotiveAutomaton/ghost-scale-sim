"""Correction identities, independent products, full native and evidence roles."""
import gzip
import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import provenance_restoration as P, source_omission as O
from ghostscale.validation.soundingline.v19 import unknown_change as U, unknown_review as R, transient_filter as T, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_known_answer_controls():
    assert all(P.controls().values())


def test_corrections_preserve_evidence_and_reject_conflicts():
    a=dict(step=1,source_step=1,source_id='a',context=0,endpoint=0)
    original=[a,dict(a,step=2),dict(a,step=3,source_step=3,source_id='b',endpoint=1)]
    omitted=O.supplied_observations(original)
    fixed=P.correct(omitted,[[2,1]])
    assert fixed[0]['source_id']==fixed[1]['source_id']!=fixed[2]['source_id']
    for old,new in zip(omitted,fixed):
        assert {k:v for k,v in old.items() if k!='source_id'}=={k:v for k,v in new.items() if k!='source_id'}
    assert len(T.unique_sources(P.correct(omitted,[[2,1]],rename=True)))==3
    for pairs in ([[3,1]],[[2,1],[2,1]],[[1,2]],[[4,1]]):
        with pytest.raises(ValueError):P.correct(omitted,pairs)


def test_nested_orders_deduplicate_and_independent_products():
    law=np.random.default_rng(19).dirichlet(np.ones(8),size=(16,4))
    for kind in U.FLIPS:
        original=U.make_stream(law,190969,190201,3,32,True,True,kind)
        supplied=O.supplied_observations(original)
        pairs=P.available_pairs(original)
        cs=P.conditions(pairs);assert len(cs)==6 and sum(len(c['aliases']) for c in cs)==8
        assert len(P.conditions([]))==1
        for condition in cs:
            observed=P.correct(supplied,condition['pairs'])
            assert len(T.unique_sources(observed))==32-len(condition['pairs'])
            for arm in U.ARMS:
                expected=R.product_checkpoints(law,observed,arm,32,kind,[32])[32]
                actual=U.filter_checkpoints(law,observed,arm,32,kind,[32])[32]
                assert np.allclose(actual[0],expected[0],atol=1e-12,rtol=0)
                assert np.allclose(actual[1].reshape(-1),expected[1],atol=1e-12,rtol=0)
        assert P.correct(supplied,pairs)==original


def test_complete_native_fixture_joint_mapping_and_reader_allowlist(tmp_path):
    lineage=190969;records=L.enumerate_world(L.law(lineage))
    parent=tmp_path/'parent';p=parent/'inputs'/f'lineage-{lineage}_points.json.gz';p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    write(parent/'SUMMARY.json',U.run(parent,{'design':cfg},lambda **kw:None))
    omission=tmp_path/'omission';(omission/'inputs').mkdir(parents=True)
    shutil.copyfile(p,omission/'inputs'/p.name);shutil.copytree(parent,omission/'inputs/parent')
    cfg['input_files']={q.relative_to(omission/'inputs').as_posix():file_digest(q) for q in (omission/'inputs').rglob('*') if q.is_file()}
    write(omission/'SUMMARY.json',O.run(omission,{'design':cfg},lambda **kw:None))
    root=tmp_path/'restoration';(root/'inputs').mkdir(parents=True)
    shutil.copyfile(p,root/'inputs'/p.name)
    for source,name in ((parent,'parent'),(omission,'omission')):
        dest=root/'inputs'/name;dest.mkdir();shutil.copytree(source/'raw',dest/'raw')
    streams=json.loads(gzip.decompress((parent/'raw'/f'{lineage}-observations_points.json.gz').read_bytes()))
    roster={}
    for stream in streams:
        for step in T.checkpoints(32):
            pairs=P.available_pairs(stream['observations'][:step])
            key=f'32-{int(stream["duplicates"])}-{step}'
            spec=dict(available_pairs=pairs,conditions=P.conditions(pairs))
            assert key not in roster or roster[key]==spec
            roster[key]=spec
    write(root/'inputs/CORRECTION_ROSTER.json',roster)
    cfg['input_files']={q.relative_to(root/'inputs').as_posix():file_digest(q) for q in (root/'inputs').rglob('*') if q.is_file()}
    result=P.run(root,{'design':cfg},lambda **kw:None)
    assert result['streams']==128 and result['rows']==9280 and len(result['cells'])==580
    assert result['identities']=={'zero':3200,'label_only':3200,'full':3200}
    rows=json.loads(gzip.decompress((root/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
    mapping=read(root/'evaluator'/f'{lineage}-joint-map.json')
    with np.load(root/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as z:
        for row in rows:
            meta=mapping[row['joint_array']];joint=z[row['joint_array']][row['joint_row']]
            assert meta['rows'][row['joint_row']]==[row['stream'],row['step'],row['condition']]
            current=np.bincount(U.state_indices(meta['hypotheses'],row['step']),weights=joint,minlength=16)
            assert np.allclose(current,row['posterior'],atol=1e-14,rtol=0)
    for case in read(root/'PUBLIC_PACKET.json')['cases']:
        assert set(case)=={'input_sha256','inputs'}
        assert set(case['inputs'])=={'contexts','observations','source_equivalences'}
        assert all(set(r)=={'step','source_step','source_id','context','endpoint'} for r in case['inputs']['observations'])
    # Frozen roster corruption must fail, independently of input-file hash checks.
    key=next(k for k,v in roster.items() if v['available_pairs'])
    roster[key]['available_pairs'][0][1]+=1;write(root/'inputs/CORRECTION_ROSTER.json',roster,immutable=False)
    cfg['input_files']['CORRECTION_ROSTER.json']=file_digest(root/'inputs/CORRECTION_ROSTER.json')
    with pytest.raises(ValueError,match='roster'):
        P.run(root,{'design':cfg},lambda **kw:None)
