"""Known-answer, source-time and parent-identity controls for timing transfer."""
from itertools import product
import gzip
import json
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import change_timing as C, unknown_change as U, transient_filter as T, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_controls_and_frozen_unknown_roster():
    assert all(C.controls().values())
    for kind,arm,at in product(U.FLIPS,U.ARMS,(8,24)):
        hs,p=C.hypotheses(arm,32,kind,at)
        assert math.isclose(math.fsum(p),1,abs_tol=1e-14)
        if arm!='known-time-type':
            old,prior=U.hypotheses(arm,32,kind)
            assert hs==old and np.array_equal(p,prior)
    with pytest.raises(ValueError,match='roster'):C.hypotheses('static',32,'purpose',7)


def test_complete_filter_products_at_both_times():
    law=np.random.default_rng(198001).uniform(.1,1,(16,4,8));law/=law.sum(-1,keepdims=True)
    rows=[dict(step=i,source_step=i,source_id=str(i),context=i%4,endpoint=i%8) for i in range(1,33)]
    # Include a copy which must not contribute an independent factor.
    rows[11]=dict(rows[10],step=12)
    for kind,arm,at in product(U.FLIPS,U.ARMS,(8,24)):
        hs,prior=C.hypotheses(arm,32,kind,at)
        actual=C.filter_checkpoints(law,rows,arm,32,kind,at,[8,9,24,25,32])
        for step,(p,joint,_) in actual.items():
            obs=T.unique_sources(rows[:step])
            if arm=='reset-16':obs=obs[-16:]
            weights=[]
            expected=np.zeros(16)
            for (k,t,m),weight in zip(hs,prior):
                for row in obs:
                    state=int(U.FLIPS[k][m]) if k!='none' and row['source_step']>t else m
                    weight*=law[state,row['context'],row['endpoint']]
                weights.append(weight)
                state=int(U.FLIPS[k][m]) if k!='none' and step>t else m
                expected[state]+=weight
            expected/=expected.sum();weights=np.asarray(weights);weights/=weights.sum()
            assert np.allclose(p,expected,atol=1e-14,rtol=0)
            assert np.allclose(joint.reshape(-1),weights,atol=1e-14,rtol=0)


def test_duplicate_source_time_and_future_projection():
    law=np.full((16,4,8),.1/7)
    for m in range(16):law[m,:,L.MAKERS[m][0]]=.9
    a=dict(step=8,source_step=8,source_id='a',context=0,endpoint=0)
    copies=[a,dict(a,step=9)]
    out=C.filter_checkpoints(law,copies,'known-time-type',32,'purpose',8,[8,9])
    assert np.array_equal(out[8][1],out[9][1])
    assert not np.allclose(out[8][0],out[9][0])
    late=C.filter_checkpoints(law,copies,'known-time-type',32,'purpose',24,[8,9])
    assert np.array_equal(late[8][0],late[9][0])
    with pytest.raises(ValueError,match='conflicting'):
        C.filter_checkpoints(law,[a,dict(a,step=9,endpoint=1)],'known-time-type',32,'purpose',8,[9])
    with pytest.raises(ValueError,match='compatible'):
        C.filter_checkpoints(np.zeros_like(law),[a],'known-time-type',32,'skill',8,[8])


def test_midpoint_identity_and_unknown_input_privilege():
    law=np.random.default_rng(198002).uniform(.1,1,(16,4,8));law/=law.sum(-1,keepdims=True)
    for kind,duplicates in product(U.FLIPS,(False,True)):
        rows=C.make_stream(law,190969,190201,3,32,duplicates,kind,16)
        assert rows==U.make_stream(law,190969,190201,3,32,True,duplicates,kind)
        for arm in U.ARMS:
            old=U.filter_checkpoints(law,rows,arm,32,kind,[8,17,32])
            new=C.filter_checkpoints(law,rows,arm,32,kind,16,[8,17,32])
            for step in old:
                assert np.array_equal(old[step][0],new[step][0])
                assert np.array_equal(old[step][1],new[step][1])
                if arm!='known-time-type':
                    a=C.filter_checkpoints(law,rows,arm,32,kind,8,[step])[step]
                    b=C.filter_checkpoints(law,rows,arm,32,kind,24,[step])[step]
                    assert np.array_equal(a[0],b[0])


def test_actual_stream_boundary_and_copy():
    law=np.zeros((16,4,8))
    for m in range(16):law[m,:,L.MAKERS[m][0]+2*L.MAKERS[m][1]]=1
    for kind,at in product(U.FLIPS,(8,24)):
        for copied in (False,True):
            rows=C.make_stream(law,190969,190201,0,32,copied,kind,at)
            for r in rows:
                m=int(U.FLIPS[kind][0]) if r['source_step']>at else 0
                assert r['endpoint']==L.MAKERS[m][0]+2*L.MAKERS[m][1]
            if copied:
                assert rows[at-1]['source_step']==at-1
                assert rows[at]['source_step']==at+1


def test_complete_native_fixture_parent_reproduction_and_roles(tmp_path):
    lineage=190969;records=L.enumerate_world(L.law(lineage))
    parent=tmp_path/'parent';p=parent/'inputs'/f'lineage-{lineage}_points.json.gz';p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    summary=U.run(parent,{'design':cfg},lambda **kw:None);write(parent/'SUMMARY.json',summary)
    root=tmp_path/'new';(root/'inputs').mkdir(parents=True)
    shutil.copyfile(p,root/'inputs'/p.name)
    target=root/'inputs/parent';target.mkdir()
    for n in ('raw','evaluator'):shutil.copytree(parent/n,target/n)
    shutil.copyfile(parent/'SUMMARY.json',target/'SUMMARY.json')
    cfg.update(quarters=[1,3],input_files={q.relative_to(root/'inputs').as_posix():file_digest(q) for q in (root/'inputs').rglob('*') if q.is_file()})
    result=C.run(root,{'design':cfg},lambda **kw:None)
    assert all(result['controls'].values()) and result['streams']==128 and result['rows']==3200
    assert len(result['cells'])==200 and all(x['makers']==16 for x in result['cells'])
    assert read(root/'PARENT_REPRODUCTION.json')['lineages'][0]['rows']==3200
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
    bad=read(target/'SUMMARY.json');bad['cells'][0]['endpoint_loss']+=.1
    (target/'SUMMARY.json').write_bytes(canonical(bad))
    with pytest.raises(ValueError,match='reconstruction'):
        C.reproduce_parent(target,lineage,T.endpoint_law(records))
