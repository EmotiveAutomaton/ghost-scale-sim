"""Known-answer and complete native controls for unknown-change filtering."""
import gzip
import json
import math
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import unknown_change as U, transient_filter as T, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, file_digest, read


def test_complete_priors_and_unknown_answer_controls():
    assert all(U.controls().values())
    for kind in U.FLIPS:
        hs,p=U.hypotheses('unknown-time-type',128,kind)
        assert len(hs)==16*(1+2*113)
        assert set(t for k,t,m in hs if k!='none')==set(range(8,121))
        assert abs(math.fsum(p)-1)<1e-13


def test_all_filters_match_independent_products_and_duplicates():
    r=np.random.default_rng(916);law=r.uniform(.01,1,(16,4,8));law/=law.sum(-1,keepdims=True)
    rows=[dict(step=i,source_step=i,source_id=str(i),context=i%4,endpoint=i%8) for i in range(1,22)]
    for kind,arm in product(U.FLIPS,U.ARMS):
        hs,prior=U.hypotheses(arm,32,kind)
        actual=U.filter_checkpoints(law,rows,arm,32,kind,[8,17,21])
        for step in (8,17,21):
            p,joint,_=actual[step];reference=np.zeros(16);weights=[]
            obs=rows[:step][-16:] if arm=='reset-16' else rows[:step]
            for (k,t,m),weight in zip(hs,prior):
                axis=0 if k=='purpose' else 1
                changed=L.MAKERS.index(tuple(1-v if i==axis else v for i,v in enumerate(L.MAKERS[m])))
                for row in obs:
                    state=changed if k!='none' and row['source_step']>t else m
                    weight*=law[state,row['context'],row['endpoint']]
                reference[changed if k!='none' and step>t else m]+=weight;weights.append(weight)
            reference/=reference.sum();weights=np.asarray(weights);weights/=weights.sum()
            assert np.allclose(p,reference,atol=1e-14,rtol=0)
            assert np.allclose(joint.reshape(-1),weights,atol=1e-14,rtol=0)
        copied=rows+[dict(rows[-1],step=22)]
        p,_,_=U.filter_checkpoints(law,copied,arm,32,kind,[22])[22]
        if arm in ('static','reset-16','known-time-type'):
            assert np.allclose(p,actual[21][0],atol=1e-14,rtol=0)
        with pytest.raises(ValueError,match='conflicting'):
            U.filter_checkpoints(law,rows+[dict(rows[-1],step=22,endpoint=7)],arm,32,kind,[22])
    old,_=T.posterior(law,rows,'coherent-mixture',16,21)
    assert np.allclose(old,U.filter_checkpoints(law,rows,'known-time-type',32,'purpose',[21])[21][0],atol=1e-14,rtol=0)
    with pytest.raises(ValueError,match='compatible'):
        U.filter_checkpoints(np.zeros_like(law),rows,'unknown-time-type',32,'skill',[8])


def test_duplicate_changes_time_without_adding_likelihood():
    # A duplicate adds no evidence, but an uncertain process can change since its source time.
    law=np.full((16,4,8),.1/7)
    for m in range(16):law[m,:,L.MAKERS[m][0]]=.9
    a=dict(step=8,source_step=8,source_id='a',context=0,endpoint=0)
    outputs=U.filter_checkpoints(law,[a,dict(a,step=9)],'unknown-time-type',32,'purpose',[8,9])
    assert np.array_equal(outputs[8][1],outputs[9][1])
    assert not np.allclose(outputs[8][0],outputs[9][0])


def test_native_fixture_support_and_role_split(tmp_path):
    lineage=190969;records=L.enumerate_world(L.law(lineage))
    p=tmp_path/'inputs'/f'lineage-{lineage}_points.json.gz';p.parent.mkdir()
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    result=U.run(tmp_path,{'design':cfg},lambda **kw:None)
    assert all(result['controls'].values()) and result['streams']==128 and result['rows']==3200
    assert len(result['cells'])==200 and all(c['makers']==16 for c in result['cells'])
    streams=json.loads(gzip.decompress((tmp_path/'raw'/f'{lineage}-observations_points.json.gz').read_bytes()))
    for maker,duplicate in product(range(16),(False,True)):
        pair=[s for s in streams if s['maker']==maker and not s['switched'] and s['duplicates']==duplicate]
        assert pair[0]['observations']==pair[1]['observations']
    mapping=read(tmp_path/'evaluator'/f'{lineage}-joint-map.json')
    rows=json.loads(gzip.decompress((tmp_path/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
    with np.load(tmp_path/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as archive:
        for row in rows:
            key=row['joint_array'];index=row['joint_row'];joint=archive[key][index];meta=mapping[key]
            assert meta['rows'][index]==[row['stream'],row['step']]
            current=np.bincount(U.state_indices(meta['hypotheses'],row['step']),weights=joint,minlength=16)
            assert np.allclose(current,row['posterior'],rtol=0,atol=1e-14)
        copy=tmp_path/'copy.npz';np.savez_compressed(copy,**{k:archive[k] for k in archive.files})
    assert copy.read_bytes()==(tmp_path/'raw'/f'{lineage}-joint_points.npz').read_bytes()
    for packet in read(tmp_path/'PUBLIC_PACKET.json')['cases']:
        assert set(packet)=={'input_sha256','inputs'} and set(packet['inputs'])=={'contexts','observations'}
        assert all(set(r)=={'step','source_step','source_id','context','endpoint'} for r in packet['inputs']['observations'])
