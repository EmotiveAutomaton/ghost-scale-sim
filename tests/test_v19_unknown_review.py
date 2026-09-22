import gzip
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import unknown_review as V
from ghostscale.validation.soundingline.v19 import unknown_change as F, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_independent_known_answers():
    assert all(V.controls().values())


def test_direct_products_time_type_and_duplicate_semantics():
    table=np.full((16,4,8),.1/7)
    for m in range(16): table[m,:,2*V.MAKERS[m][0]+V.MAKERS[m][1]]=.9
    obs=[dict(step=i,source_step=i,source_id=str(i),context=0,endpoint=i%4) for i in range(1,19)]
    obs[8]=dict(obs[7],step=9)
    for arm in V.ARMS:
        for kind in ('purpose','skill'):
            current,joint,hs=V.product_checkpoints(table,obs,arm,32,kind,[18])[18]
            _,prior=V.hypothesis_roster(arm,32,kind)
            evidence=V.distinct(obs)[-16:] if arm=='reset-16' else V.distinct(obs)
            expected=[]
            for (k,t,m),p in zip(hs,prior):
                expected.append(p*math.prod(table[V.complement(m,k) if k!='none' and row['source_step']>t else m,row['context'],row['endpoint']] for row in evidence))
            expected=np.array(expected)/math.fsum(expected)
            assert np.allclose(joint,expected,rtol=0,atol=1e-14)
            remapped=np.zeros(16)
            for (k,t,m),p in zip(hs,expected): remapped[V.complement(m,k) if k!='none' and 18>t else m]+=p
            assert np.allclose(current,remapped,rtol=0,atol=1e-14)
    with pytest.raises(ValueError,match='conflicting'):
        V.product_checkpoints(table,[obs[0],dict(obs[0],endpoint=5)],'unknown-time-type',32,'purpose',[2])
    with pytest.raises(ValueError,match='empty support'):
        V.product_checkpoints(np.zeros_like(table),obs,'unknown-time',32,'skill',[18])


def test_future_change_and_prior_by_time():
    hs,p=V.hypothesis_roster('unknown-time-type',128,'purpose')
    for kind in ('purpose','skill'):
        for time in range(8,121):
            assert abs(math.fsum(w for (k,t,m),w in zip(hs,p) if k==kind and t==time)-.25/113)<1e-15
    for i,(k,t,m) in enumerate(hs):
        if k!='none':
            assert V.current_indices([hs[i]],t)[0]==m
            assert V.current_indices([hs[i]],t+1)[0]==V.complement(m,k)


def test_reader_identity_rejection():
    visible={'inputs':{'contexts':[], 'observations':[]}, 'input_sha256':'fixture'}
    V.check_packet({'schema':'v19.unknown-change.reader.1','cases':[visible]}, {'fixture':visible})
    with pytest.raises(ValueError,match='reader projection'):
        V.check_packet({'schema':'v19.unknown-change.reader.1','cases':[dict(visible,maker=0)]}, {'fixture':visible})


def test_complete_native_fixture_and_population(tmp_path):
    original=tmp_path/'inputs/original';(original/'inputs').mkdir(parents=True)
    lineage=190969;records=L.enumerate_world(L.law(lineage))
    p=original/'inputs'/f'lineage-{lineage}_points.json.gz'
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(arms=list(F.ARMS),lineages=[lineage],draws=[190201,190202],lengths=[32,128],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    write(original/'PLAN.json',dict(design=cfg))
    summary=F.run(original,dict(design=cfg),lambda **kw:None);write(original/'SUMMARY.json',summary)
    plan=dict(design=dict(input_files={},target_plan_sha256=file_digest(original/'PLAN.json'),bootstrap_seed=190501,bootstrap_resamples=20))
    checked=V.run(tmp_path,plan,lambda **kw:None)
    assert all(checked['controls'].values()) and checked['paths']==13824 and checked['cells']==880
    assert checked['rows']==14080 and checked['streams']==512 and checked['max_error']<1e-10
    regroup=read(tmp_path/'INDEPENDENT_REGROUP.json')
    assert len(regroup['contrasts'])==4400 and len(regroup['means'])==440
    for mean in regroup['means']:
        subset=[r for r in summary['cells'] if all(r[k]==mean[k] for k in (*V.AXES,'arm'))]
        assert len(subset)==2
        for metric in V.METRICS: assert abs(mean[metric]-sum(r[metric] for r in subset)/2)<1e-10
    # Corrupt the first mapping entry so the full checker must reject its binding.
    path=original/'evaluator'/f'{lineage}-joint-map.json';mapping=read(path)
    mapping['32-purpose-static']['rows'][0][0]=999
    write(path,mapping,immutable=False)
    with pytest.raises(ValueError,match='joint index mapping'): V.run(tmp_path,plan,lambda **kw:None)
