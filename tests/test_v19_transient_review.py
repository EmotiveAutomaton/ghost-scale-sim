import gzip
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import transient_review as V
from ghostscale.validation.soundingline.v19 import transient_filter as F, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_independent_known_answers():
    assert all(V.controls().values())


def test_source_and_distribution_corruption_rejected():
    a=dict(step=1,source_step=1,source_id='a',context=0,endpoint=0)
    with pytest.raises(ValueError,match='conflicting'):
        V.distinct([a,dict(a,source_step=2)])
    with pytest.raises(ValueError,match='mismatch'):
        V.near([.1,.9],[.9,.1])


def test_future_switch_does_not_change_present_state():
    table=np.full((16,4,8),1/8)
    for m in range(16):
        table[m,:,:]=.1/7;table[m,:,V.MAKERS[m][0]]=.9
    obs=[dict(step=1,source_step=1,source_id='a',context=0,endpoint=0)]
    present,_=V.hypothesis_product(table,obs,'coherent-mixture',4,1)
    static,_=V.hypothesis_product(table,obs,'static',4,1)
    assert np.array_equal(present,static)
    after,_=V.hypothesis_product(table,obs,'coherent-mixture',4,5)
    assert np.allclose(after,1/16,atol=1e-14,rtol=0)


def test_complete_native_fixture_and_paired_population(tmp_path):
    original=tmp_path/'inputs/original';(original/'inputs').mkdir(parents=True)
    lineage=190969; records=L.enumerate_world(L.law(lineage))
    p=original/'inputs'/f'lineage-{lineage}_points.json.gz'
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(arms=list(F.ARMS),lineages=[lineage],draws=[190201,190202],lengths=[32,128],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    write(original/'PLAN.json',dict(design=cfg))
    summary=F.run(original,dict(design=cfg),lambda **kw:None);write(original/'SUMMARY.json',summary)
    plan=dict(design=dict(input_files={},target_plan_sha256=file_digest(original/'PLAN.json'),bootstrap_seed=190501,bootstrap_resamples=20))
    checked=V.run(tmp_path,plan,lambda **kw:None)
    assert all(checked['controls'].values()) and checked['paths']==13824 and checked['cells']==264
    assert checked['rows']==4224 and checked['streams']==256 and checked['max_error']<1e-10
    regroup=read(tmp_path/'INDEPENDENT_REGROUP.json')
    assert len(regroup['contrasts'])==660 and len(regroup['means'])==132
    for mean in regroup['means']:
        subset=[r for r in summary['cells'] if all(r[k]==mean[k] for k in (*V.AXES,'arm'))]
        assert len(subset)==2
        for metric in V.METRICS: assert abs(mean[metric]-sum(r[metric] for r in subset)/2)<1e-10
    # A valid source packet carrying evaluator identity must be rejected too.
    packet=read(original/'PUBLIC_PACKET.json');packet['cases'][0]['maker']=0
    write(original/'PUBLIC_PACKET.json',packet,immutable=False)
    with pytest.raises(ValueError,match='reader projection'): V.run(tmp_path,plan,lambda **kw:None)
