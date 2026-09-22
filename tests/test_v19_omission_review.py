"""Independent identity omission: scalar products, pairing and corruption."""
from itertools import product
import gzip
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import omission_review as V
from ghostscale.validation.soundingline.v19 import source_omission as C, unknown_change as U, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_controls_scalar_products_and_source_time():
    assert all(V.controls().values())
    law=np.random.default_rng(198021).uniform(.1,1,(16,4,8));law/=law.sum(-1,keepdims=True)
    rows=[dict(step=i,source_step=i,source_id=str(i),context=i%4,endpoint=i%8) for i in range(1,33)]
    rows[16]=dict(rows[15],step=17)  # A copy crosses the possible change.
    omitted=V.omit_identity(rows)
    for arm,kind in product(V.ARMS,('purpose','skill')):
        outputs=V.product_checkpoints(law,omitted,arm,32,kind,[16,17,32])
        hs,prior=V.hypothesis_roster(arm,32,kind)
        for step,(cur,joint,_) in outputs.items():
            obs=omitted[:step];obs=obs[-16:] if arm=='reset-16' else obs
            weights=np.array([p*math.prod(law[V.complement(m,k) if k!='none' and r['source_step']>t else m,r['context'],r['endpoint']] for r in obs) for (k,t,m),p in zip(hs,prior)])
            weights/=math.fsum(weights);expected=np.zeros(16)
            for (k,t,m),p in zip(hs,weights):expected[V.complement(m,k) if k!='none' and step>t else m]+=p
            assert np.allclose(joint,weights,atol=1e-14,rtol=0)
            assert np.allclose(cur,expected,atol=1e-14,rtol=0)
    assert omitted[16]['source_step']==16 and omitted[16]['source_id']!=omitted[15]['source_id']
    with pytest.raises(ValueError,match='empty support'):
        V.product_checkpoints(np.zeros_like(law),omitted,'static',32,'purpose',[32])
    with pytest.raises(ValueError,match='conflicting'):
        V.omit_identity([rows[0],dict(rows[0],step=2,endpoint=7)])
    with pytest.raises(ValueError,match='nonconsecutive'):
        V.omit_identity([dict(rows[0],step=2)])


def test_paired_population_and_draws():
    cells=[];parents=[]
    for lineage,draw,arm in product((11,12),(101,102),V.ARMS):
        value=lineage+draw+V.ARMS.index(arm)
        for switched in (False,True):
            base=dict(lineage=lineage,draw=draw,length=32,kind='purpose',switched=switched,duplicates=True,step=32,arm=arm)
            cells.append(dict(base,**{m:value+2+switched for m in V.METRICS}))
            parents.append(dict(base,**{m:value for m in V.METRICS}))
    result=V.regroup(cells,dict(bootstrap_seed=190501,bootstrap_resamples=100),parents)
    assert len(result['means'])==10 and len(result['contrasts'])==150
    for row in result['contrasts']:
        if row['comparison']=='omission-minus-identity-aware':
            assert row['mean']==row['low']==row['high']==2+row['switched']
            assert row['draw_means']==[2+row['switched']]*2
    with pytest.raises(ValueError,match='duplicate'):
        V.regroup(cells+[cells[0]],dict(bootstrap_seed=1,bootstrap_resamples=10),parents)


def test_complete_native_fixture_and_corruption(tmp_path):
    original=tmp_path/'inputs/original';lineage=190979
    records=L.enumerate_world(L.law(lineage))
    p=original/'inputs'/f'lineage-{lineage}_points.json.gz';p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    parent=original/'inputs/parent';(parent/'inputs').mkdir(parents=True)
    shutil.copyfile(p,parent/'inputs'/p.name)
    cfg=dict(arms=list(V.ARMS),lineages=[lineage],draws=[190201,190202],lengths=[32],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    summary=U.run(parent,dict(design=cfg),lambda **kw:None);write(parent/'SUMMARY.json',summary)
    cfg['input_files']={p.relative_to(original/'inputs').as_posix():file_digest(p) for p in (original/'inputs').rglob('*') if p.is_file()}
    write(original/'PLAN.json',dict(design=cfg))
    result=C.run(original,dict(design=cfg),lambda **kw:None);write(original/'SUMMARY.json',result)
    plan=dict(design=dict(input_files={},target_plan_sha256=file_digest(original/'PLAN.json'),bootstrap_seed=190501,bootstrap_resamples=20))
    review=V.run(tmp_path,plan,lambda **kw:None)
    assert all(review['controls'].values()) and review['rows']==6400 and review['cells']==400 and review['streams']==256
    assert review['independent_identity_rows']==3200 and review['max_error']<1e-10
    regroup=read(tmp_path/'INDEPENDENT_REGROUP.json')
    assert len(regroup['means'])==200 and len(regroup['contrasts'])==3000 and len(regroup['parent_cells'])==400
    for row in regroup['means']:
        matched=[r for r in result['cells'] if all(r[k]==row[k] for k in (*V.AXES,'arm'))]
        for m in V.METRICS:assert abs(row[m]['mean']-math.fsum(r[m] for r in matched)/2)<1e-10
    counts=V.zipped(tmp_path/'RETAINED_SOURCE_COUNTS_points.json.gz')
    assert any(r['arm']=='reset-16' and r['supplied']==16 and r['true']==12 for r in counts)
    packet=read(original/'PUBLIC_PACKET.json');packet['cases'][0]['maker']=0
    with pytest.raises(ValueError,match='reader projection'):
        V.check_packet(packet,{r['input_sha256']:r for r in read(original/'PUBLIC_PACKET.json')['cases']})
    bad=read(parent/'SUMMARY.json');bad['cells'][0]['endpoint_loss']+=.1;write(parent/'SUMMARY.json',bad,immutable=False)
    with pytest.raises(ValueError):V.parent_cells(parent,lineage,V.endpoint_law(records))
    write(parent/'SUMMARY.json',summary,immutable=False)
    mapping=read(original/'evaluator'/f'{lineage}-joint-map.json');mapping['32-purpose-static']['rows'][0][0]=999
    write(original/'evaluator'/f'{lineage}-joint-map.json',mapping,immutable=False)
    with pytest.raises(ValueError,match='joint index mapping'):V.run(tmp_path,plan,lambda **kw:None)
