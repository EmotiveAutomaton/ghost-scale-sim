"""Independent time-transfer checker: known answers, corruption and full fixture."""
from itertools import product
import gzip
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import timing_review as V
from ghostscale.validation.soundingline.v19 import change_timing as C, unknown_change as U, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_independent_controls_and_scalar_products():
    assert all(V.controls().values())
    law=np.random.default_rng(198011).uniform(.1,1,(16,4,8));law/=law.sum(-1,keepdims=True)
    rows=[dict(step=i,source_step=i,source_id=str(i),context=i%4,endpoint=i%8) for i in range(1,33)]
    rows[8]=dict(rows[7],step=9)
    for arm,kind,at in product(V.ARMS,('purpose','skill'),(8,24)):
        outputs=V.product_checkpoints(law,rows,arm,32,kind,at,[8,9,24,25,32])
        hs,prior=V.hypothesis_roster(arm,32,kind,at)
        for step,(cur,joint,_) in outputs.items():
            obs=V.distinct(rows[:step]);obs=obs[-16:] if arm=='reset-16' else obs
            weights=np.array([p*math.prod(law[V.complement(m,k) if k!='none' and r['source_step']>t else m,r['context'],r['endpoint']] for r in obs) for (k,t,m),p in zip(hs,prior)])
            weights/=math.fsum(weights);expected=np.zeros(16)
            for (k,t,m),p in zip(hs,weights):expected[V.complement(m,k) if k!='none' and step>t else m]+=p
            assert np.allclose(joint,weights,atol=1e-14,rtol=0)
            assert np.allclose(cur,expected,atol=1e-14,rtol=0)
    with pytest.raises(ValueError,match='empty support'):
        V.product_checkpoints(np.zeros_like(law),rows,'known-time-type',32,'purpose',8,[32])
    with pytest.raises(ValueError,match='conflicting'):
        V.product_checkpoints(law,[rows[0],dict(rows[0],endpoint=7)],'static',32,'purpose',8,[2])


def test_parent_pairing_and_draw_uncertainty():
    cells=[];parents=[]
    for lineage,draw,arm in product((11,12),(101,102),V.ARMS):
        value=lineage+draw+V.ARMS.index(arm)
        for quarter in (1,3):
            cells.append(dict(lineage=lineage,draw=draw,length=32,kind='purpose',quarter=quarter,duplicates=False,step=32,arm=arm,**{m:value+quarter for m in V.METRICS}))
        parents.append(dict(lineage=lineage,draw=draw,length=32,kind='purpose',switched=True,duplicates=False,step=32,arm=arm,**{m:value for m in V.METRICS}))
    result=V.regroup(cells,dict(bootstrap_seed=190501,bootstrap_resamples=100),parents)
    assert len(result['means'])==10 and len(result['contrasts'])==150
    for row in result['contrasts']:
        if row['comparison']=='actual-time-minus-midpoint':
            assert row['mean']==row['low']==row['high']==row['quarter']
            assert row['draw_means']==[row['quarter']]*2
    with pytest.raises(ValueError,match='duplicate'):
        V.regroup(cells+[cells[0]],dict(bootstrap_seed=1,bootstrap_resamples=10),parents)


def test_complete_native_fixture_and_corruption(tmp_path):
    original=tmp_path/'inputs/original';lineage=190969
    records=L.enumerate_world(L.law(lineage))
    p=original/'inputs'/f'lineage-{lineage}_points.json.gz';p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    parent=original/'inputs/parent';(parent/'inputs').mkdir(parents=True)
    shutil.copyfile(p,parent/'inputs'/p.name)
    cfg=dict(arms=list(V.ARMS),lineages=[lineage],draws=[190201,190202],lengths=[32],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    summary=U.run(parent,dict(design=cfg),lambda **kw:None);write(parent/'SUMMARY.json',summary)
    cfg.update(quarters=[1,3],input_files={p.relative_to(original/'inputs').as_posix():file_digest(p) for p in (original/'inputs').rglob('*') if p.is_file()})
    write(original/'PLAN.json',dict(design=cfg))
    result=C.run(original,dict(design=cfg),lambda **kw:None);write(original/'SUMMARY.json',result)
    plan=dict(design=dict(input_files={},target_plan_sha256=file_digest(original/'PLAN.json'),bootstrap_seed=190501,bootstrap_resamples=20))
    review=V.run(tmp_path,plan,lambda **kw:None)
    assert all(review['controls'].values()) and review['rows']==6400 and review['cells']==400 and review['streams']==256
    assert review['max_error']<1e-10
    regroup=read(tmp_path/'INDEPENDENT_REGROUP.json')
    assert len(regroup['means'])==200 and len(regroup['contrasts'])==3000 and len(regroup['parent_cells'])==400
    for row in regroup['means']:
        matched=[r for r in result['cells'] if all(r[k]==row[k] for k in (*V.AXES,'arm'))]
        for m in V.METRICS:assert abs(row[m]['mean']-math.fsum(r[m] for r in matched)/2)<1e-10
    packet=read(original/'PUBLIC_PACKET.json');packet['cases'][0]['maker']=0
    with pytest.raises(ValueError,match='reader projection'):V.check_packet(packet,{r['input_sha256']:r for r in read(original/'PUBLIC_PACKET.json')['cases']})
    bad=read(parent/'SUMMARY.json');bad['cells'][0]['endpoint_loss']+=.1;write(parent/'SUMMARY.json',bad,immutable=False)
    with pytest.raises(ValueError):V.parent_cells(parent,lineage,V.endpoint_law(records))
    mapping=read(original/'evaluator'/f'{lineage}-joint-map.json');mapping['32-purpose-1-static']['rows'][0][0]=999
    write(original/'evaluator'/f'{lineage}-joint-map.json',mapping,immutable=False)
    # Restore the parent aggregate, so rejection must come from the joint binding.
    write(parent/'SUMMARY.json',summary,immutable=False)
    with pytest.raises(ValueError,match='joint index mapping'):V.run(tmp_path,plan,lambda **kw:None)
