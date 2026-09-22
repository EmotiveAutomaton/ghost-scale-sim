"""Known-answer, pairing and corruption controls before campaign reconstruction."""
from itertools import product
import gzip
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import tempering_review as V
from ghostscale.validation.soundingline.v19 import provenance_tempering as P
from ghostscale.validation.soundingline.v19 import unknown_change as U, source_omission as O, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_scalar_powers_source_and_query_time():
    assert all(V.controls().values())
    law=np.random.default_rng(190911).dirichlet(np.ones(8),size=(16,4))
    obs=[dict(step=i,source_step=i,source_id=str(i),context=i%4,endpoint=i%8) for i in range(1,33)]
    obs[16]=dict(obs[15],step=17)
    out,hs,prior=V.products(law,obs,32,'purpose',[16,17,32])
    for step,power in product((16,17,32),V.POWERS):
        weights=np.asarray([p*math.prod(law[V.complement(m,k) if k!='none' and r['source_step']>t else m,r['context'],r['endpoint']]**power for r in V.distinct(obs[:step])) for (k,t,m),p in zip(hs,prior)])
        weights/=math.fsum(weights);current=np.zeros(16)
        for (k,t,m),w in zip(hs,weights):current[V.complement(m,k) if k!='none' and step>t else m]+=w
        assert np.allclose(out[step,power][0],current,atol=1e-13,rtol=0)
        assert np.allclose(out[step,power][1],weights,atol=1e-13,rtol=0)
    with pytest.raises(ValueError,match='nonconsecutive'):V.products(law,obs[:-1],32,'purpose',[32])
    bad=[*obs];bad[16]=dict(bad[16],endpoint=(bad[16]['endpoint']+1)%8)
    with pytest.raises(ValueError,match='conflicting'):V.products(law,bad,32,'purpose',[32])
    with pytest.raises(ValueError,match='empty support'):V.products(np.zeros_like(law),obs,32,'purpose',[32])


def test_paired_draws_and_all_contrasts():
    cells=[]
    for l,d,e,p in product((11,12),(101,102),V.EVIDENCE,V.POWERS):
        cells.append(dict(lineage=l,draw=d,evidence=e,power=p,length=32,kind='purpose',switched=True,duplicates=True,step=32,
            **{m:l+d+p+(2 if e=='omitted' else 0) for m in V.METRICS}))
    got=V.regroup(cells,dict(bootstrap_seed=190501,bootstrap_resamples=100))
    assert len(got['means'])==6 and len(got['contrasts'])==60
    for r in got['contrasts']:
        delta=r['power']-r['baseline_power']+(2 if r['evidence']=='omitted' else 0)-(2 if r['baseline_evidence']=='omitted' else 0)
        assert r['mean']==r['low']==r['high']==delta and r['draw_means']==[delta,delta]
    with pytest.raises(ValueError,match='duplicate'):V.regroup(cells+[cells[0]],dict(bootstrap_seed=190501,bootstrap_resamples=10))


def test_complete_native_fixture_and_corruption(tmp_path):
    lineage=190967; records=L.enumerate_world(L.law(lineage))
    parent=tmp_path/'parent';p=parent/'inputs'/f'lineage-{lineage}_points.json.gz';p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32,128],paths_per_lineage=13824,input_files={p.name:file_digest(p)})
    write(parent/'SUMMARY.json',U.run(parent,{'design':cfg},lambda **kw:None))
    omission=tmp_path/'omission';(omission/'inputs').mkdir(parents=True)
    shutil.copyfile(p,omission/'inputs'/p.name);shutil.copytree(parent,omission/'inputs/parent')
    cfg['input_files']={q.relative_to(omission/'inputs').as_posix():file_digest(q) for q in (omission/'inputs').rglob('*') if q.is_file()}
    write(omission/'SUMMARY.json',O.run(omission,{'design':cfg},lambda **kw:None))
    original=tmp_path/'review/inputs/original'
    for name,source in (('aware',parent),('omitted',omission)):
        for folder in ('raw','evaluator'):shutil.copytree(source/folder,original/'inputs'/name/folder)
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32,128],powers=list(V.POWERS),input_files={})
    write(original/'PLAN.json',dict(design=cfg))
    result=P.run(original,{'design':cfg},lambda **kw:None);write(original/'SUMMARY.json',result)
    review=tmp_path/'review';plan=dict(design=dict(input_files={},target_plan_sha256=file_digest(original/'PLAN.json'),bootstrap_seed=190501,bootstrap_resamples=20))
    got=V.verify(review,plan,lambda **kw:None)
    assert got['rows']==8448 and got['paired_streams']==512 and got['cells']==528 and got['max_error']<1e-10
    assert got['parent_identity_rows']==2816 and got['independent_source_identity_rows']==2112
    grouped=read(review/'INDEPENDENT_REGROUP.json')
    assert len(grouped['means'])==528 and len(grouped['contrasts'])==5280
    # Corrupt the first stream's source timestamp; the checker must reject it.
    path=original/'inputs/aware/raw'/f'{lineage}-observations_points.json.gz'
    streams=V.zipped(path);streams[0]['observations'][0]['source_step']=2
    path.write_bytes(gzip.compress(canonical(streams),mtime=0))
    with pytest.raises(ValueError,match='stream reconstruction'):V.verify(review,plan,lambda **kw:None)
