import gzip
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import transient_filter as T, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, file_digest, read


def test_known_answers_and_duplicate_coverage():
    assert all(T.controls().values())
    neutral=np.full((16,4,8),1/8)
    values,_=T.score(neutral,np.full(16,1/16),7)
    assert values['coverage90']==1 and values['size90']==16
    with pytest.raises(ValueError,match='compatible'):
        T.normalize(np.full((2,16),-np.inf))


def test_filter_matches_explicit_hypothesis_products():
    random=np.random.default_rng(17); law=random.uniform(.01,1,(16,4,8));law/=law.sum(-1,keepdims=True)
    rows=[dict(step=i,source_step=i,source_id=str(i),context=i%4,endpoint=i%8) for i in range(1,7)]
    for arm in T.ARMS:
        p,_=T.posterior(law,rows,arm,3,6)
        reference=np.zeros(16)
        for initial,changed in product(range(16),(False,True) if arm=='coherent-mixture' else (False,)):
            weight=1.
            for r in rows:
                state=int(T.FLIP[initial]) if changed and r['source_step']>3 else initial
                weight*=law[state,r['context'],r['endpoint']]
            final=int(T.FLIP[initial]) if changed else initial
            reference[final]+=weight
        reference/=reference.sum()
        assert np.allclose(p,reference,rtol=0,atol=1e-14)


def test_complete_native_fixture_and_observation_roles(tmp_path):
    lineage=190969;records=L.enumerate_world(L.law(lineage))
    table=T.endpoint_law(records);assert np.allclose(table.sum(-1),1)
    path=tmp_path/'inputs'/f'lineage-{lineage}_points.json.gz';path.parent.mkdir();path.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(lineages=[lineage],draws=[190201],lengths=[32],paths_per_lineage=13824,input_files={path.name:file_digest(path)})
    result=T.run(tmp_path,{'design':cfg},lambda **kw:None)
    assert result['streams']==64 and result['observations']==2048 and result['rows']==960
    assert len(result['cells'])==60 and all(r['makers']==16 for r in result['cells'])
    for packet in read(tmp_path/'PUBLIC_PACKET.json')['cases']:
        assert set(packet)=={'input_sha256','inputs'}
        assert set(packet['inputs'])=={'contexts','observations'}
        for row in packet['inputs']['observations']:
            assert set(row)=={'step','source_step','source_id','context','endpoint'}
    streams=__import__('json').loads(gzip.decompress((tmp_path/'raw'/f'{lineage}-observations_points.json.gz').read_bytes()))
    assert all(len(T.unique_sources(s['observations']))==(24 if s['duplicates'] else 32) for s in streams)
    for switched in (False,True):
        ordinary=next(s for s in streams if s['maker']==0 and s['switched']==switched and not s['duplicates'])
        duplicated=next(s for s in streams if s['maker']==0 and s['switched']==switched and s['duplicates'])
        assert all(a==b for a,b in zip(ordinary['observations'],duplicated['observations']) if a['step']%4)
