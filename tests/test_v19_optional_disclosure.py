import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import optional_disclosure as O, metadata_disclosure as D
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from test_v19_metadata_disclosure import fixture
import shutil


def test_known_null_price_ties_and_expected_random_scores():
    assert all(O.controls().values())
    assert np.array_equal(O.selection([0,.02,.02001,.1],.02),[0,0,1,1])
    p=np.array([[1.,0,0,0,0,0,0,0]])
    q=np.roll(p,1,axis=1)
    got=O.scores(p,q,np.array([0]),np.array([1.]),np.array([.25]),.1)
    assert got['infinite_loss_mass']==.25 and got['finite_loss_contribution']==0
    assert got['net_finite_loss']==.025
    with pytest.raises(ValueError):O.selection([np.nan],.1)


def test_full_fixed_law_fixture_and_matched_rates(tmp_path):
    original=tmp_path/'original';design=fixture(original)
    D.run(original,dict(design=design),lambda **kw:None)
    root=tmp_path/'optional';base=root/'inputs';base.mkdir(parents=True)
    for src,name in ((original/'evaluator/DISCLOSURE_LAWS.json','DISCLOSURE_LAWS.json'),(original/'inputs/MEMBERSHIP.json','MEMBERSHIP.json')):shutil.copyfile(src,base/name)
    for folder in ('reader','forecasts'):shutil.copytree(original/folder,base/folder)
    cfg=dict(lineages=[1],queries=4,rules=design['rules'],models=list(O.MODELS),policies=list(O.POLICIES),costs=list(O.COSTS),input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()})
    result=O.run(root,dict(design=cfg),lambda **kw:None)
    assert len(result['cells'])==96 and result['decisions']==24
    by={(r['rule'],r['model'],r['weighting'],r['cost'],r['policy']):r for r in result['cells']}
    for key,row in by.items():
        assert abs(row['request_rate']-row['skill_request_rate']-row['belief_request_rate'])<1e-12
        if key[-1]=='optional':assert abs(row['request_rate']-by[(*key[:-1],'matched-rate')]['request_rate'])<1e-12
        if key[-1]=='none':assert row['request_rate']==0
        if key[-1]=='forced':assert abs(row['request_rate']-1)<1e-12
    for p in (root/'reader').glob('*.json'):assert set(read(p))<={'initial','operations','requested_field','disclosed_value'}
