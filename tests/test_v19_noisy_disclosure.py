from itertools import product
import copy
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import noisy_disclosure as N, metadata_disclosure as D
from ghostscale.validation.soundingline.v18_3.io import read, file_digest
from test_v19_metadata_disclosure import fixture


def test_scalar_bayes_known_channel_and_constant_null():
    assert all(N.controls().values())
    legal=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    x=N.channel(legal,[0,1,2,3],[.1,.2,.3,.4],.75)
    assert np.allclose(x['tables']['skill'][0][:4],np.array([.075,.15,.075,.1])/.4)
    assert x['reply_mass']['skill'][0]==pytest.approx(.4)
    for field,axis in [('skill',0),('belief',1)]:
        for reply in (0,1):
            joint=np.array([w*(.75 if q[axis]==reply else .25) for w,q in zip([.1,.2,.3,.4],legal)])
            assert np.allclose(x['tables'][field][reply][:4],joint/joint.sum(),atol=1e-15)
    N.validate_channel(x)
    bad=copy.deepcopy(x);bad['tables']['skill'][0][0]+=.1
    with pytest.raises(ValueError,match='normalization'):N.validate_channel(bad)
    bad=copy.deepcopy(x);bad['reply_mass']['belief'][1]+=.1
    with pytest.raises(ValueError,match='mass'):N.validate_channel(bad)


def test_half_channel_identity_and_truthful_parent():
    legal=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    ends=[0,1,2,3];weights=[.1,.2,.3,.4]
    half=N.channel(legal,ends,weights,.5)
    for table in half['tables'].values():
        for p in table.values():assert np.array_equal(p,half['prior'])
    full=N.channel(legal,ends,weights,1.)
    prior,tables,entropies,choice=D.distributions(legal,ends,weights)
    assert np.allclose(full['prior'],prior)
    for field in tables:
        for bit,p in tables[field].items():assert np.allclose(full['tables'][field][bit],p,atol=1e-15)
        assert full['expected_entropy'][field]==pytest.approx(entropies[field])
    assert full['choice']==choice


def test_zero_mass_and_impossible_literal_reply_fallback():
    legal=[(1,b,2,1,4,4) for b in (0,1)]
    full=N.channel(legal,[0,1],[1.,0.],1.)
    assert np.array_equal(full['tables']['skill'][0][:2],[.5,.5])
    assert full['fallbacks']['skill'][0]=='uniform-whole-legal-group'
    assert np.array_equal(full['tables']['belief'][1][:2],[0.,1.])
    assert full['fallbacks']['belief'][1]=='uniform-likelihood-supported-legal'
    noisy=N.channel(legal,[0,1],[1.,0.],.75)
    assert np.array_equal(noisy['tables']['skill'][0][:2],[1.,0.])
    with pytest.raises(ValueError):N.channel(legal,[0,1],[1.,0.],float('nan'))


def test_expected_scores_preserve_infinity_and_do_not_score_average():
    p=np.zeros((1,2,8));p[0,0,0]=1;p[0,1,1]=1
    result=N.expected_scores(p,np.array([[.75,.25]]),np.array([0]),np.array([1.]))
    assert result['infinite_loss_mass']==.25 and result['finite_loss_contribution']==0
    assert result['squared_error']==.5 and result['true_probability']==.75
    with pytest.raises(ValueError,match='reply probabilities'):
        N.expected_scores(p,np.array([[.75,.75]]),np.array([0]),np.array([1.]))


def noisy_fixture(tmp_path):
    original=tmp_path/'original';design=fixture(original)
    parent=D.run(original,dict(design=design),lambda **kw:None)
    root=tmp_path/'noisy';base=root/'inputs';base.mkdir(parents=True)
    shutil.copyfile(original/'evaluator/DISCLOSURE_LAWS.json',base/'DISCLOSURE_LAWS.json')
    shutil.copyfile(original/'inputs/MEMBERSHIP.json',base/'MEMBERSHIP.json')
    shutil.copytree(original/'forecasts',base/'forecasts')
    cfg=dict(lineages=[1],queries=4,rules=design['rules'],models=list(D.MODELS),policies=list(N.POLICIES),costs=list(D.COSTS),reliabilities=list(N.RELIABILITIES),readers=list(N.READERS),input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()})
    return root,cfg,parent


def test_full_fixture_channel_identities_costs_roles_and_reply_coverage(tmp_path):
    root,cfg,parent=noisy_fixture(tmp_path);result=N.run(root,dict(design=cfg),lambda **kw:None)
    assert len(result['cells'])==576 and result['fits']==0
    by={(r['rule'],r['model'],r['reliability'],r['reader'],r['policy'],r['weighting'],r['cost']):r for r in result['cells']}
    original={(r['rule'],r['model'],r['policy'],r['weighting'],r['cost']):r for r in parent['cells']}
    for key,row in by.items():
        assert abs(row['net_finite_loss']-row['finite_loss_contribution']-row['cost']*row['request_rate'])<1e-12
        assert abs(row['skill_request_rate']+row['belief_request_rate']-row['request_rate'])<1e-12
        metrics=('finite_loss_contribution','infinite_loss_mass','squared_error','true_probability')
        if row['reliability']==.5 and row['reader']=='channel-aware':
            none=by[(*key[:4],'none',*key[5:])]
            for metric in metrics:assert row[metric]==pytest.approx(none[metric],abs=1e-14)
        if row['reliability']==1:
            old=original[(key[0],key[1],key[4],key[5],key[6])]
            for metric in metrics:assert row[metric]==pytest.approx(old[metric],abs=1e-14)
    rows=read(root/'evaluator/CHANNEL_LAWS.json')
    for row in rows:
        assert row['choice']==('skill' if row['expected_entropy']['skill']<=row['expected_entropy']['belief'] else 'belief')
    for p in (root/'reader').glob('*.json'):
        packet=read(p);assert set(packet)<={'initial','operations','requested_field','reply_value','reply_reliability'}
    for p in (root/'forecasts').glob('*.npz'):
        with np.load(p) as data:
            for policy in N.POLICIES:assert data[policy].shape==(4,2,8)


def test_frozen_reply_law_corruption_is_rejected(tmp_path):
    root,cfg,_=noisy_fixture(tmp_path)
    p=root/'inputs/DISCLOSURE_LAWS.json';p.write_text('[]\n',encoding='utf-8')
    with pytest.raises(ValueError,match='input binding'):N.run(root,dict(design=cfg),lambda **kw:None)
