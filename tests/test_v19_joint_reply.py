from itertools import product
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import joint_reply as J, metadata_disclosure as D
from ghostscale.validation.soundingline.v18_3.io import read, file_digest
from test_v19_metadata_disclosure import fixture


LEGAL = [(s,b,2,1,4,4) for s,b in product((0,1),repeat=2)]


def test_joint_laws_same_marginals_different_dependence():
    for accuracy in (.5,.75,1.):
        laws = [J.likelihoods('both',accuracy,m)[1] for m in J.CHANNELS]
        for law in laws:
            assert np.array_equal(law[[0,1]].sum(0),[accuracy,accuracy,1-accuracy,1-accuracy])
            assert np.array_equal(law[[0,2]].sum(0),[accuracy,1-accuracy,accuracy,1-accuracy])
        assert np.array_equal(laws[0],laws[1]) == (accuracy==1)
        for request in J.REQUESTS[:-1]:
            assert np.array_equal(J.likelihoods(request,accuracy,J.CHANNELS[0])[1],J.likelihoods(request,accuracy,J.CHANNELS[1])[1])


def test_known_scalar_bayes_and_shared_parity():
    p,m,_ = J.posterior(LEGAL,[0,1,2,3],[.1,.2,.3,.4],J.likelihoods('both',.75,'shared-flip')[1])
    assert np.allclose(p[0,:4],[3/7,0,0,4/7],atol=1e-15)
    assert m[0] == pytest.approx(.175)
    p,m,_ = J.posterior(LEGAL,[0,1,2,3],[.1,.2,.3,.4],J.likelihoods('both',.75,'independent')[1])
    assert np.allclose(p[0,:4],np.array([.05625,.0375,.05625,.025])/.175,atol=1e-15)
    assert all(J.controls().values())


def test_half_truthful_no_request_and_constant_endpoints():
    weights=[.1,.2,.3,.4]
    for mode,request in product(J.CHANNELS,J.REQUESTS):
        p,m,_ = J.posterior(LEGAL,[2]*4,weights,J.likelihoods(request,.75,mode)[1])
        assert np.all(p[:,2]==1)
        p,m,_ = J.posterior(LEGAL,[0,1,2,3],weights,J.likelihoods('none',.75,mode)[1])
        assert np.allclose(p[0,:4],weights)
        p,m,_ = J.posterior(LEGAL,[0,1,2,3],weights,J.likelihoods('both',1.,mode)[1])
        assert np.array_equal(p[:,:4],np.eye(4))
        assert np.array_equal(m,weights)


def test_explicit_zero_support_fallbacks():
    legal=LEGAL[:2]
    p,m,f = J.posterior(legal,[0,1],[1.,0.],J.likelihoods('both',1.,'shared-flip')[1])
    assert f==[None,'uniform-likelihood-supported-legal','uniform-whole-legal-group','uniform-whole-legal-group']
    assert np.array_equal(m,[1,0,0,0])
    assert np.array_equal(p[1,:2],[0,1]) and np.array_equal(p[2,:2],[.5,.5])


def test_corrupt_likelihoods_and_bad_design_rejected():
    table=J.likelihoods('both',.75,'shared-flip')[1]
    for value in (-1.,float('nan'),.1):
        bad=table.copy();bad[0,0]=value
        with pytest.raises(ValueError,match='normalization'):J.posterior(LEGAL,[0,1,2,3],[.25]*4,bad)
    for args in [('both',float('nan'),'shared-flip'),('both',.4,'independent'),('other',.75,'independent')]:
        with pytest.raises(ValueError):J.likelihoods(*args)


def test_expected_realized_loss_preserves_infinity():
    p=np.zeros((1,4,8));p[0,:,0]=[1,.5,.25,0];p[0,:,1]=1-p[0,:,0]
    probs=np.array([[.75,0,0,.25]])
    score=J.expected_scores(p,probs,np.array([0]),np.array([1.]))
    assert score['infinite_loss_mass']==.25 and score['finite_loss_contribution']==0
    probs=np.full((1,4),.25)
    score=J.expected_scores(p,probs,np.array([0]),np.array([1.]))
    assert score['finite_loss_contribution']==pytest.approx(.25*(math.log(2)+math.log(4)))


def joint_fixture(tmp_path):
    parent=tmp_path/'parent';design=fixture(parent)
    D.run(parent,dict(design=design),lambda **kw:None)
    root=tmp_path/'joint';base=root/'inputs';base.mkdir(parents=True)
    for src,dst in [('evaluator/DISCLOSURE_LAWS.json','DISCLOSURE_LAWS.json'),('inputs/MEMBERSHIP.json','MEMBERSHIP.json')]:shutil.copyfile(parent/src,base/dst)
    shutil.copytree(parent/'forecasts',base/'forecasts')
    cfg=dict(lineages=[1],queries=4,rules=design['rules'],models=list(D.MODELS),requests=list(J.REQUESTS),channels=list(J.CHANNELS),readers=list(J.READERS),costs=list(D.COSTS),reliability=.75,input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()})
    return root,cfg


def test_complete_fixture_costs_roles_and_identity(tmp_path):
    root,cfg=joint_fixture(tmp_path);result=J.run(root,dict(design=cfg),lambda **kw:None)
    assert len(result['cells'])==384 and result['fits']==0
    for row in result['cells']:
        count={'none':0,'skill':1,'belief':1,'both':2}[row['request']]
        assert row['returned_replies']==count
        assert row['net_finite_loss']==pytest.approx(row['finite_loss_contribution']+row['cost']*count)
    for packet in read(root/'reader/REPLIES.json').values():
        assert set(packet)<={'initial','operations','requested_fields','replies','marginal_reliability','joint_source_relation'}
    for f in (root/'forecasts').glob('*.npz'):
        with np.load(f,allow_pickle=False) as a:
            if '-independent-' in f.name or not f.name.endswith('-both_points.npz'):
                assert np.array_equal(a['joint-aware'],a['marginals-product'])
            if f.name.endswith('-none_points.npz'):
                prefix=f.name.split('-independent-')[0].split('-shared-flip-')[0]
                with np.load(root/'inputs/forecasts'/f'{prefix}_points.npz',allow_pickle=False) as parent:
                    assert np.allclose(a['joint-aware'][:,0],parent['none'],atol=1e-15)


def test_input_corruption_rejected(tmp_path):
    root,cfg=joint_fixture(tmp_path);(root/'inputs/DISCLOSURE_LAWS.json').write_text('[]\n',encoding='utf-8')
    with pytest.raises(ValueError,match='input binding'):J.run(root,dict(design=cfg),lambda **kw:None)
