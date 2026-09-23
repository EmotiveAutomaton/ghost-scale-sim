from itertools import product
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import repeated_disclosure as R, metadata_disclosure as D, noisy_disclosure as N
from ghostscale.validation.soundingline.v18_3.io import read, file_digest
from test_v19_metadata_disclosure import fixture


def test_known_odds_independent_and_copied():
    legal=[(0,0,2,1,4,4),(1,0,2,1,4,4)]
    for length in (1,2,4):
        strings,table=R.likelihoods(length,.75,'independent')
        p,mass,fall=R.posterior(legal,[0,1],[.5,.5],0,table)
        assert p[0,0]==pytest.approx(3**length/(3**length+1))
        for i,bits in enumerate(strings):
            odds=3**(length-2*sum(bits))
            assert p[i,0]==pytest.approx(odds/(1+odds))
        _,copy=R.likelihoods(length,.75,'copied')
        q,m,_=R.posterior(legal,[0,1],[.5,.5],0,copy)
        assert q[0,0]==.75 and q[-1,0]==.25
        assert m[0]==m[-1]==.5
        if length>1:assert np.count_nonzero(m)==2
    assert all(R.controls().values())


def test_complete_strings_probability_and_corruption():
    for length,mode in product(R.LENGTHS,R.CHANNELS):
        strings,table=R.likelihoods(length,.75,mode)
        assert len(strings)==2**length and len(set(strings))==len(strings)
        assert np.allclose(table.sum(0),1,atol=1e-15)
        if mode=='copied' and length:
            for row,bits in zip(table,strings):
                assert bool(np.any(row))==(len(set(bits))==1)
        bad=table.copy();bad[0,0]+=.1
        with pytest.raises(ValueError,match='normalization'):R.validate_likelihoods(bad)
    with pytest.raises(ValueError):R.likelihoods(3,.75,'independent')
    with pytest.raises(ValueError):R.likelihoods(1,float('nan'),'copied')


def test_half_truthful_and_constant_controls():
    legal=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    for length,mode in product((1,2,4),R.CHANNELS):
        _,half=R.likelihoods(length,.5,mode)
        p,m,_=R.posterior(legal,[0,1,2,3],[.1,.2,.3,.4],0,half)
        assert np.allclose(p[m>0,:4],[.1,.2,.3,.4],atol=1e-15)
        _,full=R.likelihoods(length,1.,mode)
        p,m,_=R.posterior(legal,[0,1,2,3],[.1,.2,.3,.4],0,full)
        assert np.allclose(p[0,:4],[1/3,2/3,0,0])
        assert np.allclose(p[-1,:4],[0,0,3/7,4/7])
        p,m,_=R.posterior(legal,[2]*4,[.1,.2,.3,.4],0,half)
        assert np.all(p[:,2]==1)


def test_zero_support_and_impossible_copies():
    legal=[(1,0,2,1,4,4),(1,1,2,1,4,4)]
    _,table=R.likelihoods(2,1.,'copied')
    p,m,fall=R.posterior(legal,[0,1],[1.,0.],1,table)
    assert fall==[None,'uniform-whole-legal-group','uniform-whole-legal-group','uniform-likelihood-supported-legal']
    assert np.array_equal(p[1,:2],[.5,.5]) and np.array_equal(p[-1,:2],[0.,1.])
    assert np.array_equal(m,[1.,0.,0.,0.])


def test_expected_scores_and_infinite_mass():
    p=np.zeros((1,4,8));p[0,:,0]=[1,.5,.25,0];p[0,:,1]=1-p[0,:,0]
    probs=np.array([[.5625,.1875,.1875,.0625]])
    s=R.expected_scores(p,probs,np.array([0]),np.array([1.]))
    assert s['infinite_loss_mass']==.0625
    assert s['finite_loss_contribution']==pytest.approx(.1875*math.log(2)+.1875*math.log(4))
    assert s['true_probability']==pytest.approx((probs@p[0,:,0]).item())
    bad=probs.copy();bad[0,0]=1
    with pytest.raises(ValueError):R.expected_scores(p,bad,np.array([0]),np.array([1.]))


def repeated_fixture(tmp_path):
    parent=tmp_path/'parent';design=fixture(parent)
    D.run(parent,dict(design=design),lambda **kw:None)
    root=tmp_path/'repeated';base=root/'inputs';base.mkdir(parents=True)
    for src,dst in [('evaluator/DISCLOSURE_LAWS.json','DISCLOSURE_LAWS.json'),('inputs/MEMBERSHIP.json','MEMBERSHIP.json')]:shutil.copyfile(parent/src,base/dst)
    shutil.copytree(parent/'forecasts',base/'forecasts')
    cfg=dict(lineages=[1],queries=4,rules=design['rules'],models=list(D.MODELS),fields=list(R.FIELDS),channels=list(R.CHANNELS),readers=list(R.READERS),lengths=list(R.LENGTHS),costs=list(D.COSTS),cost_modes=list(R.COST_MODES),reliability=.75,input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()})
    return root,cfg


def test_full_fixture_costs_roles_and_copy_identity(tmp_path):
    root,cfg=repeated_fixture(tmp_path);result=R.run(root,dict(design=cfg),lambda **kw:None)
    assert len(result['cells'])==1536 and result['fits']==0
    axes=('rule','model','field','channel','length','reader','weighting','cost','cost_mode')
    cells={tuple(r[k] for k in axes):r for r in result['cells']}
    for key,r in cells.items():
        assert r['net_finite_loss']==pytest.approx(r['finite_loss_contribution']+r['cost']*r['charged_replies'])
        assert r['returned_replies']==r['length']
        assert r['acquisitions']==(int(r['length']>0) if r['channel']=='copied' else r['length'])
        if r['length']>1 and r['channel']=='copied' and r['reader']=='channel-aware':
            baseline=cells[(*key[:4],1,*key[5:])]
            for m in ('finite_loss_contribution','infinite_loss_mass','squared_error','true_probability','residual_ambiguous_mass'):assert r[m]==pytest.approx(baseline[m],abs=1e-14)
    for packet in read(root/'reader/REPLIES.json').values():assert set(packet)<={'initial','operations','requested_field','replies','reply_reliability','reply_source_mode'}
    for f in (root/'forecasts').glob('*.npz'):
        with np.load(f) as x:
            length=x['strings'].shape[1];assert x['channel-aware'].shape==(4,2**length,8)
            assert x['reply_probabilities'].shape==(4,2**length)
            if length<=1 or '-independent-' in f.name:assert np.array_equal(x['channel-aware'],x['independence-assumed'])


def test_single_reply_agrees_with_parent_noisy_channel():
    legal=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    old=N.channel(legal,[0,1,2,3],[.1,.2,.3,.4],.75)
    for field,axis in [('skill',0),('belief',1)]:
        p,m,f=R.posterior(legal,[0,1,2,3],[.1,.2,.3,.4],axis,R.likelihoods(1,.75,'independent')[1])
        for bit in (0,1):assert np.allclose(p[bit],old['tables'][field][bit],atol=1e-15)


def test_input_corruption_rejected(tmp_path):
    root,cfg=repeated_fixture(tmp_path);(root/'inputs/DISCLOSURE_LAWS.json').write_text('[]\n',encoding='utf-8')
    with pytest.raises(ValueError,match='input binding'):R.run(root,dict(design=cfg),lambda **kw:None)
