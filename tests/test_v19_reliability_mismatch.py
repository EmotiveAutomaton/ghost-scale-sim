from itertools import product
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import reliability_mismatch as M
from ghostscale.validation.soundingline.v19 import joint_reply_review as V
from ghostscale.validation.soundingline.v18_3.io import read
from test_v19_joint_reply import joint_fixture, LEGAL


def test_known_bayes_odds_and_half_identity():
    for assumed in M.ACCURACIES:
        d = M.channel(LEGAL,[0,1,2,3],[.1,.2,.3,.4],assumed)
        for field,axis in [('skill',0),('belief',1)]:
            for bit in (0,1):
                weights = [w*(assumed if q[axis]==bit else 1-assumed) for q,w in zip(LEGAL,[.1,.2,.3,.4])]
                assert np.allclose(d['tables'][field][bit][:4],np.array(weights)/math.fsum(weights),atol=1e-15,rtol=0)
    assert all(M.controls().values())


def test_actual_accuracy_separate_from_assumed():
    d = M.channel(LEGAL,[0,1,2,3],[.25]*4,1.)
    p = np.tile(np.stack([d['tables']['skill'][b] for b in (0,1)]),(4,1,1))
    for actual in M.ACCURACIES:
        probabilities = M.reply_probabilities(LEGAL,'skill',actual)
        s = M.expected_scores(p,probabilities,np.arange(4),np.full(4,.25))
        assert s['infinite_loss_mass'] == pytest.approx(1-actual)
        assert s['finite_loss_contribution'] == pytest.approx(actual*math.log(2))
    assert np.array_equal(M.reply_probabilities(LEGAL,'none',.5),np.tile([1.,0.],(4,1)))


def test_truthful_parent_and_zero_support():
    prior,_,fields,_ = M.D.distributions(LEGAL,[0,1,2,3],[.25]*4)
    d = M.channel(LEGAL,[0,1,2,3],[.25]*4,1.)
    # Separate scalar implementation of the exact truthful channel.
    for request in ('skill','belief'):
        expected,_,_ = V.conditional(LEGAL,[0,1,2,3],[.25]*4,V.channel(request,1.,'independent')[1])
        assert np.array_equal(np.stack(list(d['tables'][request].values())),expected)
    d = M.channel(LEGAL[:2],[0,1],[1.,0.],1.)
    assert d['fallbacks']['belief'][1]=='uniform-likelihood-supported-legal'
    assert d['fallbacks']['skill'][1]=='uniform-whole-legal-group'
    assert np.array_equal(d['tables']['skill'][1][:2],[.5,.5])


def fixture(tmp_path):
    root,cfg = joint_fixture(tmp_path)
    cfg = {k:cfg[k] for k in ('lineages','queries','rules','models','costs','input_files')}
    cfg.update(actual_accuracies=list(M.ACCURACIES),assumed_accuracies=list(M.ACCURACIES),requests=list(M.REQUESTS))
    return root,cfg


def test_complete_fixture_scalar_reconstruction_roles_costs(tmp_path):
    root,cfg = fixture(tmp_path); result = M.run(root,dict(design=cfg),lambda **kw:None)
    assert len(result['cells'])==648 and result['fits']==0
    groups = read(root/'inputs/MEMBERSHIP.json')['omit-both']
    laws = {(r['lineage'],r['rule'],r['model'],r['reader_id']):r for r in read(root/'inputs/DISCLOSURE_LAWS.json')}
    for lin,rule,model,assumed in product(cfg['lineages'],cfg['rules'],cfg['models'],M.ACCURACIES):
        with np.load(root/'forecasts'/f'{lin}-{rule}-{model}-assumed-{assumed:g}_points.npz',allow_pickle=False) as data:
            for request in M.REQUESTS:
                for key,g in groups.items():
                    law = laws[(lin,rule,model,key)]
                    table = V.channel(request,assumed,'independent')[1]
                    pred,_,_ = V.conditional(g['legal_completions'],law['endpoints'],law['conditional_weights'],table)
                    if request=='none': pred = np.repeat(pred,2,axis=0)
                    assert np.allclose(data[request][g['indices']],pred,atol=1e-15,rtol=0)
    for row in result['cells']:
        assert row['net_finite_loss']==pytest.approx(row['finite_loss_contribution']+row['cost']*int(row['request']!='none'))
    for packet in read(root/'reader/REPLIES.json').values():
        assert set(packet)<={'initial','operations','requested_field','reply_value','stated_accuracy'}


def test_corruption_rejected(tmp_path):
    root,cfg = fixture(tmp_path)
    (root/'inputs/DISCLOSURE_LAWS.json').write_text('[]\n',encoding='utf-8')
    with pytest.raises(ValueError,match='input binding'): M.run(root,dict(design=cfg),lambda **kw:None)


def test_invalid_design_and_probability_rejected(tmp_path):
    root,cfg = fixture(tmp_path); cfg['actual_accuracies']=[.75]
    with pytest.raises(ValueError,match='design'): M.run(root,dict(design=cfg),lambda **kw:None)
    with pytest.raises(ValueError): M.reply_probabilities(LEGAL,'skill',float('nan'))
    p = np.zeros((4,2,8)); p[:,:,0]=1
    for invalid in (-.1,float('nan'),.6):
        bad = np.full((4,2),.5); bad[0,0]=invalid
        with pytest.raises(ValueError): M.expected_scores(p,bad,np.zeros(4,dtype=int),np.full(4,.25))
