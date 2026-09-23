import json
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import joint_reply_review as V, joint_reply as R
from ghostscale.validation.soundingline.v18_3.io import read, write
from test_v19_joint_reply import joint_fixture


def materialize(tmp_path):
    root,design = joint_fixture(tmp_path)
    summary = R.run(root,dict(design=design),lambda **kw:None)
    write(root/'PLAN.json',dict(design=design)); write(root/'SUMMARY.json',summary)
    out = tmp_path/'review'; out.mkdir()
    return root,out,dict(bootstrap_seed=190980,bootstrap_resamples=20)


def test_scalar_joint_odds_marginals_and_half_parity():
    legal = [(s,b,2,1,4,4) for s,b in product((0,1),repeat=2)]
    weights = [.1,.2,.3,.4]
    for mode in V.CHANNELS:
        p,m,f = V.conditional(legal,[0,1,2,3],weights,V.channel('both',.75,mode)[1])
        expected = np.array([.075,0,0,.1]) if mode == 'shared-flip' else np.array([.05625,.0375,.05625,.025])
        assert np.allclose(p[0,:4],expected/expected.sum()) and m[0] == pytest.approx(.175)
        p,m,f = V.conditional(legal,[0,1,2,3],weights,V.channel('both',1.,mode)[1])
        assert np.array_equal(p[:,:4],np.eye(4))
        for request in V.REQUESTS:
            p,m,f = V.conditional(legal,[2]*4,weights,V.channel(request,.75,mode)[1])
            assert np.all(p[:,2] == 1.)
    _,shared = V.channel('both',.5,'shared-flip')
    _,independent = V.channel('both',.5,'independent')
    for ids in ([0,1],[0,2]):
        assert np.array_equal(shared[ids].sum(0),independent[ids].sum(0))
    p,_,_ = V.conditional(legal,[0,1,2,3],[.25]*4,shared)
    assert np.array_equal(p[0,:4],[.5,0,0,.5])
    p,_,_ = V.conditional(legal,[0,1,2,3],[.25]*4,independent)
    assert np.array_equal(p[:,:4],np.full((4,4),.25))
    for request in V.REQUESTS[:-1]:
        assert np.array_equal(V.channel(request,.75,'shared-flip')[1],V.channel(request,.75,'independent')[1])
    p,m,f = V.conditional(legal[:2],[0,1],[1.,0.],V.channel('both',1.,'shared-flip')[1])
    assert f == [None,'uniform-likelihood-supported-legal','uniform-whole-legal-group','uniform-whole-legal-group']
    assert np.array_equal(p[2,:2],[.5,.5]) and p[1,1] == 1.
    with pytest.raises(ValueError): V.channel('both',float('nan'),'shared-flip')
    bad = shared.copy(); bad[0,0] += .1
    with pytest.raises(ValueError): V.conditional(legal,[0,1,2,3],[.25]*4,bad)


def test_expected_scores_infinity_and_constant_null():
    p = np.zeros((1,4,8)); p[0,:,0] = [1,.5,.25,0]; p[0,:,1] = 1-p[0,:,0]
    x = V.scores(p,np.array([[.5625,.1875,.1875,.0625]]),[0],[1.])
    assert x['infinite_loss_mass'] == .0625
    assert x['finite_loss_contribution'] == pytest.approx(.1875*np.log(2)+.1875*np.log(4))
    p[:] = 0; p[:,:,0] = 1
    x = V.scores(p,np.array([[.5625,.1875,.1875,.0625]]),[0],[1.])
    assert x['squared_error'] == x['infinite_loss_mass'] == 0.
    with pytest.raises(ValueError): V.scores(p,np.ones((1,4)),[0],[1.])


def test_complete_fixture_and_paired_strata(tmp_path):
    root,out,cfg = materialize(tmp_path); x = V.review(root,out,cfg)
    assert x['passed'] and x['cells'] == 384 and x['max_error'] < 1e-12
    estimates = read(out/'INDEPENDENT_REGROUP.json')['estimates']
    assert {r['contrast'] for r in estimates} == {'mean','minus-none','minus-skill','minus-belief','minus-joint-aware','shared-minus-independent'}
    assert {r['request'] for r in estimates} == set(V.REQUESTS)
    assert len(estimates) == 7488


@pytest.mark.parametrize('mutation',['forecast','reply','target','strings','likelihood','cost','law','fallback','reader','denominator'])
def test_corruption_rejected(tmp_path,mutation):
    root,out,cfg = materialize(tmp_path)
    if mutation in ('forecast','reply','target','strings'):
        p = next((root/'forecasts').glob('*uniform-legal-shared-flip-both*'))
        with np.load(p) as z: a = {k:z[k].copy() for k in z.files}
        if mutation == 'forecast': a['joint-aware'][0,0] = np.roll(a['joint-aware'][0,0],1)
        elif mutation == 'reply': a['reply_probabilities'][0] = a['reply_probabilities'][0][::-1]
        elif mutation == 'target': a['targets'][0] = (a['targets'][0]+1)%8
        else: a['strings'][0,0] = 1
        np.savez_compressed(p,**a)
    else:
        p = root/({'likelihood':'evaluator/CHANNELS.json','law':'inputs/DISCLOSURE_LAWS.json','fallback':'evaluator/REPLY_LAWS.json','reader':'reader/REPLIES.json'}.get(mutation,'SUMMARY.json'))
        data = read(p)
        if mutation == 'likelihood': data['shared-flip-both']['likelihood'][0][0] += .1
        elif mutation == 'law': data[0]['conditional_weights'][0] += .1
        elif mutation == 'fallback': data[0]['fallbacks'][0] = 'wrong'
        elif mutation == 'reader': next(iter(data.values()))['hidden_truth'] = 0
        elif mutation == 'cost': data['cells'][0]['net_finite_loss'] += .1
        else: data['law_rows'] += 1
        p.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError): V.review(root,out,cfg)
