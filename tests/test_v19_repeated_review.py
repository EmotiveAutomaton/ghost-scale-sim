import json
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import repeated_review as V, repeated_disclosure as R
from ghostscale.validation.soundingline.v18_3.io import read, write
from test_v19_repeated_disclosure import repeated_fixture


def materialize(tmp_path):
    root,design = repeated_fixture(tmp_path)
    summary = R.run(root,dict(design=design),lambda **kw:None)
    write(root/'PLAN.json',dict(design=design)); write(root/'SUMMARY.json',summary)
    out = tmp_path/'review'; out.mkdir()
    return root,out,dict(bootstrap_seed=190979,bootstrap_resamples=20)


def test_scalar_odds_copy_half_truthful_and_fallback():
    legal = [(s,0,2,1,4,4) for s in (0,1)]
    for length in (1,2,4):
        bits,table = V.channel(length,.75,'independent')
        p,m,f = V.conditional(legal,[0,1],[.5,.5],0,table)
        for i,seq in enumerate(bits):
            odds = 3**(length-2*sum(seq))
            assert p[i,0] == pytest.approx(odds/(1+odds))
        p,m,f = V.conditional(legal,[0,1],[.5,.5],0,V.channel(length,.75,'copied')[1])
        assert p[0,0] == .75 and p[-1,0] == .25
        assert sum(x > 0 for x in m) == 2
        for mode in V.CHANNELS:
            p,m,f = V.conditional(legal,[0,1],[.2,.8],0,V.channel(length,.5,mode)[1])
            assert np.allclose(p[np.array(m)>0,:2],[.2,.8])
            p,m,f = V.conditional(legal,[0,1],[.2,.8],0,V.channel(length,1.,mode)[1])
            assert p[0,0] == p[-1,1] == 1.
            p,m,f = V.conditional(legal,[2,2],[.2,.8],0,V.channel(length,.75,mode)[1])
            assert np.all(p[:,2] == 1.)
    p,m,f = V.conditional(legal,[0,1],[1.,0.],0,V.channel(2,1.,'copied')[1])
    assert f == [None,'uniform-whole-legal-group','uniform-whole-legal-group','uniform-likelihood-supported-legal']
    assert np.array_equal(p[1,:2],[.5,.5]) and p[-1,1] == 1.
    with pytest.raises(ValueError): V.channel(2,float('nan'),'copied')
    bad = V.channel(2,.75,'copied')[1]; bad[0,0] += .1
    with pytest.raises(ValueError): V.conditional(legal,[0,1],[.5,.5],0,bad)


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
    assert x['passed'] and x['cells'] == 1536 and x['max_error'] < 1e-12
    estimates = read(out/'INDEPENDENT_REGROUP.json')['estimates']
    assert {r['contrast'] for r in estimates} == {'mean','minus-zero','minus-one','minus-channel-aware','copied-minus-independent'}
    assert {r['length'] for r in estimates} == {0,1,2,4}
    assert {r['cost_mode'] for r in estimates} == set(V.COST_MODES)


@pytest.mark.parametrize('mutation',['forecast','reply','target','strings','likelihood','cost','law','fallback','reader','denominator'])
def test_corruption_rejected(tmp_path,mutation):
    root,out,cfg = materialize(tmp_path)
    if mutation in ('forecast','reply','target','strings'):
        p = next((root/'forecasts').glob('*uniform-legal-skill-copied-2*'))
        with np.load(p) as z: a = {k:z[k].copy() for k in z.files}
        if mutation == 'forecast': a['channel-aware'][0,0] = np.roll(a['channel-aware'][0,0],1)
        elif mutation == 'reply': a['reply_probabilities'][0] = a['reply_probabilities'][0][::-1]
        elif mutation == 'target': a['targets'][0] = (a['targets'][0]+1)%8
        else: a['strings'][0,0] = 1
        np.savez_compressed(p,**a)
    else:
        p = root/({'likelihood':'evaluator/CHANNELS.json','law':'inputs/DISCLOSURE_LAWS.json','fallback':'evaluator/REPLY_LAWS.json','reader':'reader/REPLIES.json'}.get(mutation,'SUMMARY.json'))
        data = read(p)
        if mutation == 'likelihood': data['copied-2']['likelihood'][0][0] += .1
        elif mutation == 'law': data[0]['conditional_weights'][0] += .1
        elif mutation == 'fallback': data[0]['fallbacks'][0] = 'wrong'
        elif mutation == 'reader': next(iter(data.values()))['hidden_truth'] = 0
        elif mutation == 'cost': data['cells'][0]['net_finite_loss'] += .1
        else: data['law_rows'] += 1
        p.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError): V.review(root,out,cfg)
