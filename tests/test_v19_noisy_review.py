import copy
import json
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import noisy_review as V, noisy_disclosure as N
from ghostscale.validation.soundingline.v18_3.io import read, write
from test_v19_noisy_disclosure import noisy_fixture


def materialize(tmp_path):
    root,design,_=noisy_fixture(tmp_path)
    summary=N.run(root,dict(design=design),lambda **kw:None)
    write(root/'PLAN.json',dict(design=design));write(root/'SUMMARY.json',summary)
    out=tmp_path/'review';out.mkdir()
    return root,out,dict(bootstrap_seed=190978,bootstrap_resamples=20)


def test_scalar_known_bayes_half_identity_truthful_and_fallback():
    legal=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    x=V.conditional(legal,[0,1,2,3],[.1,.2,.3,.4],.75)
    assert np.allclose(x['tables']['skill'][0][:4],np.array([.075,.15,.075,.1])/.4)
    assert x['reply_mass']['skill'][0]==pytest.approx(.4)
    half=V.conditional(legal,[0,1,2,3],[.1,.2,.3,.4],.5)
    for table in half['tables'].values():
        for p in table.values():assert np.array_equal(p,half['prior'])
    full=V.conditional(legal,[0,1,2,3],[.1,.2,.3,.4],1)
    assert np.allclose(full['tables']['belief'][1][:4],[0,1/3,0,2/3])
    fallback=V.conditional(legal[2:],[0,1],[1.,0.],1)
    assert np.array_equal(fallback['tables']['skill'][0][:2],[.5,.5])
    assert fallback['fallbacks']['skill'][0]=='uniform-whole-legal-group'
    assert fallback['fallbacks']['belief'][1]=='uniform-likelihood-supported-legal'
    with pytest.raises(ValueError):V.conditional(legal,[0,1,2,3],[.1,.2,.3,.4],float('nan'))


def test_expected_realized_scores_preserve_infinity_and_constant_null():
    p=np.zeros((1,2,8));p[0,0,0]=1;p[0,1,1]=1
    x=V.scores(p,np.array([[.75,.25]]),[0],[1.])
    assert x['infinite_loss_mass']==.25 and x['finite_loss_contribution']==0
    assert x['squared_error']==.5 and x['true_probability']==.75
    p[0,1]=p[0,0];x=V.scores(p,np.array([[.75,.25]]),[0],[1.])
    assert x['infinite_loss_mass']==0 and x['squared_error']==0
    with pytest.raises(ValueError):V.scores(p,np.array([[.75,.5]]),[0],[1.])


def test_complete_fixture_and_all_strata(tmp_path):
    root,out,cfg=materialize(tmp_path);x=V.review(root,out,cfg)
    assert x['passed'] and x['cells']==576 and x['max_error']<1e-12
    estimates=read(out/'INDEPENDENT_REGROUP.json')['estimates']
    assert {r['contrast'] for r in estimates}=={'mean','minus-none','minus-channel-aware','minus-skill','minus-belief'}
    assert {r['reliability'] for r in estimates}=={.5,.75,1.}


@pytest.mark.parametrize('mutation',['forecast','reply','target','choice','cost','law','entropy','fallback','sensitivity','reader','denominator'])
def test_corruption_rejected(tmp_path,mutation):
    root,out,cfg=materialize(tmp_path)
    if mutation in ('forecast','reply','target','choice'):
        p=next((root/'forecasts').glob('*uniform-legal-0.75-channel-aware*'))
        with np.load(p) as z:a={k:z[k].copy() for k in z.files}
        if mutation=='forecast':a['none'][0,0]=np.roll(a['none'][0,0],1)
        elif mutation=='reply':a['skill-reply-probabilities'][0]=a['skill-reply-probabilities'][0][::-1]
        elif mutation=='target':a['targets'][0]=(a['targets'][0]+1)%8
        else:a['choices'][0]=1-a['choices'][0]
        np.savez_compressed(p,**a)
    else:
        p=next((root/'reader').glob('*.json')) if mutation=='reader' else root/'inputs/DISCLOSURE_LAWS.json' if mutation=='law' else root/'evaluator/CHANNEL_LAWS.json' if mutation in ('entropy','fallback') else root/'evaluator/TIE_SENSITIVITY.json' if mutation=='sensitivity' else root/'SUMMARY.json'
        r=read(p)
        if mutation=='reader':r['hidden_truth']=0
        elif mutation=='law':r[0]['conditional_weights'][0]+=.1
        elif mutation=='entropy':r[0]['expected_entropy']['skill']+=.1
        elif mutation=='fallback':r[0]['fallbacks']['trusting']['skill']['0']='wrong'
        elif mutation=='sensitivity':r[0]['changed_queries']+=1
        elif mutation=='cost':r['cells'][0]['net_finite_loss']+=.1
        else:r['law_rows']+=1
        p.write_text(json.dumps(r),encoding='utf-8')
    with pytest.raises(ValueError):V.review(root,out,cfg)
