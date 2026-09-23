import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import optional_review as V, optional_disclosure as O, metadata_disclosure as D
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from test_v19_metadata_disclosure import fixture


def materialize(root):
    parent=root.parent/'parent';cfg=fixture(parent);D.run(parent,dict(design=cfg),lambda **kw:None)
    base=root/'inputs';base.mkdir(parents=True)
    shutil.copyfile(parent/'evaluator/DISCLOSURE_LAWS.json',base/'DISCLOSURE_LAWS.json')
    shutil.copyfile(parent/'inputs/MEMBERSHIP.json',base/'MEMBERSHIP.json')
    for folder in ('reader','forecasts'):shutil.copytree(parent/folder,base/folder)
    design=dict(lineages=[1],rules=cfg['rules'],queries=4,models=list(O.MODELS),policies=list(O.POLICIES),costs=list(O.COSTS),
        input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()})
    summary=O.run(root,dict(design=design),lambda **kw:None)
    write(root/'PLAN.json',dict(design=design));write(root/'SUMMARY.json',summary)
    return dict(bootstrap_seed=190977,bootstrap_resamples=20)


def test_known_answers_strict_ties_and_random_policy_scores():
    assert np.array_equal(V.decide([0,.02,.02001,.1],.02),[0,0,1,1])
    assert not V.decide([0],0)[0] and V.decide([np.log(2)],.1)[0]
    p=np.array([[.5,.5,0,0,0,0,0,0]]);q=np.array([[1.,0,0,0,0,0,0,0]])
    r=V.expected_scores(p,q,[0],[1.],[.5],.1)
    assert abs(r['finite_loss_contribution']-.5*np.log(2))<1e-12
    assert abs(r['net_finite_loss']-(.5*np.log(2)+.05))<1e-12
    assert abs(r['finite_loss_contribution']+np.log(.75))>.05
    r=V.expected_scores(q,np.roll(q,1,axis=1),[0],[1.],[.25],.1)
    assert r['infinite_loss_mass']==.25 and r['finite_loss_contribution']==0
    with pytest.raises(ValueError):V.decide([np.nan],0)
    with pytest.raises(ValueError):V.expected_scores(p,q,[0],[1.],[2.],0)


def test_complete_reconstruction_and_all_strata(tmp_path):
    root=tmp_path/'original';cfg=materialize(root);out=tmp_path/'review';out.mkdir()
    result=V.review(root,out,cfg)
    assert result['passed'] and result['cells']==96 and result['decisions']==24
    assert result['max_error']<1e-12
    estimates=read(out/'INDEPENDENT_REGROUP.json')['estimates']
    assert {r['contrast'] for r in estimates}=={'mean','minus-none','minus-forced','minus-matched-rate'}
    assert len(estimates)==1944


@pytest.mark.parametrize('mutation',['forecast','target','field','price','decision','matched-rate','law','reader','denominator'])
def test_corruption_rejected(tmp_path,mutation):
    root=tmp_path/'original';cfg=materialize(root);out=tmp_path/'review';out.mkdir()
    if mutation in ('forecast','target','field'):
        p=next((root/'inputs/forecasts').glob('*uniform-legal*'))
        with np.load(p) as z:a={k:z[k].copy() for k in z.files}
        if mutation=='forecast':a['none'][0]=np.roll(a['none'][0],1)
        elif mutation=='target':a['targets'][0]=(a['targets'][0]+1)%8
        else:a['choices'][0]=1-a['choices'][0]
        np.savez_compressed(p,**a)
    else:
        p=next((root/'reader').glob('*.json')) if mutation=='reader' else root/'inputs/DISCLOSURE_LAWS.json' if mutation=='law' else root/'evaluator/DECISIONS.json' if mutation in ('decision','matched-rate') else root/'SUMMARY.json'
        r=read(p)
        if mutation=='reader':r['target']=0
        elif mutation=='law':r[0]['prior_entropy']+=.1
        elif mutation=='decision':r[0]['optional'][0]=1-r[0]['optional'][0]
        elif mutation=='matched-rate':r[0]['matched_request_probability']+=.1
        elif mutation=='price':r['cells'][0]['net_finite_loss']+=.1
        else:r['decisions']+=1
        p.write_text(json.dumps(r),encoding='utf-8')
    with pytest.raises(ValueError):V.review(root,out,cfg)
