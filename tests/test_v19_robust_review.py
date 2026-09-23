import json
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import robust_reply as R,robust_review as V
from ghostscale.validation.soundingline.v18_3.io import read,write
from test_v19_uncertain_reliability import fixture


def prepared(tmp_path):
    root,cfg=fixture(tmp_path);cfg.update(arms=list(R.ARMS),costs=list(R.COSTS));write(root/'PLAN.json',dict(design=cfg))
    write(root/'SUMMARY.json',R.run(root,dict(design=cfg),lambda **kw:None));out=tmp_path/'review';out.mkdir()
    return root,out,dict(bootstrap_seed=190983,bootstrap_resamples=50)


def test_complete_scalar_fixture(tmp_path):
    root,out,cfg=prepared(tmp_path);r=V.review(root,out,cfg)
    assert r['passed'] and r['cells']==432 and r['max_error']<1e-12
    assert len(read(out/'INDEPENDENT_REGROUP.json')['estimates'])==5760


@pytest.mark.parametrize('key',['mixture','envelope','normalizer','compatible','skill-equal-mixture-true-probability','skill-none-squared-error'])
def test_saved_decision_or_score_corruption_rejected(tmp_path,key):
    root,out,cfg=prepared(tmp_path);p=next((root/'forecasts').glob('*.npz'))
    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    if a[key].dtype==bool:a[key].flat[0]=not a[key].flat[0]
    else:a[key].flat[0]+=.1
    np.savez_compressed(p,**a)
    with pytest.raises(ValueError):V.review(root,out,cfg)


def test_hidden_reader_truth_rejected(tmp_path):
    root,out,cfg=prepared(tmp_path);p=root/'reader/REPLIES.json';v=read(p);next(iter(v.values()))['true_reliability']=1
    p.write_text(json.dumps(v),encoding='utf-8')
    with pytest.raises(ValueError,match='reader'):V.review(root,out,cfg)
