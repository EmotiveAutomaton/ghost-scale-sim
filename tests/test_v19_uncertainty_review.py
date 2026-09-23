import json
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import uncertain_reliability as U, uncertainty_review as V
from ghostscale.validation.soundingline.v18_3.io import write,read
from test_v19_uncertain_reliability import fixture


def prepared(tmp_path):
    root,cfg=fixture(tmp_path);write(root/'PLAN.json',dict(design=cfg))
    write(root/'SUMMARY.json',U.run(root,dict(design=cfg),lambda **kw:None))
    output=tmp_path/'review';output.mkdir()
    return root,output,dict(bootstrap_seed=190982,bootstrap_resamples=50)


def test_complete_scalar_fixture(tmp_path):
    root,out,cfg=prepared(tmp_path);result=V.review(root,out,cfg)
    assert result['passed'] and result['cells']==48 and result['max_error']<1e-12
    assert len(read(out/'INDEPENDENT_REGROUP.json')['estimates'])==336


@pytest.mark.parametrize('defect',['candidate','bound','pair','mask','summary','reader','index','fallback','omission'])
def test_record_corruption_is_rejected(tmp_path,defect):
    root,out,cfg=prepared(tmp_path)
    if defect in ('candidate','bound','pair','mask'):
        p=next((root/'sets').glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        key={'candidate':'candidates','bound':'lower','pair':'pairwise_tv','mask':'compatible'}[defect]
        if defect=='mask':values[key].flat[0]=not values[key].flat[0]
        else:values[key].flat[0]+=.125
        np.savez_compressed(p,**values)
    elif defect=='omission':next((root/'sets').glob('*.npz')).unlink()
    else:
        p=root/{'summary':'SUMMARY.json','reader':'reader/REPLIES.json','index':'evaluator/INDEX.json','fallback':'evaluator/FALLBACKS.json'}[defect]
        data=read(p)
        if defect=='summary':data['cells'][0]['diameter']+=.1
        elif defect=='reader':next(iter(data.values()))['hidden_truth']=1
        elif defect=='index':data['reliabilities']=[.5,1.]
        else:data.append(dict(invented=True))
        p.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises((ValueError,FileNotFoundError,AssertionError)):V.review(root,out,cfg)
