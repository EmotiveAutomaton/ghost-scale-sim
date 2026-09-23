import json
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import disclosure_review as V, metadata_disclosure as D
from ghostscale.validation.soundingline.v18_3.io import read, write
from test_v19_metadata_disclosure import fixture


def materialize(root):
    cfg=fixture(root);summary=D.run(root,dict(design=cfg),lambda **kw:None)
    write(root/'PLAN.json',dict(design=cfg));write(root/'SUMMARY.json',summary)
    return dict(bootstrap_seed=190976,bootstrap_resamples=20)


def test_independent_informative_field_null_and_zero_mass():
    legal=[(0,0),(0,1),(1,0),(1,1)]
    assert V.conditional(legal,[0,1,0,1],[.25]*4)[3]=='belief'
    assert V.conditional(legal,[2]*4,[.25]*4)[3]=='skill'
    _,tables,_,_=V.conditional(legal,[0,1,2,3],[.5,.5,0,0])
    assert np.array_equal(tables['skill'][1],[0,0,.5,.5,0,0,0,0])


def test_full_reconstruction_and_expected_request_scores(tmp_path):
    cfg=materialize(tmp_path/'original');out=tmp_path/'review';out.mkdir()
    result=V.review(tmp_path/'original',out,cfg)
    assert result['passed'] and result['cells']==144 and result['native_paths']==8
    assert result['max_error']<1e-12
    assert len(read(out/'INDEPENDENT_REGROUP.json')['estimates'])==2304


@pytest.mark.parametrize('mutation',['forecast','choice','cost','reader','law','denominator'])
def test_corrupt_evidence_rejected(tmp_path,mutation):
    root=tmp_path/'original';cfg=materialize(root);out=tmp_path/'review';out.mkdir()
    if mutation in ('forecast','choice'):
        p=next((root/'forecasts').glob('*uniform-legal*'))
        with np.load(p) as z: arrays={k:z[k].copy() for k in z.files}
        if mutation=='forecast': arrays['skill'][0]=np.roll(arrays['skill'][0],1)
        else: arrays['choices'][0]=1-arrays['choices'][0]
        np.savez_compressed(p,**arrays)
    else:
        p=next((root/'reader').glob('*.json')) if mutation=='reader' else root/'evaluator/DISCLOSURE_LAWS.json' if mutation=='law' else root/'SUMMARY.json'
        row=read(p)
        if mutation=='reader': row['target']=0
        elif mutation=='law': row[0]['conditional_weights'][0]+=.1
        elif mutation=='cost': row['cells'][0]['net_finite_loss']+=.1
        else: row['native_paths']+=1
        p.write_text(json.dumps(row),encoding='utf-8')
    with pytest.raises(ValueError): V.review(root,out,cfg)
