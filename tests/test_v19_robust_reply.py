import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import robust_reply as R
from ghostscale.validation.soundingline.v18_3.io import read
from test_v19_uncertain_reliability import fixture


def test_known_coding_bound_and_duplicate_control():
    assert all(R.controls().values())
    p=np.zeros((3,8));p[0,0]=1;p[1,1]=1;p[2,2]=1
    d=R.decisions(p,[.5]*3)
    assert d['normalizer']==3 and np.array_equal(d['mixture'],d['envelope'])
    assert np.all(d['envelope']*d['normalizer']>=p)


@pytest.mark.parametrize('defect',['empty','negative','mass','normalization','nan'])
def test_invalid_inputs_rejected(defect):
    p=np.zeros((3,8));p[:,0]=1;m=np.ones(3)
    if defect=='empty':m[:]=0
    elif defect=='negative':p[0,1]=-.1
    elif defect=='mass':m[0]=1.1
    elif defect=='normalization':p[0,0]=.9
    else:p[0,1]=np.nan
    with pytest.raises(ValueError):R.decisions(p,m)


def test_complete_fixture_scores_bound_and_roles(tmp_path):
    root,cfg=fixture(tmp_path);cfg.update(arms=list(R.ARMS),costs=list(R.COSTS))
    result=R.run(root,dict(design=cfg),lambda **kw:None)
    assert len(result['cells'])==432 and result['fits']==0
    for row in result['cells']:
        count=int(row['arm']!='none')
        assert row['request_rate']==count
        assert row['net_finite_loss']==pytest.approx(row['finite_loss_contribution']+row['cost']*count)
    for packet in read(root/'reader/REPLIES.json').values():
        assert set(packet)<={'initial','operations','requested_field','reply_value','reliability_candidates'}
    for p in (root/'forecasts').glob('*.npz'):
        with np.load(p,allow_pickle=False) as z:
            for idx in np.ndindex(z['normalizer'].shape):
                valid=z['compatible'][idx];candidates=z['candidates'][idx][valid]
                assert np.all(z['envelope'][idx]*z['normalizer'][idx]+1e-14>=candidates)
                assert z['envelope'][idx].sum()==pytest.approx(1.)
                assert np.allclose(z['mixture'][idx],candidates.mean(0))
            for field in R.FIELDS:
                for arm in R.ARMS:
                    prefix=f'{field}-{arm}';ptrue=z[prefix+'-true-probability'];positive=ptrue>0
                    assert np.allclose(z[prefix+'-finite-log-loss'][positive],-np.log(ptrue[positive]))
                    assert np.array_equal(z[prefix+'-infinite-loss'],~positive)
