import numpy as np
import pytest
import torch
from ghostscale.validation.soundingline.v18_3 import purpose_worker as P,torch_worker as T


def test_frozen_features_and_role_label_boundary(tmp_path):
    data=dict(history=np.zeros((4,32,8),np.float32),length=np.ones(4,np.int64),world=np.eye(4,dtype=np.float32))
    model=T.Reader('flat',8,4,24);path=tmp_path/'encoder.pt';torch.save(dict(spec=model.spec,state=model.state_dict()),path)
    before=T.sha(path);features=P.features(data,path)
    assert features.shape==(4,28) and T.sha(path)==before
    assert np.array_equal(features[:,-4:],data['world'])
    np.savez(tmp_path/'public.npz',**data)
    assert set(P.dataset(tmp_path/'public.npz',public=True))==set(data)
    np.savez(tmp_path/'tainted.npz',**data,target=np.ones((4,9))/9)
    with pytest.raises(ValueError,match='hidden role truth'):P.dataset(tmp_path/'tainted.npz',public=True)
