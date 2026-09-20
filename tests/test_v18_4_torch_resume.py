"""Run separately in the existing optional CPU Torch interpreter."""
import pytest
torch=pytest.importorskip('torch')
from ghostscale.validation.soundingline.v18_4 import torch_worker as T


def test_checkpoint_resume_matches_uninterrupted_weights(tmp_path):
    torch.manual_seed(123)
    data=dict(history=torch.randn(16,32,5),length=torch.full((16,),8,dtype=torch.long),
        sample=torch.arange(16),query=torch.randn(16,3),target=torch.softmax(torch.randn(16,16),1))
    identity=dict(kind='flat',width=6,seed=1804)
    config=dict(epochs=3,batch_size=8,learning_rate=.003)
    full=T.fit(tmp_path/'full',identity,data,data,config,lambda **k:None,lambda:False)
    counter=[0]
    def limit():
        counter[0]+=1
        return counter[0]>3
    assert T.fit(tmp_path/'resume',identity,data,data,config,lambda **k:None,limit) is None
    resumed=T.fit(tmp_path/'resume',identity,data,data,config,lambda **k:None,lambda:False)
    a=torch.load(tmp_path/'full/BEST.pt',weights_only=True)['state']
    b=torch.load(tmp_path/'resume/BEST.pt',weights_only=True)['state']
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert resumed['curve']==full['curve']
    assert len(list((tmp_path/'resume').glob('checkpoint-*.pt')))<=2
