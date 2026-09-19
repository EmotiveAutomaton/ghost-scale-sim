"""Run with the explicitly configured shared CPU PyTorch interpreter."""
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch',reason='optional shared CPU PyTorch runtime required')
from ghostscale.validation.soundingline.v18_3 import torch_worker as T


def data():
    h=torch.zeros(32,32,20);h[:,0,:16]=torch.eye(16).repeat(2,1)
    return dict(history=h,length=torch.ones(32,dtype=torch.long),sample=torch.arange(32),
                query=torch.zeros(32,4),target=torch.eye(16).repeat(2,1))


def test_cpu_gradient_and_all_model_positive_controls():
    x=torch.tensor([1.,2.,3.],requires_grad=True);(x*x).sum().backward()
    assert x.grad.tolist()==[2.,4.,6.] and not torch.cuda.is_initialized()
    for kind in ('direct','flat','split'):
        assert T.positive_control(kind,20,4,24)['passed']
    assert torch.get_num_threads()==1 and torch.get_num_interop_threads()==1


def test_optimizer_resume_and_checkpoint_integrity(tmp_path):
    d=data();config=dict(epochs=2,batch_size=8,learning_rate=.01)
    identity=dict(kind='split',seed=8,width=24)
    counter=[0]
    def stop():
        counter[0]+=1;return counter[0]>3
    partial=tmp_path/'partial';full=tmp_path/'full'
    assert T.fit(partial,identity,d,d,config,lambda **kw:None,stop) is None
    a=T.fit(partial,identity,d,d,config,lambda **kw:None,lambda:False)
    b=T.fit(full,identity,d,d,config,lambda **kw:None,lambda:False)
    sa=torch.load(partial/'BEST.pt',weights_only=True)['state'];sb=torch.load(full/'BEST.pt',weights_only=True)['state']
    assert all(torch.equal(sa[k],sb[k]) for k in sa)
    assert a['curve']==b['curve']
    (partial/'BEST.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='completed fit changed'):T.fit(partial,identity,d,d,config,lambda **kw:None,lambda:False)


def test_public_capsule_and_cached_forecast_equivalence(tmp_path):
    model=T.Reader('flat',20,4,24);d=data()
    payload=tmp_path/'best.pt';torch.save(dict(spec=model.spec,state=model.state_dict(),identity_sha256='fixture'),payload)
    with pytest.raises(ValueError,match='test truth'):T.forecast(payload,d,tmp_path/'bad.npz')
    public={k:v for k,v in d.items() if k!='target'}
    receipt=T.forecast(payload,public,tmp_path/'good.npz')
    assert receipt['rows']==32
    with torch.no_grad():expected=torch.softmax(model(*T.batch(d,torch.arange(32))),1).numpy()
    actual=np.load(tmp_path/'good.npz',allow_pickle=False)['probabilities']
    assert np.allclose(actual,expected,atol=1e-7)
    with pytest.raises(ValueError):T.confined(tmp_path,'../outside.npz')
