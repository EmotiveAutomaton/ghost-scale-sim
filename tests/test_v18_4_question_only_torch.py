"""Known-answer and no-history controls, run in the optional CPU interpreter."""
import pytest
torch=pytest.importorskip('torch')
from ghostscale.validation.soundingline.v18_4 import torch_worker as T


def test_question_only_history_noninterference_and_live_query():
    torch.manual_seed(199)
    model=T.architecture('question-only',34,34,6)
    direct=T.architecture('direct',34,34,6)
    assert T.count(model)==T.count(direct)
    h=torch.randn(12,32,34);length=torch.full((12,),8,dtype=torch.long);q=torch.randn(12,34)
    a=model(h,length,q);b=model(torch.randn_like(h)*100,length*3,q)
    assert torch.equal(a,b)
    assert not torch.equal(a,model(h,length,torch.zeros_like(q)))
    assert T.positive_control('question-only',34,34,6)['passed']


def test_question_only_resume_and_forecast(tmp_path):
    torch.manual_seed(199)
    data=dict(history=torch.randn(16,32,5),length=torch.full((16,),8,dtype=torch.long),
        sample=torch.arange(16),query=torch.randn(16,20),target=torch.softmax(torch.randn(16,16),1))
    identity=dict(kind='question-only',width=6,seed=1804)
    config=dict(epochs=2,batch_size=8,learning_rate=.003)
    a=T.fit(tmp_path/'fit',identity,data,data,config,lambda **kw:None,lambda:False)
    b=T.fit(tmp_path/'fit',identity,data,data,config,lambda **kw:None,lambda:False)
    assert a==b
    public={k:v for k,v in data.items() if k!='target'}
    T.forecast(tmp_path/'fit/BEST.pt',public,tmp_path/'pred.npz')
    import numpy as np
    with np.load(tmp_path/'pred.npz') as z:assert np.allclose(z['probabilities'].sum(1),1)
