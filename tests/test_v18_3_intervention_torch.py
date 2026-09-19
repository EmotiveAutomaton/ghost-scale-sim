import numpy as np
import pytest
import torch
from ghostscale.validation.soundingline.v18_3 import intervention_worker as I


def fixture():
    bits=torch.tensor([[int(bool(i&(1<<j))) for j in range(3)] for i in range(8)])
    h=torch.zeros(8,32,8);h[:,0,:3]=bits.float()*2-1
    rows=[(a,b,r) for a in range(8) for b in range(8) for r in range(3)]
    a,b,r=map(torch.tensor,zip(*rows));mixed=bits[a].clone();mixed[torch.arange(len(a)),r]=bits[b,r]
    labels=(mixed*torch.tensor([1,2,4])).sum(1)
    return dict(history=h,length=torch.ones(8,dtype=torch.long),base=a,source=b,role=r,query=torch.zeros(len(a),4),
        target=torch.nn.functional.one_hot(labels,16).float(),behavior_target=torch.nn.functional.one_hot(a,16).float())


def test_swap_and_compatible_coordinate_partition_known_answer():
    model=I.Model('split-IIT',8,4,12);b=torch.arange(24).reshape(2,12).float();s=-b-1;r=torch.tensor([0,2]);q=torch.zeros(2,4)
    actual=model.transplant(b,s,r)
    assert torch.equal(actual[0,:4],s[0,:4]) and torch.equal(actual[0,4:],b[0,4:])
    assert torch.equal(actual[1,8:],s[1,8:]) and torch.equal(actual[1,:8],b[1,:8])
    with torch.no_grad():assert I.permutation_check(model,b,s,q,r)==0
    mixed=torch.where(torch.arange(12)[None,:]%3==r[:,None],s,b)
    assert not torch.equal(mixed,actual)
    assert torch.equal(model.incompatible_partition(b,s,q,r),model.reader.decode(mixed,q))
    with pytest.raises(ValueError):I.Model('direct-pair',8,4,12).incompatible_partition(b,s,q,r)


def test_intervention_learning_positive_control_all_informed_rivals():
    d=fixture();idx=torch.arange(len(d['base']))
    for method in ('split-IIT','flat-IIT','direct-pair'):
        torch.manual_seed(180301);model=I.Model(method,8,4,12,head=32,direct_hidden=8);opt=torch.optim.Adam(model.parameters(),lr=.02)
        for _ in range(180):
            opt.zero_grad();cf,b=model(*I.batch(d,idx));value=I.objective(cf,b,d,idx,method);value.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
        with torch.no_grad():cf,b=model(*I.batch(d,idx));accuracy=(cf.argmax(1)==d['target'].argmax(1)).float().mean()
        assert accuracy>.95,(method,float(accuracy))


def test_intervention_resume_and_test_label_boundary(tmp_path):
    d=fixture();identity=dict(method='split-IIT',seed=8,width=12);config=dict(epochs=2,batch_size=32,learning_rate=.01)
    counter=[0]
    def stop():counter[0]+=1;return counter[0]>2
    assert I.fit(tmp_path/'partial',identity,d,d,config,lambda **kw:None,stop) is None
    a=I.fit(tmp_path/'partial',identity,d,d,config,lambda **kw:None,lambda:False)
    b=I.fit(tmp_path/'full',identity,d,d,config,lambda **kw:None,lambda:False)
    assert a['curve']==b['curve']
    sa=torch.load(tmp_path/'partial/BEST.pt',weights_only=True)['state'];sb=torch.load(tmp_path/'full/BEST.pt',weights_only=True)['state']
    assert all(torch.equal(sa[k],sb[k]) for k in sa)
    np.savez(tmp_path/'truth.npz',**{k:v.numpy() for k,v in d.items()})
    with pytest.raises(ValueError,match='hidden test label'):I.dataset(tmp_path/'truth.npz',public=True)
