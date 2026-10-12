import numpy as np
from ghostscale.validation.soundingline.v21.provenance import controls,group_likelihood,conservative
from ghostscale.validation.soundingline.v21.provenance_check import reference


def test_copy_and_independent_known_answers():
    import json
    assert json.loads(json.dumps(controls()))==controls()
    assert all(controls().values())
    assert np.allclose(group_likelihood([0,1],[0,1],2,'hidden',0),[.5,.5])
    assert np.allclose(group_likelihood([1]*8,[0]*8,1,'hidden',0),[.2,.8])


def test_hidden_corrupt_and_permuted_groups():
    for mode,ids in [('hidden',[-1,1,-1,0]),('corrupt',[0,1,1,0])]:
        for rate in (.1,.3):
            p=group_likelihood([1,0,0,1],ids,2,mode,rate)
            assert np.allclose(p,reference([1,0,0,1],ids,2,mode,rate),atol=1e-12)
            renamed=[1-x if x>=0 else -1 for x in ids]
            assert np.allclose(p,group_likelihood([1,0,0,1],renamed,2,mode,rate),atol=1e-12)
    p=group_likelihood([1,1,0,0,1,0,1,0],[-1]*8,4,'hidden',.3)
    assert np.allclose(p,[.5,.5])


def test_conservative_envelope_does_not_count_copies():
    joint=np.array([[.5,0],[0,.5]])
    a,ba=conservative(joint,[1]*8,1);b,bb=conservative(joint,[1],1)
    assert np.allclose(a,b) and np.allclose(ba,bb)
    p,bounds=conservative(joint,[0,0,1,1],3)
    assert bounds[0]<.5<bounds[1] and np.allclose(p,[.5,.5])
