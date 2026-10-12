import numpy as np
from ghostscale.validation.soundingline.v21.reliability import update,controls
from ghostscale.validation.soundingline.v21.reliability_check import reference


def test_known_binary_bayes_and_uninformative_record():
    p=np.zeros((1,128));p[0,0]=.75;p[0,4]=.25
    with np.errstate(divide='ignore'):lp=np.log(p)
    answer,_=update(lp,[1],.8)
    assert np.allclose(np.exp(answer)[0,[0,4]],[3/7,4/7])
    assert np.allclose(np.exp(update(lp,[1],.5)[0]),p)
    assert np.allclose(np.exp(answer),reference(p,[1],.8))


def test_false_authoritative_record_and_empty_support_conflict():
    lp=np.full((1,128),-np.log(128))
    soft,_=update(lp,[1],.95);hard,_=update(lp,[1],.95,True)
    assert np.isfinite(soft[0,0]) and np.isneginf(hard[0,0])
    point=np.full((1,128),-np.inf);point[0,0]=0
    fallback,conflict=update(point,[1],1,True)
    assert conflict[0] and np.array_equal(fallback,point)
    assert all(controls().values())
