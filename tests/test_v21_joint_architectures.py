import numpy as np
from ghostscale.validation.soundingline.v21.joint_architectures import JointReader,KINDS,head
from ghostscale.validation.soundingline.v21.joint_check import reference


def test_scalar_probabilities_and_known_answers():
    rng=np.random.default_rng(913)
    for kind in KINDS:
        n=128 if kind=='direct' else 32 if kind=='low-rank' else 28
        zero=np.zeros((2,n));assert np.allclose(np.exp(head(zero,kind)[0]),1/128)
        output=rng.normal(0,.5,(3,n));lp,_=head(output,kind)
        assert np.allclose(np.exp(lp).sum(1),1)
        assert np.allclose(np.exp(lp),reference(output,kind),atol=1e-12)
        positive=zero.copy();positive[:,0 if kind!='low-rank' else 4]=4
        assert not np.allclose(np.exp(head(positive,kind)[0]),1/128)


def test_independent_finite_difference_gradients():
    rng=np.random.default_rng(917);z=rng.normal(size=(5,3));y=np.array([0,23,51,86,127])
    for kind in KINDS:
        model=JointReader(3,kind,512);_,grads=model.gradient(z,y)
        for p,g in zip(model.params,grads):
            for index in sorted(set([0,p.size//2,p.size-1])):
                old=p.flat[index];eps=1e-5
                p.flat[index]=old+eps;plus=model.gradient(z,y)[0]
                p.flat[index]=old-eps;minus=model.gradient(z,y)[0];p.flat[index]=old
                assert abs((plus-minus)/(2*eps)-g.flat[index])<1e-7,(kind,index)


def test_fit_live_and_placebo():
    # Null predictor is uniform; a supervised informative input must improve it.
    x=np.zeros((256,13),dtype=np.int8);x[:,0]=np.arange(256)%2;y=np.where(x[:,0],127,0)
    from ghostscale.validation.soundingline.v20.readers import features
    for kind in KINDS:
        model=JointReader(features(x).shape[1],kind,2048).fit(x,y,x,y,max_epochs=120)
        assert -model.predict(x)[np.arange(len(y)),y].mean()<3
        assert model.metadata['training_labels']==256
