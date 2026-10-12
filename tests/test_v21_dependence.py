import numpy as np
from ghostscale.validation.soundingline.v21.dependence import copied, controls, EpisodeGroups, latent_predict
from ghostscale.validation.soundingline.v21.dependence_check import latent_reference,grouped_reference
from ghostscale.validation.soundingline.v21.readers import SharedMaker


def test_copying_keeps_current_and_tracks_roots():
    assert all(controls().values())
    x=np.arange(52*20).reshape(20,52)
    changed,keep=copied(x,1,'test')
    assert np.array_equal(x[:,:13],changed[:,:13])
    assert np.all(keep.sum(1)==1)


def test_independent_readers_and_copy_ablation():
    rng=np.random.default_rng(1);x=rng.integers(0,2,(64,39));y=np.arange(64)%128
    latent=SharedMaker(x,y,states=4,iterations=3)
    groups=EpisodeGroups(x,y);xx,keep=copied(x[:5],1,'test');support=np.ones((5,128),bool)
    for roots in (None,keep):
        assert np.allclose(np.exp(latent_predict(latent,xx,support,keep=roots)),latent_reference(latent,xx,support,roots),atol=1e-12)
    for power in (.25,.5,1.):assert np.allclose(np.exp(groups.predict(xx,support,power)),grouped_reference(groups,xx,support,power),atol=1e-12)
    one=xx[:,:26]
    assert np.allclose(latent_predict(latent,xx,support,keep=keep),latent_predict(latent,one,support),atol=1e-12)
    assert not np.allclose(latent_predict(latent,xx,support),latent_predict(latent,one,support))
