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


def test_exact_table_matches_independent_counts_and_prior_fallback():
    from ghostscale.validation.soundingline.v21.dependence import ExactTable
    x=np.array([[0,1],[0,1],[1,0],[1,1]],dtype=np.int8);y=np.array([2,3,2,4]);q=np.array([[0,1],[0,0]],dtype=np.int8)
    support=np.zeros((2,128),bool);support[:,[2,3,4]]=True
    actual=np.exp(ExactTable(x,y).predict(q,support))
    reference=[]
    for row,mask in zip(q,support):
        prior=np.array([(sum(int(t)==j for t in y)+.5)/(len(y)+64) for j in range(128)])
        mass=np.array([sum(np.array_equal(xx,row) and int(t)==j for xx,t in zip(x,y))+.5*prior[j] for j in range(128)])*mask
        reference.append(mass/sum(mass))
    assert np.allclose(actual,reference,atol=1e-14,rtol=0)
    assert actual[0,2]>actual[0,4] and actual[1,2]>actual[1,3]
