import numpy as np
from ghostscale.validation.soundingline.v21.omission import controls,predict,posterior,READERS,UNIFORM_ORDER
from ghostscale.validation.soundingline.v21.omission_check import predict as reference,event_loss

def test_known_answers():
    assert all(controls().values())

def test_complete_finite_grid_and_label_independence():
    prior=np.arange(1,101,dtype=float);prior/=prior.sum()
    for keep in (100,90,75,50):
      for budget in (8,16,32,64):
        for reader in READERS:
            q,mass_mask,m=predict(prior,[8,2],keep,budget,reader,97);ref,ids,rm=reference(prior,[8,2],keep,budget,reader,UNIFORM_ORDER.tolist(),97)
            assert np.allclose(q,ref,atol=1e-14) and np.array_equal(np.flatnonzero(mass_mask),sorted(ids)) and abs(m-rm)<1e-14
            if reader!='truth-supplied':assert np.array_equal(q,predict(prior,[8,2],keep,budget,reader,3)[0])

def test_unknown_event_does_not_replace_full_universe_score():
    prior=np.full(100,.01);q,mask,m=predict(prior,[8,2],50,8,'finite-unknown')
    coarse=event_loss(q.tolist(),np.flatnonzero(mask).tolist(),82)
    assert np.isclose(-np.log(q[82])-coarse,np.log(50)) and m>0
    for reader in ('uniform-expansion','evidence-ranked-expansion'):
        last=np.zeros(100,bool)
        for budget in (8,16,32,64):
            q,mask,m=predict(prior,[8,2],50,budget,reader);assert np.all(mask[last]);last=mask
        assert np.array_equal(q,posterior(prior,[8,2]))
