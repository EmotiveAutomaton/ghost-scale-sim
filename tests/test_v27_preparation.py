import numpy as np
from ghostscale.validation.soundingline.v27 import setpoint_particles as particle, registry


def test_particle_candidate_has_independent_weights_and_resampling():
    r=np.random.default_rng(703)
    for law in range(5):
        stock=r.integers(0,21,(64,2)).astype(float)
        edge=r.integers(6,15,(64,2)).astype(float)
        pi=r.uniform(.1,10,(64,2));streak=r.integers(0,21,(64,2))
        a=[stock.copy(),edge.copy(),pi.copy(),streak.copy()]
        b=[x.copy() for x in a]
        wa=particle.step_weights(*a,2,0.,0,law,2,0.,.02)
        wb=particle.reference_weights(*b,2,0.,0,law,2,0.,.02)
        np.testing.assert_allclose(wa,wb,rtol=0,atol=1e-12)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=0,atol=1e-12)
        w=wa/wa.sum();u=.372
        expected=np.searchsorted(np.cumsum(w),(np.arange(len(w))+u)/len(w))
        np.testing.assert_array_equal(particle.reference_ancestors(w,u),expected)


def test_type_mix_remains_declared_for_reader_priors():
    assert all(s['type_mix']=='no-Ls-Ah' for s in registry.settings_with('K1',dict(type_mix='no-Ls-Ah')))
    assert all(s['truth'] not in ('Ls','Ah') for s in registry.settings_with('K1',dict(type_mix='no-Ls-Ah')))
