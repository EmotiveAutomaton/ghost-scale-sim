import numpy as np
from ghostscale.validation.soundingline.v21 import authorship as A
from ghostscale.validation.soundingline.v21.authorship_check import reference


def test_topology_witnesses_and_exact_alias_controls():
    assert all(A.controls().values())


def test_every_finite_record_matches_independent_enumeration():
    for roles in (2,3):
      for conflict in (False,True):
       for prior in (np.array([.1,.2,.7]),np.array([.6,.3,.1])):
        states=A.catalog(roles,conflict,prior)
        for tier in ('none','partial','full'):
         for reader in A.READERS:
          for s in states:
            evidence=A.view(s,tier);q,_=A.forecast(states,evidence,tier,reader)
            assert np.allclose(q,reference(roles,conflict,prior,evidence,tier,reader),atol=1e-12)
        assert all(sum(s['mass'] for s in states if s['endpoint']==e)==.5 for e in (0,1))
