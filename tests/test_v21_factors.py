import itertools
import numpy as np
from ghostscale.validation.soundingline.v21.factors import controls,posterior,direct_table,law,READERS,REVEALS
from ghostscale.validation.soundingline.v21.factors_check import reference

def test_known_answers():
    assert all(controls().values())

def test_every_observation_against_scalar_generator():
    success=law(71000781)
    for bits in REVEALS.values():
      for values in itertools.product((0,1),repeat=len(bits)):
       for y in (0,1):
        for reader in READERS:
            cues=dict(zip(bits,values));q=posterior(y,cues,success,reader)
            assert np.max(abs(q-reference(y,cues,success,reader)))<1e-12
            if reader=='joint':assert np.max(abs(q-direct_table(success,bits)[(y,*values)]))<1e-12

def test_goal_fixed_knowledge_permission_twins():
    s=np.array([.6,.9]);a=12;b=9
    q=posterior(1,{},s,'joint');assert q[a]==q[b]
    q=posterior(1,{1:1},s,'joint');assert q[a]>0 and q[b]==0
