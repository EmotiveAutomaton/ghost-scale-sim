import itertools
import numpy as np
from ghostscale.validation.soundingline.v21.sequential import Planner,controls
from ghostscale.validation.soundingline.v21.sequential_check import posterior,choice


def test_known_answers():
    assert all(controls().values())


def test_already_identified_order():
    for prior,history in (([1.,0.],((0,1),(1,1))),([0.,1.],((0,1),(1,0)))):
        planner=Planner(prior)
        assert np.array_equal(planner.order(history),prior)
        assert np.array_equal(posterior(prior,history),prior)
        assert planner.choice((),8,0.,'two-step-lookahead',.1)==-1


def test_independent_posterior_and_planning():
    for prior in ([.5,.5],[.2,.8]):
      p=Planner(prior)
      for size in range(3):
       for actions in itertools.combinations(range(4),size):
        for bits in itertools.product((0,1),repeat=size):
         history=tuple(zip(actions,bits))
         assert np.allclose(p.order(history),posterior(prior,history),atol=1e-12)
         for policy in ('greedy-information','two-step-lookahead','fixed-pair-first','uncertainty-threshold'):
          for fee in (0,.05,.2,1):
           assert p.choice(history,2,fee,policy,.1)==choice(prior,history,2,fee,policy,.1)
