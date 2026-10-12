import numpy as np
from ghostscale.validation.soundingline.v21 import learned_persistence as P, world as W


def test_known_answer_contamination_controls():
    assert all(P.controls().values())


def test_matched_contexts_and_unchanged_positive_endpoint():
    rows=W.enumerate_world(71000761)
    chosen=W.sampled(rows,8,'learned-persistence-control')
    x,_=W.inputs(rows,chosen,'context',4,'learned-persistence-control',history_tier='complete',history_observed=True)
    result=P.observations(x,rows,71000761)
    assert np.array_equal(result['1.0']['evidence'],x)
    for value in result.values():
        assert np.array_equal(value['evidence'][:,:13],x[:,:13])
        assert np.array_equal(value['evidence'][:,13:].reshape(8,4,13)[:,:,3:6],x[:,13:].reshape(8,4,13)[:,:,3:6])
    assert np.all(result['shuffled-author']['foreign_maker']==result['shuffled-author']['foreign_maker'][:,:1])
