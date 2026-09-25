from copy import deepcopy
from fractions import Fraction as F
import pytest
from ghostscale.validation.soundingline.v19 import prior_information as producer
from ghostscale.validation.soundingline.v19 import prior_information_review as review
from ghostscale.validation.soundingline.v19.randomized_source_storage import fixture


def sample():
    return fixture([1,2,3], [1,2,1], 3, {'a':[1,2], 'b':[2,3], 'c':[1,3]},
                   [[F(1,2),F(1,4),F(1,4)], [F(1,4),F(1,4),F(1,2)], [F(1,3)]*3])


def test_known_controls():
    assert all(review.controls().values())


@pytest.mark.parametrize('counts', review.MIXTURES)
def test_independent_exhaustive_policies(counts):
    assert review.value(sample(),counts) == producer.value(sample(),counts)


def test_corrupt_byte_charge_rejected():
    s=deepcopy(sample());s['candidates'][0]['used_bytes']=4
    with pytest.raises(AssertionError): review.value(s,(2,1,1))


def test_order_invariance_and_invalid_mixture():
    s=sample();reverse=deepcopy(s);reverse['candidates'].reverse()
    assert review.value(s,(2,1,1)) == review.value(reverse,(2,1,1))
    with pytest.raises(AssertionError): review.value(s,(1,1,1))
