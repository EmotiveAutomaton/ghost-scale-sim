from fractions import Fraction
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19.schedule_likelihood_envelope import evaluate, schedule_bound, controls, fixture


def test_controls(): assert all(controls().values())


@pytest.mark.parametrize('length', [8, 32, 128])
def test_explicit_rational_likelihood_products(length):
    original = [[Fraction(1, 5), Fraction(4, 5)], [Fraction(1, 3), Fraction(2, 3)], [Fraction(1, 2)]*2, [Fraction(2, 5), Fraction(3, 5)]]
    rounded = [[x+Fraction(1 if i == 0 else -1, 1000*(c+1)) for i, x in enumerate(row)] for c, row in enumerate(original)]
    spans = [max(math.log(rounded[c][s]/original[c][s]) for s in range(2))-min(math.log(rounded[c][s]/original[c][s]) for s in range(2)) for c in range(4)]
    schedule = [0, 1, 2, 3]*(length//4); total, bound = schedule_bound(spans, schedule)
    assert math.isclose(total, math.fsum(spans[c] for c in schedule), abs_tol=1e-13)
    for i in range(1, 20):
        p = Fraction(i, 20)
        a = [math.prod(original[c][s] for c in schedule) for s in range(2)]
        b = [math.prod(rounded[c][s] for c in schedule) for s in range(2)]
        exact = p*a[0]/(p*a[0]+(1-p)*a[1]); cast = p*b[0]/(p*b[0]+(1-p)*b[1])
        assert float(abs(exact-cast)) <= bound+1e-15


@pytest.mark.parametrize('dtype', ['float64', 'float32', 'float16'])
def test_balanced_order_identity_and_universal_bound(dtype):
    law = np.random.default_rng(733).uniform(.00001, 1, (16, 4, 8)); law /= law.sum(-1, keepdims=True)
    r = evaluate(law, dtype)
    assert np.max(abs(r['schedule_posterior_tv_bounds'][0]-r['schedule_posterior_tv_bounds'][1])) < 1e-15
    assert np.all(r['schedule_posterior_tv_bounds'] <= r['posterior_tv_bounds']+1e-15)
    assert r['attaining_endpoints'].any(axis=1).all()


def test_exact_cast_and_constant_contexts():
    r = evaluate(np.full((16, 4, 8), .125), 'float16'); assert not r['schedule_posterior_tv_bounds'].any()
    for length in (8, 32, 128):
        assert abs(schedule_bound([.01]*4, [0, 1, 2, 3]*(length//4))[1]-math.tanh(.01*length/4)) < 1e-15


def test_support_loss_stays_unbounded():
    law = np.zeros((16, 4, 8)); law[:, :, 0] = 1-1e-9; law[:, :, 1] = 1e-9
    r = evaluate(law, 'float16'); assert np.all(r['schedule_posterior_tv_bounds'] == 1)
    assert not r['attaining_endpoints'][:, 2:].any()


def test_permuted_context_and_schedule():
    ranges = [.01, .02, .03, .04]; perm = [2, 0, 3, 1]; order = [0, 0, 1, 2, 3]
    assert schedule_bound(ranges, order) == schedule_bound([ranges[i] for i in perm], [perm.index(i) for i in order])


def test_undersized_bound_and_attaining_prior():
    ratios = [Fraction(9, 1), Fraction(4, 1), Fraction(1), Fraction(1)]
    span = [math.log(x) for x in ratios]; _, bound = schedule_bound(span, [0, 1, 2, 3])
    p = Fraction(1, 7); q = 36*p/(36*p+1-p)
    assert abs(float(q-p)-bound) < 1e-15 and float(q-p) > bound-1e-5


@pytest.mark.parametrize('ranges,schedule', [([-1, 0, 0, 0], [0]), ([float('nan'), 0, 0, 0], [0]), ([0]*4, []), ([0]*4, [4])])
def test_invalid_inputs(ranges, schedule):
    with pytest.raises(ValueError): schedule_bound(ranges, schedule)
