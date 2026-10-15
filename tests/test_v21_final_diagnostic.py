import numpy as np
import pytest
from runners.finalize_v21_diagnostic import decision, pooled_worlds, same_arrays, expected_loss, interval


def test_unequal_partitions_and_fits_are_not_independent_worlds():
    rows = [dict(lineage=w, fit=f, reader='r', headroom_nats=x+f*2)
            for w, x in [(1, 0), (2, 3), (3, 6)] for f in (0, 1)]
    assert pooled_worlds(rows, [1, 2, 3], 'r') == [1, 4, 7]
    assert np.mean(pooled_worlds(rows[::-1], [1, 2, 3], 'r')) == 4


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'foreign'])
def test_incomplete_or_repeated_fit_is_rejected(change):
    rows = [dict(lineage=1, fit=f, reader='r', headroom_nats=0) for f in (0, 1)]
    if change == 'missing': rows.pop()
    if change == 'duplicate': rows.append(rows[0])
    if change == 'foreign': rows[1]['lineage'] = 2
    with pytest.raises(ValueError): pooled_worlds(rows, [1], 'r')


def test_validation_cap_does_not_pass_parity():
    assert decision(dict(lower=0, upper=.01), False) == 'UNRESOLVED'
    assert decision(dict(lower=.03, upper=.05), True) == 'EXCESS READER ERROR'
    assert decision(dict(lower=0, upper=.02), True) == 'PARITY'
    assert decision(dict(lower=.01, upper=.03), True) == 'UNRESOLVED'


def test_training_weight_identity_detects_changed_fit(tmp_path):
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    np.savez(a, weights=np.array([1., 2.]))
    np.savez(b, weights=np.array([1., 2.]))
    same_arrays(a, b)
    np.savez(b, weights=np.array([1., 2.001]))
    with pytest.raises(ValueError): same_arrays(a, b)


def test_expected_score_preserves_structural_zeros_and_infinities():
    p = np.array([[1., 0.]])
    assert expected_loss(p, np.array([[0., -np.inf]]))[0] == 0
    assert np.isinf(expected_loss(p, np.array([[-np.inf, 0.]]))[0])


def test_infinite_uncapped_loss_cannot_become_finite_parity():
    result = interval([0.001, np.inf])
    assert result['nonfinite_worlds'] == 1
    assert result['mean'] is None
    assert decision(result, True) == 'UNRESOLVED'
