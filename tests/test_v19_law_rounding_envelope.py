"""Known-answer and corruption controls for law-only rounding bounds."""
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import law_rounding_envelope as E


def scalar_audit(raw):
    """Independent scalar extrema, support and expected-loss arithmetic."""
    source = raw['original']; stored = raw['stored'].astype(float)
    estimate = raw['reconstructed']
    assert np.array_equal(raw['lost_support'], (source > 0) & (stored == 0))
    for c in range(4):
        squared = []
        for s in range(16):
            differences = [float(estimate[s, c, e])-float(source[s, c, e]) for e in range(8)]
            squared.append(math.fsum(x*x for x in differences))
            assert raw['row_squared'][s, c] == pytest.approx(squared[-1], rel=1e-14, abs=1e-30)
            loss = math.fsum(float(source[s, c, e])*math.fsum(
                differences[j]*(float(estimate[s, c, j])+float(source[s, c, j])-2*int(j == e))
                for j in range(8)) for e in range(8))
            assert raw['expected_one_hot_loss_change'][s, c] == pytest.approx(loss, rel=1e-11, abs=1e-20)
        assert raw['squared_bound'][c] == pytest.approx(max(squared), rel=1e-14, abs=1e-30)
        for e in range(8):
            d = [float(estimate[s, c, e])-float(source[s, c, e]) for s in range(16)]
            assert raw['lower'][c, e] == min(d) and raw['upper'][c, e] == max(d)
            assert raw['lower_attainers'][:, c, e].tolist() == [v == min(d) for v in d]
            assert raw['upper_attainers'][:, c, e].tolist() == [v == max(d) for v in d]


@pytest.mark.parametrize('dtype', E.DTYPES)
@pytest.mark.parametrize('mode', E.MODES)
def test_full_support_scalar_and_rational_mixtures(dtype, mode):
    raw = E.evaluate(E.fixture(), dtype, mode); scalar_audit(raw)
    for a in range(16):
        assert E.check_mixture(raw, np.eye(16)[a])
        for b in range(a):
            for v in (.25, .5, .75):
                w = np.zeros(16); w[a] = v; w[b] = 1-v
                assert E.check_mixture(raw, w)
    assert E.check_mixture(raw, np.full(16, 1/16))


@pytest.mark.parametrize('axis', (0, 1, 2))
def test_state_context_endpoint_permutation(axis):
    law = E.fixture(); original = E.evaluate(law, 'float16', 'direct')
    changed = E.evaluate(np.flip(law, axis), 'float16', 'direct'); scalar_audit(changed)
    assert np.array_equal(changed['delta'], np.flip(original['delta'], axis))
    assert np.array_equal(changed['lower'], original['lower'] if axis == 0 else np.flip(original['lower'], axis-1))


def test_deterministic_exact_zeros_and_rare_endpoint_underflow():
    law = np.zeros((16, 4, 8)); law[:, :, 0] = 1
    raw = E.evaluate(law, 'float16', 'row-normalized')
    assert not raw['delta'].any() and not raw['lost_support'].any()
    law[:, :, 1] = 1e-9; law[:, :, 0] -= 1e-9
    rare = E.evaluate(law, 'float16', 'direct'); scalar_audit(rare)
    assert rare['lost_support'].sum() == 64
    assert np.array_equal(rare['lost_support_mass'], np.full((16, 4), 1e-9))


def test_corrupted_support_and_undersized_bound_fail():
    raw = E.evaluate(E.fixture(), 'float16', 'direct')
    broken = dict(raw, lost_support=np.ones((16, 4, 8), dtype=bool))
    with pytest.raises(AssertionError): scalar_audit(broken)
    broken = dict(raw, squared_bound=np.zeros(4))
    assert not E.check_mixture(broken, np.eye(16)[0])
    broken = dict(raw, upper=raw['upper']-1e-4)
    assert not E.check_mixture(broken, np.full(16, 1/16))


@pytest.mark.parametrize('bad', ('empty', 'negative', 'nan', 'wrong-shape', 'unnormalized'))
def test_invalid_law_fails_without_repair(bad):
    law = E.fixture()
    if bad == 'empty': law[0, 0] = 0
    elif bad == 'negative': law[0, 0, 0] = -1
    elif bad == 'nan': law[0, 0, 0] = np.nan
    elif bad == 'wrong-shape': law = law[:1]
    else: law *= .5
    with pytest.raises(ValueError): E.evaluate(law, 'float16', 'row-normalized')


def test_recorded_gates():
    assert all(E.controls().values())


def test_handler_output_inventory(tmp_path):
    from ghostscale.validation.soundingline.v18_3.io import write, file_digest, read
    (tmp_path/'inputs').mkdir()
    write(tmp_path/'inputs/190000-law.json', E.fixture().tolist())
    cfg = dict(storage_dtypes=list(E.DTYPES), reconstruction_modes=list(E.MODES), lineages=[190000],
        input_files={'190000-law.json': file_digest(tmp_path/'inputs/190000-law.json')})
    result = E.run(tmp_path, dict(design=cfg), lambda **kwargs: None)
    assert result['rows'] == 24 and result['numerical_acceptance'] is False
    assert len(list((tmp_path/'raw').glob('*.npz'))) == 6
    assert read(tmp_path/'EVIDENCE_ROLES.json')['reader'] == 'no new reader inputs'
