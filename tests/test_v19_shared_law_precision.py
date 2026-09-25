import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import shared_law_precision as S
from test_v19_aggregate_report_state import inputs


@pytest.mark.parametrize('mode', ['random', 'uniform', 'zeros', 'disjoint'])
@pytest.mark.parametrize('sparse', [False, True])
def test_scalar_future_law_and_explicit_one_hot_loss(mode, sparse):
    st, w, law, ids = inputs(mode, sparse)
    raw = S.evaluate(w, st, law, ids); S.summarize(raw)
    for vi, (dtype, repair) in enumerate(S.VARIANTS):
        quantized = law.astype(dtype).astype(float)
        if repair == 'row-normalized': quantized /= quantized.sum(-1, keepdims=True)
        for ai, alpha in enumerate(S.ALPHAS):
            for report in range(8):
                numerator = np.array([math.fsum(w[h]*(alpha*int(old == report)+(1-alpha)*law[st['past'][t-1, h], c, report])/len(ids) for t, c, old in ids) for h in range(len(w))])
                mass = math.fsum(numerator)
                assert raw['report_probability'][ai, report] == pytest.approx(mass, abs=2e-15)
                assert raw['defined'][vi, ai, report] == (mass > 0)
                if not mass: continue
                weights = numerator/mass
                changes = []; errors = []; maxima = []; drifts = []
                for time in range(st['future'].shape[1]):
                    for context in range(4):
                        truth = sum(weights[h]*law[k, context] for h, k in enumerate(st['future'][:, time]))
                        pred = sum(weights[h]*quantized[k, context] for h, k in enumerate(st['future'][:, time]))
                        changes.append(math.fsum(truth[e]*(sum((pred-np.eye(8)[e])**2)-sum((truth-np.eye(8)[e])**2)) for e in range(8)))
                        errors.append(sum((pred-truth)**2)); maxima.append(max(abs(pred-truth)))
                        drifts.append(abs(math.fsum(pred)-1))
                for field, value in [('expected_one_hot_loss_difference', np.mean(changes)), ('future_squared_error', np.mean(errors)), ('max_future_probability_error', max(maxima)), ('max_future_normalization_drift', max(drifts))]:
                    assert raw[field][vi, ai, report] == pytest.approx(value, abs=2e-15)


def test_rare_endpoint_loss_keeps_exact_report_support_and_state():
    st, w, law, ids = inputs('uniform')
    law[:] = 0; law[:, :, 0] = 1-1e-9; law[:, :, 1] = 1e-9; ids[:, 2] = 0
    raw = S.evaluate(w, st, law, ids); summary = S.summarize(raw)
    assert raw['possible'][0, 1] and raw['defined'][4, 0, 1]
    assert raw['law_underflow_count'][4] == 64
    assert np.nanmax(raw['lost_future_support_probability'][4]) > 0
    assert np.isfinite(raw['expected_one_hot_loss_difference'][4, 0, 1])
    assert summary['unusable_report_probability'].max() == 0
    assert raw['shared_law_bytes'].tolist() == [4096, 4096, 2048, 2048, 1024, 1024]


def test_normalization_and_empty_row_failures():
    law = np.full((16, 4, 8), .125, dtype='float16'); law[0, 0, 0] += .01
    direct, failed = S.reconstruct(law, 'direct'); assert not failed
    normalized, failed = S.reconstruct(law, 'row-normalized'); assert not failed
    assert direct[0, 0].sum() != 1
    np.testing.assert_allclose(normalized.sum(-1), 1, atol=1e-15, rtol=0)
    law[0, 0] = 0
    assert all(S.reconstruct(law, mode)[1] for mode in S.MODES)
    law[1, 0, 0] = -1
    with pytest.raises(ValueError): S.reconstruct(law, 'direct')


def test_source_context_endpoint_permutations():
    st, w, law, ids = inputs(); raw = S.evaluate(w, st, law, ids)
    permutation = [2, 0, 3, 1]; inverse = np.argsort(permutation)
    changed = ids[::-1].copy(); changed[:, 1] = inverse[changed[:, 1]]
    other = S.evaluate(w, st, law[:, permutation], changed)
    for k in S.summarize(raw): np.testing.assert_allclose(S.summarize(raw)[k], S.summarize(other)[k], atol=2e-15, rtol=0)
    changed = ids.copy(); changed[:, 2] = 7-changed[:, 2]
    other = S.evaluate(w, st, law[:, :, ::-1], changed)
    np.testing.assert_allclose(raw['future_squared_error'], other['future_squared_error'][:, :, ::-1], atol=2e-15, rtol=0)
    assert all(S.controls().values())


@pytest.mark.parametrize('field', ['defined', 'possible', 'copy_counts', 'joint_report_mass', 'future_squared_error', 'shared_law_bytes'])
def test_corruption_rejected(field):
    st, w, law, ids = inputs('zeros'); raw = S.evaluate(w, st, law, ids)
    if raw[field].dtype == bool: raw[field].flat[0] = not raw[field].flat[0]
    elif field == 'future_squared_error': raw[field].flat[0] = np.nan if np.isfinite(raw[field].flat[0]) else 0
    else: raw[field].flat[0] += 99
    with pytest.raises(ValueError): S.summarize(raw)


def test_complete_native_integration(tmp_path):
    from test_v19_source_identity import complete_source_fixture
    import gzip, json
    root, plan = complete_source_fixture(tmp_path)
    plan['design'].update(report_state='exact-state-shared-future-law-precision', storage_dtypes=list(S.DTYPES), reconstruction_modes=list(S.MODES))
    result = S.run(root, plan, lambda **kw: None)
    assert result['posterior_rows'] == 512 and result['sources'] == 768
    assert all(result['controls'].values()) and not result['numerical_acceptance']
    rows = json.loads(gzip.decompress((root/'raw/shared_law_precision_summary_points.json.gz').read_bytes()))
    assert len(rows) == 15360 and len(list((root/'raw').glob('*.npz'))) == 16
    assert {(r['storage_dtype'], r['reconstruction']) for r in rows} == set(S.VARIANTS)
    stored=list((root/'evaluator').glob('*-stored-law_points.npz'))
    assert len(stored) == 1
    with np.load(stored[0], allow_pickle=False) as z:
        assert set(z.files) == set(S.DTYPES)
        for dtype in S.DTYPES: assert z[dtype].dtype == np.dtype(dtype)


def test_exact_state_export_reconstructible_from_frozen_inputs():
    import hashlib
    st, w, law, ids = inputs()
    raw = S.evaluate(w, st, law, ids); saved = S.export_raw(raw)
    assert 'group_mass' not in saved and 'joint_report_mass' not in saved
    for field in ('group_mass', 'joint_report_mass'):
        assert bytes(saved[field+'_sha256']) == hashlib.sha256(raw[field].astype('<f8').tobytes()).digest()
    assert raw['group_mass'].size and raw['joint_report_mass'].size
