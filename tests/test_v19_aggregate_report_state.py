import gzip
import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import aggregate_report_state as S, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from test_v19_reachable_retrospective import fixture, law
from test_v19_source_identity import complete_source_fixture


def inputs(mode='random', sparse=False):
    st = Q.prepare(fixture(), [.2, .3, .4, .1])
    w = np.array([0, .5, 0, .5] if sparse else [.6, .1, .2, .1])
    p = law(mode)
    ids = []
    for t, c in ((1, 0), (2, 1)):
        e = next(e for e in range(8) if sum(w[h]*p[k, c, e] for h, k in enumerate(st['past'][t-1])) > 0)
        ids.append([t, c, e])
    return st, w, p, np.array(ids)


@pytest.mark.parametrize('mode', ['random', 'uniform', 'zeros', 'disjoint'])
@pytest.mark.parametrize('sparse', [False, True])
def test_scalar_joint_bayes_all_report_and_future_coordinates(mode, sparse):
    st, w, p, ids = inputs(mode, sparse)
    raw = S.evaluate(w, st, p, ids); summary = S.summarize(raw)
    groups = len(st['signatures'])
    joint = np.zeros((groups, 8)); q = np.zeros(groups)
    for h, g in enumerate(st['mapping']):
        q[g] += w[h]
        for t, c, old in ids:
            for e in range(8):
                joint[g, e] += w[h]*p[st['past'][t-1, h], c, e]/len(ids)
    np.testing.assert_allclose(raw['group_mass'], q, rtol=0, atol=2e-15)
    np.testing.assert_allclose(raw['joint_report_mass'], joint, rtol=0, atol=2e-15)
    probs, mask, posterior = S.update(q, joint, raw['copy_counts'])
    for ai, alpha in enumerate(S.ALPHAS):
        for e in range(8):
            numerator = np.array([math.fsum(w[h]*(alpha*int(old == e)+(1-alpha)*p[st['past'][t-1, h], c, e])/len(ids) for t, c, old in ids) for h in range(len(w))])
            den = math.fsum(numerator)
            assert probs[ai, e] == pytest.approx(den, abs=2e-15)
            assert mask[ai, e] == (den > 0)
            if not den:
                assert np.isnan(raw['max_future_probability_error'][ai, e])
                continue
            exact = numerator/den
            for time in range(st['future'].shape[1]):
                for context in range(4):
                    expected = sum(exact[h]*p[st['future'][h, time], context] for h in range(len(w)))
                    actual = sum(posterior[ai, e, g]*p[st['signatures'][g, time], context] for g in range(groups))
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
                    regret = math.fsum(expected[y]*(sum((actual-np.eye(8)[y])**2)-sum((expected-np.eye(8)[y])**2)) for y in range(8))
                    assert abs(regret) < 2e-15
    mixed = cells = 0
    for t, c, old in ids:
        sets = [{int(st['past'][t-1, h]) for h, g in enumerate(st['mapping']) if g == i} for i in range(groups)]
        cells += sum(map(len, sets)); mixed += sum(len(x) for x in sets if len(x) > 1)
    assert raw['source_table_float64_count'] == groups+mixed
    assert raw['source_table_int32_count'] == 8+3*len(ids)+2*cells
    assert raw['aggregate_float64_count'] == 9*groups
    assert np.all(summary['aggregate_state_bytes'] == 72*groups+32)
    assert np.all(summary['source_table_state_bytes'] == 8*(groups+mixed)+4*(8+3*len(ids)+2*cells))


def test_source_and_group_and_endpoint_permutations():
    st, w, p, ids = inputs()
    raw = S.evaluate(w, st, p, ids)
    swapped = S.evaluate(w, st, p, ids[::-1])
    for name in raw:
        if name != 'source_rows':
            np.testing.assert_allclose(raw[name], swapped[name], rtol=0, atol=2e-15)
    q, joint, counts = raw['group_mass'], raw['joint_report_mass'], raw['copy_counts']
    prob, mask, post = S.update(q, joint, counts)
    perm = np.arange(len(q))[::-1]
    pp, mm, qq = S.update(q[perm], joint[perm], counts)
    np.testing.assert_allclose(pp, prob, rtol=0, atol=2e-15)
    np.testing.assert_allclose(qq, post[:, :, perm], rtol=0, atol=2e-15)
    pp, mm, qq = S.update(q, joint[:, ::-1], counts[::-1])
    np.testing.assert_allclose(pp, prob[:, ::-1], rtol=0, atol=2e-15)
    np.testing.assert_allclose(qq, post[:, ::-1], rtol=0, atol=2e-15)
    assert all(S.controls().values())


@pytest.mark.parametrize('field', ['joint_report_mass', 'copy_counts', 'group_mass', 'report_probability', 'possible', 'max_future_probability_error', 'future_squared_error', 'updated_group_total_variation', 'aggregate_float64_count', 'aggregate_int32_count'])
def test_corrupted_state_and_undefined_masks_rejected(field):
    st, w, p, ids = inputs('zeros')
    raw = S.evaluate(w, st, p, ids)
    if field == 'possible': raw[field][0, 0] = not raw[field][0, 0]
    elif field in ('max_future_probability_error', 'future_squared_error', 'updated_group_total_variation'): raw[field][0, 0] = 0
    else: raw[field].flat[0] += 1
    with pytest.raises(ValueError): S.summarize(raw)


@pytest.mark.parametrize('bad', ['weight', 'weight_nan', 'law', 'law_nan', 'empty', 'fractional', 'time', 'context', 'endpoint', 'duplicate', 'unsupported'])
def test_malformed_inputs_rejected(bad):
    st, w, p, ids = inputs()
    if bad == 'weight': w[0] = -.1
    if bad == 'weight_nan': w[0] = np.nan
    if bad == 'law': p[0, 0, 0] = -.1
    if bad == 'law_nan': p[0, 0, 0] = np.nan
    if bad == 'empty': ids = []
    if bad == 'fractional': ids = ids.astype(float); ids[0, 0] = 1.1
    if bad == 'time': ids[0, 0] = 3
    if bad == 'context': ids[0, 1] = 4
    if bad == 'endpoint': ids[0, 2] = 8
    if bad == 'duplicate': ids[1] = ids[0]
    if bad == 'unsupported': p = law('zeros'); ids[0, 2] = 0
    with pytest.raises(ValueError): S.evaluate(w, st, p, ids)


@pytest.mark.parametrize('corrupt', [None, 'input_hash', 'pair', 'source_count', 'map', 'roster', 'forecast', 'design'])
def test_complete_native_parent_integration(tmp_path, corrupt):
    root, plan = complete_source_fixture(tmp_path)
    plan['design']['report_state'] = 'fixed-uniform-source-joint-group-endpoint'
    if corrupt:
        path = root/'inputs/bindings/1-omitted-4-2-bindings.json'; d = read(path)
        if corrupt in ('input_hash', 'pair'): d['sources'][0][2] = 2
        if corrupt == 'source_count': d['rows'][0]['report_sources'] += 1
        if corrupt == 'map': d['rows'][0]['joint_row'] = 1
        if corrupt == 'roster': d['rows'][0]['draw'] = 99
        if corrupt == 'forecast': d['rows'][0]['forecast'][0][0] += .01
        if corrupt == 'design': plan['design']['report_state'] = 'arbitrary-prior'
        write(path, d, immutable=False)
        if corrupt != 'input_hash': plan['design']['input_files'][path.relative_to(root/'inputs').as_posix()] = file_digest(path)
        with pytest.raises(ValueError): S.run(root, plan, lambda **kw: None)
    else:
        result = S.run(root, plan, lambda **kw: None)
        assert (result['posterior_rows'], result['sources'], result['report_queries']) == (512, 768, 20480)
        assert all(result['controls'].values()) and not result['numerical_acceptance']
        rows = json.loads(gzip.decompress((root/'raw/aggregate_report_summary_points.json.gz').read_bytes()))
        assert len(rows) == 2560 and len(list((root/'raw').glob('*.npz'))) == 16
        assert all(r['maximum_max_future_probability_error'] < 1e-12 for r in rows)
