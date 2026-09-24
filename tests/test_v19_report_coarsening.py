import gzip
import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import report_coarsening as S, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from test_v19_reachable_retrospective import fixture, law
from test_v19_source_identity import complete_source_fixture


@pytest.mark.parametrize('mode', ['random', 'uniform', 'zeros', 'disjoint'])
@pytest.mark.parametrize('sparse', [False, True])
def test_scalar_bayes_all_reports_and_explicit_one_hot_loss(mode, sparse):
    st = Q.prepare(fixture(), [.2, .3, .4, .1]); w = np.array([.6, .1, .2, .1]) if not sparse else np.array([0, .5, 0, .5])
    p = law(mode); ids = []
    for t, c in ((1, 0), (2, 1)):
        old = next(e for e in range(8) if sum(w[h]*p[s, c, e] for h, s in enumerate(st['past'][t-1])) > 0)
        ids.append([t, c, old])
    raw = S.evaluate(w, st, p, ids); summary = S.summarize(raw)
    baseline = np.array([sum(w[h]*p[st['future'][h, t]] for h in range(len(w))) for t in range(st['future'].shape[1])])
    for ai, a in enumerate(S.ALPHAS):
        values = {}
        for ri in range(14):
            endpoints = [ri] if ri < 8 else [e for e in range(8) if ((e >> ((ri-8)//2)) & 1) == (ri-8)%2]
            joint = np.array([[w[h]*(a*(old in endpoints)+(1-a)*sum(p[st['past'][t-1, h], c, e] for e in endpoints))/len(ids) for h in range(len(w))] for t, c, old in ids])
            den = math.fsum(joint.ravel()); assert raw['report_probability'][ai, ri] == pytest.approx(den, abs=3e-15)
            if den == 0:
                assert not raw['possible'][ai, ri]
                for k in ('source_entropy', 'future_optimal_squared_loss', 'future_squared_gain'): assert np.isnan(raw[k][ai, ri])
                continue
            posterior = joint.sum(0)/den; source = joint.sum(1)/den
            np.testing.assert_allclose(raw['source_posterior'][ai, :, ri], source, atol=3e-15, rtol=0)
            future = np.array([sum(posterior[h]*p[st['future'][h, t]] for h in range(len(w))) for t in range(st['future'].shape[1])])
            optimal = before = 0.
            for t in range(len(future)):
                for c in range(4):
                    for e in range(8):
                        one = np.eye(8)[e]; mass = future[t, c, e]/(len(future)*4)
                        optimal += mass*sum((future[t, c]-one)**2)
                        before += mass*sum((baseline[t, c]-one)**2)
            entropy = -sum(q*math.log(q) for q in source if q)
            assert raw['future_optimal_squared_loss'][ai, ri] == pytest.approx(optimal, abs=4e-15)
            assert raw['future_squared_gain'][ai, ri] == pytest.approx(before-optimal, abs=4e-15)
            assert raw['source_entropy'][ai, ri] == pytest.approx(entropy, abs=4e-15)
            values[ri] = (den, optimal, before-optimal, entropy, future)
        for label, indices in [('fine', range(8)), ('bit0', range(8, 10)), ('bit1', range(10, 12)), ('bit2', range(12, 14))]:
            for key, col in [('future_optimal_squared_loss', 1), ('future_squared_gain', 2), ('source_entropy', 3)]:
                expected = math.fsum(values[r][0]*values[r][col] for r in indices if r in values)
                assert summary[label+'_expected_'+key][ai] == pytest.approx(expected, abs=4e-15)
        for bit in range(3):
            for e in range(8):
                if e not in values:
                    assert np.isnan(raw['fine_to_coarse_squared_regret'][ai, bit, e]); continue
                coarse = 8+2*bit+((e >> bit) & 1)
                d = values[e][4]-values[coarse][4]
                assert raw['fine_to_coarse_squared_regret'][ai, bit, e] == pytest.approx((d*d).sum(-1).mean(), abs=4e-15)


def test_single_source_permutation_constant_and_informative_bit():
    st = Q.prepare(fixture(), [.2, .3, .4, .1]); w = np.array([.6, .1, .2, .1]); p = law('random'); ids = [[1, 0, 0], [2, 1, 1]]
    a = S.summarize(S.evaluate(w, st, p, ids)); b = S.summarize(S.evaluate(w, st, p, ids[::-1]))
    for k in a: np.testing.assert_allclose(a[k], b[k], atol=4e-15, rtol=0)
    one = S.summarize(S.evaluate(w, st, p, ids[:1]))
    for k, v in one.items():
        if 'entropy' in k: assert abs(v).max() < 1e-14
    assert all(S.controls().values())
    assert np.array_equal(S.CONTENT[:, :8], np.eye(8))
    for b in range(3): assert np.array_equal(S.CONTENT[:, 8+2*b:10+2*b].sum(1), np.ones(8))


@pytest.mark.parametrize('bad', ['weight', 'weight_nan', 'law', 'law_nan', 'empty', 'fractional', 'time', 'context', 'endpoint', 'duplicate', 'unsupported'])
def test_malformed_inputs_rejected(bad):
    st = Q.prepare(fixture(), [.2, .3, .4, .1]); w = np.array([.6, .1, .2, .1]); p = law('random'); ids = [[1, 0, 0], [2, 1, 1]]
    if bad == 'weight': w[0] = -.1
    if bad == 'weight_nan': w[0] = np.nan
    if bad == 'law': p[0, 0, 0] = -.1
    if bad == 'law_nan': p[0, 0, 0] = np.nan
    if bad == 'empty': ids = []
    if bad == 'fractional': ids[0][0] = 1.1
    if bad == 'time': ids[0][0] = 3
    if bad == 'context': ids[0][1] = 4
    if bad == 'endpoint': ids[0][2] = 8
    if bad == 'duplicate': ids[1] = ids[0]
    if bad == 'unsupported': p = law('zeros')
    with pytest.raises(ValueError): S.evaluate(w, st, p, ids)


@pytest.mark.parametrize('corrupt', [None, 'input_hash', 'pair', 'source_count', 'map', 'roster', 'forecast'])
def test_complete_native_parent_integration(tmp_path, corrupt):
    root, plan = complete_source_fixture(tmp_path)
    if corrupt:
        path = root/'inputs/bindings/1-omitted-4-2-bindings.json'; d = read(path)
        if corrupt in ('input_hash', 'pair'): d['sources'][0][2] = 2
        if corrupt == 'source_count': d['rows'][0]['report_sources'] += 1
        if corrupt == 'map': d['rows'][0]['joint_row'] = 1
        if corrupt == 'roster': d['rows'][0]['draw'] = 99
        if corrupt == 'forecast': d['rows'][0]['forecast'][0][0] += .01
        write(path, d, immutable=False)
        if corrupt != 'input_hash': plan['design']['input_files'][path.relative_to(root/'inputs').as_posix()] = file_digest(path)
        with pytest.raises(ValueError): S.run(root, plan, lambda **kw: None)
    else:
        result = S.run(root, plan, lambda **kw: None)
        assert result['posterior_rows'] == 512 and result['sources'] == 768 and result['report_queries'] == 35840
        assert all(result['controls'].values()) and not result['numerical_acceptance']
        rows = json.loads(gzip.decompress((root/'raw/report_coarsening_summary_points.json.gz').read_bytes()))
        assert len(rows) == 2560 and len(list((root/'raw').glob('*.npz'))) == 16
        for r in rows:
            for bit in range(3): assert r[f'bit{bit}_lost_future_gain'] >= -1e-12
