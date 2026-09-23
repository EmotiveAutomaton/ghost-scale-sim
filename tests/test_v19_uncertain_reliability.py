from itertools import product
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import uncertain_reliability as U
from ghostscale.validation.soundingline.v19 import noisy_review as V
from ghostscale.validation.soundingline.v18_3.io import read
from test_v19_joint_reply import joint_fixture, LEGAL


def test_known_bounds_pairs_and_nonjoint_envelope():
    p = np.zeros((3, 8)); p[0, 0] = 1; p[1, 1] = 1; p[2, :2] = .5
    result = U.describe(p, [.5]*3)
    assert np.array_equal(result['pairwise_tv'], [1, .5, .5])
    assert result['diameter'] == 1 and result['sum_range'] == 2
    assert result['upper'].sum() == 2 and result['lower'].sum() == 0
    assert all(U.controls().values())


def test_impossible_candidate_fallback_excluded():
    channels = [U.channel(LEGAL[:2], [0, 1], [1., 0.], a) for a in U.ACCURACIES]
    for field in U.FIELDS:
        p = np.array([ch['tables'][field][1] for ch in channels])
        mass = np.array([ch['reply_mass'][field][1] for ch in channels])
        d = U.describe(p, mass)
        assert np.array_equal(d['compatible'], [True, True, False])
        assert np.array_equal(d['pairwise_tv'], [0, -1, -1])
        assert d['collapsed_exact'] and np.array_equal(d['lower'], p[0])


def test_half_identity_truthful_conditioning_and_containment():
    channels = [U.channel(LEGAL, [0, 1, 2, 3], [.1, .2, .3, .4], a) for a in U.ACCURACIES]
    for field, bit in product(U.FIELDS, (0, 1)):
        p = np.array([ch['tables'][field][bit] for ch in channels])
        mass = np.array([ch['reply_mass'][field][bit] for ch in channels])
        d = U.describe(p, mass)
        assert np.allclose(p[0], channels[0]['prior'], atol=1e-15, rtol=0)
        axis = U.FIELDS.index(field)
        raw = [w if q[axis] == bit else 0 for q, w in zip(LEGAL, [.1, .2, .3, .4])]
        assert np.allclose(p[2, :4], np.array(raw)/math.fsum(raw), atol=1e-15, rtol=0)
        assert np.all(p >= d['lower']) and np.all(p <= d['upper'])


def test_exact_and_tolerance_collapse_separate():
    p = np.zeros((3, 8)); p[:, 0] = 1; p[2, 0] -= 1e-14; p[2, 1] = 1e-14
    d = U.describe(p, [.5]*3)
    assert not d['collapsed_exact'] and d['collapsed_tolerance']


def test_mass_roundoff_admits_normalized_candidates_without_clipping():
    p = np.zeros((3, 8)); p[:, 0] = 1
    masses = np.array([.5000000000000001, .7500000000000001, 1.0000000000000002])
    d = U.describe(p, masses)
    assert d['collapsed_exact'] and d['excluded_candidates'] == 0
    assert masses[-1] > 1 and np.array_equal(d['upper'], p[0])


@pytest.mark.parametrize('defect', ['omitted', 'negative', 'nan', 'unnormalized', 'empty', 'invalid_mass'])
def test_invalid_or_omitted_candidate_rejected(defect):
    p = np.zeros((3, 8)); p[:, 0] = 1; m = np.full(3, .5)
    if defect == 'omitted': p = p[:2]; m = m[:2]
    if defect == 'negative': p[1, 1] = -.1
    if defect == 'nan': p[1, 1] = np.nan
    if defect == 'unnormalized': p[1, 0] = .9
    if defect == 'empty': m[:] = 0
    if defect == 'invalid_mass': m[0] = 1.1
    with pytest.raises(ValueError): U.describe(p, m)


def fixture(tmp_path):
    root, cfg = joint_fixture(tmp_path)
    cfg = {k: cfg[k] for k in ('lineages', 'queries', 'rules', 'models', 'input_files')}
    cfg.update(reliabilities=list(U.ACCURACIES), fields=list(U.FIELDS))
    return root, cfg


def test_full_fixture_scalar_sets_populations_and_roles(tmp_path):
    root, cfg = fixture(tmp_path); result = U.run(root, dict(design=cfg), lambda **kw: None)
    groups = read(root/'inputs/MEMBERSHIP.json')['omit-both']; ids = sorted(groups)
    laws = {(r['lineage'], r['rule'], r['model'], r['reader_id']): r for r in read(root/'inputs/DISCLOSURE_LAWS.json')}
    assert len(result['cells']) == 48 and result['sets'] == 16*len(ids)
    assert set(read(root/'reader/REPLIES.json')) == {
        U.digest(dict(g['reader'], requested_field=field, reply_value=bit, reliability_candidates=list(U.ACCURACIES)))
        for g in groups.values() for field, bit in product(('skill', 'belief_error'), (0, 1))}
    for packet in read(root/'reader/REPLIES.json').values():
        assert set(packet) == {'initial', 'operations', 'requested_field', 'reply_value', 'reliability_candidates'}
    for lin, rule, model in product(cfg['lineages'], cfg['rules'], cfg['models']):
        with np.load(root/'sets'/f'{lin}-{rule}-{model}_points.npz', allow_pickle=False) as data:
            for gi, key in enumerate(ids):
                law = laws[(lin, rule, model, key)]
                for fi, field in enumerate(U.FIELDS):
                    for bit in (0, 1):
                        scalar = [V.conditional(law['legal_completions'], law['endpoints'], law['conditional_weights'], a) for a in U.ACCURACIES]
                        vectors = np.array([d['tables'][field][bit] for d in scalar]); masses = np.array([d['reply_mass'][field][bit] for d in scalar])
                        assert np.allclose(data['candidates'][gi, fi, bit], vectors, atol=1e-15, rtol=0)
                        assert np.allclose(data['modeled_reply_mass'][gi, fi, bit], masses, atol=1e-15, rtol=0)
                        valid = masses > 0; assert np.array_equal(data['compatible'][gi, fi, bit], valid)
                        assert np.allclose(data['lower'][gi, fi, bit], vectors[valid].min(0), atol=1e-15, rtol=0)
                        assert np.allclose(data['upper'][gi, fi, bit], vectors[valid].max(0), atol=1e-15, rtol=0)
            for row in result['cells']:
                if (row['lineage'], row['rule'], row['model']) != (lin, rule, model): continue
                fi = U.FIELDS.index(row['field']); n = len(data['queries'])
                weights = data['mass'] if row['weighting'] == 'native' else np.full(n, 1/n)
                for metric in ('diameter', 'max_range', 'sum_range', 'collapsed_exact', 'collapsed_tolerance', 'excluded_candidates'):
                    terms = [float(weights[i])*(row['actual_accuracy'] if q[fi] == bit else 1-row['actual_accuracy'])*float(data[metric][data['query_group'][i], fi, bit]) for i, q in enumerate(data['queries']) for bit in (0, 1)]
                    assert row[metric] == pytest.approx(math.fsum(terms), abs=1e-14)


def test_binding_and_design_corruption(tmp_path):
    root, cfg = fixture(tmp_path); cfg['reliabilities'] = [.5, 1.]
    with pytest.raises(ValueError, match='design'): U.run(root, dict(design=cfg), lambda **kw: None)
    cfg['reliabilities'] = list(U.ACCURACIES)
    (root/'inputs/DISCLOSURE_LAWS.json').write_text('[]\n', encoding='utf-8')
    with pytest.raises(ValueError, match='input binding'): U.run(root, dict(design=cfg), lambda **kw: None)
