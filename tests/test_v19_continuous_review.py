import json
import shutil
from fractions import Fraction
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import continuous_review as V, continuous_reliability as C, uncertain_reliability as U
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from test_v19_continuous_reliability import fixture


def prepared(tmp_path):
    root, cfg = fixture(tmp_path)
    # A singular native field is deliberate, rather than left to fixture chance.
    laws = read(root/'inputs/DISCLOSURE_LAWS.json')
    for p in (root/'inputs/forecasts').glob('*.npz'):
        with np.load(p, allow_pickle=False) as z: d = {k: z[k].copy() for k in z.files}
        d['mass'][:] = [0., 0., .25, .75]
        np.savez_compressed(p, **d)
    for row in laws:
        if row['model'] == 'native-law': row['conditional_weights'] = [0., 0., .25, .75]
    (root/'inputs/DISCLOSURE_LAWS.json').write_text(json.dumps(laws), encoding='utf-8')
    cfg['input_files'] = {p.relative_to(root/'inputs').as_posix(): file_digest(p)
                         for p in (root/'inputs').rglob('*') if p.is_file()}
    finite = tmp_path/'finite';finite.mkdir();shutil.copytree(root/'inputs', finite/'inputs')
    fcfg = {k: cfg[k] for k in ('lineages', 'queries', 'rules', 'models', 'fields', 'input_files')}
    fcfg['reliabilities'] = [.5, .75, 1.]
    write(finite/'PLAN.json', dict(design=fcfg));write(finite/'SUMMARY.json', U.run(finite, dict(design=fcfg), lambda **kw: None))
    write(root/'PLAN.json', dict(design=cfg));write(root/'SUMMARY.json', C.run(root, dict(design=cfg), lambda **kw: None))
    out = tmp_path/'review';out.mkdir()
    return root, out, finite


def test_complete_scalar_review_with_singular_native_support(tmp_path):
    root, out, finite = prepared(tmp_path);result = V.review(root, out, finite)
    assert result['passed'] and result['cells'] == 4 and result['certificates'] == 16
    assert result['singular_endpoints'] == 2 and result['compatible_grid_vectors'] == 270
    assert result['finite_reference_vectors'] == 46
    assert result['max_error'] < 1e-12 and result['finite_reference_max_error'] < 1e-12
    assert len(read(out/'RECONSTRUCTED_CELLS.json')) == 4


def test_independent_rational_and_nonbalanced_bayes_controls():
    assert V.rational_checks() == dict(equalities=84, impossible=1)
    legal = [(0, 0), (1, 0)]
    p, z = V.scalar(legal, [0, 1], [.25, .75], 0, 0, Fraction(3, 4))
    assert z == .375 and p[:2] == [.5, .5]
    assert V.scalar(legal, [0, 1], [0., 1.], 0, 0, 1) == (None, 0.)


@pytest.mark.parametrize('defect', ['hit', 'mass', 'limit', 'bound', 'diameter', 'grid',
    'weight', 'error', 'mask', 'collapse', 'impossible-vector', 'summary', 'reader',
    'index', 'roles', 'finite', 'omission', 'law', 'population'])
def test_independent_checker_rejects_corruption(tmp_path, defect):
    root, out, finite = prepared(tmp_path)
    arrays = {'hit': 'hit', 'mass': 'hit_mass', 'limit': 'high_limit', 'bound': 'upper',
              'diameter': 'diameter', 'grid': 'grid_forecasts', 'weight': 'grid_weight',
              'error': 'grid_interpolation_error', 'mask': 'high_compatible', 'collapse': 'collapsed_binary64'}
    if defect in arrays or defect == 'impossible-vector':
        p = next((root/'certificates').glob('*native-law*.npz'))
        with np.load(p, allow_pickle=False) as z: d = {k: z[k].copy() for k in z.files}
        if defect == 'impossible-vector':
            ix = tuple(np.argwhere(~d['grid_compatible'])[0]);d['grid_forecasts'][ix][0] = 1.
        else:
            k = arrays[defect]
            if defect in ('mask', 'collapse'): d[k].flat[0] = not d[k].flat[0]
            else: d[k].flat[0] += .125
        np.savez_compressed(p, **d)
    elif defect == 'finite':
        p = next((finite/'sets').glob('*.npz'))
        with np.load(p, allow_pickle=False) as z: d = {k: z[k].copy() for k in z.files}
        d['upper'].flat[0] += .125;np.savez_compressed(p, **d)
    elif defect == 'omission':
        next((root/'certificates').glob('*.npz')).unlink()
    else:
        paths = {'summary': 'SUMMARY.json', 'reader': 'reader/REPLIES.json', 'index': 'evaluator/INDEX.json',
                 'roles': 'EVIDENCE_ROLES.json', 'law': 'inputs/DISCLOSURE_LAWS.json', 'population': 'PLAN.json'}
        p = root/paths[defect];d = read(p)
        if defect == 'summary': d['cells'][0]['certificates'] += 1
        elif defect == 'reader': next(iter(d.values()))['hidden_truth'] = 1
        elif defect == 'index': d['grid'] = []
        elif defect == 'roles': d['reader'] = 'includes evaluator truth'
        elif defect == 'law': d[0]['endpoints'][0] = (d[0]['endpoints'][0]+1)%8
        else: d['design']['queries'] += 1
        p.write_text(json.dumps(d), encoding='utf-8')
    with pytest.raises((ValueError, FileNotFoundError, AssertionError)):
        V.review(root, out, finite)
