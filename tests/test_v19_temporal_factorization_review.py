"""Independent known answers, complete fixture and deliberate corruptions."""
import ast
import copy
import gzip
import json
import math
from pathlib import Path
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import temporal_factorization_review as R
from ghostscale.validation.soundingline.v18_3.io import write, read, canonical, file_digest


def test_known_controls(): assert all(R.controls().values())


def test_scalar_arithmetic_and_product_known_answers():
    q = np.zeros(R.N); q[[0, 2634]] = .5
    steps = R.marginals(q); p, error = R.reconstruct(q, steps)
    assert error == 0 and np.count_nonzero(p) == 4
    assert np.array_equal(p[[0, 1980, 654, 2634]], [.25]*4)
    assert np.array_equal(R.reconstruct(p, R.marginals(p))[0], p)


def test_same_marginals_different_joints():
    q = np.zeros(R.N); q[[0, 2634]] = .5
    p = np.zeros(R.N); p[[1980, 654]] = .5
    assert np.array_equal(R.marginals(q), R.marginals(p))
    assert not np.array_equal(q, p)


def test_full_scalar_label_axes():
    from test_v19_temporal_factorization import scalar
    q = np.arange(1, R.N+1, dtype=float); q /= q.sum()
    expected, m = scalar(q)
    actual, err = R.reconstruct(q, R.marginals(q))
    assert err == 0 and np.allclose(actual, expected, rtol=0, atol=1e-17)
    assert np.allclose(R.marginals(q), m, rtol=0, atol=1e-15)
    wrong = np.prod(m[np.arange(3)[None,:], R.INDICES[:,::-1]], axis=1)
    assert max(abs(actual-wrong)) > 1e-6


def test_uniform_and_point_mass():
    for q in (np.full(R.N, 1/R.N), np.eye(1, R.N, R.N-1)[0]):
        assert np.allclose(R.reconstruct(q, R.marginals(q))[0], q, rtol=0, atol=1e-16)


def test_unknown_mass_and_tie_priority():
    q = R.expand([32/33], np.array([172]), 32)
    ref = (np.array([5831]), np.array([[1.]]), np.array([1.]))
    answer = R.score(q, R.priority_for([172]), ref)
    assert abs(answer['loss'][0]-math.log(33*5831)) < 1e-12
    assert abs(answer['unknown_mass'][0]-1/33) < 1e-12
    q = np.zeros(R.N); q[[0, 217]] = .5
    ref = (np.array([0]), np.array([[1.]]), np.array([1.]))
    a = R.score(q, R.priority_for([217, 0]), ref)
    b = R.score(q, R.priority_for(R.LABELS), ref)
    assert a['top_incompatible'][0] == 1 and b['top_incompatible'][0] == 0
    assert a['candidate_size'][0] == b['candidate_size'][0] == 2
    assert a['candidate_coverage'][0] == b['candidate_coverage'][0] == 1


@pytest.mark.parametrize('problem', ['negative', 'nan', 'mass', 'shape'])
def test_invalid_distributions(problem):
    q = np.full(R.N, 1/R.N)
    if problem == 'negative': q[0] = -1
    if problem == 'nan': q[0] = np.nan
    if problem == 'mass': q *= 2
    if problem == 'shape': q = q[None, :]
    with pytest.raises(ValueError): R.marginals(q)


@pytest.mark.parametrize('alphabet', [[0, 0], [.5], [-1], [5832]])
def test_invalid_alphabets(alphabet):
    with pytest.raises(ValueError): R.expand(np.ones(len(alphabet)), np.array(alphabet), 32)


def test_zero_probability_target_rejected():
    q = np.zeros(R.N); q[0] = 1.
    with pytest.raises(ValueError):
        R.score(q, R.priority_for([0]), (np.array([1]), np.array([[1.]]), np.array([1.])))


def fixture(root):
    from test_v19_joint_factorization import fixture as parent_fixture
    from ghostscale.validation.soundingline.v19 import temporal_factorization as F
    producer = root/'producer'; plan = parent_fixture(producer)
    write(producer/'PLAN.json', plan)
    write(producer/'SUMMARY.json', F.run(producer, plan, lambda **kw: None))
    target = root/'checker'; inputs = target/'inputs'
    shutil.copytree(producer/'inputs', inputs/'parent')
    original = inputs/'original'; original.mkdir()
    for n in ('reader', 'evaluator', 'marginals'): shutil.copytree(producer/n, original/n)
    for n in ('PLAN.json', 'SUMMARY.json', 'temporal_points.json.gz'): shutil.copyfile(producer/n, original/n)
    return target, dict(design=dict(target_plan_sha256=file_digest(original/'PLAN.json'),
        input_files={p.relative_to(inputs).as_posix(): file_digest(p) for p in inputs.rglob('*') if p.is_file()}))


def test_complete_handler(tmp_path):
    root, plan = fixture(tmp_path); s = R.run(root, plan, lambda **kw: None)
    assert (s['rows'], s['native_rows'], s['original_cells_reproduced']) == (256, 8, 128)
    assert (s['frame_forecasts'], s['packets'], s['native_frame_laws']) == (64, 2, 4)
    assert (s['contrasts'], s['learning_areas'], s['means']) == (16, 8, 32)
    assert all(s[k] < 1e-10 for k in s if k.startswith('max_'))
    assert all(s['controls'].values())


@pytest.mark.parametrize('problem', ['loss', 'localization', 'coverage', 'unknown', 'missing', 'duplicate',
    'reference', 'private', 'hash', 'summary', 'marginal', 'native_score', 'native_duplicate'])
def test_complete_corruptions(tmp_path, problem):
    root, plan = fixture(tmp_path); base = root/'inputs'; original = base/'original'
    if problem in ('loss', 'localization', 'coverage', 'unknown', 'missing', 'duplicate'):
        path = original/'temporal_points.json.gz'; rows = json.loads(gzip.decompress(path.read_bytes()))
        if problem == 'missing': rows.pop()
        elif problem == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
        else: rows[0][{'loss':'loss', 'localization':'goal_0', 'coverage':'candidate_coverage', 'unknown':'unknown_mass'}[problem]] += .1
        path.write_bytes(gzip.compress(canonical(rows), mtime=0))
    elif problem == 'marginal':
        path = next((original/'marginals').glob('*.npz'))
        with np.load(path) as data: steps = data['steps']
        steps[0, 0, 0] += .01; steps[0, 0, 1] -= .01
        np.savez_compressed(path, steps=steps)
    elif problem.startswith('native_'):
        path = original/'evaluator/NATIVE_SCORES.json'; rows = read(path)
        if problem == 'native_score': rows[0]['loss'] += .1
        else: rows[-1] = copy.deepcopy(rows[0])
        write(path, rows, immutable=False)
    elif problem == 'reference':
        path = original/'evaluator/REFERENCES.json'; refs = read(path); refs[0]['frames'][0]['mass'] = .5; write(path, refs, immutable=False)
    elif problem == 'private':
        path = original/'reader/PACKETS.json'; p = read(path); next(iter(p['packets'].values()))['inputs']['truth'] = 1; write(path, p, immutable=False)
    else:
        path = original/'SUMMARY.json'; s = read(path); s['contrasts'][0]['mean'] += .1; write(path, s, immutable=False)
    if problem != 'hash': plan['design']['input_files'][path.relative_to(base).as_posix()] = file_digest(path)
    with pytest.raises(ValueError): R.run(root, plan, lambda **kw: None)


def test_no_producer_imports():
    tree = ast.parse(Path(R.__file__).read_text())
    imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(n.endswith(('temporal_factorization', 'joint_factorization', 'joint_support', 'joint_readout', 'joint_uncertainty', 'joint_review')) for n in imports)
