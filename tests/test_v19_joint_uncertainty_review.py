"""Known scalar answers, complete synthetic reconstruction and corruptions."""
import ast
import copy
import gzip
import json
import math
from pathlib import Path
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import joint_uncertainty_review as R
from ghostscale.validation.soundingline.v18_3.io import write, read, canonical, file_digest


def test_controls():
    assert all(R.controls().values())


def test_correlated_chain_and_axis_convention():
    q = [1/R.N]*R.N
    answer, error = R.components(q, [R.native([[0, .25], [216, .75]])])
    a = answer[0]
    assert abs(a['loss']-math.log(R.N)) < 1e-12
    assert abs(a['operation_loss']-math.log(216)) < 1e-12
    assert abs(a['conditional_goal_loss']-math.log(27)) < 1e-12
    assert a['operation_entropy'] == 0 and a['conditional_goal_entropy'] > 0
    assert error < 1e-12
    q = [1.]*R.N; q[0] = q[216] = 100.; q = [x/math.fsum(q) for x in q]
    a = R.components(q, [R.native([[0, .5], [216, .5]])])[0][0]
    assert abs(a['operation_loss']+math.log(math.fsum(q[:27]))) > .05


def test_perfect_sparse_joint_and_distinct_operations():
    q = [0.]*R.N; q[0], q[1], q[216] = .2, .3, .5
    a = R.components(q, [R.native([[0, .2], [1, .3], [216, .5]])])[0][0]
    assert all(abs(a[k]) < 1e-12 for k in R.EXCESSES)
    assert abs(a['operation_entropy']+.7*math.log(.7)+.3*math.log(.3)) < 1e-12


def test_unknown_tail_is_not_discarded():
    q = R.expand([32/33], np.array([0]), 32)
    a = R.components(q, [R.native([[5831, 1.]])])[0][0]
    assert abs(a['loss']-math.log(33*5831)) < 1e-12
    assert a['joint_entropy'] == 0


@pytest.mark.parametrize('problem', ['negative', 'nan', 'mass', 'zero_truth'])
def test_invalid_distributions(problem):
    q = [1/R.N]*R.N
    if problem == 'negative': q[0] = -1
    if problem == 'nan': q[0] = float('nan')
    if problem == 'mass': q = [2*x for x in q]
    if problem == 'zero_truth': q[1] += q[0]; q[0] = 0
    with pytest.raises(ValueError): R.components(q, [R.native([[0, 1.]])])


@pytest.mark.parametrize('target', [[[0, .5], [0, .5]], [[0, .9]], [[5832, 1.]], [[0, -1.]], [[0, float('nan')]]])
def test_invalid_targets(target):
    with pytest.raises(ValueError): R.native(target)


@pytest.mark.parametrize('alphabet', [[0, 0], [.5], [-1], [5832]])
def test_invalid_alphabets(alphabet):
    with pytest.raises(ValueError): R.expand(np.ones(len(alphabet)), np.array(alphabet), 32)


def fixture(root):
    from test_v19_joint_uncertainty import fixture as parent_fixture
    from ghostscale.validation.soundingline.v19 import joint_uncertainty as U
    producer = root/'producer'; plan = parent_fixture(producer)
    write(producer/'PLAN.json', plan)
    write(producer/'SUMMARY.json', U.run(producer, plan, lambda **kw: None))
    target = root/'checker'; inputs = target/'inputs'
    shutil.copytree(producer/'inputs', inputs/'parent')
    original = inputs/'original'; original.mkdir()
    for n in ('reader', 'evaluator'): shutil.copytree(producer/n, original/n)
    for n in ('PLAN.json', 'SUMMARY.json', 'uncertainty_points.json.gz'): shutil.copyfile(producer/n, original/n)
    return target, dict(design=dict(target_plan_sha256=file_digest(original/'PLAN.json'),
        input_files={p.relative_to(inputs).as_posix(): file_digest(p) for p in inputs.rglob('*') if p.is_file()}))


def test_whole_handler_and_independent_original_regroup(tmp_path):
    root, plan = fixture(tmp_path)
    s = R.run(root, plan, lambda **kw: None)
    assert s['rows'] == s['original_cells_reproduced'] == 128
    assert s['frame_forecasts_scored'] == 64 and s['packets'] == 2 and s['native_frame_laws'] == 4
    assert (s['contrasts'], s['learning_areas'], s['means']) == (72, 36, 16)
    assert all(s[k] < 1e-12 for k in s if k.startswith('max_'))
    assert all(s['controls'].values())


@pytest.mark.parametrize('corruption', ['loss', 'entropy', 'inherited', 'missing', 'duplicate', 'reference', 'private', 'hash', 'summary'])
def test_complete_corruptions(tmp_path, corruption):
    root, plan = fixture(tmp_path); base = root/'inputs'; original = base/'original'
    if corruption in ('loss', 'entropy', 'inherited', 'missing', 'duplicate'):
        path = original/'uncertainty_points.json.gz'; rows = json.loads(gzip.decompress(path.read_bytes()))
        if corruption == 'missing': rows.pop()
        elif corruption == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
        else: rows[0][{'loss': 'operation_loss', 'entropy': 'joint_entropy', 'inherited': 'goal_accuracy'}[corruption]] += .1
        path.write_bytes(gzip.compress(canonical(rows), mtime=0))
    elif corruption == 'reference':
        path = original/'evaluator/REFERENCES.json'; refs = read(path); refs[0]['frames'][0]['mass'] = .5; write(path, refs, immutable=False)
    elif corruption == 'private':
        path = original/'reader/PACKETS.json'; p = read(path); next(iter(p['packets'].values()))['inputs']['truth'] = 1; write(path, p, immutable=False)
    else:
        path = original/'SUMMARY.json'; s = read(path); s['contrasts'][0]['mean'] += .1; write(path, s, immutable=False)
    if corruption != 'hash': plan['design']['input_files'][path.relative_to(base).as_posix()] = file_digest(path)
    with pytest.raises(ValueError): R.run(root, plan, lambda **kw: None)


def test_no_producer_imports():
    tree = ast.parse(Path(R.__file__).read_text())
    imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(any(s in n for s in ('joint_uncertainty', 'joint_support', 'joint_review', 'joint_readout')) for n in imports)
