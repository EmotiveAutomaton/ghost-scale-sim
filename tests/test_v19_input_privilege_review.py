import gzip
from itertools import product
import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import input_privilege as P
from ghostscale.validation.soundingline.v19 import input_privilege_review as V
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def fixture(root):
    base = root/'inputs'; (base/'raw').mkdir(parents=True)
    qs = [(s, b, 2, 1, 4, 4) for s, b in product(range(2), repeat=2)]
    truth = [dict(query=list(q), targets={r: V.T.endpoint(q, r) for r in V.T.RULES}) for q in qs]
    write(base/'QUERY_TRUTH.json', truth); write(base/'MEMBERSHIP.json', P.membership(qs))
    for rule in V.T.RULES:
        rows = [dict(maker=[0, q[0], q[1], 0], initial=list(V.V.ARTS[q[2]]),
            steps=[dict(operation=V.V.OPS[o]) for o in q[3:]], final=list(V.V.ARTS[V.T.endpoint(q, rule)]),
            probability=w) for q, w in zip(qs, [.1, .2, .7, 0.], strict=True)]
        (base/'raw'/f'1-{rule}_points.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
    pins = {p.relative_to(base).as_posix(): file_digest(p) for p in base.rglob('*') if p.is_file()}
    cfg = dict(views=list(P.VIEWS), rules=list(P.M.RULES), queries=4, lineages=[1], input_files=pins)
    write(root/'PLAN.json', dict(design=cfg))
    write(root/'SUMMARY.json', P.run(root, dict(design=cfg), lambda **kw: None))
    return root


def test_known_ambiguity_null_and_zero_support_are_distinct():
    qs = [(1, 0, 2, 1, 4, 4), (1, 1, 2, 1, 4, 4)]
    targets, rows, u, n = V.reconstruct(qs, V.groups_for(qs, 'omit-belief'), [1., 0.], 'original')
    assert rows[0]['legal_ambiguous'] and not rows[0]['native_ambiguous']
    assert rows[0]['native_entropy'] == 0 and math.isclose(rows[0]['legal_entropy'], math.log(2))
    assert V.scores(n, targets, [.5, .5])['infinite_loss_mass'] == .5
    assert V.scores(u, targets, [.5, .5])['infinite_loss_mass'] == 0
    _, full, _, _ = V.reconstruct(qs, V.groups_for(qs, 'full'), [1., 0.], 'original')
    assert all(not r['legal_ambiguous'] for r in full)
    assert full[1]['zero_native_mass'] and full[1]['native_zero_mass_fallback'] == 'uniform-legal'
    same = [(1, b, 2, 4, 4, 4) for b in range(2)]
    _, null, _, _ = V.reconstruct(same, V.groups_for(same, 'omit-belief'), [.5, .5], 'original')
    assert not null[0]['legal_ambiguous'] and not null[0]['native_ambiguous']


def test_legal_metadata_is_larger_than_retained_policy_support():
    q = (0, 0, 2, 0, 4, 4)
    group = next(iter(V.groups_for([q], 'omit-both').values()))
    assert len(group['legal_completions']) == 4 and group['indices'] == [0]
    tool = next(iter(V.groups_for([(1, 0, 2, 3, 4, 4)], 'omit-both').values()))
    assert len(tool['legal_completions']) == 2 and all(q[0] == 1 for q in tool['legal_completions'])


def test_complete_fixture_and_paired_regroup(tmp_path):
    root = fixture(tmp_path/'original'); out = tmp_path/'review'; out.mkdir()
    cfg = dict(bootstrap_seed=190974, bootstrap_resamples=20)
    result = V.review(root, out, cfg)
    assert result['passed'] and result['cells'] == 32 and result['group_rows'] == 18
    assert result['forecast_vectors'] == 64 and result['max_error'] < 1e-12
    rows = read(out/'INDEPENDENT_REGROUP.json')['estimates']
    assert len(rows) == 432
    for row in rows:
        assert row['low'] == row['mean'] == row['high']
        if row['contrast'] == 'native-minus-uniform' and row['weighting'] == 'native' and row['metric'] == 'finite_loss_contribution':
            assert row['mean'] <= 1e-12


@pytest.mark.parametrize('kind', ['forecast', 'native_mass', 'target', 'query', 'group_mass',
    'legal_completions', 'group_entropy', 'group_flag', 'group_denominator', 'group_count',
    'summary_score', 'summary_count', 'membership', 'reader_leak', 'reader_missing', 'extra_forecast'])
def test_reviewer_rejects_corrupt_science_or_evidence(tmp_path, kind):
    root = fixture(tmp_path/'original'); out = tmp_path/'review'; out.mkdir()
    if kind in ('forecast', 'native_mass', 'target', 'query'):
        path = next((root/'forecasts').glob('*.npz'))
        with np.load(path) as z: arrays = {k: z[k].copy() for k in z.files}
        key = {'forecast':'native', 'native_mass':'mass', 'target':'targets', 'query':'queries'}[kind]
        arrays[key].flat[0] += .1 if key in ('native', 'mass') else 1
        np.savez(path, **arrays)
    elif kind.startswith('group_') or kind == 'legal_completions':
        path = root/'evaluator/GROUPS_points.json.gz'; rows = json.loads(gzip.decompress(path.read_bytes()))
        if kind == 'group_count': rows.pop()
        elif kind == 'legal_completions': rows[0]['legal_completions'].pop()
        elif kind == 'group_denominator': rows[0]['query_indices'].append(1)
        elif kind == 'group_flag': rows[0]['legal_ambiguous'] = not rows[0]['legal_ambiguous']
        else: rows[0]['native_mass' if kind == 'group_mass' else 'legal_entropy'] += .1
        path.write_bytes(gzip.compress(canonical(rows), mtime=0))
    elif kind.startswith('summary_'):
        path = root/'SUMMARY.json'; s = read(path)
        if kind == 'summary_count': s['native_paths'] += 1
        else: s['cells'][0]['finite_loss_contribution'] += .1
        write(path, s, immutable=False)
    elif kind == 'membership':
        path = root/'evaluator/MEMBERSHIP.json'; s = read(path); s.pop('full'); write(path, s, immutable=False)
    elif kind == 'reader_leak':
        path = next((root/'reader').glob('*.json')); s = read(path); s['evaluator_truth'] = 1; write(path, s, immutable=False)
    elif kind == 'reader_missing': next((root/'reader').glob('*.json')).unlink()
    elif kind == 'extra_forecast': np.savez(root/'forecasts/extra.npz', junk=[1])
    with pytest.raises((ValueError, AssertionError)):
        V.review(root, out, dict(bootstrap_seed=190974, bootstrap_resamples=20))
