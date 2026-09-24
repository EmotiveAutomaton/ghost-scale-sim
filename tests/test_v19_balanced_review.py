import gzip
import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import balanced_review as V, balanced_retention as S, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import read, write, canonical, file_digest
from test_v19_reachable_retrospective import fixture, law
from test_v19_source_identity import complete_source_fixture


def example(mode='random', sparse=False, single=False):
    spec = fixture(); st = V.structure(spec); p = law(mode)
    w = np.array([0., .5, 0., .5] if sparse else [.6, .1, .2, .1])
    ids = []
    for t, c in ((1, 0), (2, 1)):
        old = next(e for e in range(8) if sum(w[h]*p[s, c, e] for h, s in enumerate(st['past'][t-1])) > 0)
        ids.append([t, c, old])
    if single: ids = ids[:1]
    saved = S.evaluate(w, Q.prepare(spec, np.full(len(w), 1/len(w))), p, ids)
    return w, st, p, ids, saved


@pytest.mark.parametrize('mode', ['random', 'uniform', 'zeros', 'disjoint'])
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('single', [False, True])
def test_all_source_weights_posteriors_and_future_coordinates(mode, sparse, single):
    w, st, p, ids, saved = example(mode, sparse, single)
    actual = V.reconstruct(w, st, p, ids, saved); expected = S.summarize(saved)
    assert set(actual) == set(expected)
    for key in actual: np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=0)
    assert all(V.controls().values())


@pytest.mark.parametrize('field', list(example()[4]))
def test_each_retained_field_is_checked(field):
    w, st, p, ids, saved = example()
    saved = {k:v.copy() for k, v in saved.items()}
    a = saved[field]
    if a.dtype.kind == 'b': a.flat[0] = not a.flat[0]
    else: a.flat[0] += 1
    with pytest.raises(ValueError): V.reconstruct(w, st, p, ids, saved)


def test_undefined_error_and_permutation():
    w, st, p, ids, saved = example('disjoint')
    for key in ('future_squared_regret', 'max_future_probability_error', 'updated_group_total_variation'):
        broken = {k:v.copy() for k, v in saved.items()}
        pos = np.flatnonzero(np.isnan(broken[key]))
        assert len(pos)
        broken[key].flat[pos[0]] = 0.
        with pytest.raises(ValueError, match='undefined mask'): V.reconstruct(w, st, p, ids, broken)
    reversed_saved = S.evaluate(w, Q.prepare(fixture(), w), p, ids[::-1])
    a = V.reconstruct(w, st, p, ids, saved); b = V.reconstruct(w, st, p, ids[::-1], reversed_saved)
    for key in a: np.testing.assert_allclose(a[key], b[key], atol=1e-12, rtol=0)


def complete(tmp_path):
    original, producer = complete_source_fixture(tmp_path/'producer')
    producer['design']['context_orders'] = [list(x) for x in V.ORDERS]
    write(original/'PLAN.json', producer)
    result = S.run(original, producer, lambda **kw:None); write(original/'SUMMARY.json', result)
    root = tmp_path/'checker'; shutil.copytree(original/'inputs', root/'inputs/parent')
    for path in original.iterdir():
        if path.name == 'inputs': continue
        target = root/'inputs/original'/path.name; target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir(): shutil.copytree(path, target)
        else: shutil.copyfile(path, target)
    return root, dict(design=dict(target_plan_sha256=file_digest(original/'PLAN.json'), input_files={p.relative_to(root/'inputs').as_posix():file_digest(p) for p in (root/'inputs').rglob('*') if p.is_file()}))


@pytest.mark.parametrize('corrupt', [None, 'hash', 'summary', 'duplicate_summary', 'timing', 'extra_raw', 'pair', 'count'])
def test_complete_integration_and_corruptions(tmp_path, corrupt):
    root, plan = complete(tmp_path)
    if corrupt in ('hash', 'pair', 'count'):
        path = root/'inputs/parent/bindings/1-omitted-4-2-bindings.json'; d = read(path)
        if corrupt == 'count': d['rows'][0]['report_sources'] += 1
        else: d['sources'][0][2] = 2
        write(path, d, immutable=False)
    elif corrupt in ('summary', 'duplicate_summary'):
        path = root/'inputs/original/raw/balanced_retention_summary_points.json.gz'
        rows = json.loads(gzip.decompress(path.read_bytes()))
        if corrupt == 'summary': rows[0]['recent_half_expected_squared_regret'] += .1
        else: rows.append(rows[0])
        path.write_bytes(gzip.compress(canonical(rows), mtime=0))
    elif corrupt == 'timing':
        path = root/'inputs/original/TIMING.jsonl'; rows = [json.loads(s) for s in path.read_text().splitlines()]; rows[0]['sources'] += 1
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    elif corrupt == 'extra_raw':
        path = root/'inputs/original/raw/extra_points.npz'; np.savez(path, x=[1])
    if corrupt:
        if corrupt != 'hash': plan['design']['input_files'] = {p.relative_to(root/'inputs').as_posix():file_digest(p) for p in (root/'inputs').rglob('*') if p.is_file()}
        with pytest.raises(ValueError): V.run(root, plan, lambda **kw:None)
    else:
        result = V.run(root, plan, lambda **kw:None)
        assert result['passed'] and result['rows'] == 2560 and result['sources'] == 768
        assert result['strata'] == 20 and not result['numerical_acceptance']
        assert len(read(root/'PAIRED_STRATA.json')) == 20


@pytest.mark.parametrize('order', [0, 1])
def test_independent_roster_ranks_match_round_robin(order):
    # Deliberately unbalanced contexts with exhausted queues and shuffled rows.
    ids = np.array([[9, 0, 7], [1, 3, 2], [3, 0, 6], [8, 2, 1],
                    [4, 0, 3], [7, 0, 0], [2, 1, 5], [6, 0, 4]])
    masks = V.select(ids, 10)
    for k, wi in ((int(masks[1].sum()), 3+2*order), (int(masks[2].sum()), 4+2*order)):
        np.testing.assert_array_equal(masks[wi], S.balanced_mask(ids, k, V.ORDERS[order]))
    changed = ids.copy(); changed[:, 2] = (changed[:, 2]+3)%8
    np.testing.assert_array_equal(V.select(changed, 10), masks)
    perm = [5, 2, 7, 1, 0, 6, 3, 4]
    np.testing.assert_array_equal(V.select(ids[perm], 10), masks[:, perm])


def test_single_context_and_empty_retention():
    ids = np.array([[1, 2, 1], [2, 2, 0], [4, 2, 4], [8, 2, 2]])
    masks = V.select(ids, 8)
    for wi, ri in ((3, 1), (4, 2), (5, 1), (6, 2)):
        np.testing.assert_array_equal(masks[wi], masks[ri])
    masks = V.select(ids[:2], 8)
    assert not masks[1:].any()


def test_reversal_is_a_real_tiebreaker():
    ids = np.array([[1, 0, 0], [2, 1, 0], [3, 2, 0], [4, 3, 0]])
    masks = V.select(ids, 4)
    np.testing.assert_array_equal(masks[3], [True, True, False, False])
    np.testing.assert_array_equal(masks[5], [False, False, True, True])
