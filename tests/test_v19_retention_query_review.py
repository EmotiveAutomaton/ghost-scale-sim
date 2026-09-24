import gzip
import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import retention_query_review as V, retention_query as S, reachable_retrospective as Q
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
    for key in ('future_squared_regret', 'max_future_probability_error', 'updated_group_total_variation', 'query_squared_regret', 'query_max_future_probability_error'):
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
    producer['design']['selection_rule'] = 'midpoint-sorted-time-ranks'
    producer['design']['query_contexts'] = list(S.CONTEXTS)
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
        path = root/'inputs/original/raw/retention_query_summary_points.json.gz'
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


@pytest.mark.parametrize('n', range(1, 33))
def test_exact_midpoint_ranks_all_counts_and_irregular_times(n):
    # Changing the checkpoint realizes every possible recent-half source count.
    times = np.arange(1, n+1)*3
    ids = np.array([[t, i%4, i%8] for i, t in enumerate(times)])
    for k in range(n+1):
        cp = 2*(0 if k == n else times[n-k-1])
        masks = V.select(ids, cp)
        assert masks[1].sum() == k
        for recent, spaced in ((1, 3), (2, 4)):
            np.testing.assert_array_equal(masks[spaced], S.spaced_mask(ids, int(masks[recent].sum())))
        altered = ids.copy(); altered[:, 1:] = 0
        np.testing.assert_array_equal(V.select(altered, cp), masks)
        np.testing.assert_array_equal(V.select(ids[::-1], cp), masks[:, ::-1])


def test_fixed_midpoint_roster_and_empty_selection():
    ids = np.array([[1, 2, 1], [2, 2, 0], [4, 2, 4], [8, 2, 2]])
    masks = V.select(ids, 4)
    np.testing.assert_array_equal(masks[3], [False, True, False, True])
    masks = V.select(ids[:2], 8)
    assert not masks[1:].any()


@pytest.mark.parametrize('future_steps', [0, 1, 2, 3, 7])
def test_all_contexts_permutation_and_recombination(future_steps):
    cp=2
    spec=dict(hypotheses=[['none',0,i] for i in range(16)],length=cp+future_steps,checkpoint=cp,signatures=[[i]*(future_steps+1) for i in range(16)],membership=list(range(16)))
    w=np.arange(1,17,dtype=float);w/=w.sum();p=law('random');ids=np.array([[1,0,0],[2,1,1]])
    saved=S.evaluate(w,Q.prepare(spec,w),p,ids)
    actual=V.reconstruct(w,V.structure(spec),p,ids,saved)
    expected=S.summarize(saved)
    assert len(actual)==90
    for key in actual:np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=0)
    order=np.array([2,0,3,1]);inverse=np.argsort(order);changed=ids.copy();changed[:,1]=inverse[ids[:,1]]
    base,summary=V.calculate(w,V.structure(spec),p,ids)
    perm,ps=V.calculate(w,V.structure(spec),p[:,order],changed)
    for k in ('query_squared_regret','query_max_future_probability_error'):
        np.testing.assert_allclose(perm[k],base[k][order],atol=1e-12,rtol=0)
    np.testing.assert_allclose(base['query_squared_regret'].mean(0),base['future_squared_regret'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(base['query_max_future_probability_error'].max(0),base['max_future_probability_error'],atol=1e-12,rtol=0)


def test_query_recombination_cannot_hide_swapped_contexts():
    w,st,p,ids,saved=example()
    broken={k:v.copy() for k,v in saved.items()}
    for k in ('query_squared_regret','query_max_future_probability_error'):
        broken[k]=broken[k][[1,0,2,3]]
    np.testing.assert_allclose(broken['query_squared_regret'].mean(0),saved['query_squared_regret'].mean(0),atol=1e-12)
    with pytest.raises(ValueError):V.reconstruct(w,st,p,ids,broken)


@pytest.mark.parametrize('field',['query_contexts','selection_rule','alphas'])
def test_changed_frozen_design_is_rejected(tmp_path,field):
    root,plan=complete(tmp_path)
    path=root/'inputs/original/PLAN.json';d=read(path);d['design'][field]=[]
    write(path,d,immutable=False)
    plan['design']['target_plan_sha256']=file_digest(path)
    plan['design']['input_files'][path.relative_to(root/'inputs').as_posix()]=file_digest(path)
    with pytest.raises(ValueError,match='design/parent'):V.run(root,plan,lambda **kw:None)
