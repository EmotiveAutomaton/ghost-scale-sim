from itertools import product
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import support_review as V
from ghostscale.validation.soundingline.v18_3.io import write, file_digest


def test_known_answers_and_corruption():
    assert all(V.controls().values())
    with pytest.raises(ValueError): V.near([.1,.9], [.9,.1])


def test_propagation_matches_explicit_path_sum():
    random=np.random.default_rng(48); table=V.normalize(random.uniform(.1,1,(2,2,8,8,6,8)))
    q=(1,0,2,3,1,5); expected=np.zeros(8)
    for a,b,c in product(range(8),repeat=3):
        expected[c]+=table[1,0,2,2,3,a]*table[1,0,a,2,1,b]*table[1,0,b,a,5,c]
    assert np.allclose(V.propagate(table,q),expected,atol=1e-14)


def test_tied_retrieval_averages_counts_before_normalization():
    q=(1,0,2,4,4,4); a=q[:-1]+(1,); b=q[:-1]+(2,)
    p=np.arange(1.,9.); r=np.arange(8.,0.,-1)*3
    assert np.allclose(V.nearest({a:p,b:r},q),(p+r)/(p+r).sum())


def test_undo_support_requires_true_buffer():
    q=(1,0,2,3,1,5); counts=np.ones((2,2,8,8,6,8)); a=b=q[2]
    for op in q[3:]:
        n=V.code(V.execute(V.ARTS[a],V.ARTS[b],V.OPS[op],1,0,'original'))
        counts[1,0,a,b,op,n]+=1; b,a=a,n
    _,diag,_=V.forecast(counts,{},q,np.full((16,3),.5))
    assert diag['all_primitives_seen'] and not diag['query_seen']
    counts[1,0,6,6,5]=1
    assert not V.forecast(counts,{},q,np.full((16,3),.5))[1]['all_primitives_seen']


def test_complete_native_reconstruction_and_damaged_forecast(tmp_path):
    from ghostscale.validation.soundingline.v19 import forward_support as S, local_world as L, rollout as R
    original=tmp_path/'original'; original.mkdir(); inputs=original/'inputs'
    records=L.enumerate_world(L.law(190966))
    selected=[next(r for r in records if V.held(V.query(r))),next(r for r in records if not V.held(V.query(r)))]
    write(inputs/'auxiliary/selection-190301.json',dict(paths=selected))
    counts,direct=R.fit(selected);keys=sorted(direct);(inputs/'models').mkdir()
    np.savez(inputs/'models/190301-2048.npz',transition_counts=counts,direct_keys=keys,direct_counts=[direct[k] for k in keys])
    design=dict(arms=list(S.ARMS),epsilon=1/32,lineages=[190966],training_draws=[190301],input_files={p.relative_to(inputs).as_posix():file_digest(p) for p in inputs.rglob('*') if p.is_file()})
    write(original/'PLAN.json',dict(design=design))
    result=S.run(original,dict(design=design),lambda **kw:None);write(original/'SUMMARY.json',result)
    root=tmp_path/'verify';shutil.copytree(original,root/'inputs/original')
    config=dict(input_files={p.relative_to(root/'inputs').as_posix():file_digest(p) for p in (root/'inputs').rglob('*') if p.is_file()},target_plan_sha256=file_digest(original/'PLAN.json'),bootstrap_seed=190501,bootstrap_resamples=20)
    check=V.run(root,dict(design=config),lambda **kw:None)
    assert check['paths']==13824 and check['rows']==8960 and check['max_error']<1e-10
    damaged=tmp_path/'damaged';shutil.copytree(original,damaged/'inputs/original')
    path=damaged/'inputs/original/forecasts/190301-original.npz'
    with np.load(path) as z: arrays={k:z[k].copy() for k in z.files}
    arrays['direct'][0,0]+=.01;arrays['direct'][0,1]-=.01;np.savez(path,**arrays)
    config=dict(config,input_files={p.relative_to(damaged/'inputs').as_posix():file_digest(p) for p in (damaged/'inputs').rglob('*') if p.is_file()})
    with pytest.raises(ValueError,match='reconstruction mismatch'):V.run(damaged,dict(design=config),lambda **kw:None)
