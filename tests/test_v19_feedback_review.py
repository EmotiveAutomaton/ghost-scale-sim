from itertools import product
import gzip
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import feedback_review as V, primitive_feedback as F
from ghostscale.validation.soundingline.v19 import rollout as R, forward_support as S, rollout_transfer as T
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def fixture(root):
    base=root/'inputs';base.mkdir(parents=True)
    qs=[(1,0,2,3,4,4),(1,0,2,3,5,4)]
    counts=np.ones((2,2,8,8,6,8));counts[1,0,2,2,3,4]=5
    write(base/'QUERY_TRUTH.json',[dict(query=list(q),targets={r:T.oracle(q,r) for r in F.M.RULES}) for q in qs])
    for folder in ('models','forecasts','raw'):(base/folder).mkdir()
    for mode in ('original','composition-holdout'):
        np.savez(base/'models'/f'1-{mode}.npz',transition_counts=counts)
        np.savez(base/'forecasts'/f'1-{mode}.npz',queries=qs,**{'learned-exact':np.array([S.smooth(R.propagate(R.normalized(counts),q)) for q in qs])})
        write(base/'forecasts'/f'1-{mode}-support.json',[dict(query_seen=False,all_primitives_seen=False,visited_primitives=0,withheld_composition=False) for q in qs])
    for rule in F.M.RULES:
        records=[dict(maker=[0,1,0,0],initial=list(R.ARTIFACTS[q[2]]),steps=[dict(operation=F.L.OPERATIONS[o]) for o in q[3:]],final=list(R.ARTIFACTS[T.oracle(q,rule)]),probability=p) for q,p in zip(qs,(.8,.2))]
        (base/'raw'/f'2-{rule}_points.json.gz').write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(arms=list(F.ARMS),budgets=list(F.BUDGETS),orders=list(F.ORDERS),feedback=list(F.FEEDBACK),epsilon=S.EPSILON,saved_prior=1,
             input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()},queries=len(qs),lineages=[2],training_draws=[1])
    write(root/'PLAN.json',dict(design=cfg));write(root/'SUMMARY.json',F.run(root,dict(design=cfg),lambda **kw:None))


def test_known_tool_wrong_feedback_and_balanced_roster():
    assert V.inputs('reverse')==V.inputs('forward')[::-1]
    assert len(set(V.inputs('forward')))==128
    for rule,order in product(V.T.RULES,('forward','reverse')):
        for key,target in V.reports(order,64,rule,False): assert target==F.observation(key,rule)
        for key,target in V.reports(order,64,rule,True): assert target==F.observation(key,rule,True)


def test_single_prior_duplicates_nonobserved_and_undo_rows():
    counts=np.ones((2,2,8,8,6,8));obs=V.reports('forward',4,'original',False)
    for arm in V.ARMS:
        assert np.array_equal(V.revised(counts,[],arm),counts/8)
        assert np.array_equal(V.revised(counts,obs*2,arm),V.revised(counts,obs,arm))
        assert np.array_equal(V.revised(counts,obs,arm)[... ,5,:],(counts/8)[... ,5,:])
    key,target=obs[0];assert V.revised(counts,obs,'add-one')[key][target]==2/9
    with pytest.raises(ValueError,match='conflicting'):V.revised(counts,[(key,target),(key,target^1)],'add-one')


def test_weighting_empty_strata_and_shape_nan_rejection():
    d=[dict(query_seen=True,withheld_composition=False,all_primitives_seen=True)]*2
    cells=V.strata(dict(loss=np.array([2.,0.]),squared_error=np.zeros(2),true_probability=np.ones(2)),np.array([.9,.1]),d,[True,False],{})
    selected={r['weighting']:r for r in cells if r['change']=='all' and r['subset']=='all'}
    assert selected['native']['loss']==1.8 and selected['equal-query']['loss']==1
    assert all(r['loss'] is None for r in cells if r['subset']=='heldout-composition')
    for a,b in (([1.],[np.nan]),([1.,2.],[1.]),([1.],[.9])):
        with pytest.raises(ValueError):V.near(a,b)


def test_complete_native_fixture_and_corruptions(tmp_path):
    original=tmp_path/'original';fixture(original);output=tmp_path/'audit';output.mkdir()
    result=V.review(original,output,dict(bootstrap_resamples=20,bootstrap_seed=190971))
    assert result['passed'] and result['forecast_vectors']==192*2 and result['cells']==5760
    for folder,field in (('models','table'),('forecasts','predictions'),('forecasts','loss')):
        p=next((original/folder).glob('*.npz'));old=p.read_bytes()
        with np.load(p) as z:arrays={k:z[k] for k in z.files}
        arrays[field]=arrays[field].copy();arrays[field].flat[0]+=.1;np.savez_compressed(p,**arrays)
        with pytest.raises(ValueError,match='numerical'):V.review(original,output,dict(bootstrap_resamples=20,bootstrap_seed=190971))
        p.write_bytes(old)
    p=original/'evaluator/POPULATIONS.json';old=p.read_bytes();rows=read(p);rows[0]['masses'][0]+=.1;write(p,rows,immutable=False)
    with pytest.raises(ValueError,match='numerical'):V.review(original,output,dict(bootstrap_resamples=20,bootstrap_seed=190971))
    p.write_bytes(old)
    p=original/'SUMMARY.json';rows=read(p);rows['cells'][0]['queries']+=1;write(p,rows,immutable=False)
    with pytest.raises(ValueError,match='denominator'):V.review(original,output,dict(bootstrap_resamples=20,bootstrap_seed=190971))
