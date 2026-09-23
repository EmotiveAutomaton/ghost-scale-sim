from itertools import product
import gzip
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import replacement_review as V, pooled_replacement as P
from ghostscale.validation.soundingline.v19 import rollout_transfer as T
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def fixture(root):
    base=root/'inputs';base.mkdir(parents=True)
    qs=[(1,0,2,3,4,4),(1,0,2,3,5,4)]
    counts=np.ones((2,2,8,8,6,8));counts[1,0,2,2,3,4]=5
    write(base/'QUERY_TRUTH.json',[dict(query=list(q),targets={r:T.oracle(q,r) for r in P.F.M.RULES}) for q in qs])
    for folder in ('models','forecasts','raw'): (base/folder).mkdir()
    for mode in ('original','composition-holdout'):
        np.savez(base/'models'/f'1-{mode}.npz',transition_counts=counts)
        np.savez(base/'forecasts'/f'1-{mode}.npz',queries=qs,**{'learned-exact':np.array([P.F.S.smooth(P.F.R.propagate(P.F.R.normalized(counts),q)) for q in qs])})
        write(base/'forecasts'/f'1-{mode}-support.json',[dict(query_seen=False,all_primitives_seen=False,visited_primitives=0,withheld_composition=False) for q in qs])
    for rule in P.F.M.RULES:
        records=[dict(maker=[0,1,0,0],initial=list(P.F.R.ARTIFACTS[q[2]]),steps=[dict(operation=P.F.L.OPERATIONS[o]) for o in q[3:]],final=list(P.F.R.ARTIFACTS[T.oracle(q,rule)]),probability=p) for q,p in zip(qs,(.8,.2))]
        (base/'raw'/f'2-{rule}_points.json.gz').write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(arms=list(P.ARMS),budgets=list(P.BUDGETS),orders=list(P.F.ORDERS),feedback=list(P.F.FEEDBACK),epsilon=1/32,saved_prior=1,
             input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()},queries=len(qs),lineages=[2],training_draws=[1])
    write(root/'PLAN.json',dict(design=cfg));write(root/'SUMMARY.json',P.run(root,dict(design=cfg),lambda **kw:None))


def test_scalar_replacement_keeps_one_prior_and_unobserved_counts():
    counts=np.ones((2,2,8,8,6,8)); counts[1,0,2,0,3,6]=41
    observed=[((1,0,2,0,3),3),((1,0,2,1,3),4)]
    for arm in V.ARMS:
        actual,table,info=V.revised(counts,observed*2,arm)
        assert np.array_equal(actual,P.updated_counts(counts,observed,arm)[0])
        assert np.allclose(table.sum(-1),1)
        assert info['unique_inputs']==2
        assert np.array_equal(actual[:,:,:,:,5],counts[:,:,:,:,5])
    result,_,_=V.revised(counts,observed,'pool-undo__replace')
    assert result[1,0,2,0,3].sum()==10
    assert result[1,0,2,0,3,3]==result[1,0,2,0,3,4]==2
    assert result[1,0,2,0,3,6]==1
    with pytest.raises(ValueError,match='conflicting'):
        V.revised(counts,[observed[0],(observed[0][0],4)],'pool-undo__replace')


def test_zero_feedback_and_uniform_null():
    counts=np.ones((2,2,8,8,6,8));counts[1,0,2,0,3,6]=41
    for mapping in V.MAPS:
        assert np.array_equal(V.revised(counts,[],mapping+'__add')[0],V.revised(counts,[],mapping+'__replace')[0])
        assert np.array_equal(V.revised(np.ones_like(counts),[],mapping+'__replace')[0],np.ones_like(counts))


def test_paired_regroup_retains_draws_and_update_contrast():
    rows=[]
    for lineage,draw,arm,budget in product(range(8),(1,2),V.ARMS,(0,16)):
        value=lineage+draw*10+(2 if arm.startswith('pool-undo') else 0)+(3 if budget else 0)+(5 if arm.endswith('replace') else 0)
        rows.append(dict(lineage=lineage,draw=draw,mode='original',rule='original',order='forward',feedback='true',budget=budget,arm=arm,change='all',subset='all',weighting='native',loss=value))
    result=V.regroup(rows,list(range(8)),[1,2],dict(bootstrap_seed=190975,bootstrap_resamples=20))
    assert all(len(r['complete_lineages'])==8 and len(r['draw_means'])==2 for r in result['estimates'])
    for r in result['estimates']:
        if r['contrast']=='replace-minus-add': assert r['mean']==r['low']==r['high']==5
        if r['contrast']=='feedback-minus-zero': assert r['mean']==r['low']==r['high']==3
    with pytest.raises(ValueError,match='duplicate'):V.regroup(rows+rows[:1],list(range(8)),[1,2],{})


def test_complete_native_fixture_and_deliberate_corruption(tmp_path):
    original=tmp_path/'original';fixture(original);output=tmp_path/'audit';output.mkdir()
    cfg=dict(bootstrap_resamples=20,bootstrap_seed=190975)
    result=V.review(original,output,cfg)
    assert result['passed'] and result['forecast_vectors']==384 and result['cells']==5760
    for folder,field in (('models','counts'),('models','table'),('forecasts','predictions'),('forecasts','loss')):
        p=next((original/folder).glob('*.npz'));old=p.read_bytes()
        with np.load(p) as z:arrays={k:z[k] for k in z.files}
        arrays[field]=arrays[field].copy();arrays[field].flat[0]+=.1;np.savez_compressed(p,**arrays)
        with pytest.raises(ValueError,match='numerical'):V.review(original,output,cfg)
        p.write_bytes(old)
    for name,mutate,match in (
        ('evaluator/POOL_MAPS.json',lambda x:x['pool-undo__replace'][0]['group'].__setitem__(3,0),'pool map'),
        ('evaluator/GROUP_COUNTS.json',lambda x:x[0].__setitem__('unique_inputs',99),'group denominator'),
        ('evaluator/POPULATIONS.json',lambda x:x[0]['masses'].__setitem__(0,.5),'numerical'),
        ('reader/QUERIES.json',lambda x:x[0].__setitem__('truth',1),'projection'),
        ('SUMMARY.json',lambda x:x['cells'][0].__setitem__('queries',99),'denominator')):
        p=original/name;old=p.read_bytes();obj=read(p);mutate(obj);write(p,obj,immutable=False)
        with pytest.raises(ValueError,match=match):V.review(original,output,cfg)
        p.write_bytes(old)
