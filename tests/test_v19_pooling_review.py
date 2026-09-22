from itertools import product
import gzip
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import pooling_review as V, primitive_pooling as P
from ghostscale.validation.soundingline.v19 import rollout_transfer as T
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def fixture(root):
    base=root/'inputs';base.mkdir(parents=True)
    qs=[(1,0,2,3,4,4),(1,0,2,3,5,4)]
    counts=np.ones((2,2,8,8,6,8)); counts[1,0,2,2,3,4]=5
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


def test_known_scalar_pooling_prior_and_preserved_undo():
    counts=np.ones((2,2,8,8,6,8))
    for a,u in product(range(8),range(8)): counts[1,0,a,u,3,(a+u)%8]+=a+u+1
    for arm in V.ARMS:
        revised,table,info=V.revised(counts,[],arm)
        assert np.array_equal(revised,P.pooled_counts(counts,[],arm)[0])
        assert np.array_equal(revised[:,:,:,:,5],counts[:,:,:,:,5])
        assert np.allclose(table.sum(-1),1)
        assert info['total_tool_groups']==(128 if arm=='unpooled' else 16)
    flat,_,_=V.revised(np.ones_like(counts),[],'pool-undo')
    assert np.array_equal(flat,np.ones_like(counts))


def test_duplicate_inputs_count_once_distinct_aliases_count_separately():
    counts=np.ones((2,2,8,8,6,8)); obs=[((1,0,2,u,3),3) for u in range(8)]
    revised,_,info=V.revised(counts,obs*2,'pool-undo')
    assert revised[1,0,2,0,3,3]==9 and info['unique_inputs']==8 and info['observed_groups']==1
    with pytest.raises(ValueError,match='conflicting'): V.revised(counts,[obs[0],(obs[0][0],4)],'pool-undo')
    with pytest.raises(ValueError,match='prior'): V.revised(counts/2,[],'unpooled')


def test_correct_and_wrong_group_semantics():
    for rule,b,a in product(V.T.RULES,range(2),range(8)):
        values={target for key,target in V.reports('forward',128,rule,False) if key[1]==b and key[2]==a}
        assert len(values)==1
    assert any(len({target for key,target in V.reports('forward',128,r,False) if key[1]==b and key[3]==u})>1
               for r,b,u in product(V.T.RULES,range(2),range(8)))


def test_regroup_pairs_fits_inside_lineages_and_feedback_baseline():
    rows=[]
    for lineage,draw,arm,budget in product(range(8),(1,2),V.ARMS,(0,16)):
        value=lineage+draw*10+(2 if arm=='pool-undo' else 0)+(3 if budget else 0)
        rows.append(dict(lineage=lineage,draw=draw,mode='original',rule='original',order='forward',feedback='true',budget=budget,arm=arm,change='all',subset='all',weighting='native',loss=value))
    result=V.regroup(rows,list(range(8)),[1,2],dict(bootstrap_seed=190972,bootstrap_resamples=20))
    assert all(len(r['complete_lineages'])==8 and len(r['draw_means'])==2 for r in result['estimates'])
    for r in result['estimates']:
        if r['contrast']=='feedback-minus-zero': assert r['mean']==r['low']==r['high']==3
        if r['contrast']=='pool-minus-unpooled' and r['arm']=='pool-undo': assert r['mean']==r['low']==r['high']==2
    with pytest.raises(ValueError,match='duplicate'): V.regroup(rows+rows[:1],list(range(8)),[1,2],{})


def test_complete_native_fixture_and_corruptions(tmp_path):
    original=tmp_path/'original';fixture(original);output=tmp_path/'audit';output.mkdir()
    cfg=dict(bootstrap_resamples=20,bootstrap_seed=190972)
    result=V.review(original,output,cfg)
    assert result['passed'] and result['forecast_vectors']==192 and result['cells']==2880
    for folder,field in (('models','counts'),('models','table'),('forecasts','predictions'),('forecasts','loss')):
        p=next((original/folder).glob('*.npz'));old=p.read_bytes()
        with np.load(p) as z: arrays={k:z[k] for k in z.files}
        arrays[field]=arrays[field].copy();arrays[field].flat[0]+=.1;np.savez_compressed(p,**arrays)
        with pytest.raises(ValueError,match='numerical'): V.review(original,output,cfg)
        p.write_bytes(old)
    for name,mutate,match in (
        ('evaluator/POOL_MAPS.json',lambda x:x['pool-undo'][0]['group'].__setitem__(3,0),'pool map'),
        ('evaluator/GROUP_COUNTS.json',lambda x:x[0].__setitem__('unique_inputs',99),'group denominator'),
        ('evaluator/POPULATIONS.json',lambda x:x[0]['masses'].__setitem__(0,.5),'numerical'),
        ('reader/QUERIES.json',lambda x:x[0].__setitem__('truth',1),'projection'),
        ('SUMMARY.json',lambda x:x['cells'][0].__setitem__('queries',99),'denominator')):
        p=original/name;old=p.read_bytes();obj=read(p);mutate(obj);write(p,obj,immutable=False)
        with pytest.raises(ValueError,match=match): V.review(original,output,cfg)
        p.write_bytes(old)
