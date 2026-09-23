"""Known answers and corruptions for independent support adjudication."""
import copy
from itertools import product
from pathlib import Path
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import joint_support_review as R
from ghostscale.validation.soundingline.v18_3.io import write, read, digest, file_digest


def packet(tier='E0', ops=(4,4,4), initial=(0,1,0), skill=1):
    a=prev=initial;events=[]
    for step,op in enumerate(ops):
        b=R.transition(a,prev,op,skill,0)
        events.append(dict(step=step,operation=R.OPS[op],before=list(a),after=list(b),tool_proposal=list(b) if op==3 else None))
        prev,a=a,b
    inputs=dict(artifact=list(a))
    if tier!='E0':inputs.update(initial=list(initial),requested_purpose=0)
    if tier.startswith('E2'):inputs['observations']=events[:1] if tier=='E2-sparse' else events
    return dict(schema='v19.local.public.1',tier=tier,inputs=inputs)


@pytest.fixture(scope='module')
def index():
    idx,counts=R.enumerate_support()
    assert counts['executions']==5456 and counts['primitive_transitions']==16368
    return idx


def test_full_witness_goal_union(index):
    mask=R.mask_for(packet('E2-full',ops=(0,1,5)),index)
    assert np.array_equal(np.flatnonzero(mask),216*np.arange(27)+11)


def test_hidden_start_and_request_union(index):
    p=packet();m=R.mask_for(p,index)
    assert m[0] and m[172] and m.sum()>27
    p=packet('E1');q=copy.deepcopy(p);q['inputs']['requested_purpose']=1
    assert np.array_equal(R.mask_for(p,index),R.mask_for(q,index))
    assert np.all(R.mask_for(p,index)<=m)


@pytest.mark.parametrize('ops',[(3,5,3),(5,5,5),(0,1,5)])
def test_tool_and_undo(index,ops):
    m=R.mask_for(packet('E2-full',ops=ops),index)
    assert m.sum()==27 and m[36*ops[0]+6*ops[1]+ops[2]]


@pytest.mark.parametrize('field',['goal','maker','policy','probability'])
def test_reader_leak_rejected(field):
    p=packet();p['inputs'][field]=0
    with pytest.raises(ValueError):R.validate(p)


def test_schema_and_witness_corruption(index):
    p=packet('E2-full');p['inputs']['observations'][0]['undo_buffer']=[0,0,0]
    with pytest.raises(ValueError):R.validate(p)
    p=packet('E2-full');p['inputs']['artifact']=[1,1,1]
    assert not R.mask_for(p,index).any()


def test_conditions_and_invalids():
    assert all(R.controls().values())
    p=np.array([[1.,0.]])
    for mask in (np.array([[0,0]],bool),np.array([[0,1]],bool)):
        q,z,ok=R.condition(p,mask)
        assert not ok[0] and z[0]==0 and np.array_equal(q,p)
    with pytest.raises(ValueError):R.condition(np.array([[np.nan,0.]]),np.ones((1,2),bool))


def test_unknown_and_stable_tie_scores():
    full=R.expand(np.array([[32/33]]),np.array([0]),32)
    mask=np.zeros_like(full,bool);mask[0,[1,2]]=True
    q,_,_=R.condition(full,mask)
    refs=[(np.array([1,2]),np.array([[.25,.75]]),np.array([1.]))]
    m=R.score(q,[0],refs)
    assert np.isclose(m['loss'][0],np.log(2)) and m['candidate_coverage'][0]==1 and m['candidate_size'][0]==2
    assert m['operation_2'][0]==.25 and m['unknown_mass'][0]==1 and m['top_incompatible'][0]==0
    # Known label takes priority over a lower-numbered unseen label at a tie.
    q[:]=0;q[0,[0,1]]=.5
    m=R.score(q,[1],[(np.array([0,1]),np.array([[0.,1.]]),np.array([1.]))])
    assert m['operation_2'][0]==1 and m['abstain'][0]==0


def test_native_compatibility_is_narrower_than_legality():
    q=np.zeros((1,R.N));q[0,[0,1,2]]=[.2,.5,.3]
    m=R.score(q,[0,1,2],[(np.array([0,2]),np.array([[.25,.75]]),np.array([1.]))])
    assert m['compatible_mass'][0]==.5 and m['top_incompatible'][0]==1
    assert np.isclose(m['loss'][0],-.25*np.log(.2)-.75*np.log(.3))
    assert np.isclose(m['squared_error'][0],(.2-.25)**2+.5**2+(.3-.75)**2)


def test_true_exclusion_and_corrupted_numeric_rejected():
    refs=[dict(lineage=1,tier='E0',frames=[dict(frame='a',mass=1.,target=[[0,1.]])])]
    with pytest.raises(ValueError,match='excluded'):R.references(refs,'E0',['a'],[1],np.zeros((1,R.N),bool))
    with pytest.raises(ValueError):R.close([.1],[.2])
    with pytest.raises(ValueError):R.expand(np.array([[.5,.5]]),np.array([0,0]),32)


def test_whole_handler_and_mask_corruption(tmp_path):
    # Synthetic complete producer fixture; checker imports no producer code.
    from ghostscale.validation.soundingline.v19 import joint_support as S, joint_review as V
    root=tmp_path/'produced';parent=root/'inputs'
    for n in ('reader','evaluator','forecasts'):(parent/n).mkdir(parents=True)
    cfg=dict(tiers=['E0','E2-full'],budgets=[32,128],training_draws=[1],fit_seeds=[2],development_lineages=[3,4],bootstrap_seed=191003,bootstrap_resamples=10000)
    write(parent/'PLAN.json',dict(design=cfg));packets={};refs=[];cells=[]
    for tier in cfg['tiers']:
        p=packet(tier);key=digest(p);packets[key]=p;truth={172:.75,388:.25}
        for lineage in cfg['development_lineages']:
            refs.append(dict(lineage=lineage,tier=tier,frames=[dict(frame=key,mass=1.,target=list(truth.items()))]))
        for budget,arm in product(cfg['budgets'],S.ARMS):
            stem=f'1-{tier}-2-{budget}-{arm}';pred=np.array([[.6,budget/(budget+1)-.6]]);alphabet=np.array([172,388])
            np.savez_compressed(parent/'forecasts'/(stem+'.npz'),probabilities=pred,alphabet=alphabet)
            write(parent/'forecasts'/(stem+'-frames.json'),[key])
            for lineage in cfg['development_lineages']:
                cells.append(dict(draw=1,tier=tier,seed=2,budget=budget,arm=arm,lineage=lineage,frames=1,**V.score(pred[0],alphabet,truth,budget)))
    # Producer score returns numpy booleans; serialize explicit primitives.
    for row in cells:
        for k,v in row.items():
            if isinstance(v,np.generic):row[k]=v.item()
    write(parent/'reader/PACKETS.json',dict(packets=packets));write(parent/'evaluator/REFERENCES.json',refs);write(parent/'SUMMARY.json',dict(cells=cells))
    cfg=dict(cfg,input_files={p.relative_to(parent).as_posix():file_digest(p) for p in parent.rglob('*') if p.is_file()})
    write(root/'PLAN.json',dict(design=cfg));summary=S.run(root,dict(design=cfg),lambda **kw:None);write(root/'SUMMARY.json',summary)
    out=tmp_path/'review';inputs=out/'inputs';shutil.copytree(parent,inputs/'parent')
    original=inputs/'original';original.mkdir()
    for name in ('reader','evaluator','raw'):shutil.copytree(root/name,original/name)
    for name in ('PLAN.json','SUMMARY.json'):shutil.copyfile(root/name,original/name)
    design=dict(target_plan_sha256=file_digest(original/'PLAN.json'),input_files={p.relative_to(inputs).as_posix():file_digest(p) for p in inputs.rglob('*') if p.is_file()})
    answer=R.run(out,dict(design=design),lambda **kw:None)
    assert answer['rows']==72 and answer['original_cells_reproduced']==32 and answer['max_score_error']<1e-10
    assert answer['contrasts']==32 and answer['learning_areas']==16
    damaged=arrays_path=original/'evaluator/E0-support.npz'
    mask=R.arrays(damaged)['masks'];mask[0,0]=~mask[0,0];np.savez_compressed(damaged,masks=mask)
    # Integrity failure occurs before scientific input use, even with valid data elsewhere.
    second=tmp_path/'second';second.mkdir();shutil.copytree(inputs,second/'inputs')
    with pytest.raises(ValueError,match='input hash'):R.run(second,dict(design=design),lambda **kw:None)


def test_no_producer_dependency():
    import ast
    tree=ast.parse(Path(R.__file__).read_text())
    imported=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(any(s in n for s in ('joint_support','joint_review','local_world')) for n in imported)
