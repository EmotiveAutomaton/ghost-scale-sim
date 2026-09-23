"""Known-answer mass/conditional decomposition and complete frozen execution."""
import copy
import gzip
import json
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import temporal_mass_control as F, temporal_factorization as T
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,canonical
from test_v19_joint_factorization import fixture as joint_fixture


def fixture(root):
    plan=joint_fixture(root);base=root/'inputs';temp=root/'parent-execution'
    temp.mkdir();shutil.copytree(base,temp/'inputs')
    summary=T.run(temp,plan,lambda **kw:None)
    write(base/'temporal/PLAN.json',dict(design={k:v for k,v in plan['design'].items() if k!='input_files'}))
    for src,dst in [('temporal_points.json.gz','temporal_points.json.gz'),('evaluator/NATIVE_SCORES.json','NATIVE_SCORES.json')]:shutil.copyfile(temp/src,base/'temporal'/dst)
    write(base/'temporal/SUMMARY.json',summary)
    plan['design']['bootstrap_seed']=191009
    plan['design']['input_files']={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}
    return plan


def test_controls():assert all(F.controls().values())


def test_scalar_group_conditionals_and_loss_decomposition():
    rng=np.random.default_rng(1201);q=rng.random(F.N);q/=q.sum();p=rng.random(F.N);p/=p.sum();target=rng.random(F.N);target/=target.sum()
    alphabet=np.arange(0,F.N,3);known=set(alphabet.tolist())
    matched,before,after=F.mass_match(q[None],p[None],alphabet)
    expected=np.zeros(F.N);mass_loss=conditional_loss=0.
    for group in ([k for k in range(F.N) if k in known],[k for k in range(F.N) if k not in known]):
        a=math.fsum(q[k] for k in group);b=math.fsum(p[k] for k in group);w=math.fsum(target[k] for k in group)
        for k in group:expected[k]=q[k]*b/a
        mass_loss-=w*math.log(b)
        conditional_loss-=math.fsum(target[k]*math.log(q[k]/a) for k in group)
    assert np.allclose(matched[0],expected,rtol=0,atol=2e-18)
    actual=-math.fsum(target[k]*math.log(matched[0,k]) for k in range(F.N))
    assert abs(actual-mass_loss-conditional_loss)<1e-12
    assert np.allclose(before.sum(1),1) and np.allclose(after.sum(1),1)
    for mask in (np.isin(np.arange(F.N),alphabet),~np.isin(np.arange(F.N),alphabet)):
        assert abs(matched[0,mask].sum()-p[mask].sum())<1e-12


@pytest.mark.parametrize('alphabet',[np.array([],int),np.arange(F.N),np.array([0])])
def test_empty_groups_and_point_identity(alphabet):
    q=np.zeros((1,F.N));q[0,0]=1
    assert np.array_equal(F.mass_match(q,q,alphabet)[0],q)


def test_uniform_identity_and_zero_target_group():
    q=np.ones((1,F.N))/F.N;p=q.copy()
    assert np.allclose(F.mass_match(q,p,np.arange(13))[0],q,rtol=0,atol=1e-18)
    p[:]=0;p[0,0]=1
    result=F.mass_match(q,p,np.array([0]))[0]
    assert np.allclose(result,p,rtol=0,atol=1e-15)
    assert np.count_nonzero(result)==1


def test_positive_mass_into_undefined_conditional_rejected():
    q=np.zeros((1,F.N));q[0,0]=1;p=q.copy();p[0,0]=.5;p[0,1]=.5
    with pytest.raises(ValueError,match='undefined'):F.mass_match(q,p,np.array([0]))


@pytest.mark.parametrize('problem',['negative','nan','normalization','shape','alphabet-type','alphabet-duplicate','alphabet-range'])
def test_invalid_inputs(problem):
    q=np.ones((1,F.N))/F.N;p=q.copy();a=np.array([0,1])
    if problem=='negative':q[0,0]=-1
    elif problem=='nan':p[0,0]=np.nan
    elif problem=='normalization':q*=2
    elif problem=='shape':q=q[0]
    elif problem=='alphabet-type':a=a.astype(float)
    elif problem=='alphabet-duplicate':a=np.array([0,0])
    else:a=np.array([F.N])
    with pytest.raises(ValueError):F.mass_match(q,p,a)


def test_disjoint_native_support_and_candidate_ties():
    q=np.zeros((1,F.N));q[0,:4]=.25;p=q.copy();p[0,:4]=[.1,.1,.4,.4]
    result,_,_=F.mass_match(q,p,np.array([0,1]))
    refs=[dict(lineage=1,tier='E0',frames=[dict(frame='x',mass=1.,target=[[2,.5],[3,.5]])])]
    ref=F.S.reference_arrays(refs,'E0',['x'],[1],np.ones_like(q,bool))
    score=F.S.scores(result,np.array([0,1]),ref)
    assert np.isclose(score['loss'][0],-math.log(.4))
    assert np.isclose(score['compatible_mass'][0],.8)
    assert score['candidate_coverage'][0]==1 and score['top_incompatible'][0]==0


def test_complete_synthetic_handler(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=F.run(root,plan,lambda **kw:None)
    assert s['rows']==384 and s['original_cells_reproduced']==128 and s['temporal_cells_reproduced']==256
    assert all(s['controls'].values()) and s['original_max_error']<1e-10 and s['temporal_max_error']<1e-10
    assert len(s['contrasts'])==32 and len(s['normalized_log_budget_area'])==16 and len(s['means'])==48
    assert file_digest(root/'reader/PACKETS.json')==file_digest(root/'inputs/reader/PACKETS.json')
    for path in (root/'masses').glob('*.npz'):
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)=={'steps','original','product'}
            assert z['original'].shape==(1,2) and z['steps'].shape==(1,3,18)


@pytest.mark.parametrize('problem',['original-score','temporal-score','private','hash','missing','duplicate','reference','native','population'])
def test_corruption_rejected(tmp_path,problem):
    root=tmp_path/'packet';plan=fixture(root);base=root/'inputs';p=base/'SUMMARY.json';v=read(p)
    if problem=='temporal-score':
        p=base/'temporal/temporal_points.json.gz';v=json.loads(gzip.decompress(p.read_bytes()));v[0]['loss']+=.1;p.write_bytes(gzip.compress(canonical(v),mtime=0))
    else:
        if problem=='private':p=base/'reader/PACKETS.json';v=read(p);next(iter(v['packets'].values()))['inputs']['goal']=0
        elif problem=='reference':p=base/'evaluator/REFERENCES.json';v=read(p);v[0]['frames'][0]['target'][0][1]=.1
        elif problem=='native':p=base/'temporal/NATIVE_SCORES.json';v=read(p);v[0]['loss']+=.1
        elif problem=='population':p=base/'temporal/PLAN.json';v=read(p);v['design']['budgets']=[2,3]
        elif problem=='missing':v['cells'].pop()
        elif problem=='duplicate':v['cells'].append(copy.deepcopy(v['cells'][0]))
        else:v['cells'][0]['loss']+=.1
        write(p,v,immutable=False)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):F.run(root,plan,lambda **kw:None)


def test_missing_regroup_stratum():
    cfg=dict(tiers=['E0'],budgets=[32,128],training_draws=[1],fit_seeds=[2],development_lineages=[3],bootstrap_seed=191009,bootstrap_resamples=5)
    with pytest.raises(ValueError):F.aggregate([],cfg)
