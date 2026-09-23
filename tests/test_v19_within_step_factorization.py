"""Known dependence, identity and corruption fixtures before scientific outcomes."""
import copy
import gzip
import json
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import within_step_factorization as F
from ghostscale.validation.soundingline.v19 import temporal_factorization as T
from ghostscale.validation.soundingline.v19 import within_step_factorization_review as I
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest, canonical
from test_v19_joint_factorization import fixture as joint_fixture


def fixture(root):
    plan=joint_fixture(root)
    prior=root.parent/'temporal-fixture';prior.mkdir()
    shutil.copytree(root/'inputs',prior/'inputs')
    T.run(prior,plan,lambda **kw:None)
    for src,dest in [('temporal_points.json.gz','temporal_points.json.gz'),('evaluator/NATIVE_SCORES.json','NATIVE_SCORES.json')]:
        out=root/'inputs/temporal'/dest;out.parent.mkdir(exist_ok=True)
        shutil.copyfile(prior/src,out)
        plan['design']['input_files']['temporal/'+dest]=file_digest(out)
    return plan


def scalar(q):
    membership=[(k//1944,(k//648)%3,(k//216)%3,(k//36)%6,(k//6)%6,k%6) for k in range(F.N)]
    marg=[np.array([math.fsum(float(q[k]) for k in range(F.N) if membership[k][a]==v) for v in range(size)]) for a,size in enumerate((3,3,3,6,6,6))]
    return np.array([math.prod(marg[a][row[a]] for a in range(6)) for row in membership]),marg


def test_known_controls():assert all(F.controls().values())


def test_scalar_six_marginals_products_and_axis_order():
    q=np.arange(1,F.N+1,dtype=float);q/=q.sum()
    pair,single,steps,goals,ops=F.factorize(q[None])
    expected,m=scalar(q)
    assert np.allclose(single[0],expected,rtol=0,atol=1e-17)
    assert np.allclose(goals[0],np.array(m[:3]),rtol=0,atol=1e-15)
    assert np.allclose(ops[0],np.array(m[3:]),rtol=0,atol=1e-15)
    ip,ig,io=I.marginals(q)
    assert np.allclose(ig,np.array(m[:3]),rtol=0,atol=1e-15)
    assert np.allclose(io,np.array(m[3:]),rtol=0,atol=1e-15)
    pp,ss,e=I.reconstruct(q,steps[0],goals[0],ops[0])
    assert e<1e-12 and np.array_equal(pp,pair[0]) and np.array_equal(ss,single[0])
    wrong=np.ones(F.N)
    for t in range(3):wrong*=goals[0,t,F.PAIRS[:,2-t]//6]*ops[0,t,F.PAIRS[:,2-t]%6]
    assert np.max(abs(single[0]-wrong))>1e-6


@pytest.mark.parametrize('field',['steps','goals','operations'])
def test_independent_reconstruction_rejects_marginal_corruption(field):
    q=np.arange(1,F.N+1,dtype=float);q/=q.sum()
    _,_,s,g,o=F.factorize(q[None]);data=dict(steps=s[0],goals=g[0],operations=o[0])
    data[field][0,0]+=.01
    with pytest.raises(ValueError):I.reconstruct(q,data['steps'],data['goals'],data['operations'])


def test_equal_singles_different_pairs_and_log_penalty():
    a=np.zeros((1,F.N));b=a.copy();a[0,[0,1980]]=.5;b[0,[36,1944]]=.5
    ap,aa,*_=F.factorize(a);bp,bb,*_=F.factorize(b)
    assert np.array_equal(aa,bb) and not np.array_equal(ap,bp)
    refs=[dict(lineage=1,tier='E0',frames=[dict(frame='a',mass=1.,target=[[0,.5],[1980,.5]])])]
    ref=F.S.reference_arrays(refs,'E0',['a'],[1],np.ones_like(a,bool))
    pair=F.S.scores(ap,F.S.LABELS,ref);single=F.S.scores(aa,F.S.LABELS,ref)
    assert abs(single['loss'][0]-pair['loss'][0]-math.log(2))<1e-12
    assert single['compatible_mass'][0]==.5 and single['candidate_size'][0]==4
    assert single['candidate_coverage'][0]==1


def test_uniform_unknown_and_point_identity():
    q=np.ones((2,F.N))/F.N;q[1]=0;q[1,-1]=1
    pair,single,*_=F.factorize(q)
    assert np.allclose(pair,q,rtol=0,atol=1e-16)
    assert np.allclose(single,q,rtol=0,atol=1e-16)


@pytest.mark.parametrize('problem',['negative','nan','mass','shape'])
def test_invalid_distribution(problem):
    q=np.ones((1,F.N))/F.N
    if problem=='negative':q[0,0]=-1
    if problem=='nan':q[0,0]=np.nan
    if problem=='mass':q*=2
    if problem=='shape':q=q[0]
    with pytest.raises(ValueError):F.factorize(q)


def test_complete_synthetic_handler(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=F.run(root,plan,lambda **kw:None)
    assert s['rows']==384 and s['native_rows']==12 and s['packets']==2
    assert s['original_cells_reproduced']==128 and s['temporal_max_error']<1e-10
    assert all(s['controls'].values()) and len(s['contrasts'])==16 and len(s['normalized_log_budget_area'])==8
    assert len(s['means'])==48
    for f in (root/'marginals').glob('*.npz'):
        with np.load(f,allow_pickle=False) as z:
            assert set(z.files)=={'steps','goals','operations'}
            assert z['steps'].shape==(1,3,18) and z['goals'].shape==(1,3,3) and z['operations'].shape==(1,3,6)
    assert file_digest(root/'reader/PACKETS.json')==file_digest(root/'inputs/reader/PACKETS.json')


@pytest.mark.parametrize('problem',['score','private','hash','missing','duplicate','reference','temporal-score','temporal-missing','native-temporal'])
def test_corruption_detection(tmp_path,problem):
    root=tmp_path/'packet';plan=fixture(root);base=root/'inputs';p=base/'SUMMARY.json';v=read(p)
    if problem=='private':
        p=base/'reader/PACKETS.json';v=read(p);next(iter(v['packets'].values()))['inputs']['goal']=0
    elif problem=='reference':
        p=base/'evaluator/REFERENCES.json';v=read(p);v[0]['frames'][0]['target'][0][1]=.1
    elif problem.startswith('temporal-'):
        p=base/'temporal/temporal_points.json.gz';v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='temporal-missing':v.pop()
        else:next(r for r in v if r['arm'].endswith('-product'))['loss']+=.1
    elif problem=='native-temporal':
        p=base/'temporal/NATIVE_SCORES.json';v=read(p);v[0]['loss']+=.1
    elif problem=='missing':v['cells'].pop()
    elif problem=='duplicate':v['cells'].append(copy.deepcopy(v['cells'][0]))
    else:v['cells'][0]['loss']+=.1
    if p.suffix=='.gz':p.write_bytes(gzip.compress(canonical(v),mtime=0))
    else:write(p,v,immutable=False)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):F.run(root,plan,lambda **kw:None)


def test_missing_strata_rejected():
    cfg=dict(tiers=['E0'],budgets=[32,128],training_draws=[1],fit_seeds=[2],development_lineages=[3],bootstrap_seed=1,bootstrap_resamples=2)
    with pytest.raises(ValueError):F.aggregate([],cfg)
