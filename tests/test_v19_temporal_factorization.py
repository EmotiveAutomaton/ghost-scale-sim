"""Known answers for between-step dependence and full frozen readout execution."""
from itertools import product
import copy
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import temporal_factorization as F
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from test_v19_joint_factorization import fixture


def scalar(q):
    memberships=[]
    for k in range(5832):
        goals=k//216;ops=k%216
        memberships.append([(goals//9)*6+ops//36,((goals//3)%3)*6+(ops//6)%6,(goals%3)*6+ops%6])
    marg=[[math.fsum(q[k] for k in range(5832) if memberships[k][t]==v) for v in range(18)] for t in range(3)]
    return np.array([math.prod(marg[t][memberships[k][t]] for t in range(3)) for k in range(5832)]),np.array(marg)


def test_known_controls():assert all(F.controls().values())


def test_scalar_marginals_axis_order_and_full_products():
    q=np.arange(1,5833,dtype=float);q/=q.sum()
    p,m=F.factorize(q[None]);expected,em=scalar(q)
    assert np.allclose(p[0],expected,rtol=0,atol=1e-17)
    assert np.allclose(m[0],em,rtol=0,atol=1e-15)
    # Opposite time order must not reconstruct the same nonuniform labels.
    wrong=np.prod(m[0,np.arange(3)[None,:],F.PAIRS[:,::-1]],axis=1)
    assert max(abs(p[0]-wrong))>1e-6


def test_same_step_marginals_different_paths():
    a=np.zeros((1,F.N));b=a.copy();a[0,[0,2634]]=.5;b[0,[1980,654]]=.5
    p,m=F.factorize(a);q,n=F.factorize(b)
    assert np.array_equal(m,n) and np.array_equal(p,q) and not np.array_equal(a,b)
    assert np.count_nonzero(p)==4


def test_full_unknown_and_point_mass():
    q=np.ones((2,F.N))/F.N;q[1]=0;q[1,-1]=1
    assert np.allclose(F.factorize(q)[0],q,rtol=0,atol=1e-16)


@pytest.mark.parametrize('problem',['negative','nan','mass','shape'])
def test_invalid_distribution(problem):
    q=np.ones((1,F.N))/F.N
    if problem=='negative':q[0,0]=-1
    if problem=='nan':q[0,0]=np.nan
    if problem=='mass':q*=2
    if problem=='shape':q=q[0]
    with pytest.raises(ValueError):F.factorize(q)


def test_native_incompatible_cross_paths():
    refs=[dict(lineage=1,tier='E0',frames=[dict(frame='a',mass=1.,target=[[0,.5],[2634,.5]])])]
    ref=F.S.reference_arrays(refs,'E0',['a'],[1],np.ones((1,F.N),bool))
    q=np.zeros((1,F.N));q[0,[0,2634]]=.5
    joint=F.S.scores(q,F.S.LABELS,ref);fact=F.S.scores(F.factorize(q)[0],F.S.LABELS,ref)
    assert fact['compatible_mass'][0]==.5 and joint['compatible_mass'][0]==1
    assert np.isclose(fact['loss'][0]-joint['loss'][0],math.log(2))
    assert fact['candidate_size'][0]==4 and fact['candidate_coverage'][0]==1


def test_complete_synthetic_handler(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=F.run(root,plan,lambda **kw:None)
    assert s['rows']==256 and s['packets']==2 and s['native_rows']==8 and all(s['controls'].values())
    assert s['original_cells_reproduced']==128 and s['original_max_error']<1e-10
    assert len(s['contrasts'])==16 and len(s['normalized_log_budget_area'])==8
    # Fixture native labels vary only in the final goal; time factorization is identity.
    native=read(root/'evaluator/NATIVE_SCORES.json')
    for tier,lin in product(plan['design']['tiers'],plan['design']['development_lineages']):
        a,b=[r for r in native if r['tier']==tier and r['lineage']==lin]
        assert abs(a['loss']-b['loss'])<1e-12
    for f in (root/'marginals').glob('*.npz'):
        with np.load(f,allow_pickle=False) as z:assert set(z.files)=={'steps'} and z['steps'].shape==(1,3,18)
    assert file_digest(root/'reader/PACKETS.json')==file_digest(root/'inputs/reader/PACKETS.json')


@pytest.mark.parametrize('problem',['score','private','hash','missing','duplicate','reference'])
def test_corruption_detection(tmp_path,problem):
    root=tmp_path/'packet';plan=fixture(root);base=root/'inputs';p=base/'SUMMARY.json';v=read(p)
    if problem=='private':
        p=base/'reader/PACKETS.json';v=read(p);next(iter(v['packets'].values()))['inputs']['goal']=0
    elif problem=='reference':
        p=base/'evaluator/REFERENCES.json';v=read(p);v[0]['frames'][0]['target'][0][1]=.1
    elif problem=='missing':v['cells'].pop()
    elif problem=='duplicate':v['cells'].append(copy.deepcopy(v['cells'][0]))
    else:v['cells'][0]['loss']+=.1
    write(p,v,immutable=False)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):F.run(root,plan,lambda **kw:None)


def test_missing_strata_rejected():
    cfg=dict(tiers=['E0'],budgets=[32,128],training_draws=[1],fit_seeds=[2],development_lineages=[3],bootstrap_seed=1,bootstrap_resamples=2)
    with pytest.raises(ValueError):F.aggregate([],cfg)
