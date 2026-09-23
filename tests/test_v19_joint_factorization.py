"""Independent known-answer dependence controls and complete synthetic execution."""
from itertools import product
import copy
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import joint_factorization as F, joint_review as V
from ghostscale.validation.soundingline.v18_3.io import read, write, digest, file_digest


def test_known_controls(): assert all(F.controls().values())


def test_scalar_outer_product_and_axis_order():
    q=np.arange(1,F.N+1,dtype=float);q/=q.sum()
    result,goals,ops=F.factorize(q[None,:])
    g=[math.fsum(q[k] for k in range(i*216,(i+1)*216)) for i in range(27)]
    o=[math.fsum(q[k] for k in range(j,F.N,216)) for j in range(216)]
    expected=[g[k//216]*o[k%216] for k in range(F.N)]
    assert np.allclose(result[0],expected,rtol=0,atol=1e-16)
    assert np.allclose(goals[0],g,rtol=0,atol=1e-15) and np.allclose(ops[0],o,rtol=0,atol=1e-15)
    assert np.max(abs(result[0]-(q.reshape(216,27).sum(1)[:,None]*q.reshape(216,27).sum(0)[None,:]).ravel()))>1e-6


def test_same_marginals_different_dependence():
    a=np.zeros((1,F.N));b=a.copy();a[0,[0,217]]=.5;b[0,[1,216]]=.5
    p,_,_=F.factorize(a);q,_,_=F.factorize(b)
    assert np.array_equal(p,q) and not np.array_equal(a,b)
    assert np.isclose(-math.fsum(.5*math.log(p[0,k]) for k in (0,217)),math.log(4))


def test_full_unknown_and_independent_identity():
    q=np.ones((2,F.N))/F.N
    assert np.allclose(F.factorize(q)[0],q,rtol=0,atol=1e-16)
    q[1]=0;q[1,5831]=1
    assert np.array_equal(F.factorize(q)[0][1],q[1])


@pytest.mark.parametrize('problem',['negative','nan','mass','shape'])
def test_invalid_forecasts(problem):
    q=np.ones((1,F.N))/F.N
    if problem=='negative':q[0,0]=-1
    if problem=='nan':q[0,0]=np.nan
    if problem=='mass':q*=2
    if problem=='shape':q=q[0]
    with pytest.raises(ValueError): F.factorize(q)


def test_native_product_compatibility_and_loss():
    refs=[dict(lineage=1,tier='E0',frames=[dict(frame='a',mass=1.,target=[[0,.5],[217,.5]])])]
    ref=F.S.reference_arrays(refs,'E0',['a'],[1],np.ones((1,F.N),bool))
    q=np.zeros((1,F.N));q[0,[0,217]]=.5
    joint=F.S.scores(q,F.S.LABELS,ref);factor=F.S.scores(F.factorize(q)[0],F.S.LABELS,ref)
    assert joint['compatible_mass'][0]==1 and factor['compatible_mass'][0]==.5
    assert np.isclose(joint['loss'][0],math.log(2)) and np.isclose(factor['loss'][0],math.log(4))
    assert factor['candidate_size'][0]==4 and factor['candidate_coverage'][0]==1


def fixture(root):
    base=root/'inputs';(base/'forecasts').mkdir(parents=True)
    cfg=dict(tiers=['E0','E2-full'],budgets=[32,128],training_draws=[1,2],fit_seeds=[3,4],development_lineages=[5,6],bootstrap_seed=191005,bootstrap_resamples=10000)
    write(base/'PLAN.json',dict(design=cfg));packets={};refs=[];cells=[]
    for tier in cfg['tiers']:
        a=dict(artifact=[0,0,0])
        if tier=='E2-full':a.update(initial=[0,0,0],requested_purpose=0,observations=[dict(step=i,operation='inspect',before=[0,0,0],after=[0,0,0],tool_proposal=None) for i in range(3)])
        packet=dict(schema='v19.local.public.1',tier=tier,inputs=a);key=digest(packet);packets[key]=packet
        for lin in cfg['development_lineages']:
            truth={172:.6 if lin==5 else .4,388:.4 if lin==5 else .6}
            refs.append(dict(lineage=lin,tier=tier,frames=[dict(frame=key,mass=1.,target=list(truth.items()))]))
        for draw,seed,budget,arm in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],F.ARMS):
            stem=f'{draw}-{tier}-{seed}-{budget}-{arm}';p=np.array([[.5,budget/(budget+1)-.5]]);alphabet=np.array([172,388])
            np.savez_compressed(base/'forecasts'/(stem+'.npz'),probabilities=p,alphabet=alphabet);write(base/'forecasts'/(stem+'-frames.json'),[key])
            for lin in cfg['development_lineages']:
                truth={172:.6 if lin==5 else .4,388:.4 if lin==5 else .6}
                cells.append(dict(tier=tier,budget=budget,arm=arm,lineage=lin,draw=draw,seed=seed,frames=1,
                                  **{k:float(v) for k,v in V.score(p[0],alphabet,truth,budget).items()}))
    write(base/'reader/PACKETS.json',dict(schema='fixture',packets=packets));write(base/'evaluator/REFERENCES.json',refs);write(base/'SUMMARY.json',dict(cells=cells))
    cfg['input_files']={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}
    return dict(design=cfg)


def test_complete_synthetic_handler(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=F.run(root,plan,lambda **kw:None)
    assert s['rows']==256 and s['packets']==2 and s['native_rows']==8 and all(s['controls'].values())
    assert s['original_cells_reproduced']==128 and s['original_max_error']<1e-10
    assert len(s['contrasts'])==16 and len(s['normalized_log_budget_area'])==8
    native=read(root/'evaluator/NATIVE_SCORES.json')
    for tier,lin in product(plan['design']['tiers'],plan['design']['development_lineages']):
        a,b=[r for r in native if r['tier']==tier and r['lineage']==lin]
        assert abs(a['loss']-b['loss'])<1e-12  # one operation: goal/operation independent
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


def test_regroup_missing_strata_rejected():
    cfg=dict(tiers=['E0'],budgets=[32,128],training_draws=[1],fit_seeds=[2],development_lineages=[3],bootstrap_seed=1,bootstrap_resamples=2)
    with pytest.raises(ValueError):F.aggregate([],cfg)
