"""Independent scalar chain-rule fixtures and complete synthetic diagnostics."""
from itertools import product
import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import joint_uncertainty as U
from ghostscale.validation.soundingline.v18_3.io import write, digest, file_digest


def scalar(q, truth):
    operation={}
    native={}
    for label,probability in enumerate(q):
        operation[label%216]=operation.get(label%216,0.)+probability
    for label,p in truth.items():native[label%216]=native.get(label%216,0.)+p
    joint=-math.fsum(p*math.log(q[k]) for k,p in truth.items())
    op=-math.fsum(p*math.log(operation[k%216]) for k,p in truth.items())
    goal=-math.fsum(p*math.log(q[k]/operation[k%216]) for k,p in truth.items())
    entropy=-math.fsum(p*math.log(p) for p in truth.values())
    op_entropy=-math.fsum(p*math.log(p) for p in native.values())
    goal_entropy=-math.fsum(p*math.log(p/native[k%216]) for k,p in truth.items())
    return dict(loss=joint,operation_loss=op,conditional_goal_loss=goal,joint_entropy=entropy,
                operation_entropy=op_entropy,conditional_goal_entropy=goal_entropy,
                joint_excess=joint-entropy,operation_excess=op-op_entropy,conditional_goal_excess=goal-goal_entropy)


def test_controls():assert all(U.controls().values())


def test_scalar_reconstruction_all_components():
    q=np.arange(1,U.N+1,dtype=float);q/=q.sum()
    truth={0:.1,216:.2,1:.3,5800:.4}
    actual,error=U.components(q,np.array(list(truth)),np.array([list(truth.values())]))
    for k,v in scalar(q,truth).items():assert abs(actual[k][0]-v)<1e-12,k
    assert error<1e-12


def test_swapped_axes_are_detectably_wrong():
    q=np.ones(U.N);q[[0,216]]=100;q/=q.sum()
    actual,_=U.components(q,np.array([0,216]),np.array([[.25,.75]]))
    wrong=q.reshape(216,27).sum(1)
    assert abs(actual['operation_loss'][0]+math.log(wrong[0]))>.05
    assert actual['operation_entropy'][0]==0
    assert actual['conditional_goal_entropy'][0]>0


def test_perfect_native_joint():
    q=np.zeros(U.N);q[[0,1,216]]=[.2,.3,.5]
    actual,_=U.components(q,np.array([0,1,216]),np.array([[.2,.3,.5]]))
    for k in U.COMPONENTS[3:]:assert abs(actual[k][0])<1e-12


def test_unknown_only_truth():
    q=U.distribution(np.array([[32/33]]),np.array([0]),32)[0]
    values,_=U.components(q,np.array([5831]),np.array([[1.]]))
    assert np.isclose(values['loss'][0],math.log(33*5831))
    assert values['joint_entropy'][0]==0


@pytest.mark.parametrize('problem',['negative','nonfinite','mass','zero_truth'])
def test_invalid_forecasts(problem):
    q=np.ones(U.N)/U.N
    if problem=='negative':q[0]=-1
    if problem=='nonfinite':q[0]=np.nan
    if problem=='mass':q*=2
    if problem=='zero_truth':q[1]+=q[0];q[0]=0
    with pytest.raises(ValueError):U.components(q,np.array([0]),np.array([[1.]]))


@pytest.mark.parametrize('alphabet',[[0,0],[0.5],[-1],[5832]])
def test_invalid_alphabets(alphabet):
    with pytest.raises(ValueError):U.distribution(np.ones((1,len(alphabet)))*.5,np.array(alphabet),32)


def fixture(root):
    base=root/'inputs';(base/'forecasts').mkdir(parents=True)
    cfg=dict(tiers=['E0','E2-full'],budgets=[32,128],training_draws=[1,2],fit_seeds=[3,4],development_lineages=[5,6],bootstrap_seed=191004,bootstrap_resamples=10000)
    write(base/'PLAN.json',dict(design=cfg));packets={};refs=[];cells=[]
    for tier in cfg['tiers']:
        a=dict(artifact=[0,0,0])
        if tier=='E2-full':a.update(initial=[0,0,0],requested_purpose=0,observations=[dict(step=i,operation='inspect',before=[0,0,0],after=[0,0,0],tool_proposal=None) for i in range(3)])
        packet=dict(schema='v19.local.public.1',tier=tier,inputs=a);key=digest(packet);packets[key]=packet
        for lin in cfg['development_lineages']:
            truth={172:.6 if lin==5 else .4,388:.4 if lin==5 else .6}
            refs.append(dict(lineage=lin,tier=tier,frames=[dict(frame=key,mass=1.,target=list(truth.items()))]))
        for draw,seed,budget,arm in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets'],U.ARMS):
            stem=f'{draw}-{tier}-{seed}-{budget}-{arm}';p=np.array([[.5,budget/(budget+1)-.5]]);alphabet=np.array([172,388])
            np.savez_compressed(base/'forecasts'/(stem+'.npz'),probabilities=p,alphabet=alphabet)
            write(base/'forecasts'/(stem+'-frames.json'),[key])
            for lin in cfg['development_lineages']:
                truth={172:.6 if lin==5 else .4,388:.4 if lin==5 else .6}
                loss=-sum(v*math.log(p[0,j]) for j,v in enumerate(truth.values()))
                cells.append(dict(tier=tier,budget=budget,arm=arm,lineage=lin,draw=draw,seed=seed,loss=loss,goal_accuracy=.5,operation_accuracy=1.,abstain=True))
    write(base/'reader/PACKETS.json',dict(schema='fixture',packets=packets));write(base/'evaluator/REFERENCES.json',refs);write(base/'SUMMARY.json',dict(cells=cells))
    cfg['input_files']={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}
    return dict(design=cfg)


def test_complete_synthetic_handler(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=U.run(root,plan,lambda **kw:None)
    assert s['rows']==128 and s['packets']==2 and all(s['controls'].values())
    assert len(s['contrasts'])==72 and len(s['normalized_log_budget_area'])==36
    assert all(abs(r['mean'])<1e-12 for r in s['contrasts'])
    assert s['max_chain_error']<1e-12 and s['max_original_loss_error']<1e-12
    assert file_digest(root/'reader/PACKETS.json')==file_digest(root/'inputs/reader/PACKETS.json')


@pytest.mark.parametrize('corruption',['score','private','hash'])
def test_complete_corruption_detection(tmp_path,corruption):
    root=tmp_path/'packet';plan=fixture(root);base=root/'inputs'
    if corruption=='private':
        p=base/'reader/PACKETS.json';d=json.loads(p.read_text());next(iter(d['packets'].values()))['inputs']['truth']=1
    else:
        p=base/'SUMMARY.json';d=json.loads(p.read_text());d['cells'][0]['loss']+=.1
    write(p,d,immutable=False)
    if corruption!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):U.run(root,plan,lambda **kw:None)
