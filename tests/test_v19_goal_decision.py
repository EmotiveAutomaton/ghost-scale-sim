"""Decision objectives differ; controls must not demand the same winner."""
import copy
import gzip
import json
import shutil
import math
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_decision as D, goal_mass_control as M
from ghostscale.validation.soundingline.v18_3.io import read, write, canonical, file_digest
from test_v19_goal_mass_control import fixture as mass_fixture


def fixture(root):
    parent=root/'mass';parent.mkdir(parents=True);plan=mass_fixture(parent)
    M.run(parent,plan,lambda **kw:None)
    base=root/'inputs';base.mkdir()
    for directory in ('reader','evaluator','masses'):shutil.copytree(parent/directory,base/directory)
    shutil.copytree(parent/'inputs/forecasts',base/'forecasts')
    write(base/'parent/PLAN.json',dict(design={k:v for k,v in plan['design'].items() if k!='input_files'}))
    shutil.copyfile(parent/'goal_mass_points.json.gz',base/'parent/goal_mass_points.json.gz')
    plan['design']['bootstrap_seed']=191011
    plan['design']['input_files']={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}
    return plan


def test_known_objectives_and_controls():
    assert all(D.controls().values())
    q=np.zeros((1,27));q[0,[0,12,10,4]]=[.30,.26,.26,.18]
    d=D.decisions(q);assert d['joint'][0]==0 and d['coordinate'][0]==9
    assert q[0,d['coordinate'][0]]==0
    j=D.score(q,'joint',q[None],np.ones((1,1)));c=D.score(q,'coordinate',q[None],np.ones((1,1)))
    assert c['goal_accuracy'][0]>j['goal_accuracy'][0] and c['path_accuracy'][0]<j['path_accuracy'][0]
    assert c['top_incompatible'][0]==1 and j['top_incompatible'][0]==0
    assert c['loss'][0]==j['loss'][0]


def test_independent_scalar_optima_scores_and_ties():
    q=np.array([[k+1 for k in range(27)],[1]*27],float);q/=q.sum(1)[:,None]
    goals=list(product(range(3),repeat=3)); d=D.decisions(q)
    target=np.array([[k+3 for k in range(27)],[28-k for k in range(27)]],float);target/=target.sum(1)[:,None]
    mass=np.array([[.2,.8]])
    for decision in ('joint','coordinate'):
        selected=[]
        for row in q:
            if decision=='joint':choice=max(range(27),key=lambda k:row[k])
            else:
                m=[[math.fsum(row[k] for k,g in enumerate(goals) if g[t]==v) for v in range(3)] for t in range(3)]
                choice=goals.index(tuple(max(range(3),key=lambda v:m[t][v]) for t in range(3)))
            selected.append(choice)
        assert np.array_equal(d[decision],selected)
        s=D.score(q,decision,target[None],mass)
        accuracy=math.fsum(mass[0,i]*math.fsum(target[i,k]*sum(a==b for a,b in zip(goals[k],goals[selected[i]]))/3 for k in range(27)) for i in range(2))
        assert abs(s['goal_accuracy'][0]-accuracy)<1e-14
        assert abs(s['path_accuracy'][0]-sum(mass[0,i]*target[i,selected[i]] for i in range(2)))<1e-14
    assert d['joint_ties'][1]==27 and np.array_equal(d['coordinate_ties'][1],[3,3,3])


def test_equal_marginals_distinct_joint_choices():
    q=np.zeros((2,27));q[0,[0,8]]=.5;q[1,[2,6]]=.5
    d=D.decisions(q);assert np.array_equal(d['marginals'][0],d['marginals'][1])
    assert d['joint'][0]!=d['joint'][1] and d['coordinate'][0]==d['coordinate'][1]


@pytest.mark.parametrize('problem',['shape','negative','nan','mass'])
def test_invalid_forecast(problem):
    q=np.ones((1,27))/27
    if problem=='shape':q=q[:,:26]
    elif problem=='negative':q[0,0]=-1
    elif problem=='nan':q[0,0]=np.nan
    else:q*=2
    with pytest.raises(ValueError):D.decisions(q)


def test_complete_handler_and_prediction_identity(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=D.run(root,plan,lambda **kw:None)
    assert all(s['controls'].values()) and s['rows']==256 and s['parent_loss_cells']==128 and s['native_rows']==4
    assert len(s['contrasts'])==96 and len(s['normalized_log_budget_area'])==48 and len(s['means'])==32
    assert s['parent_max_error']<1e-10
    assert all(r['mean']==0 for r in s['contrasts'] if r['metric']=='loss')
    for r in s['contrasts']:
        if r['metric']=='forecast_hamming_loss':assert r['mean']<=1e-12
        if r['metric']=='forecast_path_loss':assert r['mean']>=-1e-12


@pytest.mark.parametrize('problem',['loss','missing','duplicate','hash','reader','witness','reference','mass','population','forecast-order'])
def test_corruptions(tmp_path,problem):
    root=tmp_path/'packet';plan=fixture(root);base=root/'inputs'
    if problem in ('loss','missing','duplicate','hash'):
        p=base/'parent/goal_mass_points.json.gz';v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        else:v[0]['loss']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    elif problem in ('reader','witness'):
        p=base/'reader/PACKETS.json';v=read(p);packet=next(iter(v['packets'].values()))
        if problem=='reader':packet['inputs']['goal']=0
        else:packet['inputs']['observations'][0]['operation']='undo'
        write(p,v,immutable=False)
    elif problem in ('reference','mass'):
        p=base/'evaluator/REFERENCES.json';v=read(p)
        if problem=='reference':v[0]['frames'][0]['target'][0][1]=.5
        else:v[0]['frames'][0]['mass']=2
        write(p,v,immutable=False)
    elif problem=='population':
        p=base/'parent/PLAN.json';v=read(p);v['design']['fit_seeds']=[-1];write(p,v,immutable=False)
    else:
        p=next((base/'forecasts').glob('*-E2-full-*-frames.json'));write(p,[],immutable=False)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):D.run(root,plan,lambda **kw:None)
