"""Abstention objectives, zero denominators, support and input corruption."""
import copy
import gzip
import json
import math
import shutil
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_abstention as A, goal_decision as D
from ghostscale.validation.soundingline.v18_3.io import read, write, canonical, file_digest
from test_v19_goal_decision import fixture as decision_fixture


def fixture(root):
    parent = root/'decision'; parent.mkdir(parents=True)
    plan = decision_fixture(parent); D.run(parent, plan, lambda **kw: None)
    base = root/'inputs'; base.mkdir()
    for directory in ('reader', 'decisions'):
        shutil.copytree(parent/directory, base/directory)
    write(base/'evaluator/REFERENCES.json', read(parent/'evaluator/REFERENCES.json'))
    write(base/'parent/PLAN.json', dict(design={k:v for k,v in plan['design'].items() if k != 'input_files'}))
    shutil.copyfile(parent/'goal_decision_points.json.gz', base/'parent/goal_decision_points.json.gz')
    plan['design'].update(costs=list(A.COSTS), bootstrap_seed=191012)
    plan['design']['input_files'] = {p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}
    return plan


def scalar_cost(row, reported, cost):
    goals = list(product(range(3), repeat=3))
    return math.fsum(float(row[k])*math.fsum(cost if value < 0 else float(value != goals[k][t])
        for t,value in enumerate(reported))/3 for k in range(27))


def test_controls_and_exhaustive_partial_reports():
    assert all(A.controls().values())
    q = np.array([[k+1 for k in range(27)], [27-k for k in range(27)]], float)
    q /= q.sum(1)[:,None]
    possible = list(product((-1,0,1,2), repeat=3))
    for cost in A.COSTS:
        r = A.reports(q,cost)
        for i,row in enumerate(q):
            best = min(scalar_cost(row,x,cost) for x in possible)
            assert abs(scalar_cost(row,r['step-abstention'][i],cost)-best) < 1e-12
            joint = tuple(A.GOALS[max(range(27),key=lambda k:row[k])])
            assert abs(scalar_cost(row,r['joint-all-or-none'][i],cost)-min(cost,scalar_cost(row,joint,cost))) < 1e-12


def test_boundary_equality_and_near_equality():
    q = np.zeros((3,27));q[:,0]=[.75,np.nextafter(.75,1),np.nextafter(.75,0)];q[:,13]=1-q[:,0]
    r = A.reports(q,.25)
    for policy in A.POLICIES[1:]:
        assert np.all(r[policy][0] == -1)
        assert np.all(r[policy][1] == 0)
        assert np.all(r[policy][2] == -1)


def test_joint_path_probability_is_not_step_risk():
    # Independent 80%-likely zeros: modal path .512, but mean step error .2.
    goals = list(product(range(3),repeat=3))
    q = np.array([[math.prod(.8 if v==0 else .2 if v==1 else 0 for v in g) for g in goals]])
    assert q.max() < .75
    assert np.all(A.reports(q,.25)['joint-all-or-none'] == 0)


def test_partial_support_and_scalar_native_cost():
    q = np.full((2,27),1/27);truth=np.zeros((1,2,27));truth[:,:,0]=1;mass=np.array([[.2,.8]])
    report=np.array([[1,-1,-1],[-1,-1,-1]])
    s=A.score(q,report,truth,mass,.25)
    assert abs(s['coverage'][0]-.2/3)<1e-14
    assert s['correct'][0]==0 and s['incompatible'][0]==.2 and s['full_incompatible'][0]==0
    expected=sum(mass[0,i]*scalar_cost(truth[0,i],report[i],.25) for i in range(2))
    assert abs(s['cost_value'][0]-expected)<1e-14
    expected=sum(mass[0,i]*scalar_cost(q[i],report[i],.25) for i in range(2))
    assert abs(s['forecast_cost'][0]-expected)<1e-14


def test_exact_modal_ties_and_all_none_partial():
    q=np.zeros((1,27));q[0,[0,13]]=.5
    assert np.all(A.reports(q,.5)['always-coordinate']==0)
    assert np.all(A.reports(q,.5)['step-abstention']==-1)
    q[:]=0;q[0,0]=1
    assert all(np.all(v==0) for v in A.reports(q,.1).values())
    q[:]=0;q[0,[0,1,2]]=1/3
    assert np.array_equal(A.reports(q,.1)['step-abstention'],[[0,0,-1]])


@pytest.mark.parametrize('problem',['shape','negative','nan','normalization','cost'])
def test_invalid_forecast(problem):
    q=np.ones((1,27))/27;cost=.1
    if problem=='shape':q=q[:,:26]
    elif problem=='negative':q[0,0]=-1
    elif problem=='nan':q[0,0]=np.nan
    elif problem=='normalization':q*=2
    else:cost=.2
    with pytest.raises(ValueError):A.reports(q,cost)


def test_complete_handler_and_zero_coverage(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=A.run(root,plan,lambda **kw:None)
    assert s['rows']==1152 and s['native_rows']==18 and s['parent_cells']==128
    assert len(s['contrasts'])==768 and len(s['normalized_log_budget_area'])==384 and len(s['means'])==144
    assert all(r['mean']==0 for r in s['contrasts'] if r['metric']=='loss')
    rows=json.loads(gzip.decompress((root/'goal_abstention_points.json.gz').read_bytes()))
    assert all(r['conditional_error'] is None for r in rows if r['coverage']==0)
    assert all(abs(r['coverage']-r['incorrect']-r['correct'])<1e-12 for r in rows)
    assert all(r['mean']<=1e-12 for r in s['contrasts'] if r['policy']=='step-abstention' and r['metric']=='forecast_cost')
    # Force a valid all-abstaining synthetic score roster to exercise pooled nulls.
    changed=copy.deepcopy(rows)
    for r in changed:
        if r['policy']!='always-coordinate':
            r.update(coverage=0.,correct=0.,incorrect=0.,conditional_error=None,cost_value=r['cost'],forecast_cost=r['cost'])
    group=A.aggregate(changed,plan['design'])
    assert all(r['conditional_error'] is None for r in group['means'] if r['policy']!='always-coordinate')


@pytest.mark.parametrize('problem',['hash','missing','duplicate','parent-score','reader','witness','reference','mass','population','cost','probabilities','decision','operations'])
def test_corruptions(tmp_path,problem):
    root=tmp_path/'packet';plan=fixture(root);base=root/'inputs'
    if problem in ('hash','missing','duplicate','parent-score'):
        p=base/'parent/goal_decision_points.json.gz';v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        else:
            for row in v:row['loss']+=.1
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
    elif problem=='cost':
        plan['design']['costs']=[.2];p=None
    else:
        p=next((base/'decisions').glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        if problem=='probabilities':v['probabilities']*=2
        elif problem=='decision':v['coordinate'][0]=(v['coordinate'][0]+1)%27
        else:v['operations'][0]=(v['operations'][0]+1)%216
        np.savez_compressed(p,**v)
    if problem!='hash' and p is not None:plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):A.run(root,plan,lambda **kw:None)
