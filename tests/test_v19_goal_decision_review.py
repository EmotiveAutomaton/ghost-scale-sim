"""Independent executed ties, decision objectives and hostile record checks."""
import copy
import gzip
import json
import math
import shutil
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_decision_review as V, goal_decision as D
from ghostscale.validation.soundingline.v18_3.io import read, write, canonical, file_digest
from test_v19_goal_decision import fixture as decision_fixture


def fixture(root):
    original = root/'producer'; plan = decision_fixture(original)
    write(original/'PLAN.json', plan)
    write(original/'SUMMARY.json', D.run(original, plan, lambda **kw: None))
    base = root/'inputs'; base.mkdir()
    shutil.copytree(original/'inputs', base/'parent')
    dest = base/'original'; dest.mkdir()
    for n in ('PLAN.json', 'SUMMARY.json', 'goal_decision_points.json.gz'):
        shutil.copyfile(original/n, dest/n)
    for n in ('reader', 'evaluator', 'decisions'): shutil.copytree(original/n, dest/n)
    cfg = dict(target_plan_sha256=file_digest(dest/'PLAN.json'),
               input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()})
    return dict(design=cfg)


def test_known_answers_and_complete_reconstruction(tmp_path):
    assert all(V.controls().values())
    root = tmp_path/'case'; plan = fixture(root); result = V.run(root, plan, lambda **kw: None)
    assert result['passed'] and result['rows'] == 256 and result['native_rows'] == 4
    assert result['parent_loss_cells'] == 128
    assert max(v for k,v in result.items() if k.startswith('max_')) < 1e-10
    assert len(read(root/'INDEPENDENT_REGROUP.json')['contrasts']) == 96


def test_scalar_marginals_risks_and_equal_marginal_choices():
    a = np.zeros(27); a[[0,8]] = .5
    b = np.zeros(27); b[[2,6]] = .5
    da,_ = V.decide(a); db,_ = V.decide(b)
    assert np.array_equal(da['marginals'],db['marginals'])
    assert da['joint'] != db['joint'] and da['coordinate'] == db['coordinate']
    q = np.arange(1,28,dtype=float);q/=q.sum();d,_ = V.decide(q)
    for chosen,g in enumerate(V.GOALS):
        risk = math.fsum(q[k]*sum(x!=y for x,y in zip(g,h))/3 for k,h in enumerate(V.GOALS))
        assert abs(risk-d['hamming_risks'][chosen]) < 1e-14


def test_exact_ties_and_near_ties_are_distinct():
    q=np.zeros(27);q[[0,9]]=.5
    d,_=V.decide(q);assert d['joint']==d['coordinate']==0
    q[0]-=1e-15;q[9]+=1e-15
    d,_=V.decide(q);assert d['joint']==d['coordinate']==9 and d['joint_ties']==1
    # A checked saved marginal represents the producer's actual binary64 sum.
    exact=np.zeros(27);exact[[0,9]]=.5
    saved,_=V.decide(exact);saved=copy.deepcopy(saved)
    saved['marginals'][0,0]-=1e-16;saved['marginals'][0,1]+=1e-16
    saved['coordinate']=9;saved['coordinate_ties'][0]=1
    saved['coordinate_margins'][0]=saved['marginals'][0,1]-saved['marginals'][0,0]
    observed,_=V.decide(exact,saved);assert observed['coordinate']==9


@pytest.mark.parametrize('problem',['shape','negative','nan','normalization'])
def test_invalid_probability(problem):
    q=np.ones(27)/27
    if problem=='shape':q=q[:26]
    elif problem=='negative':q[0]=-1
    elif problem=='nan':q[0]=np.nan
    else:q*=2
    with pytest.raises(ValueError):V.decide(q)


@pytest.mark.parametrize('field', sorted(V.DECISION_FIELDS))
def test_corrupt_saved_decision(field):
    q=np.arange(1,28,dtype=float);q/=q.sum();d,_=V.decide(q);d=copy.deepcopy(d)
    d[field]=np.asarray(d[field])+.1
    with pytest.raises(ValueError):V.decide(q,d)


@pytest.mark.parametrize('problem',['probability','choice','native','score','missing','duplicate','hash','reader',
                                   'reference','witness','summary','population','parent-loss','forecast-order'])
def test_record_corruptions(tmp_path,problem):
    root=tmp_path/'case';plan=fixture(root);base=root/'inputs'
    if problem in ('probability','choice','native'):
        folder=base/'original'/('evaluator' if problem=='native' else 'decisions')
        p=next(folder.glob('*.npz'))
        with np.load(p) as z:v={k:z[k] for k in z.files}
        if problem=='choice':v['joint'][0]=(v['joint'][0]+1)%27
        else:v['probabilities'][0,0]+=.1
        np.savez_compressed(p,**v)
    elif problem in ('score','missing','duplicate','hash','parent-loss'):
        p=base/('parent/parent/goal_mass_points.json.gz' if problem=='parent-loss' else 'original/goal_decision_points.json.gz')
        v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        else:v[0]['loss']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    elif problem in ('reader','witness'):
        p=base/'original/reader/PACKETS.json';v=read(p);packet=next(iter(v['packets'].values()))
        if problem=='reader':packet['inputs']['goal']=0
        else:packet['inputs']['observations'][0]['operation']='undo'
        write(p,v,immutable=False)
    elif problem=='reference':
        p=base/'original/evaluator/REFERENCES.json';v=read(p);v[0]['frames'][0]['mass']=2;write(p,v,immutable=False)
    elif problem=='summary':
        p=base/'original/SUMMARY.json';v=read(p);v['contrasts'][0]['mean']+=.1;write(p,v,immutable=False)
    elif problem=='population':
        p=base/'parent/parent/PLAN.json';v=read(p);v['design']['fit_seeds']=[-1];write(p,v,immutable=False)
    else:
        p=next((base/'parent/forecasts').glob('*-E2-full-*-frames.json'));write(p,[],immutable=False)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):V.run(root,plan,lambda **kw:None)
