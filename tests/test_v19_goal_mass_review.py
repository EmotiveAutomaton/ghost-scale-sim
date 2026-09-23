"""Independent probability and full-record corruption controls."""
from pathlib import Path
import ast
import copy
import gzip
import json
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_mass_review as R
from ghostscale.validation.soundingline.v18_3.io import read, write, canonical, file_digest


def test_known_answers_and_empty_groups():
    assert all(R.controls().values())
    q=np.zeros(R.N);q[0]=1.
    for a in (np.array([],dtype=int),np.array([0]),np.arange(R.N)):
        assert np.array_equal(R.group_match(q,q,a)[0],q)
    f=q.copy();f[0]=.5;f[1]=.5
    with pytest.raises(ValueError):R.group_match(q,f,np.array([0]))


def test_scalar_log_identity_and_conditional_odds():
    q=np.zeros(R.N);q[:4]=[.1,.2,.3,.4]
    f=np.zeros(R.N);f[:4]=[.3,.3,.2,.2];a=np.array([0,1])
    r,b,e,_=R.group_match(q,f,a)
    ref=(np.array([0,3]),np.array([[.6,.4],[.2,.8]]),np.array([.3,.7]))
    direct=[mass*sum(-w*math.log(r[k]/q[k]) for k,w in zip(ref[0],weights)) for weights,mass in zip(ref[1],ref[2])]
    assert np.allclose(R.loss_shift(ref,a,b,e),direct,rtol=0,atol=1e-15)
    assert r[0]/r[1]==q[0]/q[1] and abs(r[2]/r[3]-.75)<1e-15


def test_tiny_group_saved_mass_cannot_hide_normalized_error():
    q=np.zeros(R.N);q[:2]=[1e-14,1-1e-14]
    f=np.zeros(R.N);f[:2]=[.5,.5];a=np.array([0])
    _,b,e,_=R.group_match(q,f,a)
    with pytest.raises(ValueError):R.group_match(q,f,a,b*np.array([2.,1.]),e)
    with pytest.raises(ValueError):R.group_match(q,f,a,[0,1],e)


@pytest.mark.parametrize('alphabet',[[.5],[0,0],[-1],[5832]])
def test_invalid_alphabet(alphabet):
    q=np.ones(R.N)/R.N
    with pytest.raises(ValueError):R.group_match(q,q,np.array(alphabet))


@pytest.mark.parametrize('bad',['shape','negative','nan','mass','saved_shape'])
def test_invalid_probability(bad):
    q=np.ones(R.N)/R.N;f=q.copy();a=np.array([0])
    if bad=='shape':q=q[None,:]
    elif bad=='negative':q[0]=-1
    elif bad=='nan':q[0]=np.nan
    elif bad=='mass':q*=2
    with pytest.raises(ValueError):
        R.group_match(q,f,a,[.5] if bad=='saved_shape' else None,[.5] if bad=='saved_shape' else None)


def test_verified_ties_preserve_alphabet_priority():
    q=np.zeros(R.N);q[:2]=.5
    r,_,_,_=R.group_match(q,q,np.array([1,0]))
    ref=(np.array([1]),np.ones((1,1)),np.ones(1))
    assert R.score(r,R.priority_for([1,0]),ref)['top_incompatible'][0]==0
    assert R.score(r,R.priority_for([0,1]),ref)['top_incompatible'][0]==1


def fixture(root):
    from test_v19_goal_mass_control import fixture as parent_fixture
    from ghostscale.validation.soundingline.v19 import goal_mass_control as F
    producer=root/'producer';plan=parent_fixture(producer)
    write(producer/'PLAN.json',plan);write(producer/'SUMMARY.json',F.run(producer,plan,lambda **kw:None))
    target=root/'checker';base=target/'inputs';shutil.copytree(producer/'inputs',base/'parent')
    original=base/'original';original.mkdir()
    for n in ('reader','evaluator','masses'):shutil.copytree(producer/n,original/n)
    for n in ('PLAN.json','SUMMARY.json','goal_mass_points.json.gz'):shutil.copyfile(producer/n,original/n)
    return target,dict(design=dict(target_plan_sha256=file_digest(original/'PLAN.json'),input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}))


def test_full_execution_and_separate_regroup(tmp_path):
    root,plan=fixture(tmp_path);s=R.run(root,plan,lambda **kw:None)
    assert (s['rows'],s['frame_forecasts'],s['packets'],s['parent_cells_reproduced'],s['inherited_native_rows'])==(192,32,1,128,6)
    assert all(s[k]<1e-10 for k in s if k.startswith('max_')) and all(s['controls'].values())
    g=read(root/'INDEPENDENT_REGROUP.json');o=read(root/'ORIGINAL_ROW_REGROUP.json')
    assert (len(g['contrasts']),len(g['normalized_log_budget_area']),len(g['means']))==(16,8,24)
    assert R.compare_groups(g,o)<1e-10


@pytest.mark.parametrize('problem',['loss','localization','coverage','unknown','missing','duplicate','reference','private','hash','summary',
    'goals','normalizer','operations','original_mass','product_mass','native_score','native_duplicate','parent_score','parent_missing','parent_population'])
def test_corruption_rejected(tmp_path,problem):
    root,plan=fixture(tmp_path);base=root/'inputs';original=base/'original'
    if problem in ('loss','localization','coverage','unknown','missing','duplicate','parent_score','parent_missing'):
        path=base/'parent/witnessed/witnessed_goal_points.json.gz' if problem.startswith('parent_') else original/'goal_mass_points.json.gz'
        rows=json.loads(gzip.decompress(path.read_bytes()))
        if problem in ('missing','parent_missing'):rows.pop()
        elif problem=='duplicate':rows[-1]=copy.deepcopy(rows[0])
        else:
            row=next(r for r in rows if r['arm'].endswith('-restricted'))
            row[{'loss':'loss','localization':'goal_0','coverage':'candidate_coverage','unknown':'unknown_mass','parent_score':'loss'}[problem]]+=.1
        path.write_bytes(gzip.compress(canonical(rows),mtime=0))
    elif problem in ('goals','normalizer','operations','original_mass','product_mass'):
        path=next((original/'masses').glob('*.npz'))
        with np.load(path) as z:d={k:z[k] for k in z.files}
        key={'original_mass':'original','product_mass':'product'}.get(problem,problem)
        d[key].flat[0]+=.1 if problem!='operations' else 1
        np.savez_compressed(path,**d)
    elif problem.startswith('native_'):
        path=original/'evaluator/INHERITED_NATIVE_SCORES.json';d=read(path)
        if problem=='native_duplicate':d[-1]=copy.deepcopy(d[0])
        else:d[0]['loss']+=.1
        write(path,d,immutable=False)
    elif problem=='parent_population':
        path=base/'parent/witnessed/PLAN.json';d=read(path);d['design']['fit_seeds']=[999];write(path,d,immutable=False)
    elif problem=='reference':
        path=original/'evaluator/REFERENCES.json';d=read(path);d[0]['frames'][0]['mass']=.5;write(path,d,immutable=False)
    elif problem=='private':
        path=original/'reader/PACKETS.json';d=read(path);next(iter(d['packets'].values()))['inputs']['truth']=1;write(path,d,immutable=False)
    else:
        path=original/'SUMMARY.json';d=read(path);d['contrasts'][0]['mean']+=.1;write(path,d,immutable=False)
    if problem!='hash':plan['design']['input_files'][path.relative_to(base).as_posix()]=file_digest(path)
    with pytest.raises(ValueError):R.run(root,plan,lambda **kw:None)


def test_no_producer_imports():
    tree=ast.parse(Path(R.__file__).read_text())
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(n.endswith(('goal_mass_control','temporal_mass_control','joint_support','witnessed_goal_factorization','joint_uncertainty')) for n in imports)
