"""Known coarse truth, specificity, noncommuting decisions and corrupt inputs."""
import copy
import gzip
import json
import math
import shutil
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_coarsening as C, goal_abstention as A
from ghostscale.validation.soundingline.v18_3.io import read, write, canonical, file_digest
from test_v19_goal_abstention import fixture as abstention_fixture


def fixture(root):
    parent=root/'abstention';parent.mkdir(parents=True);plan=abstention_fixture(parent)
    A.run(parent,plan,lambda **kw:None)
    base=root/'inputs';base.mkdir()
    for name in ('reader','reports'):shutil.copytree(parent/name,base/name)
    write(base/'evaluator/REFERENCES.json',read(parent/'evaluator/REFERENCES.json'))
    write(base/'parent/PLAN.json',dict(design={k:v for k,v in plan['design'].items() if k!='input_files'}))
    shutil.copyfile(parent/'goal_abstention_points.json.gz',base/'parent/goal_abstention_points.json.gz')
    plan['design'].update(partitions={k:list(v) for k,v in C.PARTITIONS.items()},bootstrap_seed=191013)
    plan['design']['input_files']={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}
    return plan


def test_all_partitions_mass_scalar_projection():
    assert all(C.controls().values())
    q=np.array([[k+1 for k in range(27)]],float);q/=q.sum()
    for name,mapping in C.PARTITIONS.items():
        g=C.geometry(name);r=C.forecasts(q,name);expected=np.zeros(g['n']**3)
        for k,goal in enumerate(product(range(3),repeat=3)):
            label=sum(mapping[v]*g['n']**(2-t) for t,v in enumerate(goal));expected[label]+=q[0,k]
        assert np.allclose(r['probabilities'][0],expected,rtol=0,atol=1e-15)
        assert abs(expected.sum()-1)<1e-15
    assert len(C.PARTITIONS)==5


def test_broad_correct_does_not_identify_fine_and_modes_do_not_commute():
    q=np.zeros((1,27));q[0,[0,13,26]]=[.4,.3,.3]
    r=C.forecasts(q,'meaning')
    assert np.all(r['mapped-fine-0.1-always']==0)
    assert np.all(r['coarse-mode-0.1-always']==1)
    truth=np.zeros((1,1,27));truth[0,0,13]=1;mass=np.ones((1,1))
    tables=C.native_tables(truth,mass,'meaning')
    v=C.score(r['probabilities'],r['coarse-mode-0.1-always'],tables,.1,[1.])
    assert v['correct'][0]==1 and v['alternatives'][0]==2
    assert v['fine_loss'][0]==1
    one=C.forecasts(q,'one-class');v=C.score(one['probabilities'],one['coarse-mode-0.1-always'],C.native_tables(truth,mass,'one-class'),.1,[1.])
    assert abs(v['correct'][0]-1)<1e-15 and v['alternatives'][0]==3 and abs(v['coarse_loss'][0])<1e-15


def test_exhaustive_partial_report_scores_against_scalar_goals():
    q=np.arange(1,28,dtype=float)[None];q/=q.sum();t=np.arange(27,0,-1,dtype=float)[None,None];t/=t.sum()
    for name,mapping in C.PARTITIONS.items():
        g=C.geometry(name);p=C.forecasts(q,name)['probabilities'];tables=C.native_tables(t,np.ones((1,1)),name)
        for r in g['reports']:
            v=C.score(p,r[None],tables,.25,[2.])
            correct=math.fsum(float(t[0,0,k])*sum(x>=0 and x==mapping[goal[s]] for s,x in enumerate(r))/3 for k,goal in enumerate(product(range(3),repeat=3)))
            coverage=sum(x>=0 for x in r)/3
            assert abs(v['correct'][0]-correct)<1e-14 and abs(v['coverage'][0]-coverage)<1e-14
            assert abs(v['cost_value'][0]-(coverage-correct+.25*(1-coverage)))<1e-14
            assert v['incompatible'][0]==0  # this reference has full fine support


def test_partial_incompatibility_and_empty_report():
    q=np.ones((1,27))/27;t=np.zeros((1,1,27));t[0,0,0]=1
    tables=C.native_tables(t,np.ones((1,1)),'meaning');p=C.forecasts(q,'meaning')['probabilities']
    v=C.score(p,np.array([[1,-1,-1]]),tables,.25,[1.])
    assert v['correct'][0]==0 and v['incompatible'][0]==1 and v['full_incompatible'][0]==0
    v=C.score(p,np.full((1,3),-1),tables,.25,[1.])
    assert v['coverage'][0]==0 and v['incorrect'][0]==0 and v['incompatible'][0]==0 and v['cost_value'][0]==.25


def test_thresholds_exact_ties_and_fine_identity():
    q=np.zeros((3,27));q[:,0]=[.75,np.nextafter(.75,1),np.nextafter(.75,0)];q[:,13]=1-q[:,0]
    r=C.forecasts(q,'fine')
    assert np.all(r['coarse-mode-0.25-abstain'][0]==-1)
    assert np.all(r['coarse-mode-0.25-abstain'][1]==0)
    assert np.all(r['coarse-mode-0.25-abstain'][2]==-1)
    assert np.array_equal(r['mapped-fine-0.25-abstain'],r['coarse-mode-0.25-abstain'])
    q[:]=0;q[:,0]=q[:,13]=.5
    assert np.all(C.forecasts(q,'fine')['coarse-mode-0.5-always']==0)


def test_complete_handler_and_all_specificity_endpoints(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=C.run(root,plan,lambda **kw:None)
    assert s['rows']==7680 and s['native_rows']==120 and s['parent_cells']==768
    assert s['parent_max_error']<1e-10 and all(s['controls'].values())
    rows=json.loads(gzip.decompress((root/'goal_coarsening_points.json.gz').read_bytes()))
    assert all(r['conditional_error'] is None and r['alternatives_per_claim'] is None for r in rows if r['coverage']==0)
    assert all(abs(r['coarse_loss'])<1e-12 and abs(r['alternatives_per_claim']-3)<1e-12 for r in rows if r['partition']=='one-class')
    assert all(abs(r['alternatives_per_claim']-1)<1e-12 for r in rows if r['partition']=='fine' and r['coverage']>0)
    assert all(r['mean']<=1e-12 for r in s['contrasts'] if r['metric']=='incorrect' and r['policy']=='abstain' and r['base_policy']=='always')
    changed=copy.deepcopy(rows);changed.pop()
    with pytest.raises(ValueError,match='roster'):C.aggregate(changed,plan['design'])


@pytest.mark.parametrize('problem',['hash','missing','duplicate','parent-score','reader','reference','mass','population','partition','probability'])
def test_corruptions(tmp_path,problem):
    root=tmp_path/'packet';plan=fixture(root);base=root/'inputs';p=None
    if problem in ('hash','missing','duplicate','parent-score'):
        p=base/'parent/goal_abstention_points.json.gz';v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        else:
            for r in v:r['loss']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    elif problem=='reader':
        p=base/'reader/PACKETS.json';v=read(p);next(iter(v['packets'].values()))['inputs']['goal']=0;write(p,v,immutable=False)
    elif problem in ('reference','mass'):
        p=base/'evaluator/REFERENCES.json';v=read(p)
        if problem=='reference':v[0]['frames'][0]['target'][0][1]=.5
        else:v[0]['frames'][0]['mass']=2
        write(p,v,immutable=False)
    elif problem=='population':
        p=base/'parent/PLAN.json';v=read(p);v['design']['fit_seeds']=[-1];write(p,v,immutable=False)
    elif problem=='partition':plan['design']['partitions']['meaning']=[0,1,2]
    else:
        p=next((base/'reports').glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        v['probabilities']*=2;np.savez_compressed(p,**v)
    if p is not None and problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):C.run(root,plan,lambda **kw:None)
