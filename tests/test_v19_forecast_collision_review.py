"""Independent centered moments, all small resamples and corrupted evidence."""
from pathlib import Path
from itertools import product
import copy
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import forecast_collision as C
from ghostscale.validation.soundingline.v19 import forecast_collision_review as R
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from test_v19_forecast_collision import fixture as producer_fixture,scalar


def fixture(root):
    original=root/'inputs/original';original.mkdir(parents=True)
    p=producer_fixture(original);summary=C.run(original,p,lambda **kw:None)
    write(original/'PLAN.json',p);write(original/'SUMMARY.json',summary)
    shutil.copytree(original/'inputs',root/'inputs/parent')
    return dict(design=dict(target_plan_sha256=file_digest(original/'PLAN.json'),
        input_files={v.relative_to(root/'inputs').as_posix():file_digest(v) for v in (root/'inputs').rglob('*') if v.is_file()}))


@pytest.mark.parametrize('equal',[False,True])
def test_all_small_law_counts_against_scalar(equal):
    rng=np.random.default_rng(391);p=rng.dirichlet([1,2,3],(7,3));p[1]=p[0];p[3]=p[2]
    t=rng.dirichlet([3,2,1],(3,7,3));w=rng.dirichlet(np.ones(7),3)
    if equal:w[:]=1/7
    w[:,6]=0;w/=w.sum(1)[:,None]
    ns=np.array([v for v in product(range(4),repeat=3) if sum(v)==3])
    s=R.sufficient(p,t,w);v=R.resample(s,ns)
    for n,x in zip(ns,v):assert np.allclose(x,scalar(p,t,w,n),rtol=0,atol=1e-14)
    for k,x in C.sufficient(p,t,w).items():assert np.allclose(s[k],x,rtol=0,atol=1e-14)
    ix=[6,4,2,1,5,0,3]
    assert np.allclose(v,R.resample(R.sufficient(p[ix],t[:,ix],w[:,ix]),ns),rtol=0,atol=1e-14)
    assert np.isnan(s['group_target'][s['mass'].sum(0)==0]).all()


def test_known_answers_and_one_bit():
    assert all(R.controls().values())
    p=np.tile([.5,.5,0.],(2,3,1));t=p[None].copy();t[0,0,:,:2]=[1,0];t[0,1,:,:2]=[0,1]
    v=R.resample(R.sufficient(p,t,[[.5,.5]]),[[1]])[0]
    assert np.allclose(v[2,:,:2],.25) and np.max(abs(v[3]))<1e-14
    p[1,0,0]=np.nextafter(.5,1);s=R.sufficient(p,t,[[.5,.5]])
    assert s['ids'].tolist()==[0,1]
    assert np.max(abs(R.resample(s,[[1]])[0,2]))<1e-14
    bad=copy.deepcopy(s);bad['ids'][:]=0
    with pytest.raises(ValueError):R.arrays(s,bad)
    p=np.tile([1.,0.,0.],(2,3,1));p[1,:,0]=np.nextafter(1.,2.)
    assert R.sufficient(p,p[None],[[.5,.5]])['ids'].tolist()==[0,1]
    p[1,0,0]=1+2e-12
    with pytest.raises(ValueError):R.sufficient(p,p[None],[[.5,.5]])


def test_complete_independent_handler(tmp_path):
    plan=fixture(tmp_path);result=R.run(tmp_path,plan,lambda **kw:None)
    assert result['passed'] and result['parent_cells']>0
    assert result['max_regroup_error']<1e-10 and result['max_original_regroup_error']<1e-10
    assert len(list((tmp_path/'reconstructed_groups').glob('*.npz')))==result['group_bundles']


@pytest.mark.parametrize('problem',['hash','ids','mass','target_sum','target_square','within','loss','group_target','group_forecast','frame_counts','law_frame_counts','positive_mass_counts','values','summary','schema','bootstrap','parent-score','population','reader','native','marginal'])
def test_corruption_rejected(tmp_path,problem):
    plan=fixture(tmp_path);base=tmp_path/'inputs';original=base/'original'
    if problem in ('hash','summary'):
        p=original/'SUMMARY.json';v=read(p);v['estimates'][0]['mean']+=.1;write(p,v,immutable=False)
    elif problem=='schema':
        p=original/'ARRAY_SCHEMA.json';v=read(p);v['frames'][0]='broken';write(p,v,immutable=False)
    elif problem=='bootstrap':
        p=original/'BOOTSTRAP.json';v=read(p);v['count_digest']='broken';write(p,v,immutable=False)
    elif problem=='population':
        p=base/'parent/parent/PLAN.json';v=read(p);v['design']['fit_seeds']=[-1];write(p,v,immutable=False)
    elif problem=='reader':
        p=original/'reader/PACKETS.json';v=read(p);next(iter(v['packets'].values()))['inputs']['goal']=0;write(p,v,immutable=False)
    elif problem=='parent-score':
        import gzip,json
        from ghostscale.validation.soundingline.v18_3.io import canonical
        p=base/'parent/parent/goal_decision_points.json.gz';v=json.loads(gzip.decompress(p.read_bytes()))
        for r in v:r['loss']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    else:
        p=original/'evaluator/NATIVE_FORECASTS.npz' if problem=='native' else next((base/'parent/decisions').glob('*.npz')) if problem=='marginal' else next((original/'groups').glob('*.npz'))
        with np.load(p) as z:v=dict(z)
        key='marginals' if problem in ('native','marginal') else problem
        v[key]=v[key].copy();v[key].flat[0]+=1 if np.issubdtype(v[key].dtype,np.integer) else .1
        np.savez_compressed(p,**v)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):R.run(tmp_path,plan,lambda **kw:None)
