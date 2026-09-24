"""Independent scalar reconstruction, complete execution and corruption checks."""
from itertools import product
import copy
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import joint_forecast_partition as J, joint_partition_review as R
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from test_v19_joint_forecast_partition import fixture as original_fixture, scalar


@pytest.fixture
def completed(tmp_path):
    original=tmp_path/'producer';plan=original_fixture(original)
    write(original/'PLAN.json',plan);summary=J.run(original,plan,lambda **kw:None);write(original/'SUMMARY.json',summary)
    root=tmp_path/'review';base=root/'inputs';base.mkdir(parents=True)
    shutil.copytree(original,base/'original',ignore=shutil.ignore_patterns('inputs','decision'))
    shutil.copytree(original/'inputs',base/'parent')
    return root,dict(design=dict(target_plan_sha256=file_digest(original/'PLAN.json'),input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}))


@pytest.mark.parametrize('kind',['joint','marginal'])
@pytest.mark.parametrize('equal',[True,False])
def test_scalar_moments_and_exhaustive_multiplicities(kind,equal):
    rng=np.random.default_rng(816);q=rng.dirichlet(np.ones(27),7);q[1]=q[0];q[3]=q[2]
    p=q if kind=='joint' else R.marginals(q)
    t=rng.dirichlet(np.ones(27),(3,7));w=rng.dirichlet(np.ones(7),3)
    if equal:w[:]=1/7
    w[:,6]=0;w/=w.sum(1)[:,None]
    s=R.sufficient(p,kind,t,w);assert R.arrays(s,J.sufficient(p,kind,t,w))<1e-12
    counts=np.array([n for n in product(range(4),repeat=3) if sum(n)==3])
    v=R.resample(s,counts,chunk=2)
    for value,n in zip(v,counts):assert np.allclose(value,scalar(p,kind,t,w,n),atol=1e-14,rtol=0)
    li=[2,0,1];ix=[6,2,1,0,5,4,3]
    assert np.allclose(v,R.resample(R.sufficient(p[ix],kind,t[li][:,ix],w[li][:,ix]),counts[:,li]),atol=1e-14,rtol=0)


def test_controls_and_exact_membership():
    assert all(R.controls().values())
    q=np.zeros((3,27));q[:,:2]=.5;q[1,0]=np.nextafter(.5,1);q[2,2]=-0.
    assert R.membership(q,'joint')[0].tolist()==[0,1,0]
    assert R.relation([0,0],[0,1])==J.refinement([0,0],[0,1])
    assert R.relation([0,1],[0,0])==J.refinement([0,1],[0,0])


def test_complete_review(completed):
    root,plan=completed;result=R.run(root,plan,lambda **kw:None)
    assert result['passed'] and result['max_regroup_error']<1e-12
    assert result['group_bundles']==4*result['settings']


@pytest.mark.parametrize('bad',['membership','mass','sum','square','within','pair','mean','value','binding','binding-membership','extra-binding','relation','summary','parent','reader','hash'])
def test_corruption(completed,bad):
    root,plan=completed;base=root/'inputs';original=base/'original'
    bindings=read(original/'GROUP_BINDINGS.json');key=next(iter(bindings));bound=bindings[key]
    if bad in ('membership','mass','sum','square','within','pair','mean','value'):
        p=original/bound['file']
        with np.load(p) as z:v=dict(z)
        field=dict(membership='ids',mass='mass',sum='target_sum',square='target_square',within='within',pair='pair_distance',mean='group_target',value='values')[bad]
        if v[field].size==0:pytest.fail('fixture lacks pair coverage')
        v[field].flat[0]+=.1 if bad!='membership' else 1;np.savez_compressed(p,**v)
        for b in bindings.values():
            if b['file']==bound['file']:b['sha256']=file_digest(p)
        write(original/'GROUP_BINDINGS.json',bindings,immutable=False)
    elif bad in ('binding','binding-membership','extra-binding'):
        if bad=='binding':bindings[key]['sha256']='bad'
        elif bad=='binding-membership':bindings[key]['partition_membership_sha256']='bad'
        else:bindings['extra']=copy.deepcopy(bound)
        write(original/'GROUP_BINDINGS.json',bindings,immutable=False)
    else:
        names=dict(relation='PARTITION_RELATIONS.json',summary='SUMMARY.json',reader='reader/PACKETS.json',hash='SUMMARY.json')
        p=base/'parent/parent/PLAN.json' if bad=='parent' else original/names[bad];v=read(p)
        if bad=='relation':v[0]['joint_groups']+=1
        elif bad in ('summary','hash'):v['estimates'][0]['mean']+=.1
        elif bad=='reader':next(iter(v['packets'].values()))['inputs']['goal']=0
        else:v['design']['fit_seeds']=[-1]
        write(p,v,immutable=False)
    if bad!='hash':plan['design']['input_files']={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}
    with pytest.raises((ValueError,KeyError)):R.run(root,plan,lambda **kw:None)
