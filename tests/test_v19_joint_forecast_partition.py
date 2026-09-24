"""Exact partition coverage and independent centered law/frame identities."""
import copy
import gzip
import math
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import joint_forecast_partition as J
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_goal_abstention import fixture as parent_fixture


def fixture(root):
    p=parent_fixture(root);p['design']['bootstrap_seed']=191023
    return p


def scalar(p,kind,t,w,n):
    keys=[tuple(x.ravel()) for x in p];groups={}
    for i,k in enumerate(keys):groups.setdefault(k,[]).append(i)
    total=within=across=0.;laws=len(t)
    for members in groups.values():
        mass=math.fsum(float(n[l]*w[l,i]) for l in range(laws) for i in members)
        if mass==0:continue
        for k in range(27):
            mean=math.fsum(float(n[l]*w[l,i]*t[l,i,k]) for l in range(laws) for i in members)/mass
            for l in range(laws):
                lm=math.fsum(float(w[l,i]) for i in members)
                if lm==0:continue
                local=math.fsum(float(w[l,i]*t[l,i,k]) for i in members)/lm
                total+=n[l]/laws*math.fsum(float(w[l,i])*(float(t[l,i,k])-mean)**2 for i in members)
                within+=n[l]/laws*math.fsum(float(w[l,i])*(float(t[l,i,k])-local)**2 for i in members)
                across+=n[l]*lm/laws*(local-mean)**2
    return np.array([total,within,across])


@pytest.mark.parametrize('equal',[True,False])
@pytest.mark.parametrize('kind',['joint','marginal'])
def test_every_small_law_multiplicity_and_permutation(kind,equal):
    rng=np.random.default_rng(733);q=rng.dirichlet(np.ones(27),7);q[1]=q[0];q[3]=q[2]
    p=q if kind=='joint' else np.stack([q@(J.D.GOALS[:,t,None]==np.arange(3)) for t in range(3)],axis=1)
    t=rng.dirichlet(np.ones(27),(3,7));w=rng.dirichlet(np.ones(7),3)
    if equal:w[:]=1/7
    w[:,6]=0;w/=w.sum(1)[:,None];counts=np.array([n for n in product(range(4),repeat=3) if sum(n)==3])
    s=J.sufficient(p,kind,t,w);v=J.resample(s,counts,chunk=2)
    for value,n in zip(v,counts):assert np.allclose(value,scalar(p,kind,t,w,n),atol=1e-14,rtol=0)
    assert np.allclose(v[:,0],v[:,1:].sum(1),atol=1e-14,rtol=0)
    assert np.isnan(s['group_target'][s['mass'].sum(0)==0]).all()
    ix=[6,2,1,0,5,4,3];li=[2,0,1]
    perm=J.resample(J.sufficient(p[ix],kind,t[li][:,ix],w[li][:,ix]),counts[:,li])
    assert np.allclose(v,perm,atol=1e-14,rtol=0)


def test_equal_marginal_dependence_and_numerical_refinement_exceptions():
    assert all(J.controls().values())
    q=np.zeros((2,27));q[:,:2]=.5;q[1,0]=np.nextafter(.5,1)
    assert J.partition(q,'joint').tolist()==[0,1]
    with pytest.raises(ValueError,match='faulty'):J.sufficient(q,'joint',q[None],[[.5,.5]],[0,0])
    assert not J.refinement([0,0],[0,1])['joint_refines_marginal']
    q[0,2]=-0.;assert J.partition(q,'joint')[0]==0
    q[:]=0;q[:,0]=np.nextafter(1.,2.);s=J.sufficient(q,'joint',q[None],[[.5,.5]])
    assert s['target_sum'][0,0,0]>1 and np.array_equal(s['target_sum'][0,0],q[0])


@pytest.mark.parametrize('bad',['forecast','nonfinite','target','weights','shape','kind','counts'])
def test_invalid_inputs(bad):
    q=np.zeros((2,27));q[:,0]=1;t=q[None].copy();w=np.full((1,2),.5);kind='joint'
    if bad=='forecast':q[0,0]=.8
    if bad=='nonfinite':q[0,0]=np.nan
    if bad=='target':t[0,0,1]=.1
    if bad=='weights':w[0,0]=-1
    if bad=='shape':t=t[:,:1]
    if bad=='kind':kind='bad'
    with pytest.raises(ValueError):
        s=J.sufficient(q,kind,t,w)
        if bad=='counts':J.resample(s,[[2]])


def test_complete_handler_membership_and_scalar_identity(tmp_path):
    root=tmp_path/'job';plan=fixture(root);summary=J.run(root,plan,lambda **kw:None)
    assert summary['refinement_exceptions']==0 and summary['parent_max_error']<1e-10
    assert summary['group_bundles']==summary['settings']*4
    bindings=read(root/'GROUP_BINDINGS.json')
    assert len(bindings)==summary['group_bundles']
    assert len({v['file'] for v in bindings.values()})==summary['unique_group_arrays']
    for b in bindings.values():assert file_digest(root/b['file'])==b['sha256']
    with np.load(root/'evaluator/NATIVE_JOINT.npz') as z:t=z['targets'];native=z['weights']
    for file in (root/'groups').glob('*.npz'):
        kind=file.stem.split('-')[-1];stem=file.stem.removesuffix('-'+kind);weighting='equal-frame' if stem.endswith('equal-frame') else 'native';decision=stem.removesuffix('-'+weighting)
        with np.load(root/'inputs/decisions'/(decision+'.npz')) as d:p=d['probabilities' if kind=='joint' else 'marginals']
        with np.load(file) as z:
            w=np.full_like(native,1/native.shape[1]) if weighting=='equal-frame' else native
            assert np.array_equal(z['ids'],J.partition(p,kind))
            assert np.allclose(z['values'],scalar(p,kind,t,w,np.ones(len(t))),atol=1e-12,rtol=0)
            assert z['frame_counts'].sum()==native.shape[1]


@pytest.mark.parametrize('bad',['hash','parent-score','missing-row','duplicate-row','reader','target','mass','population','marginal','decision','operation','missing-frame','duplicate-frame'])
def test_native_input_corruption(tmp_path,bad):
    root=tmp_path/'job';plan=fixture(root);base=root/'inputs'
    if bad in ('hash','parent-score','missing-row','duplicate-row'):
        p=base/'parent/goal_decision_points.json.gz';v=__import__('json').loads(gzip.decompress(p.read_bytes()))
        if bad=='missing-row':v.pop()
        elif bad=='duplicate-row':v.append(copy.deepcopy(v[0]))
        else:
            for row in v:row['loss']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    elif bad=='reader':
        p=base/'reader/PACKETS.json';v=read(p);next(iter(v['packets'].values()))['inputs']['goal']=0;write(p,v,immutable=False)
    elif bad in ('target','mass','missing-frame','duplicate-frame'):
        p=base/'evaluator/REFERENCES.json';v=read(p)
        if bad=='target':v[0]['frames'][0]['target'][0][1]=.5
        elif bad=='mass':v[0]['frames'][0]['mass']=2
        elif bad=='missing-frame':v[0]['frames'].pop()
        else:v[0]['frames'].append(copy.deepcopy(v[0]['frames'][0]))
        write(p,v,immutable=False)
    elif bad=='population':
        p=base/'parent/PLAN.json';v=read(p);v['design']['fit_seeds']=[-1];write(p,v,immutable=False)
    else:
        p=next((base/'decisions').glob('*.npz'))
        with np.load(p) as z:v=dict(z)
        k={'marginal':'marginals','decision':'coordinate','operation':'operations'}[bad];v[k]=v[k]+.1;np.savez_compressed(p,**v)
    if bad!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):J.run(root,plan,lambda **kw:None)
