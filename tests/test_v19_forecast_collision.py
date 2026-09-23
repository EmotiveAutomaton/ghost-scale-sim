"""Known aliases and independent explicit law/frame regrouping."""
import copy
import gzip
import math
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import forecast_collision as C
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_goal_abstention import fixture as parent_fixture


def fixture(root):
    p=parent_fixture(root);p['design']['bootstrap_seed']=191022
    return p


def scalar(p,t,w,n):
    groups={}
    for i,row in enumerate(p):groups.setdefault(tuple(row.ravel()),[]).append(i)
    out=np.zeros((4,3,3));laws=len(t)
    for step,goal in product(range(3),repeat=2):
        for members in groups.values():
            mass=math.fsum(float(n[l]*w[l,i]) for l in range(laws) for i in members)
            if not mass:continue
            mean=math.fsum(float(n[l]*w[l,i]*t[l,i,step,goal]) for l in range(laws) for i in members)/mass
            out[1,step,goal]+=mass/laws*(float(p[members[0],step,goal])-mean)**2
            for l in range(laws):
                lm=math.fsum(float(w[l,i]) for i in members)
                if not lm:continue
                local=math.fsum(float(w[l,i]*t[l,i,step,goal]) for i in members)/lm
                out[3,step,goal]+=n[l]*lm/laws*(local-mean)**2
                out[2,step,goal]+=n[l]/laws*math.fsum(float(w[l,i])*(float(t[l,i,step,goal])-local)**2 for i in members)
                out[0,step,goal]+=n[l]/laws*math.fsum(float(w[l,i])*(float(p[i,step,goal])-float(t[l,i,step,goal]))**2 for i in members)
    return out


@pytest.mark.parametrize('equal',[False,True])
def test_direct_regroup_all_law_resamples_and_zero_weights(equal):
    rng=np.random.default_rng(83);p=rng.dirichlet([1,2,3],(7,3));p[1]=p[0];p[3]=p[2]
    t=rng.dirichlet([3,2,1],(3,7,3));w=rng.dirichlet(np.ones(7),3)
    if equal:w[:]=1/7
    w[:,6]=0;w/=w.sum(1)[:,None]
    counts=np.array([n for n in product(range(4),repeat=3) if sum(n)==3])
    s=C.sufficient(p,t,w);b=C.resampled(s,counts,chunk=2)
    for v,n in zip(b,counts):assert np.allclose(v,scalar(p,t,w,n),rtol=0,atol=1e-14)
    assert np.allclose(b[:,0],b[:,1:].sum(1),rtol=0,atol=1e-14)
    ix=np.array([6,4,2,1,5,0,3]);other=C.resampled(C.sufficient(p[ix],t[:,ix],w[:,ix]),counts)
    assert np.allclose(b,other,rtol=0,atol=1e-14)


def test_controls_opposing_frames_singletons_and_one_bit():
    assert all(C.controls().values())
    p=np.tile([.5,.5,0.],(2,3,1));t=p[None].copy();t[0,0,:,:2]=[1,0];t[0,1,:,:2]=[0,1]
    v=C.resampled(C.sufficient(p,t,[[.5,.5]]),[[1]])[0]
    assert np.allclose(v[2,:,:2],.25) and np.max(abs(v[3]))==0
    p[1,0,0]=np.nextafter(.5,1)
    assert C.partition(p).tolist()==[0,1]
    # A single bit matters even when normalization agrees to display precision.
    v=C.resampled(C.sufficient(p,t,[[.5,.5]]),[[1]])[0]
    assert np.max(abs(v[2]))<1e-14
    with pytest.raises(ValueError,match='faulty merge'):C.sufficient(p,t,[[.5,.5]],ids=[0,0])


def test_normalized_native_roundoff_is_preserved_without_clipping():
    p=np.tile([1.,0.,0.],(2,3,1));p[1,:,0]=np.nextafter(1.,2.)
    t=p[None].copy();s=C.sufficient(p,t,[[.5,.5]])
    assert C.partition(p).tolist()==[0,1]
    assert np.array_equal(s['target_sum'][0],.5*t[0].reshape(2,9))
    assert np.max(abs(C.resampled(s,[[1]])))<1e-14
    t[0,0,0,0]=1+2e-12
    with pytest.raises(ValueError,match='simplex'):C.sufficient(p,t,[[.5,.5]])


@pytest.mark.parametrize('problem',['forecast','nan','target','weights','shape','merge','counts'])
def test_input_validation(problem):
    p=np.tile([.5,.5,0.],(2,3,1));t=p[None].copy();w=np.array([[.5,.5]])
    if problem=='forecast':p[0,0,0]=.2
    if problem=='nan':p[0,0,0]=np.nan
    if problem=='target':t[0,0,0]=[2,-1,0]
    if problem=='weights':w[0,0]=-1
    if problem=='shape':t=t[:,:1]
    with pytest.raises(ValueError):
        s=C.sufficient(p,t,w,ids=[0,1] if problem=='merge' else None)
        if problem=='counts':C.resampled(s,[[2]])


def test_complete_handler_and_group_reconstruction(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);summary=C.run(root,plan,lambda **kw:None)
    assert summary['parent_max_error']<1e-10 and summary['group_bundles']==2*summary['settings']
    assert len(summary['estimates'])==len(plan['design']['budgets'])*4*2*2*4*9
    with np.load(root/'evaluator/NATIVE_FORECASTS.npz') as z:target=z['marginals'];native=z['weights']
    for path in (root/'groups').glob('*.npz'):
        with np.load(path) as z:
            p=z['group_forecast'][z['ids']];w=np.full_like(native,1/native.shape[1]) if path.name.endswith('equal-frame.npz') else native
            assert np.allclose(z['values'],scalar(p,target,w,np.ones(len(target))),rtol=0,atol=1e-12)
            assert z['frame_counts'].sum()==native.shape[1]
            assert z['law_frame_counts'].sum()==native.size
            assert np.array_equal(z['ids'],C.partition(p))


@pytest.mark.parametrize('problem',['hash','parent-score','missing-row','duplicate-row','reader','target','mass','population','marginal','decision','operation','missing-frame','duplicate-frame'])
def test_native_input_corruption(tmp_path,problem):
    root=tmp_path/'packet';plan=fixture(root);base=root/'inputs';p=None
    if problem in ('hash','parent-score','missing-row','duplicate-row'):
        p=base/'parent/goal_decision_points.json.gz';v=__import__('json').loads(gzip.decompress(p.read_bytes()))
        if problem=='missing-row':v.pop()
        elif problem=='duplicate-row':v.append(copy.deepcopy(v[0]))
        else:
            for r in v:r['loss']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    elif problem=='reader':
        p=base/'reader/PACKETS.json';v=read(p);next(iter(v['packets'].values()))['inputs']['goal']=0;write(p,v,immutable=False)
    elif problem in ('target','mass','missing-frame','duplicate-frame'):
        p=base/'evaluator/REFERENCES.json';v=read(p)
        if problem=='target':v[0]['frames'][0]['target'][0][1]=.5
        elif problem=='mass':v[0]['frames'][0]['mass']=2
        elif problem=='missing-frame':v[0]['frames'].pop()
        else:v[0]['frames'].append(copy.deepcopy(v[0]['frames'][0]))
        write(p,v,immutable=False)
    elif problem=='population':
        p=base/'parent/PLAN.json';v=read(p);v['design']['fit_seeds']=[-1];write(p,v,immutable=False)
    else:
        p=next((base/'decisions').glob('*.npz'))
        with np.load(p) as z:v={k:z[k] for k in z.files}
        key={'marginal':'marginals','decision':'coordinate','operation':'operations'}[problem];v[key]=v[key]+.1;np.savez_compressed(p,**v)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):C.run(root,plan,lambda **kw:None)
