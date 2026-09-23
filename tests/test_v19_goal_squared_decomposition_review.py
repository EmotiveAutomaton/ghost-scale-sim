"""Known answers, scalar reconstruction, and corrupted retained evidence."""
import copy
import gzip
import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_squared_decomposition_review as V
from ghostscale.validation.soundingline.v19 import goal_squared_decomposition as O
from ghostscale.validation.soundingline.v18_3.io import read, write, canonical, file_digest
from test_v19_goal_squared_decomposition import fixture as producer_fixture


def fixture(root):
    original=root/'producer';original.mkdir(parents=True)
    plan=producer_fixture(original);write(original/'PLAN.json',plan)
    write(original/'SUMMARY.json',O.run(original,plan,lambda **kw:None))
    base=root/'inputs';base.mkdir();shutil.copytree(original/'inputs',base/'parent')
    dest=base/'original';dest.mkdir()
    for n in ('PLAN.json','SUMMARY.json','goal_squared_decomposition_points.json.gz','ARRAY_SCHEMA.json'):
        shutil.copyfile(original/n,dest/n)
    for n in ('reader','evaluator','groups','forecasts'):shutil.copytree(original/n,dest/n)
    return dict(design=dict(target_plan_sha256=file_digest(dest/'PLAN.json'),input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}))


def test_complete_reconstruction_and_all_paired_estimates(tmp_path):
    root=tmp_path/'case';plan=fixture(root);r=V.run(root,plan,lambda **kw:None)
    assert r['passed'] and all(v<1e-10 for k,v in r.items() if k.startswith('max_'))
    assert r['rows']==2304 and r['parent_cells']==128 and r['group_bundles']==64
    assert r['empty_bins']>0
    g=read(root/'INDEPENDENT_REGROUP.json')
    assert len(g['contrasts'])==648 and len(g['normalized_log_budget_area'])==324
    assert len(g['means'])==288 and len(g['component_estimates'])==576
    rows=json.loads(gzip.decompress((root/'reconstructed_points.json.gz').read_bytes()));cfg=read(root/'inputs/original/PLAN.json')['design']
    for broken in (rows[:-1],rows+[rows[0]],[r for r in rows if r['goal']!=2]):
        with pytest.raises(ValueError,match='roster'):V.regroup(broken,cfg)


def test_known_opposing_constant_and_unequal_residuals():
    assert all(V.controls().values())
    p=np.tile([.15,.85,0.],(2,3,1));c=p[None].copy()
    c[0,0,:,:2]+=[.05,-.05];c[0,1,:,:2]+=[-.05,.05]
    r=V.sufficient(p,c,np.array([[.25,.75]]))
    assert r['squared_bias'][0,0,0]==pytest.approx(.000625)
    assert r['within_variance'][0,0,0]==pytest.approx(.001875)
    c=p[None].copy();c[0,:,:,:2]+=[.05,-.05]
    r=V.sufficient(p,c,np.array([[.5,.5]]))
    assert np.max(abs(r['within_variance']))<1e-28
    assert r['squared_bias'][0,0,0]==pytest.approx(.0025)


def test_pairwise_variance_permutations_and_all_moments():
    import math
    from itertools import product
    rng=np.random.default_rng(191020);p=rng.dirichlet([1,2,3],(17,3));c=rng.dirichlet([2,1,3],(2,17,3));w=rng.dirichlet(np.ones(17),2)
    r=V.sufficient(p,c,w);assert V.compare_arrays(r,O.sufficient(p,c,w))<1e-14
    assert V.compare_arrays(r,V.sufficient(p[::-1],c[:,::-1],w[:,::-1]))<1e-14
    rr=V.sufficient(p[:,:,::-1],c[:,:,:,::-1],w)
    for m in V.METRICS:assert np.allclose(rr[m][...,::-1],r[m],atol=1e-14,rtol=0)
    for li,t,g,b in product(range(2),range(3),range(3),range(10)):
        ix=[i for i in range(17) if V.bin_id(float(p[i,t,g]))==b]
        mass=math.fsum(float(w[li,i]) for i in ix)
        if mass:
            e={i:float(p[i,t,g])-float(c[li,i,t,g]) for i in ix}
            pair=math.fsum(float(w[li,i])*float(w[li,j])*(e[i]-e[j])**2 for i,j in product(ix,ix))/(2*mass)
            assert r['bin_variance_mass'][li,t,g,b]==pytest.approx(pair,abs=1e-14)


@pytest.mark.parametrize('edge',range(1,10))
def test_exact_adjacent_edges(edge):
    x=edge/10;p=np.array([0.,np.nextafter(x,0),x,np.nextafter(x,1),1.])
    assert [V.bin_id(float(v)) for v in p]==[0,edge-1,edge,edge,9]
    assert np.array_equal(V.memberships(p),O.bins(p))


def test_empty_means_probability_one_and_zero_weight():
    p=np.tile([0.,1.,0.],(2,3,1));c=p[None].copy();c[0,1]=[.5,.5,0.]
    r=V.sufficient(p,c,np.array([[1.,0.]]))
    for k in ('bin_residual_mean','bin_residual_second_moment','bin_residual_variance'):
        assert np.isnan(r[k][r['bin_weight']==0]).all()
    assert r['bin_weight'][0,0,1,9]==1
    for m in V.METRICS:assert np.max(r[m])==0


@pytest.mark.parametrize('problem',['probability','target','weight','conservation','nan','shape'])
def test_invalid_inputs(problem):
    p=np.tile([.6,.3,.1],(2,3,1));c=p[None].copy();w=np.array([[.5,.5]])
    if problem=='probability':p[0,0,0]=1.1
    elif problem=='target':c[0,0,0,0]=-1
    elif problem=='weight':w*=2
    elif problem=='conservation':p[0,0,0]=.5
    elif problem=='nan':w[0,0]=np.nan
    else:w=w[:,:1]
    with pytest.raises(ValueError):V.sufficient(p,c,w)


@pytest.mark.parametrize('problem',['probability','marginal','operation','decision','sum','empty','metric','native','native-forecast','saved-marginal','saved-bin','schema','score','missing','duplicate','hash','parent-score','reader','reference','summary','component-summary','population','bins'])
def test_corruptions(tmp_path,problem):
    root=tmp_path/'case';plan=fixture(root);base=root/'inputs'
    if problem in ('probability','marginal','operation','decision','sum','empty','metric','native','native-forecast','saved-marginal','saved-bin'):
        if problem=='native':p=base/'original/evaluator/NATIVE_GROUPS.npz'
        elif problem=='native-forecast':p=base/'original/evaluator/NATIVE_FORECASTS.npz'
        else:
            folder='parent/decisions' if problem in ('probability','marginal','operation','decision') else ('original/forecasts' if problem.startswith('saved-') else 'original/groups')
            p=next((base/folder).glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        field={'probability':'probabilities','marginal':'marginals','operation':'operations','decision':'coordinate','sum':'bin_residual_sum','empty':'bin_residual_mean','metric':'within_variance','native':'bin_residual_squared_sum','native-forecast':'marginals','saved-marginal':'marginals','saved-bin':'bin_ids'}[problem]
        if problem=='empty':v[field][np.isnan(v[field])]=0
        else:v[field]=v[field]+.1
        np.savez_compressed(p,**v)
    elif problem in ('score','missing','duplicate','hash','parent-score'):
        p=base/('parent/parent/goal_decision_points.json.gz' if problem=='parent-score' else 'original/goal_squared_decomposition_points.json.gz');v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        elif problem=='parent-score':
            for r in v:r['loss']+=.1
        else:v[0]['squared_bias']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    else:
        name={'reader':'original/reader/PACKETS.json','reference':'original/evaluator/REFERENCES.json','summary':'original/SUMMARY.json','component-summary':'original/SUMMARY.json','population':'parent/parent/PLAN.json','bins':'original/PLAN.json','schema':'original/ARRAY_SCHEMA.json'}[problem]
        p=base/name;v=read(p)
        if problem=='reader':next(iter(v['packets'].values()))['inputs']['goal']=0
        elif problem=='reference':v[0]['frames'][0]['target'][0][1]=.5
        elif problem=='summary':v['contrasts'][0]['mean']+=.1
        elif problem=='component-summary':v['component_estimates'][0]['mean']+=.1
        elif problem=='population':v['design']['fit_seeds']=[-1]
        elif problem=='bins':v['design']['bin_edges'][1]=.11
        else:v['axes'].reverse()
        write(p,v,immutable=False)
        if problem=='bins':plan['design']['target_plan_sha256']=file_digest(p)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):V.run(root,plan,lambda **kw:None)
