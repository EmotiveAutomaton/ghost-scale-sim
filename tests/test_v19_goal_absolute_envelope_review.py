"""Known answers, scalar reconstruction, and corrupted retained evidence."""
import copy
import gzip
import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_absolute_envelope_review as V
from ghostscale.validation.soundingline.v19 import goal_absolute_envelope as O
from ghostscale.validation.soundingline.v18_3.io import read, write, canonical, file_digest
from test_v19_goal_absolute_envelope import fixture as producer_fixture


def fixture(root):
    original=root/'producer';original.mkdir(parents=True)
    plan=producer_fixture(original);write(original/'PLAN.json',plan)
    write(original/'SUMMARY.json',O.run(original,plan,lambda **kw:None))
    base=root/'inputs';base.mkdir();shutil.copytree(original/'inputs',base/'parent')
    dest=base/'original';dest.mkdir()
    for n in ('PLAN.json','SUMMARY.json','goal_absolute_envelope_points.json.gz','ARRAY_SCHEMA.json'):
        shutil.copyfile(original/n,dest/n)
    for n in ('reader','evaluator','groups','forecasts','residuals'):shutil.copytree(original/n,dest/n)
    return dict(design=dict(target_plan_sha256=file_digest(dest/'PLAN.json'),input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}))


def test_complete_reconstruction_and_all_paired_estimates(tmp_path):
    root=tmp_path/'case';plan=fixture(root);r=V.run(root,plan,lambda **kw:None)
    assert r['passed'] and all(v<1e-10 for k,v in r.items() if k.startswith('max_'))
    assert r['rows']==2304 and r['parent_cells']==128 and r['group_bundles']==64
    assert r['empty_bins']>0
    g=read(root/'INDEPENDENT_REGROUP.json')
    assert len(g['contrasts'])==1512 and len(g['normalized_log_budget_area'])==756
    assert len(g['means'])==288 and len(g['envelope_estimates'])==864
    rows=json.loads(gzip.decompress((root/'reconstructed_points.json.gz').read_bytes()));cfg=read(root/'inputs/original/PLAN.json')['design']
    for broken in (rows[:-1],rows+[rows[0]],[r for r in rows if r['goal']!=2]):
        with pytest.raises(ValueError,match='roster'):V.regroup(broken,cfg)


def test_known_opposing_constant_unequal_and_zero_errors():
    assert all(V.controls().values())
    p=np.tile([.15,.85,0.],(2,3,1));c=p[None].copy()
    c[0,0,:,:2]+=[.05,-.05];c[0,1,:,:2]+=[-.05,.05]
    r=V.sufficient(p,c,np.array([[.25,.75]]))
    assert r['absolute_error'][0,0,0]==pytest.approx(.05)
    assert r['gap_20'][0,0,0]==pytest.approx(.025)
    c=p[None].copy();c[0,:,:,:2]+=[.05,-.05]
    r=V.sufficient(p,c,np.array([[.5,.5]]))
    assert np.max(abs(r['gap_20']))<1e-14
    c[0,1]=p[1]
    assert np.max(V.sufficient(p,c,np.array([[0.,1.]]))['absolute_error'])==0


def test_local_scalar_bins_preserve_prior_independent_algorithm():
    from ghostscale.validation.soundingline.v19.goal_bin_resolution_review import sufficient
    rng=np.random.default_rng(191021)
    p=rng.dirichlet([1,2,3],(41,3));c=rng.dirichlet([3,2,1],(3,41,3));w=rng.dirichlet(np.ones(41),3)
    p[0]=[0.,1.,0.];p[1]=[.05,.15,.8];w[0,0]=0.;w[0]/=w[0].sum()
    old=sufficient(p,c,w);new=V.bin_sufficient(p,c,w)
    assert set(old)==set(new)
    for k in old:assert np.array_equal(old[k],new[k],equal_nan=True),k


@pytest.mark.parametrize('a,b',[(float('nan'),0.),(0.,float('inf')),(1.,1.0001)])
def test_scalar_check_still_rejects_nonfinite_or_changed_sums(a,b):
    with pytest.raises(ValueError):V._scalar_close(a,b)


def test_scalar_raw_arrays_and_permutations():
    rng=np.random.default_rng(191021);p=rng.dirichlet([1,2,3],(17,3));c=rng.dirichlet([2,1,3],(2,17,3));w=rng.dirichlet(np.ones(17),2)
    r=V.sufficient(p,c,w);assert V.compare_arrays(r,O.sufficient(p,c,w))<1e-14
    reverse=V.sufficient(p[::-1],c[:,::-1],w[:,::-1])
    for k in ('frame_residual','frame_absolute_contribution'):reverse[k]=reverse[k][:,::-1]
    assert V.compare_arrays(r,reverse)<1e-14
    rr=V.sufficient(p[:,:,::-1],c[:,:,:,::-1],w)
    for m in V.METRICS:assert np.allclose(rr[m][...,::-1],r[m],atol=1e-14,rtol=0)
    assert np.allclose(r['frame_absolute_contribution'].sum(1),r['absolute_error'],rtol=0,atol=1e-14)


@pytest.mark.parametrize('n,edge',[(n,i) for n in (5,10,20) for i in range(1,n)])
def test_exact_adjacent_edges(n,edge):
    x=edge/n;p=np.array([0.,np.nextafter(x,0),x,np.nextafter(x,1),1.])
    assert [V.bin_id(float(v),n) for v in p]==[0,edge-1,edge,edge,n-1]
    assert np.array_equal(V.memberships(p)[0],V.memberships(p)[1]//2)
    assert np.array_equal(V.memberships(p)[1],V.memberships(p)[2]//2)


def test_empty_means_probability_one_and_faulty_bounds(monkeypatch):
    p=np.tile([0.,1.,0.],(2,3,1));w=np.array([[1.,0.]])
    r=V.sufficient(p,p[None],w)
    assert np.isnan(r['bin_forecast_mean'][r['bin_weight']==0]).all()
    assert np.isnan(r['bin_correct_mean'][r['bin_weight']==0]).all()
    for ri,n in enumerate(V.RESOLUTIONS):assert r['bin_weight'][0,0,1,V.OFFSETS[ri]+n-1]==1
    original=V.bin_sufficient
    def wrong(*args):
        v=original(*args);v['bin_error_20']+=.1;return v
    monkeypatch.setattr(V,'bin_sufficient',wrong)
    with pytest.raises(ValueError):V.sufficient(p,p[None],w)


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


@pytest.mark.parametrize('problem',['raw-residual','raw-contribution','residual-schema','bin-parent-score','bin-parent-missing','bin-parent-duplicate','probability','marginal','operation','decision','sum','empty','metric','native','native-forecast','saved-marginal','saved-bin','schema','score','missing','duplicate','hash','parent-score','reader','reference','summary','envelope-summary','population','bins','resolution-design'])
def test_corruptions(tmp_path,problem):
    root=tmp_path/'case';plan=fixture(root);base=root/'inputs'
    if problem in ('raw-residual','raw-contribution'):
        p=next((base/'original/residuals').glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:v=dict(z)
        v['signed' if problem=='raw-residual' else 'absolute_contribution']+=.1
        np.savez_compressed(p,**v)
    elif problem=='residual-schema':
        p=base/'original/ARRAY_SCHEMA.json';v=read(p);v['frame'][0]+='-corrupted';write(p,v,immutable=False)
    elif problem.startswith('bin-parent-'):
        p=base/'parent/parent_bin/goal_bin_resolution_points.json.gz';v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='bin-parent-missing':v.pop()
        elif problem=='bin-parent-duplicate':v.append(copy.deepcopy(v[0]))
        else:v[0]['bin_error_20']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    elif problem in ('probability','marginal','operation','decision','sum','empty','metric','native','native-forecast','saved-marginal','saved-bin'):

        if problem=='native':p=base/'original/evaluator/NATIVE_GROUPS.npz'
        elif problem=='native-forecast':p=base/'original/evaluator/NATIVE_FORECASTS.npz'
        else:
            folder='parent/decisions' if problem in ('probability','marginal','operation','decision') else ('original/forecasts' if problem.startswith('saved-') else 'original/groups')
            p=next((base/folder).glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        field={'probability':'probabilities','marginal':'marginals','operation':'operations','decision':'coordinate','sum':'bin_forecast_sum','empty':'bin_forecast_mean','metric':'gap_20','native':'bin_correct_sum','native-forecast':'marginals','saved-marginal':'marginals','saved-bin':'bin_ids'}[problem]
        if problem=='empty':v[field][np.isnan(v[field])]=0
        else:v[field]=v[field]+.1
        np.savez_compressed(p,**v)
    elif problem in ('score','missing','duplicate','hash','parent-score'):
        p=base/('parent/parent/goal_decision_points.json.gz' if problem=='parent-score' else 'original/goal_absolute_envelope_points.json.gz');v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        elif problem=='parent-score':
            for r in v:r['loss']+=.1
        else:v[0]['bin_error_20']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    else:
        name={'reader':'original/reader/PACKETS.json','reference':'original/evaluator/REFERENCES.json','summary':'original/SUMMARY.json','envelope-summary':'original/SUMMARY.json','population':'parent/parent/PLAN.json','bins':'original/PLAN.json','resolution-design':'original/PLAN.json','schema':'original/ARRAY_SCHEMA.json'}[problem]
        p=base/name;v=read(p)
        if problem=='reader':next(iter(v['packets'].values()))['inputs']['goal']=0
        elif problem=='reference':v[0]['frames'][0]['target'][0][1]=.5
        elif problem=='summary':v['contrasts'][0]['mean']+=.1
        elif problem=='envelope-summary':v['envelope_estimates'][0]['mean']+=.1
        elif problem=='population':v['design']['fit_seeds']=[-1]
        elif problem=='bins':v['design']['bin_edges']['10'][1]=.11
        elif problem=='resolution-design':v['design']['resolutions'].reverse()
        else:v['axes'].reverse()
        write(p,v,immutable=False)
        if problem in ('bins','resolution-design'):plan['design']['target_plan_sha256']=file_digest(p)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):V.run(root,plan,lambda **kw:None)
