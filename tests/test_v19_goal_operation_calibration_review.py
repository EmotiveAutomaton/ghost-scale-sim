"""Independent operation grouping, empty denominators and corruption controls."""
import copy
import gzip
import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_operation_calibration_review as V
from ghostscale.validation.soundingline.v19 import goal_operation_calibration as O
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_goal_operation_calibration import fixture as producer_fixture


def fixture(root):
    original=root/'producer';original.mkdir(parents=True)
    plan=producer_fixture(original);write(original/'PLAN.json',plan)
    write(original/'SUMMARY.json',O.run(original,plan,lambda **kw:None))
    base=root/'inputs';base.mkdir();shutil.copytree(original/'inputs',base/'parent')
    dest=base/'original';dest.mkdir()
    for n in ('PLAN.json','SUMMARY.json','goal_operation_calibration_points.json.gz','ARRAY_SCHEMA.json'):
        shutil.copyfile(original/n,dest/n)
    for n in ('reader','evaluator','groups'):shutil.copytree(original/n,dest/n)
    return dict(design=dict(target_plan_sha256=file_digest(dest/'PLAN.json'),input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}))


def test_complete_reconstruction_and_all_paired_estimates(tmp_path):
    root=tmp_path/'case';plan=fixture(root);r=V.run(root,plan,lambda **kw:None)
    assert r['passed'] and all(v<1e-10 for k,v in r.items() if k.startswith('max_'))
    assert r['rows']==2304 and r['parent_cells']==128 and r['group_bundles']==64
    assert r['empty_groups']>0 and r['empty_bins']>0
    g=read(root/'INDEPENDENT_REGROUP.json')
    assert len(g['contrasts'])==648 and len(g['normalized_log_budget_area'])==324 and len(g['means'])==len(g['gap_estimates'])==288
    rows=json.loads(gzip.decompress((root/'reconstructed_points.json.gz').read_bytes()));cfg=read(root/'inputs/original/PLAN.json')['design']
    for broken in (rows[:-1],rows+[rows[0]],[r for r in rows if r['goal']!=2]):
        with pytest.raises(ValueError,match='roster'):V.regroup(broken,cfg)


def test_known_cancellation_permutations_mass_and_empty_means():
    assert all(V.controls().values())
    p=np.tile([.6,.3,.1],(2,3,1));c=p[None].copy()
    c[0,0,:,0]+=.1;c[0,0,:,1]-=.1;c[0,1,:,0]-=.1;c[0,1,:,1]+=.1
    w=np.array([[.5,.5]]);ops=np.array([[0]*3,[1]*3]);r=V.sufficient(p,c,w,ops)
    assert r['grouped_bin_error'][0,0,0]==pytest.approx(.1)
    assert r['pooled_bin_error'][0,0,0]==pytest.approx(0,abs=1e-14)
    assert r['conditional_signed_error'][0,0,0,:2]==pytest.approx([-.1,.1])
    assert np.isnan(r['bin_forecast_mean'][r['bin_weight']==0]).all()
    same=V.sufficient(p,np.repeat(c[:,:1],2,axis=1),w,ops)
    assert np.max(abs(same['cancellation_gap']))<1e-14
    rng=np.random.default_rng(191016);p=rng.dirichlet([1,2,3],(17,3));c=rng.dirichlet([2,1,3],(2,17,3));w=rng.dirichlet(np.ones(17),2);ops=rng.integers(6,size=(17,3))
    r=V.sufficient(p,c,w,ops);assert V.compare_arrays(r,O.sufficient(p,c,w,ops))<1e-14
    for rr in (V.sufficient(p[::-1],c[:,::-1],w[:,::-1],ops[::-1]),V.sufficient(p,c,w,5-ops),V.sufficient(p[:,:,::-1],c[:,:,:,::-1],w,ops)):
        assert np.sum(rr['grouped_bin_error'])==pytest.approx(np.sum(r['grouped_bin_error']))
        assert np.sum(rr['pooled_bin_error'])==pytest.approx(np.sum(r['pooled_bin_error']))
    assert np.allclose(r['operation_mass'].sum(3),1)


@pytest.mark.parametrize('edge',range(1,10))
def test_exact_adjacent_binary64_edges(edge):
    x=edge/10;pp=np.array([np.nextafter(x,0),x,np.nextafter(x,1)])
    p=np.repeat(np.stack([pp,1-pp,np.zeros(3)],axis=1)[:,None,:],3,axis=1)
    r=V.sufficient(p,p[None],np.full((1,3),1/3),np.zeros((3,3),int))
    assert r['bin_weight'][0,0,0,0,edge-1]==pytest.approx(1/3)
    assert r['bin_weight'][0,0,0,0,edge]==pytest.approx(2/3)
    assert np.max(r['grouped_bin_error'])==0


@pytest.mark.parametrize('problem',['probability','target','weight','operation','conservation','nan','shape'])
def test_invalid_inputs(problem):
    p=np.tile([.6,.3,.1],(2,3,1));c=p[None].copy();w=np.array([[.5,.5]]);ops=np.zeros((2,3),int)
    if problem=='probability':p[0,0,0]=1.1
    elif problem=='target':c[0,0,0,0]=-1
    elif problem=='weight':w*=2
    elif problem=='operation':ops[0,0]=6
    elif problem=='conservation':p[0,0,0]=.5
    elif problem=='nan':w[0,0]=np.nan
    else:ops=ops[:1]
    with pytest.raises(ValueError):V.sufficient(p,c,w,ops)


@pytest.mark.parametrize('problem',['probability','marginal','operation','decision','sum','empty','group-mass','signed','metric','native','schema','score','missing','duplicate','hash','parent-score','reader','reference','summary','gap-summary','population','bins','operation-design'])
def test_corruptions(tmp_path,problem):
    root=tmp_path/'case';plan=fixture(root);base=root/'inputs'
    if problem in ('probability','marginal','operation','decision','sum','empty','group-mass','signed','metric','native'):
        if problem=='native':p=base/'original/evaluator/NATIVE_GROUPS.npz'
        else:p=next((base/('parent/decisions' if problem in ('probability','marginal','operation','decision') else 'original/groups')).glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        field={'probability':'probabilities','marginal':'marginals','operation':'operations','decision':'coordinate','sum':'bin_forecast_sum','empty':'bin_forecast_mean','group-mass':'operation_mass','signed':'conditional_signed_error','metric':'cancellation_gap','native':'bin_correct_sum'}[problem]
        if problem=='empty':v[field][np.isnan(v[field])]=0
        else:v[field]=v[field]+.1
        np.savez_compressed(p,**v)
    elif problem in ('score','missing','duplicate','hash','parent-score'):
        p=base/('parent/parent/goal_decision_points.json.gz' if problem=='parent-score' else 'original/goal_operation_calibration_points.json.gz');v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        elif problem=='parent-score':
            for r in v:r['loss']+=.1
        else:v[0]['grouped_bin_error']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    else:
        name={'reader':'original/reader/PACKETS.json','reference':'original/evaluator/REFERENCES.json','summary':'original/SUMMARY.json','gap-summary':'original/SUMMARY.json','population':'parent/parent/PLAN.json','bins':'original/PLAN.json','operation-design':'original/PLAN.json','schema':'original/ARRAY_SCHEMA.json'}[problem]
        p=base/name;v=read(p)
        if problem=='reader':next(iter(v['packets'].values()))['inputs']['goal']=0
        elif problem=='reference':v[0]['frames'][0]['target'][0][1]=.5
        elif problem=='summary':v['contrasts'][0]['mean']+=.1
        elif problem=='gap-summary':v['gap_estimates'][0]['mean']+=.1
        elif problem=='population':v['design']['fit_seeds']=[-1]
        elif problem=='bins':v['design']['bin_edges'][1]=.11
        elif problem=='operation-design':v['design']['operations'].reverse()
        else:v['axes'].reverse()
        write(p,v,immutable=False)
        if problem in ('bins','operation-design'):plan['design']['target_plan_sha256']=file_digest(p)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):V.run(root,plan,lambda **kw:None)
