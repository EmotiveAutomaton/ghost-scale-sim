"""Known group cancellation and scalar sufficient-sum fixtures."""
import copy
import gzip
import json
import math
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_operation_calibration as O
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_goal_abstention import fixture as decision_inputs


def fixture(root):
    p=decision_inputs(root)
    p['design'].update(bin_edges=O.C.EDGES.tolist(),bootstrap_seed=191016,operations=list(O.D.S.L.OPERATIONS))
    return p


def scalar(p,c,w,ops):
    shape=(len(c),3,3,6,10);out={k:np.zeros(shape) for k in ('bin_weight','bin_forecast_sum','bin_correct_sum')}
    for law,t,g,op,b in product(range(len(c)),range(3),range(3),range(6),range(10)):
        ids=[i for i in range(len(p)) if ops[i,t]==op and min(sum(float(p[i,t,g])>=j/10 for j in range(1,10)),9)==b]
        out['bin_weight'][law,t,g,op,b]=math.fsum(float(w[law,i]) for i in ids)
        out['bin_forecast_sum'][law,t,g,op,b]=math.fsum(float(w[law,i])*float(p[i,t,g]) for i in ids)
        out['bin_correct_sum'][law,t,g,op,b]=math.fsum(float(w[law,i])*float(c[law,i,t,g]) for i in ids)
    return out


def test_opposite_error_hidden_by_pooling_and_native_self():
    assert all(O.controls().values())
    p=np.tile([.6,.3,.1],(2,3,1));c=p[None].copy();c[0,0,:,0]+=.1;c[0,0,:,1]-=.1;c[0,1,:,0]-=.1;c[0,1,:,1]+=.1
    w=np.array([[.5,.5]]);ops=np.array([[0]*3,[1]*3]);r=O.sufficient(p,c,w,ops)
    assert r['grouped_bin_error'][0,0,0]==pytest.approx(.1)
    assert r['pooled_bin_error'][0,0,0]==pytest.approx(0,abs=1e-14)
    assert r['conditional_signed_error'][0,0,0,:2]==pytest.approx([-.1,.1])
    assert np.isnan(r['bin_forecast_mean'][r['bin_weight']==0]).all()
    same=O.sufficient(p,np.repeat(c[:,:1],2,axis=1),w,ops)
    assert np.max(abs(same['cancellation_gap']))<1e-14


def test_all_sums_against_scalar_and_group_frame_class_permutations():
    rng=np.random.default_rng(191016);p=rng.dirichlet([1,2,3],(17,3));c=rng.dirichlet([2,1,3],(2,17,3));w=rng.dirichlet(np.ones(17),2);ops=rng.integers(6,size=(17,3))
    r=O.sufficient(p,c,w,ops)
    for k,v in scalar(p,c,w,ops).items():assert np.allclose(v,r[k],atol=1e-14,rtol=0)
    for permuted in (O.sufficient(p[::-1],c[:,::-1],w[:,::-1],ops[::-1]),O.sufficient(p,c,w,5-ops),O.sufficient(p[:,:,::-1],c[:,:,:,::-1],w,ops)):
        assert np.sum(permuted['grouped_bin_error'])==pytest.approx(np.sum(r['grouped_bin_error']))
        assert np.sum(permuted['pooled_bin_error'])==pytest.approx(np.sum(r['pooled_bin_error']))
    assert np.allclose(r['operation_mass'].sum(3),1)
    assert np.min(r['cancellation_gap'])>=-1e-14


@pytest.mark.parametrize('edge',range(1,10))
def test_exact_adjacent_edges(edge):
    x=edge/10;pp=np.array([np.nextafter(x,0),x,np.nextafter(x,1)])
    p=np.repeat(np.stack([pp,1-pp,np.zeros(3)],axis=1)[:,None,:],3,axis=1)
    r=O.sufficient(p,p[None],np.full((1,3),1/3),np.zeros((3,3),int))
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
    with pytest.raises(ValueError):O.sufficient(p,c,w,ops)


def test_complete_handler_roster_and_all_empty_groups_retained(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=O.run(root,plan,lambda **kw:None);cfg=plan['design']
    rows=json.loads(gzip.decompress((root/'goal_operation_calibration_points.json.gz').read_bytes()))
    assert s['rows']==len(rows)==s['parent_cells']*18
    assert s['max_pooled_error']<1e-10
    assert len(s['contrasts'])==len(cfg['budgets'])*2*2*3*3*3*3
    for f in (root/'groups').glob('*.npz'):
        with np.load(f,allow_pickle=False) as z:
            assert z['bin_weight'].shape==(len(cfg['development_lineages']),2,3,3,6,10)
            assert np.isnan(z['bin_forecast_mean'][z['bin_weight']==0]).all()
            assert np.allclose(z['bin_weight'].sum((-1,-2)),1)
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError,match='roster'):O.aggregate(bad,cfg)


@pytest.mark.parametrize('problem',['hash','missing','duplicate','parent-score','reader','reference','mass','population','bins','operation-design','probability','marginal','operation','decision'])
def test_corruptions(tmp_path,problem):
    root=tmp_path/'packet';plan=fixture(root);base=root/'inputs';p=None
    if problem in ('hash','missing','duplicate','parent-score'):
        p=base/'parent/goal_decision_points.json.gz';v=json.loads(gzip.decompress(p.read_bytes()))
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
    elif problem=='bins':plan['design']['bin_edges'][1]=.11
    elif problem=='operation-design':plan['design']['operations'].reverse()
    else:
        p=next((base/'decisions').glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        key={'probability':'probabilities','marginal':'marginals','operation':'operations','decision':'coordinate'}[problem]
        v[key]=v[key]+.1;np.savez_compressed(p,**v)
    if p is not None and problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):O.run(root,plan,lambda **kw:None)
