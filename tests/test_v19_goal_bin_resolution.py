"""Independent scalar sums, nested boundaries, and known calibration answers."""
import copy
import gzip
import json
import math
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_bin_resolution as O
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_goal_abstention import fixture as decision_inputs


def fixture(root):
    p=decision_inputs(root)
    p['design'].update(resolutions=[5,10,20],bin_edges={str(n):O.EDGES[n].tolist() for n in O.RESOLUTIONS},bootstrap_seed=191019)
    return p


def scalar(p,c,w):
    shape=(len(c),3,3,35)
    out={k:np.zeros(shape) for k in ('bin_weight','bin_forecast_sum','bin_correct_sum')}
    for law,t,g,ri in product(range(len(c)),range(3),range(3),range(3)):
        n=O.RESOLUTIONS[ri]
        for b in range(n):
            ix=[i for i in range(len(p)) if sum(float(p[i,t,g])>=j/n for j in range(1,n))==b]
            k=law,t,g,O.OFFSETS[ri]+b
            out['bin_weight'][k]=math.fsum(float(w[law,i]) for i in ix)
            out['bin_forecast_sum'][k]=math.fsum(float(w[law,i])*float(p[i,t,g]) for i in ix)
            out['bin_correct_sum'][k]=math.fsum(float(w[law,i])*float(c[law,i,t,g]) for i in ix)
    return out


def test_known_opposite_and_identical_errors_and_native_self():
    assert all(O.controls().values())
    p=np.repeat(np.array([[.11,.89,0.],[.16,.84,0.]])[:,None,:],3,axis=1)
    c=p[None].copy();c[0,0,:,:2]+=[.05,-.05];c[0,1,:,:2]+=[-.05,.05]
    w=np.array([[.5,.5]]);x=O.sufficient(p,c,w)
    assert x['bin_error_5'][0,0,0]==pytest.approx(0,abs=1e-14)
    assert x['bin_error_10'][0,0,0]==pytest.approx(0,abs=1e-14)
    assert x['bin_error_20'][0,0,0]==pytest.approx(.05)
    assert x['squared_error'][0,0,0]==pytest.approx(.0025)
    same=p[None].copy();same[0,:,:,:2]+=[.05,-.05]
    y=O.sufficient(p,same,w)
    assert np.max(abs(y['increment_10_5']))<1e-14
    assert np.max(abs(y['increment_20_10']))<1e-14
    assert np.max(y['bin_error_20'])==pytest.approx(.05)


def test_scalar_sums_permutations_conservation_and_coarsening():
    rng=np.random.default_rng(191019);p=rng.dirichlet([1,2,3],(17,3));c=rng.dirichlet([2,1,3],(2,17,3));w=rng.dirichlet(np.ones(17),2)
    r=O.sufficient(p,c,w)
    for k,v in scalar(p,c,w).items():
        assert np.allclose(v,r[k],atol=1e-14,rtol=0)
        assert np.allclose(v[...,5:15].reshape(2,3,3,5,2).sum(-1),v[...,:5],atol=1e-14,rtol=0)
        assert np.allclose(v[...,15:].reshape(2,3,3,10,2).sum(-1),v[...,5:15],atol=1e-14,rtol=0)
    frame=O.sufficient(p[::-1],c[:,::-1],w[:,::-1])
    classes=O.sufficient(p[:,:,::-1],c[:,:,:,::-1],w)
    for m in O.METRICS:
        assert np.allclose(frame[m],r[m],atol=1e-14,rtol=0)
        assert np.allclose(classes[m][...,::-1],r[m],atol=1e-14,rtol=0)
    for ri in range(3):assert np.allclose(r['bin_weight'][...,O.OFFSETS[ri]:O.OFFSETS[ri+1]].sum(-1),1)
    for m in O.INCREMENTS:assert np.min(r[m])>=-1e-14


@pytest.mark.parametrize('n,edge',[(n,i) for n in (5,10,20) for i in range(1,n)])
def test_exact_and_adjacent_boundaries(n,edge):
    x=edge/n;p=np.array([0.,np.nextafter(x,0),x,np.nextafter(x,1),1.])
    assert O.bins(p,n).tolist()==[0,edge-1,edge,edge,n-1]
    assert np.array_equal(O.bins(p,20)//2,O.bins(p,10))
    assert np.array_equal(O.bins(p,10)//2,O.bins(p,5))


def test_empty_means_probability_one_and_faulty_assignment(monkeypatch):
    p=np.tile([0.,1.,0.],(2,3,1));w=np.array([[1.,0.]])
    x=O.sufficient(p,p[None],w)
    assert np.isnan(x['bin_forecast_mean'][x['bin_weight']==0]).all()
    assert np.isnan(x['bin_correct_mean'][x['bin_weight']==0]).all()
    for ri,n in enumerate(O.RESOLUTIONS):assert x['bin_weight'][0,0,1,O.OFFSETS[ri]+n-1]==1
    original=O.bins
    def wrong(p,n):
        v=original(p,n)
        if n==10:v=np.zeros_like(v)
        return v
    monkeypatch.setattr(O,'bins',wrong)
    with pytest.raises(ValueError,match='nested'):O.sufficient(p,p[None],w)


@pytest.mark.parametrize('problem',['probability','target','weight','conservation','nan','shape'])
def test_invalid_inputs(problem):
    p=np.tile([.6,.3,.1],(2,3,1));c=p[None].copy();w=np.array([[.5,.5]])
    if problem=='probability':p[0,0,0]=1.1
    elif problem=='target':c[0,0,0,0]=-1
    elif problem=='weight':w*=2
    elif problem=='conservation':p[0,0,0]=.5
    elif problem=='nan':w[0,0]=np.nan
    else:w=w[:,:1]
    with pytest.raises(ValueError):O.sufficient(p,c,w)


def test_complete_handler_roster_and_fixed_unbinned_score(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=O.run(root,plan,lambda **kw:None);cfg=plan['design']
    rows=json.loads(gzip.decompress((root/'goal_bin_resolution_points.json.gz').read_bytes()))
    assert s['rows']==len(rows)==s['parent_cells']*18
    assert s['max_pooled_error']<1e-10
    assert len(s['contrasts'])==len(cfg['budgets'])*2*2*3*3*3*len(O.METRICS)
    assert len(s['increment_estimates'])==len(s['means'])*2
    for f in (root/'groups').glob('*.npz'):
        with np.load(f,allow_pickle=False) as z:
            assert z['bin_weight'].shape==(len(cfg['development_lineages']),2,3,3,35)
            assert np.isnan(z['bin_forecast_mean'][z['bin_weight']==0]).all()
    for f in (root/'forecasts').glob('*.npz'):
        with np.load(f,allow_pickle=False) as z:
            assert np.array_equal(z['bin_ids'][0],z['bin_ids'][1]//2)
            assert np.array_equal(z['bin_ids'][1],z['bin_ids'][2]//2)
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError,match='roster'):O.aggregate(bad,cfg)


@pytest.mark.parametrize('problem',['hash','missing','duplicate','parent-score','reader','reference','mass','population','bins','resolutions','probability','marginal','operation','decision'])
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
    elif problem=='bins':plan['design']['bin_edges']['10'][1]=.11
    elif problem=='resolutions':plan['design']['resolutions'].reverse()
    else:
        p=next((base/'decisions').glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        key={'probability':'probabilities','marginal':'marginals','operation':'operations','decision':'coordinate'}[problem]
        v[key]=v[key]+.1;np.savez_compressed(p,**v)
    if p is not None and problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):O.run(root,plan,lambda **kw:None)
