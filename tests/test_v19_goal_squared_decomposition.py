"""Known answers and pairwise variance independently test the residual identity."""
import copy
import gzip
import json
import math
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_squared_decomposition as O
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_goal_abstention import fixture as decision_inputs


def fixture(root):
    p=decision_inputs(root)
    p['design'].update(bin_edges=O.EDGES.tolist(),bootstrap_seed=191020)
    return p


def test_known_signed_and_opposing_residuals_native_self():
    assert all(O.controls().values())
    p=np.tile([.15,.85,0.],(2,3,1));c=p[None].copy()
    c[0,0,:,:2]+=[.05,-.05];c[0,1,:,:2]+=[-.05,.05]
    x=O.sufficient(p,c,np.array([[.5,.5]]))
    assert np.max(x['squared_bias'])<1e-28
    assert x['within_variance'][0,0,0]==pytest.approx(.0025)
    # Unequal weights give nonzero mean and variation in the same bin.
    y=O.sufficient(p,c,np.array([[.25,.75]]))
    assert y['squared_bias'][0,0,0]==pytest.approx(.000625)
    assert y['within_variance'][0,0,0]==pytest.approx(.001875)


def test_scalar_moments_pairwise_variance_and_permutations():
    rng=np.random.default_rng(191020)
    p=rng.dirichlet([1,2,3],(19,3));c=rng.dirichlet([3,2,1],(2,19,3));w=rng.dirichlet(np.ones(19),2)
    result=O.sufficient(p,c,w)
    for law,t,g,b in product(range(2),range(3),range(3),range(10)):
        ix=[i for i in range(19) if sum(float(p[i,t,g])>=j/10 for j in range(1,10))==b]
        mass=math.fsum(float(w[law,i]) for i in ix)
        residual={i:float(p[i,t,g])-float(c[law,i,t,g]) for i in ix}
        first=math.fsum(float(w[law,i])*residual[i] for i in ix)
        second=math.fsum(float(w[law,i])*residual[i]**2 for i in ix)
        key=law,t,g,b
        for k,v in [('bin_weight',mass),('bin_residual_sum',first),('bin_residual_squared_sum',second)]:
            assert result[k][key]==pytest.approx(v,abs=1e-14)
        if mass:
            # Ordered-pair identity does not subtract two near-equal moments.
            pair=math.fsum(float(w[law,i])*float(w[law,j])*(residual[i]-residual[j])**2 for i,j in product(ix,ix))/(2*mass)
            assert result['bin_variance_mass'][key]==pytest.approx(pair,abs=1e-14)
        else:
            assert math.isnan(result['bin_residual_mean'][key])
            assert result['bin_bias_mass'][key]==result['bin_variance_mass'][key]==0
    for v in [O.sufficient(p[::-1],c[:,::-1],w[:,::-1]),O.sufficient(p[:,:,::-1],c[:,:,:,::-1],w)]:
        for m in O.METRICS:
            expected=result[m] if v is not None and np.allclose(v['squared_error'],result['squared_error']) else result[m][...,::-1]
            assert np.allclose(v[m],expected,rtol=0,atol=1e-14)
    assert np.allclose(result['squared_bias']+result['within_variance'],result['squared_error'],rtol=0,atol=1e-14)
    assert np.allclose(result['bin_weight'].sum(-1),1,rtol=0,atol=1e-14)


@pytest.mark.parametrize('edge',range(1,10))
def test_boundaries_and_probability_one(edge):
    x=edge/10
    assert O.bins(np.array([0,np.nextafter(x,0),x,np.nextafter(x,1),1])).tolist()==[0,edge-1,edge,edge,9]


def test_empty_bins_zero_weight_and_point_mass():
    p=np.tile([0.,1.,0.],(2,3,1));c=p[None].copy();c[0,1]=[.5,.5,0]
    r=O.sufficient(p,c,np.array([[1.,0.]]))
    for k in ('bin_residual_mean','bin_residual_second_moment','bin_residual_variance'):
        assert np.isnan(r[k][r['bin_weight']==0]).all()
    assert r['bin_weight'][0,0,1,9]==1
    assert all(np.max(abs(r[k]))==0 for k in O.METRICS)


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


def test_complete_handler_roster_parent_identity(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=O.run(root,plan,lambda **kw:None);cfg=plan['design']
    rows=json.loads(gzip.decompress((root/'goal_squared_decomposition_points.json.gz').read_bytes()))
    assert s['rows']==len(rows)==s['parent_cells']*18
    assert s['max_pooled_error']<1e-10
    assert len(s['contrasts'])==len(cfg['budgets'])*2*2*3*3*3*len(O.METRICS)
    assert len(s['component_estimates'])==len(s['means'])*2
    for f in (root/'groups').glob('*.npz'):
        with np.load(f,allow_pickle=False) as z:
            assert z['bin_weight'].shape==(len(cfg['development_lineages']),2,3,3,10)
            assert np.isnan(z['bin_residual_mean'][z['bin_weight']==0]).all()
            assert np.allclose(z['squared_bias']+z['within_variance'],z['squared_error'],atol=1e-12,rtol=0)
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError,match='roster'):O.aggregate(bad,cfg)
    wrong=copy.deepcopy(cfg);wrong['budgets'].reverse()
    with pytest.raises(ValueError,match='budget order'):O.aggregate(rows,wrong)


@pytest.mark.parametrize('problem',['hash','missing','duplicate','parent-score','reader','reference','mass','population','bins','probability','marginal','operation','decision'])
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
    else:
        p=next((base/'decisions').glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        key={'probability':'probabilities','marginal':'marginals','operation':'operations','decision':'coordinate'}[problem]
        v[key]=v[key]+.1;np.savez_compressed(p,**v)
    if p is not None and problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):O.run(root,plan,lambda **kw:None)
