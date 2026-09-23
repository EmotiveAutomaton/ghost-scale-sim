"""Known calibration, boundary/support behavior and complete lineage roster."""
import copy
import gzip
import json
import math
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_calibration as C
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_goal_abstention import fixture as decision_inputs


def fixture(root):
    plan=decision_inputs(root)
    plan['design'].update(bin_edges=C.EDGES.tolist(),bootstrap_seed=191014)
    return plan


def test_known_answers_scalar_scores_and_reversal():
    assert all(C.controls().values())
    p=np.array([.6,.9]);c=np.array([.2,.4]);w=np.array([.25,.75])
    r=C.measure(p,c,w)
    assert abs(r['signed_error']-.475)<1e-14
    assert abs(r['bin_error']-.475)<1e-14
    assert abs(r['squared_error']-(.25*.4**2+.75*.5**2))<1e-14
    expected=math.fsum(wi*(ci*(1-pi)**2+(1-ci)*pi*pi) for pi,ci,wi in zip(p,c,w))
    assert abs(r['brier']-expected)<1e-14
    logloss=math.fsum(-wi*(ci*math.log(pi)+(1-ci)*math.log1p(-pi)) for pi,ci,wi in zip(p,c,w))
    assert abs(r['binary_loss']-logloss)<1e-14
    reversed=C.measure(c,p,w)
    assert reversed['signed_error']==-r['signed_error']
    assert reversed['squared_error']==r['squared_error']
    assert reversed['binary_loss']!=r['binary_loss']


def test_same_bin_cancellation_and_native_uncertainty():
    r=C.measure([.61,.69],[.69,.61],[.5,.5])
    assert r['bin_error']==0 and r['squared_error']>0
    own=C.measure([.61,.69],[.61,.69],[.5,.5])
    assert own['bin_error']==own['squared_error']==0 and own['brier']>0
    assert sum(v is None for v in own['bin_confidence'])==9
    assert abs(own['brier']-sum(x*(1-x)/2 for x in (.61,.69)))<1e-14


@pytest.mark.parametrize('edge',list(range(1,10)))
def test_exact_and_adjacent_bin_boundaries(edge):
    x=edge/10
    assert C.bins(np.array([np.nextafter(x,0),x,np.nextafter(x,1)])).tolist()==[edge-1,edge,edge]


def test_endpoints_clipping_impossible_events_and_zero_weight():
    assert C.bins(np.array([0.,1.,1.+1e-12])).tolist()==[0,9,9]
    r=C.measure([1.+1e-12],[1.],[1.])
    assert r['binary_loss']==0 and r['clipped_count']==1 and r['confidence']>1
    for p,c in ((0,1),(1,0)):
        r=C.measure([p],[c],[1])
        assert r['binary_loss'] is None and r['infinite_loss'] and r['infinite_mass']==1
        assert r['brier']==1
        r=C.measure([p,.5],[c,.5],[0,1])
        assert not r['infinite_loss'] and r['infinite_mass']==0
        assert abs(r['binary_loss']-math.log(2))<1e-14


@pytest.mark.parametrize('p,c,w',[([-1e-15],[0],[1]),([1+1e-8],[1],[1]),([np.nan],[0],[1]),([.5],[1.1],[1]),([.5],[.5],[-1]),([.5],[.5],[.5])])
def test_invalid_probability_and_weights(p,c,w):
    with pytest.raises(ValueError):C.measure(p,c,w)


def test_complete_handler_native_calibration_loss_identity_and_bootstrap(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=C.run(root,plan,lambda **kw:None)
    cfg=plan['design'];expected=math.prod(len(cfg[k]) for k in ('tiers','budgets','development_lineages','training_draws','fit_seeds'))*4*2*2*3
    assert s['rows']==expected and s['parent_cells']*6==expected
    assert s['parent_max_error']<1e-10 and all(s['controls'].values())
    native=read(root/'evaluator/NATIVE_CALIBRATION.json')
    assert len(native)==len(cfg['development_lineages'])*6
    assert all(r['signed_error']==r['bin_error']==r['squared_error']==0 for r in native)
    rows=json.loads(gzip.decompress((root/'goal_calibration_points.json.gz').read_bytes()))
    assert all(abs(sum(r['bin_weight'])-1)<1e-10 for r in rows)
    assert all(sum(r['bin_confidence_sum'])==pytest.approx(r['confidence']) for r in rows)
    assert len(s['contrasts'])==len(cfg['budgets'])*2*2*3*3*5
    broken=copy.deepcopy(rows);broken.pop()
    with pytest.raises(ValueError,match='roster'):C.aggregate(broken,cfg)
    # Undefined paired proper losses cannot acquire a finite bootstrap result.
    for r in rows:
        if r['arm']=='learned-bank':r['binary_loss']=None
    a=C.aggregate(rows,cfg)
    assert all(not r['defined'] and r['nonfinite_pairs']>0 for r in a['contrasts'] if r['metric']=='binary_loss')


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
    with pytest.raises(ValueError):C.run(root,plan,lambda **kw:None)
