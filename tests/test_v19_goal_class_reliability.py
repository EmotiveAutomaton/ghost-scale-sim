"""Known calibration, boundary/support behavior and complete lineage roster."""
import copy
import gzip
import json
import math
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_class_reliability as C
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_goal_abstention import fixture as decision_inputs


def fixture(root):
    plan=decision_inputs(root)
    plan['design'].update(bin_edges=C.EDGES.tolist(),bootstrap_seed=191015)
    return plan


def test_calibrated_chosen_goal_can_hide_other_class_error():
    assert all(C.controls().values())
    rows,full=C.score_classes([[.6,.35,.05]],[[.6,.2,.2]],[1.])
    assert rows[0]['squared_error']==0
    assert full['squared_forecast_error']==pytest.approx(.045)
    assert full['native_uncertainty']==pytest.approx(.56)
    assert full['multiclass_brier']==pytest.approx(.605)
    assert sum(r['signed_error'] for r in rows)==pytest.approx(0)
    assert rows[1]['signed_error']==pytest.approx(.15)
    assert rows[2]['signed_error']==pytest.approx(-.15)


def test_multiclass_loss_against_explicit_one_hot_outcomes_and_permutation():
    p=np.array([[.1,.3,.6],[.7,.2,.1]])
    c=np.array([[.2,.4,.4],[.5,.25,.25]]);w=[.25,.75]
    rr,total=C.score_classes(p,c,w)
    expected=math.fsum(wi*ci*sum((pj-(j==k))**2 for j,pj in enumerate(pv)) for pv,cv,wi in zip(p,c,w) for k,ci in enumerate(cv))
    assert total['multiclass_brier']==pytest.approx(expected,abs=1e-14)
    perm=[2,0,1];pr,pt=C.score_classes(p[:,perm],c[:,perm],w)
    assert pt['multiclass_brier']==pytest.approx(total['multiclass_brier'])
    for i,j in enumerate(perm):assert pr[i]['signed_error']==rr[j]['signed_error']
    own,ot=C.score_classes(c,c,w)
    assert ot['squared_forecast_error']==0 and ot['native_uncertainty']>0
    assert all(r['bin_error']==0 for r in own)


@pytest.mark.parametrize('edge',range(1,10))
def test_class_bin_boundaries(edge):
    x=edge/10
    assert C.bins(np.array([np.nextafter(x,0),x,np.nextafter(x,1)])).tolist()==[edge-1,edge,edge]


def test_empty_bins_cancellation_and_roundoff():
    r=C.measure([.61,.69],[.69,.61],[.5,.5])
    assert r['bin_error']==0 and r['squared_error']>0
    assert sum(x is None for x in r['bin_confidence'])==9
    rows,d=C.score_classes([[1+1e-12,0,0]],[[1,0,0]],[1])
    assert rows[0]['clipped_count']==1 and rows[0]['confidence']>1
    assert d['multiclass_brier']==d['squared_forecast_error']==0


@pytest.mark.parametrize('p,c,w',[([[.3,.3,.3]],[[.3,.3,.4]],[1]),([[1.1,0,0]],[[1,0,0]],[1]),([[.2,.3,.5]],[[.2,.2,.5]],[1]),([[.2,.3,.5]],[[.2,.3,.5]],[-1]),([[.2,.3,.5]],[[.2,.3,.5]],[.5]),([[np.nan,0,0]],[[1,0,0]],[1])])
def test_invalid_class_inputs(p,c,w):
    with pytest.raises(ValueError):C.score_classes(p,c,w)


def test_complete_handler_equal_class_means_and_roster(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=C.run(root,plan,lambda **kw:None)
    cfg=plan['design'];expected=math.prod(len(cfg[k]) for k in ('tiers','budgets','development_lineages','training_draws','fit_seeds'))*4*2*2*3*3
    assert s['rows']==expected and s['parent_cells']*18==expected
    assert s['decomposition_cells']*3==expected and s['max_brier_identity_error']<1e-10
    native=read(root/'evaluator/NATIVE_CLASS_RELIABILITY.json')
    assert len(native)==len(cfg['development_lineages'])*18
    assert all(r['signed_error']==r['bin_error']==r['squared_error']==0 for r in native)
    rows=json.loads(gzip.decompress((root/'goal_class_reliability_points.json.gz').read_bytes()))
    assert len(s['contrasts'])==len(cfg['budgets'])*2*2*3*3*3*5
    assert all(abs(r['multiclass_brier']-r['native_uncertainty']-r['squared_forecast_error'])<1e-10 for r in s['equal_class_means'])
    for r in s['equal_class_means']:
        assert r['multiclass_brier']==pytest.approx(3*r['brier'])
        assert r['signed_error']==pytest.approx(0,abs=1e-10)
    for broken in (rows[:-1],rows+[rows[0]],[r for r in rows if r['goal']!=2]):
        with pytest.raises(ValueError,match='roster'):C.aggregate(broken,cfg)


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
