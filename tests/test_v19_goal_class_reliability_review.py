"""Scalar class reliability, one-hot expectation, complete rosters and corruptions."""
import copy
import gzip
import json
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_class_reliability_review as V
from ghostscale.validation.soundingline.v19 import goal_class_reliability as C
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_goal_class_reliability import fixture as producer_fixture


def fixture(root):
    original=root/'producer';original.mkdir(parents=True)
    plan=producer_fixture(original);write(original/'PLAN.json',plan)
    write(original/'SUMMARY.json',C.run(original,plan,lambda **kw:None))
    base=root/'inputs';base.mkdir();shutil.copytree(original/'inputs',base/'parent')
    dest=base/'original';dest.mkdir()
    for n in ('PLAN.json','SUMMARY.json','goal_class_reliability_points.json.gz','BRIER_DECOMPOSITION.json'):
        shutil.copyfile(original/n,dest/n)
    for n in ('reader','evaluator','reports'):shutil.copytree(original/n,dest/n)
    return dict(design=dict(target_plan_sha256=file_digest(dest/'PLAN.json'),input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}))


def test_complete_reconstruction_and_equal_class_population(tmp_path):
    root=tmp_path/'case';plan=fixture(root);r=V.run(root,plan,lambda **kw:None)
    assert r['passed'] and all(v<1e-10 for k,v in r.items() if k.startswith('max_'))
    assert r['rows']==2304 and r['parent_cells']==128 and r['native_rows']==36 and r['decomposition_cells']==768
    g=read(root/'INDEPENDENT_REGROUP.json')
    assert len(g['contrasts'])==1080 and len(g['normalized_log_budget_area'])==540 and len(g['means'])==288 and len(g['equal_class_means'])==96
    for row in g['equal_class_means']:
        assert row['signed_error']==pytest.approx(0,abs=1e-12)
        assert row['multiclass_brier']==pytest.approx(3*row['brier'])
        assert row['multiclass_brier']==pytest.approx(row['native_uncertainty']+row['squared_forecast_error'])
    rows=json.loads(gzip.decompress((root/'reconstructed_points.json.gz').read_bytes()));cfg=read(root/'inputs/original/PLAN.json')['design']
    for broken in (rows[:-1],rows+[rows[0]],[r for r in rows if r['goal']!=2]):
        with pytest.raises(ValueError,match='roster'):V.regroup(broken,cfg)


def test_explicit_multiclass_known_answer_and_permutation():
    assert all(V.controls().values())
    p=np.array([[.6,.35,.05],[.2,.3,.5]]);c=np.array([[.6,.2,.2],[.1,.4,.5]]);w=[.25,.75]
    rr,total=V.score_classes(p,c,w)
    direct=math.fsum(wi*ci*sum((pj-(j==k))**2 for j,pj in enumerate(pv)) for pv,cv,wi in zip(p,c,w) for k,ci in enumerate(cv))
    assert total['multiclass_brier']==pytest.approx(direct,abs=1e-14)
    own,ot=V.score_classes(c,c,w)
    assert ot['squared_forecast_error']==0 and ot['native_uncertainty']>0
    assert all(r['bin_error']==0 for r in own)
    perm=[2,0,1];pr,pt=V.score_classes(p[:,perm],c[:,perm],w)
    assert pt['multiclass_brier']==pytest.approx(total['multiclass_brier'])
    for i,j in enumerate(perm):assert pr[i]['signed_error']==rr[j]['signed_error']
    uniform,ut=V.score_classes([[1/3]*3],[[1/3]*3],[1])
    assert ut['multiclass_brier']==pytest.approx(2/3) and ut['squared_forecast_error']==0


@pytest.mark.parametrize('edge',range(1,10))
def test_boundaries(edge):
    p=edge/10
    assert [V.bin_id(x) for x in (np.nextafter(p,0),p,np.nextafter(p,1))]==[edge-1,edge,edge]


def test_cancellation_empty_bins_roundoff_and_zero_mass():
    r=V.measure([.61,.69],[.69,.61],[.5,.5])
    assert r['bin_error']==0 and r['squared_error']>0
    assert sum(x is None for x in r['bin_confidence'])==9
    rows,d=V.score_classes([[1+1e-12,0,0]],[[1,0,0]],[1])
    assert rows[0]['clipped_count']==1 and rows[0]['confidence']>1
    assert d['multiclass_brier']==d['squared_forecast_error']==0
    r=V.measure([0,1],[1,1],[0,1]);assert r['brier']==0
    r=V.measure([1,0],[0,1],[.25,.75]);assert r['brier']==1


@pytest.mark.parametrize('p,c,w',[([[.3,.3,.3]],[[.3,.3,.4]],[1]),([[1.1,0,0]],[[1,0,0]],[1]),([[.2,.3,.5]],[[.2,.2,.5]],[1]),([[.2,.3,.5]],[[.2,.3,.5]],[-1]),([[.2,.3,.5]],[[.2,.3,.5]],[.5]),([[np.nan,0,0]],[[1,0,0]],[1])])
def test_invalid_probabilities_and_weights(p,c,w):
    with pytest.raises(ValueError):V.score_classes(p,c,w)


@pytest.mark.parametrize('problem',['probability','marginal','operation','decision','saved-marginal','saved-native','bin','shape','score','empty-bin','missing','duplicate','hash','parent-score','reader','reference','summary','equal-class','population','bins','native','decomposition','decomposition-missing','decomposition-duplicate'])
def test_corruptions(tmp_path,problem):
    root=tmp_path/'case';plan=fixture(root);base=root/'inputs'
    if problem in ('probability','marginal','operation','decision','saved-marginal','saved-native','bin','shape'):
        parent=problem in ('probability','marginal','operation','decision')
        p=next((base/('parent/decisions' if parent else 'original/reports')).glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        field={'probability':'probabilities','marginal':'marginals','operation':'operations','decision':'coordinate','saved-marginal':'marginals','saved-native':'native_marginals','bin':'bin_ids','shape':'marginals'}[problem]
        if problem=='shape':v[field]=v[field][:0]
        else:v[field]=v[field]+.1
        np.savez_compressed(p,**v)
    elif problem in ('score','empty-bin','missing','duplicate','hash','parent-score'):
        p=base/('parent/parent/goal_decision_points.json.gz' if problem=='parent-score' else 'original/goal_class_reliability_points.json.gz');v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        elif problem=='empty-bin':v[0]['bin_confidence'][0]=.1
        elif problem=='parent-score':
            for row in v:row['loss']+=.1
        else:v[0]['brier']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    else:
        name={'reader':'original/reader/PACKETS.json','reference':'original/evaluator/REFERENCES.json','summary':'original/SUMMARY.json','equal-class':'original/SUMMARY.json','population':'parent/parent/PLAN.json','bins':'original/PLAN.json','native':'original/evaluator/NATIVE_CLASS_RELIABILITY.json','decomposition':'original/BRIER_DECOMPOSITION.json','decomposition-missing':'original/BRIER_DECOMPOSITION.json','decomposition-duplicate':'original/BRIER_DECOMPOSITION.json'}[problem]
        p=base/name;v=read(p)
        if problem=='reader':next(iter(v['packets'].values()))['inputs']['goal']=0
        elif problem=='reference':v[0]['frames'][0]['target'][0][1]=.5
        elif problem=='summary':v['contrasts'][0]['mean']+=.1
        elif problem=='equal-class':v['equal_class_means'][0]['brier']+=.1
        elif problem=='population':v['design']['fit_seeds']=[-1]
        elif problem=='bins':v['design']['bin_edges'][1]=.11
        elif problem=='decomposition':v[0]['direct_brier']+=.1
        elif problem=='decomposition-missing':v.pop()
        elif problem=='decomposition-duplicate':v.append(copy.deepcopy(v[0]))
        else:v[0]['brier']+=.1
        write(p,v,immutable=False)
        if problem=='bins':plan['design']['target_plan_sha256']=file_digest(p)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):V.run(root,plan,lambda **kw:None)
