"""Independent calibration reconstruction, boundary controls and corruptions."""
import copy
import gzip
import json
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_calibration_review as V,goal_calibration as C
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_goal_calibration import fixture as producer_fixture


def fixture(root):
    original=root/'producer';original.mkdir(parents=True)
    plan=producer_fixture(original);write(original/'PLAN.json',plan)
    write(original/'SUMMARY.json',C.run(original,plan,lambda **kw:None))
    base=root/'inputs';base.mkdir();shutil.copytree(original/'inputs',base/'parent')
    dest=base/'original';dest.mkdir()
    for n in ('PLAN.json','SUMMARY.json','goal_calibration_points.json.gz'):shutil.copyfile(original/n,dest/n)
    for n in ('reader','evaluator','reports'):shutil.copytree(original/n,dest/n)
    return dict(design=dict(target_plan_sha256=file_digest(dest/'PLAN.json'),input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}))


def test_complete_independent_reconstruction_and_nonfinite_pairs(tmp_path):
    root=tmp_path/'case';plan=fixture(root);r=V.run(root,plan,lambda **kw:None)
    assert r['passed'] and all(v<1e-10 for k,v in r.items() if k.startswith('max_'))
    assert r['rows']==768 and r['parent_cells']==128 and r['native_rows']==12
    g=read(root/'INDEPENDENT_REGROUP.json')
    assert len(g['contrasts'])==360 and len(g['normalized_log_budget_area'])==180 and len(g['means'])==96
    rows=json.loads(gzip.decompress((root/'reconstructed_points.json.gz').read_bytes()));cfg=read(root/'inputs/original/PLAN.json')['design']
    for row in rows:
        if row['arm']=='learned-bank':row['binary_loss']=None
    g=V.regroup(rows,cfg)
    assert all(not r['defined'] and r['nonfinite_pairs']>0 for r in g['contrasts'] if r['metric']=='binary_loss')


def test_scalar_known_scores_and_pooled_cancellation():
    assert all(V.controls().values())
    r=V.measure([.6,.9],[.2,.4],[.25,.75])
    assert abs(r['signed_error']-.475)<1e-14 and abs(r['squared_error']-.2275)<1e-14
    assert abs(r['brier']-(.25*(.2*.4**2+.8*.6**2)+.75*(.4*.1**2+.6*.9**2)))<1e-14
    assert abs(r['binary_loss']-(-.25*(.2*math.log(.6)+.8*math.log(.4))-.75*(.4*math.log(.9)+.6*math.log(.1))))<1e-14
    r=V.measure([.61,.69],[.69,.61],[.5,.5]);assert r['bin_error']==0 and r['squared_error']>0
    own=V.measure([.61,.69],[.61,.69],[.5,.5]);assert own['squared_error']==0 and own['brier']>0
    assert sum(x is None for x in own['bin_confidence'])==9


@pytest.mark.parametrize('edge',range(1,10))
def test_boundaries(edge):
    p=edge/10
    assert [V.bin_id(x) for x in (np.nextafter(p,0),p,np.nextafter(p,1))]==[edge-1,edge,edge]


def test_support_clipping_and_zero_weights():
    for p,c in ((0,1),(1,0)):
        r=V.measure([p],[c],[1]);assert r['binary_loss'] is None and r['infinite_mass']==1 and r['brier']==1
        r=V.measure([p,.5],[c,.5],[0,1]);assert not r['infinite_loss'] and abs(r['binary_loss']-math.log(2))<1e-14
    r=V.measure([1+1e-12],[1],[1]);assert r['clipped_count']==1 and r['clipped_mass']==1 and r['binary_loss']==0 and r['confidence']>1
    for p,c,w in (([-1e-15],[0],[1]),([1+1e-8],[1],[1]),([np.nan],[0],[1]),([.5],[1.1],[1]),([.5],[.5],[-1]),([.5],[.5],[.5])):
        with pytest.raises(ValueError):V.measure(p,c,w)


@pytest.mark.parametrize('problem',['chosen','confidence','correctness','bin','shape','score','empty-bin','infinite','missing','duplicate','hash','parent-score','reader','reference','summary','population','bins','native','marginal','operation'])
def test_corruptions(tmp_path,problem):
    root=tmp_path/'case';plan=fixture(root);base=root/'inputs'
    if problem in ('chosen','confidence','correctness','bin','shape','marginal','operation'):
        folder=base/('parent/decisions' if problem in ('marginal','operation') else 'original/reports');p=next(folder.glob('*.npz'))
        with np.load(p,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        field={'chosen':'chosen','confidence':'confidence','correctness':'native_correctness','bin':'bin_ids','shape':'confidence','marginal':'marginals','operation':'operations'}[problem]
        if problem=='shape':v[field]=v[field][:0]
        else:v[field]=v[field]+.1
        np.savez_compressed(p,**v)
    elif problem in ('score','empty-bin','infinite','missing','duplicate','hash','parent-score'):
        p=base/('parent/parent/goal_decision_points.json.gz' if problem=='parent-score' else 'original/goal_calibration_points.json.gz');v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        elif problem=='empty-bin':v[0]['bin_confidence'][0]=.1
        elif problem=='infinite':v[0]['infinite_loss']=not v[0]['infinite_loss']
        elif problem=='parent-score':
            for row in v:row['loss']+=.1
        else:v[0]['brier']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    else:
        name={'reader':'original/reader/PACKETS.json','reference':'original/evaluator/REFERENCES.json','summary':'original/SUMMARY.json','population':'parent/parent/PLAN.json','bins':'original/PLAN.json','native':'original/evaluator/NATIVE_CALIBRATION.json'}[problem]
        p=base/name;v=read(p)
        if problem=='reader':next(iter(v['packets'].values()))['inputs']['goal']=0
        elif problem=='reference':v[0]['frames'][0]['target'][0][1]=.5
        elif problem=='summary':v['contrasts'][0]['mean']+=.1
        elif problem=='population':v['design']['fit_seeds']=[-1]
        elif problem=='bins':v['design']['bin_edges'][1]=.11
        else:v[0]['brier']+=.1
        write(p,v,immutable=False)
        if problem=='bins':plan['design']['target_plan_sha256']=file_digest(p)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):V.run(root,plan,lambda **kw:None)
