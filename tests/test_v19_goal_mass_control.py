"""Conditional goal mass matching must reproduce its accepted parents."""
import copy,gzip,json,shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_mass_control as F,witnessed_goal_factorization as W
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_witnessed_goal_factorization import fixture as witnessed_fixture


def fixture(root):
    plan=witnessed_fixture(root);base=root/'inputs';temp=root/'parent-execution'
    temp.mkdir();shutil.copytree(base,temp/'inputs');W.run(temp,plan,lambda **kw:None)
    write(base/'witnessed/PLAN.json',dict(design={k:v for k,v in plan['design'].items() if k!='input_files'}))
    for src,dst in [('witnessed_goal_points.json.gz','witnessed_goal_points.json.gz'),('evaluator/NATIVE_SCORES.json','NATIVE_SCORES.json')]:shutil.copyfile(temp/src,base/'witnessed'/dst)
    plan['design']['bootstrap_seed']=191010
    plan['design']['input_files']={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}
    return plan


def test_complete_handler_and_named_contrasts(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);summary=F.run(root,plan,lambda **kw:None)
    assert all(summary['controls'].values()) and summary['rows']==192 and summary['witnessed_cells_reproduced']==128
    assert summary['parent_max_error']<1e-10 and summary['packets']==1 and summary['inherited_native_rows']==6
    assert len(summary['contrasts'])==16 and len(summary['normalized_log_budget_area'])==8 and len(summary['means'])==24
    assert {r['arm'] for r in summary['contrasts']}=={a+s for a in F.ARMS for s in ('-mass-matched','-goal-product')}
    assert {r['baseline'] for r in summary['contrasts']}=={a+s for a in F.ARMS for s in ('-restricted','-mass-matched')}
    rows=json.loads(gzip.decompress((root/'goal_mass_points.json.gz').read_bytes()))
    for row in rows:assert abs(row['operation_accuracy']-1)<1e-12
    for p in (root/'masses').glob('*.npz'):
        with np.load(p,allow_pickle=False) as z:
            assert set(z.files)=={'goals','normalizer','original','product','operations'}
            assert np.array_equal(z['operations'],[172]) and z['goals'].shape==(1,3,3)


@pytest.mark.parametrize('problem',['restricted-score','product-score','missing','duplicate','hash','private','witness','native','reference','population'])
def test_corruption_rejected(tmp_path,problem):
    root=tmp_path/'packet';plan=fixture(root);base=root/'inputs';p=base/'witnessed/witnessed_goal_points.json.gz'
    if problem in ('restricted-score','product-score','missing','duplicate','hash'):
        v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        else:next(r for r in v if r['arm'].endswith('-restricted' if problem!='product-score' else '-goal-product'))['loss']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    else:
        if problem in ('private','witness'):
            p=base/'reader/PACKETS.json';v=read(p);packet=next(p for p in v['packets'].values() if p['tier']=='E2-full')
            if problem=='private':packet['inputs']['goal']=0
            else:packet['inputs']['observations'][0]['operation']='undo'
        elif problem=='reference':p=base/'evaluator/REFERENCES.json';v=read(p);next(r for r in v if r['tier']=='E2-full')['frames'][0]['target'][0][1]=.1
        elif problem=='native':p=base/'witnessed/NATIVE_SCORES.json';v=read(p);v[0]['loss']+=.1
        else:p=base/'witnessed/PLAN.json';v=read(p);v['design']['budgets']=[1,2]
        write(p,v,immutable=False)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):F.run(root,plan,lambda **kw:None)
