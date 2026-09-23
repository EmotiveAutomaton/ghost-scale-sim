"""Independent abstention objectives, denominators and hostile saved records."""
import copy
import gzip
import json
import math
import shutil
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_abstention_review as V, goal_abstention as A
from ghostscale.validation.soundingline.v18_3.io import read, write, canonical, file_digest
from test_v19_goal_abstention import fixture as abstention_fixture, scalar_cost


def fixture(root):
    original=root/'producer';original.mkdir(parents=True)
    plan=abstention_fixture(original);write(original/'PLAN.json',plan)
    write(original/'SUMMARY.json',A.run(original,plan,lambda **kw:None))
    base=root/'inputs';base.mkdir();shutil.copytree(original/'inputs',base/'parent')
    dest=base/'original';dest.mkdir()
    for n in ('PLAN.json','SUMMARY.json','goal_abstention_points.json.gz'):shutil.copyfile(original/n,dest/n)
    for n in ('reader','evaluator','reports'):shutil.copytree(original/n,dest/n)
    return dict(design=dict(target_plan_sha256=file_digest(dest/'PLAN.json'),input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}))


def test_full_reconstruction_and_pooled_undefined(tmp_path):
    root=tmp_path/'case';plan=fixture(root);result=V.run(root,plan,lambda **kw:None)
    assert result['passed'] and result['rows']==1152 and result['native_rows']==18 and result['parent_cells']==128
    assert all(result[k]<1e-10 for k in result if k.startswith('max_'))
    groups=read(root/'INDEPENDENT_REGROUP.json')
    assert len(groups['contrasts'])==768 and len(groups['normalized_log_budget_area'])==384
    raw=json.loads(gzip.decompress((root/'reconstructed_points.json.gz').read_bytes()))
    for r in raw:
        if r['policy']!='always-coordinate':r.update(coverage=0.,incorrect=0.,correct=0.,conditional_error=None)
    changed=V.regroup(raw,read(root/'inputs/original/PLAN.json')['design'])
    assert all(r['conditional_error'] is None for r in changed['means'] if r['policy']!='always-coordinate')
    # Pooled conditional error is ratio of masses, never mean of conditional ratios.
    for r in raw:
        if r['policy']!='always-coordinate':
            r.update(coverage=.2 if r['lineage']==raw[0]['lineage'] else .8,incorrect=.1)
    changed=V.regroup(raw,read(root/'inputs/original/PLAN.json')['design'])
    assert all(abs(r['conditional_error']-.2)<1e-12 for r in changed['means'] if r['policy']!='always-coordinate')


def test_exhaustive_reports_and_sparse_scores():
    assert all(V.controls().values())
    q=np.arange(1,28,dtype=float);q/=q.sum();d,_=V.decide(q)
    ref=(np.array([0,5,26]),np.array([[.3,.2,.5],[.8,.1,.1]]),np.array([.2,.7]))
    for cost in V.COSTS:
        report=V.choose(q,d['marginals'],cost)
        assert abs(scalar_cost(q,report['step-abstention'],cost)-min(scalar_cost(q,p,cost) for p in product((-1,0,1,2),repeat=3)))<1e-12
        for p in product((-1,0,1,2),repeat=3):
            value=V.scores(q,p,ref,cost)
            for li in range(2):
                target=np.zeros(27);target[ref[0]]=ref[1][li]
                assert abs(value['cost_value'][li]-ref[2][li]*scalar_cost(target,p,cost))<1e-12
                support=any(all(v<0 or v==V.GOALS[k][t] for t,v in enumerate(p)) for k in ref[0])
                assert value['incompatible'][li]==ref[2][li]*int(not support and any(v>=0 for v in p))


def test_strict_threshold_and_modes():
    for p,expected in ((.75,-1),(np.nextafter(.75,1),0),(np.nextafter(.75,0),-1)):
        q=np.zeros(27);q[[0,13]]=[p,1-p];d,_=V.decide(q)
        selected=V.choose(q,d['marginals'],.25)
        assert selected['step-abstention']==(expected,)*3
        assert selected['joint-all-or-none']==(expected,)*3
    q=np.zeros(27);q[[0,13]]=.5;d,_=V.decide(q)
    assert V.choose(q,d['marginals'],.5)['always-coordinate']==(0,0,0)
    assert V.choose(q,d['marginals'],.5)['step-abstention']==(-1,-1,-1)
    q=np.array([math.prod(.8 if v==0 else .2 if v==1 else 0 for v in g) for g in V.GOALS]);d,_=V.decide(q)
    assert q.max()<.75 and V.choose(q,d['marginals'],.25)['joint-all-or-none']==(0,0,0)


@pytest.mark.parametrize('problem',['report','probability','native','shape','parent-decision','parent-marginal','score','ratio','missing','duplicate','hash','parent-score','reader','reference','summary','population','cost'])
def test_corruption(tmp_path,problem):
    root=tmp_path/'case';plan=fixture(root);base=root/'inputs'
    if problem in ('report','probability','native','shape','parent-decision','parent-marginal'):
        folder=base/('parent/decisions' if problem.startswith('parent-') else 'original/evaluator' if problem=='native' else 'original/reports')
        p=next(folder.glob('*.npz'))
        with np.load(p) as z:v={k:z[k] for k in z.files}
        if problem=='report':v['0.1-step-abstention'][0]=[2,2,2]
        elif problem=='shape':v['0.1-step-abstention']=v['0.1-step-abstention'][:0]
        elif problem=='parent-decision':v['coordinate'][0]=(v['coordinate'][0]+1)%27
        elif problem=='parent-marginal':v['marginals'][0,0,0]+=.1
        else:v['probabilities'][0,0]+=.1
        np.savez_compressed(p,**v)
    elif problem in ('score','ratio','missing','duplicate','hash','parent-score'):
        p=base/('parent/parent/goal_decision_points.json.gz' if problem=='parent-score' else 'original/goal_abstention_points.json.gz')
        v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        elif problem=='ratio':v[0]['conditional_error']=None
        else:v[0]['loss']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    elif problem=='reader':
        p=base/'original/reader/PACKETS.json';v=read(p);next(iter(v['packets'].values()))['inputs']['goal']=0;write(p,v,immutable=False)
    elif problem=='reference':
        p=base/'original/evaluator/REFERENCES.json';v=read(p);v[0]['frames'][0]['mass']=2;write(p,v,immutable=False)
    elif problem=='summary':
        p=base/'original/SUMMARY.json';v=read(p);v['contrasts'][0]['mean']+=.1;write(p,v,immutable=False)
    elif problem=='population':
        p=base/'parent/parent/PLAN.json';v=read(p);v['design']['fit_seeds']=[-1];write(p,v,immutable=False)
    else:
        p=base/'original/PLAN.json';v=read(p);v['design']['costs']=[.2];write(p,v,immutable=False);plan['design']['target_plan_sha256']=file_digest(p)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):V.run(root,plan,lambda **kw:None)
