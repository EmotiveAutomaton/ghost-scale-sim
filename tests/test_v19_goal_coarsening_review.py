"""Independent partitions and report costs, exhaustive controls and corruption."""
import copy
import gzip
import json
import math
import shutil
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import goal_coarsening_review as V, goal_coarsening as C
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_goal_coarsening import fixture as producer_fixture


def fixture(root):
    original=root/'producer';original.mkdir(parents=True)
    plan=producer_fixture(original);write(original/'PLAN.json',plan)
    write(original/'SUMMARY.json',C.run(original,plan,lambda **kw:None))
    base=root/'inputs';base.mkdir();shutil.copytree(original/'inputs',base/'parent')
    dest=base/'original';dest.mkdir()
    for n in ('PLAN.json','SUMMARY.json','goal_coarsening_points.json.gz'):shutil.copyfile(original/n,dest/n)
    for n in ('reader','evaluator','reports'):shutil.copytree(original/n,dest/n)
    return dict(design=dict(target_plan_sha256=file_digest(dest/'PLAN.json'),input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}))


def test_full_reconstruction_and_undefined_pooled_ratios(tmp_path):
    root=tmp_path/'case';plan=fixture(root);r=V.run(root,plan,lambda **kw:None)
    assert r['passed'] and r['rows']==7680 and r['native_rows']==120 and r['parent_cells']==768
    assert all(r[k]<1e-10 for k in r if k.startswith('max_'))
    g=read(root/'INDEPENDENT_REGROUP.json')
    assert len(g['contrasts'])==3840 and len(g['normalized_log_budget_area'])==1920 and len(g['means'])==960
    raw=json.loads(gzip.decompress((root/'reconstructed_points.json.gz').read_bytes()))
    cfg=read(root/'inputs/original/PLAN.json')['design']
    for row in raw:
        if row['policy']=='abstain':row.update(coverage=0.,incorrect=0.,alternatives=0.,conditional_error=None,alternatives_per_claim=None)
    g=V.regroup(raw,cfg)
    assert all(row['conditional_error'] is None and row['alternatives_per_claim'] is None for row in g['means'] if row['policy']=='abstain')
    for row in raw:
        if row['policy']=='abstain':row.update(coverage=.2 if row['lineage']==raw[0]['lineage'] else .8,incorrect=.1,alternatives=1.)
    g=V.regroup(raw,cfg)
    assert all(abs(row['conditional_error']-.2)<1e-12 and abs(row['alternatives_per_claim']-2)<1e-12 for row in g['means'] if row['policy']=='abstain')


def test_all_partial_reports_native_support_and_specificity():
    assert all(V.controls().values())
    q=np.arange(1,28,dtype=float)[None];q/=q.sum()
    truth=np.zeros((2,1,27));truth[0,0,[0,13]]=[.2,.8];truth[1,0,[1,26]]=[.7,.3]
    for part,mapping in V.PARTITIONS.items():
        g=V.geometry(part);p=V.project(q,g);table=V.tables(truth,np.ones((2,1)),g)
        assert abs(p.sum()-1)<1e-14
        for r in g['reports']:
            v=V.score(q,p,np.array([r]),table,.25);cov=sum(x>=0 for x in r)/3
            for li in range(2):
                correct=math.fsum(float(truth[li,0,k])*sum(x>=0 and x==mapping[goal[t]] for t,x in enumerate(r))/3 for k,goal in enumerate(V.GOALS))
                possible=any(truth[li,0,k]>0 and all(x<0 or x==mapping[goal[t]] for t,x in enumerate(r)) for k,goal in enumerate(V.GOALS))
                assert abs(v['correct'][li]-correct)<1e-14
                assert abs(v['cost_value'][li]-(cov-correct+.25*(1-cov)))<1e-14
                assert v['incompatible'][li]==int(not possible and cov>0)
                assert abs(v['alternatives'][li]-sum(mapping.count(x) for x in r if x>=0)/3)<1e-14


def test_boundaries_modes_and_one_class():
    g=V.geometry('fine')
    for p,expected in ((.75,-1),(np.nextafter(.75,1),0),(np.nextafter(.75,0),-1)):
        q=np.zeros((1,27));q[0,[0,13]]=[p,1-p];m=V.marginals(q,V.GOALS,3)
        assert np.all(V.choose(m,m,g,'coarse-mode',.25,'abstain')==expected)
    q=np.zeros((1,27));q[0,[0,13]]=.5;m=V.marginals(q,V.GOALS,3)
    assert np.all(V.choose(m,m,g,'coarse-mode',.5,'always')==0)
    assert np.all(V.choose(m,m,g,'coarse-mode',.5,'abstain')==-1)
    g=V.geometry('one-class');p=V.project(q,g);cm=V.marginals(p,g['paths'],1)
    assert np.all(V.choose(m,cm,g,'coarse-mode',.1,'abstain')==0)


@pytest.mark.parametrize('problem',['report','probability','marginal','native','shape','score','ratio','specificity','missing','duplicate','hash','parent-score','reader','reference','summary','population','partition','cost'])
def test_corruption(tmp_path,problem):
    root=tmp_path/'case';plan=fixture(root);base=root/'inputs'
    if problem in ('report','probability','marginal','native','shape'):
        folder=base/('original/evaluator' if problem=='native' else 'original/reports');p=next(folder.glob('*.npz'))
        with np.load(p) as z:v={k:z[k] for k in z.files}
        if problem=='report':v['meaning-coarse-mode-0.1-always'][0]=(v['meaning-coarse-mode-0.1-always'][0]+1)%2
        elif problem=='shape':v['fine-coarse-mode-0.1-always']=v['fine-coarse-mode-0.1-always'][:0]
        elif problem=='marginal':v['meaning-marginals'][0,0,0]+=.1
        else:v['meaning-probabilities'][0,0]+=.1
        np.savez_compressed(p,**v)
    elif problem in ('score','ratio','specificity','missing','duplicate','hash','parent-score'):
        p=base/('parent/parent/goal_abstention_points.json.gz' if problem=='parent-score' else 'original/goal_coarsening_points.json.gz')
        v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        elif problem=='duplicate':v.append(copy.deepcopy(v[0]))
        elif problem=='ratio':v[0]['conditional_error']=None
        elif problem=='specificity':v[0]['alternatives']+=.1
        elif problem=='parent-score':
            next(row for row in v if row['policy']=='always-coordinate')['loss']+=.1
        else:v[0]['coarse_loss']+=.1
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
        p=base/'original/PLAN.json';v=read(p)
        if problem=='cost':v['design']['costs']=[.2]
        else:v['design']['partitions']['meaning']=[0,1,2]
        write(p,v,immutable=False);plan['design']['target_plan_sha256']=file_digest(p)
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):V.run(root,plan,lambda **kw:None)
