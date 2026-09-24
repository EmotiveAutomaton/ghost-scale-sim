import gzip
import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import source_review as R, retrospective_source as S, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import write, canonical, file_digest
from test_v19_reachable_retrospective import fixture, law


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('sparse',[False,True])
def test_all_factors_and_every_future_coordinate(mode,sparse):
    spec=fixture();st=R.structure(spec);W=np.array([[.6,.1,.2,.1],[0,.5,0,.5]])
    if sparse:W[0]=[1.,0,0,0]
    p=law(mode);ids=[]
    for row,w in enumerate(W):
        for t in (1,2):
            for c in range(4):
                for e in range(8):
                    if sum(w[h]*p[s,c,e] for h,s in enumerate(st['past'][t-1]))>0:ids.append([row,t,c,e])
    raw=S.evaluate(W,Q.prepare(spec,[.2,.3,.4,.1]),p,ids)
    result=R.verify_batch(W,st,p,ids,raw);expected=S.summarize(raw)
    for name in result:np.testing.assert_allclose(result[name],expected[name],atol=1e-14,rtol=0)
    for row,t,c,e in ids:
        numerator=np.array([W[row,h]*p[s,c,e] for h,s in enumerate(st['past'][t-1])])
        d=numerator/numerator.sum()-W[row]
        futures=R.future_mass(d[None,:],st)[0]@p.reshape(16,32)
        direct=np.array([sum(d[h]*p[st['signatures'][st['mapping'][h],tau]] for h in range(len(d))) for tau in range(st['signatures'].shape[1])])
        np.testing.assert_allclose(futures.reshape(-1,4,8),direct,atol=1e-14,rtol=0)


@pytest.mark.parametrize('field',['source_rows','independent_report_probability','correct_report_probability','copy_fraction','independent_possible','copy_possible','correct_possible','independent_vs_copy_group_tv','independent_vs_copy_max_future_difference','independent_vs_copy_mean_future_tv'])
def test_corrupt_factors_refused(field):
    spec=fixture();W=np.array([[.6,.1,.2,.1]]);p=law('random');ids=[[0,1,0,0]]
    raw=S.evaluate(W,Q.prepare(spec,[.2,.3,.4,.1]),p,ids)
    if raw[field].dtype==bool:raw[field].flat[0]=not raw[field].flat[0]
    elif field=='source_rows':raw[field].flat[0]+=1
    else:raw[field].flat[0]+=.01
    with pytest.raises(ValueError):R.verify_batch(W,R.structure(spec),p,ids,raw)


def test_distinct_sources_and_duplicates():
    a=dict(step=1,source_step=1,source_id='a',context=0,endpoint=1)
    b=dict(a,step=2);c=dict(step=3,source_step=3,source_id='b',context=0,endpoint=1)
    assert R.source_inventory([a,b,c],3)==[('a',1,0,1),('b',3,0,1)]
    assert R.source_inventory([a,b,c],2)==[('a',1,0,1)]
    for bad in (dict(b,endpoint=2),dict(b,source_id='b'),dict(b,step=3)):
        with pytest.raises(ValueError):R.source_inventory([a,bad],2)
    assert all(R.controls().values())


def complete_fixture(tmp_path):
    from test_v19_retrospective_source import test_complete_native_parent_integration
    test_complete_native_parent_integration(tmp_path,False)
    original=tmp_path/'source'
    design=dict(lineages=[1],draws=[1,2],lengths=[4],checkpoints=[2],alphas=list(S.ALPHAS),source_batch_size=64,input_files={p.relative_to(original/'inputs').as_posix():file_digest(p) for p in (original/'inputs').rglob('*') if p.is_file()})
    write(original/'PLAN.json',dict(design=design))
    write(original/'SUMMARY.json',dict(rows=2560,posterior_rows=512,sources=768,report_queries=30720,unavailable_checkpoints=[]))
    root=tmp_path/'review';(root/'inputs').mkdir(parents=True)
    shutil.copytree(original,root/'inputs/original');shutil.copytree(original/'inputs',root/'inputs/parent')
    return root,dict(design=dict(target_plan_sha256=file_digest(original/'PLAN.json'),input_files={}))


def test_complete_independent_review(tmp_path):
    root,plan=complete_fixture(tmp_path);r=R.run(root,plan,lambda **kw:None)
    assert r['passed'] and not r['numerical_acceptance']
    assert r['rows']==2560 and r['sources']==768 and r['strata']==20 and r['batches']==12


@pytest.mark.parametrize('bad',['timing','summary','binding','stream','raw_roster','denominator'])
def test_complete_corruption_refused(tmp_path,bad):
    root,plan=complete_fixture(tmp_path);o=root/'inputs/original'
    if bad=='timing':(o/'TIMING.jsonl').write_text('')
    elif bad=='summary':
        p=o/'raw/source_summary_points.json.gz';rows=json.loads(gzip.decompress(p.read_bytes()));rows[0]['copy_supported_mass']+=.01;p.write_bytes(gzip.compress(canonical(rows),mtime=0))
    elif bad=='binding':
        p=o/'evaluator/1-aware-4-2-bindings.json';x=json.loads(p.read_text());x['sources'][0][1]=2;p.write_text(json.dumps(x))
    elif bad=='stream':
        p=root/'inputs/parent/aware/raw/1-observations_points.json.gz';x=json.loads(gzip.decompress(p.read_bytes()));x[0]['observations'][0]['context']=3;p.write_bytes(gzip.compress(canonical(x),mtime=0))
    elif bad=='raw_roster':np.savez(o/'raw/unbound_points.npz',extra=[1])
    else:
        p=o/'raw/source_summary_points.json.gz';rows=json.loads(gzip.decompress(p.read_bytes()));rows[0]['report_sources']+=1;p.write_bytes(gzip.compress(canonical(rows),mtime=0))
    with pytest.raises((ValueError,KeyError)):R.run(root,plan,lambda **kw:None)
