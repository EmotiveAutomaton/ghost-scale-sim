import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import retrospective_source as S,reachable_retrospective as Q
from test_v19_reachable_retrospective import fixture,law


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
def test_scalar_full_hypothesis_updates_and_every_forecast(mode):
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);W=np.array([[.6,.1,.2,.1],[0,.5,0,.5]]);p=law(mode)
    sources=[]
    for row,w in enumerate(W):
        for t in (1,2):
            for ctx in range(4):
                for old in range(8):
                    if sum(w[h]*p[s,ctx,old] for h,s in enumerate(st['past'][t-1]))>0:sources.append([row,t,ctx,old])
    raw=S.evaluate(W,st,p,sources);metrics=S.summarize(raw)
    for i,(row,t,ctx,old) in enumerate(sources):
        w=W[row];expected={k:np.zeros(5) for k in metrics}
        for ai,alpha in enumerate(S.ALPHAS):
            for e in range(8):
                likelihood=np.array([p[s,ctx,e] for s in st['past'][t-1]])
                direct=w*(alpha*(e==old)+(1-alpha)*likelihood);den=sum(direct)
                assert raw['correct_report_probability'][i,ai,e]==pytest.approx(den,abs=2e-15)
                if den==0:assert not raw['correct_possible'][i,ai,e];continue
                direct/=den
                for arm,lik in [('independent',likelihood),('copy',np.full(len(w),e==old))]:
                    rival=w*lik;mass=sum(rival)
                    if mass==0:expected[arm+'_support_failure_mass'][ai]+=den;continue
                    rival/=mass;expected[arm+'_supported_mass'][ai]+=den
                    group_diff=np.array([sum(direct[h]-rival[h] for h,g in enumerate(st['mapping']) if g==j) for j in range(len(st['signatures']))])
                    diffs=np.array([sum((direct[h]-rival[h])*p[st['future'][h,tau]] for h in range(len(w))) for tau in range(st['future'].shape[1])])
                    values=dict(group_tv=.5*abs(group_diff).sum(),max_future_difference=abs(diffs).max(),mean_future_tv=.5*abs(diffs).sum(-1).mean())
                    for k,v in values.items():
                        expected[arm+'_supported_error_mass_'+k][ai]+=den*v
                        expected[arm+'_max_supported_'+k][ai]=max(expected[arm+'_max_supported_'+k][ai],v)
        for k in metrics:np.testing.assert_allclose(metrics[k][i],expected[k],atol=3e-15,rtol=0)


def test_sources_deduplicate_only_matching_identity():
    a=dict(step=1,source_step=1,source_id='a',context=0,endpoint=1)
    b=dict(a,step=2);c=dict(step=3,source_step=3,source_id='b',context=0,endpoint=1)
    assert S.sources([a,b,c],3)==[('a',1,0,1),('b',3,0,1)]
    with pytest.raises(ValueError,match='content'):S.sources([a,dict(b,endpoint=2)],2)
    with pytest.raises(ValueError,match='missing original'):S.sources([a,dict(b,source_id='b')],2)
    with pytest.raises(ValueError,match='observation times'):S.sources([a,dict(b,step=3)],2)


def test_endpoint_controls_and_source_corruption():
    assert all(S.controls().values())
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([[.6,.1,.2,.1]]);p=law('random')
    a=S.evaluate(w,st,p,[[0,1,0,0]]);b=S.evaluate(w,st,p,[[0,2,0,0]])
    assert np.max(abs(a['independent_report_probability']-b['independent_report_probability']))>.001


@pytest.mark.parametrize('bad',['weight','nan','law','row','time','context','endpoint','impossible'])
def test_invalid_inputs_refused(bad):
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([[.6,.1,.2,.1]]);p=law('random');ids=[[0,1,0,0]]
    if bad=='weight':w[0,0]=-.1
    if bad=='nan':w[0,0]=np.nan
    if bad=='law':p[0,0,0]=-.1
    if bad=='row':ids[0][0]=1
    if bad=='time':ids[0][1]=3
    if bad=='context':ids[0][2]=4
    if bad=='endpoint':ids[0][3]=8
    if bad=='impossible':p=law('zeros')
    with pytest.raises(ValueError):S.evaluate(w,st,p,ids)


@pytest.mark.parametrize('corrupt',[False,True])
def test_complete_native_parent_integration(tmp_path,corrupt):
    import gzip,json,shutil
    from itertools import product
    from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
    from test_v19_sufficient_review import complete_fixture
    fixture_root,_=complete_fixture(tmp_path/'fixture')
    root=tmp_path/'source';shutil.copytree(fixture_root/'inputs/parent',root/'inputs')
    streams=[]
    for identity in product([1,2],range(16),('purpose','skill'),(False,True),(False,True)):
        a=dict(step=1,source_step=1,source_id='a',context=0,endpoint=0)
        b=dict(a,step=2) if identity[4] else dict(step=2,source_step=2,source_id='b',context=1,endpoint=1)
        streams.append(dict(draw=identity[0],maker=identity[1],kind=identity[2],switched=identity[3],duplicates=identity[4],length=4,observations=[a,b]))
    if corrupt:streams[0]['draw']=99
    (root/'inputs/aware/raw/1-observations_points.json.gz').write_bytes(gzip.compress(canonical(streams),mtime=0))
    design=dict(lineages=[1],draws=[1,2],lengths=[4],checkpoints=[2],alphas=list(S.ALPHAS),source_batch_size=64,input_files={p.relative_to(root/'inputs').as_posix():file_digest(p) for p in (root/'inputs').rglob('*') if p.is_file()})
    if corrupt:
        with pytest.raises(ValueError,match='stream binding'):S.run(root,dict(design=design),lambda **kw:None)
    else:
        result=S.run(root,dict(design=design),lambda **kw:None)
        assert result['posterior_rows']==512 and result['sources']==768 and result['report_queries']==30720
        assert all(result['controls'].values())
        rows=json.loads(gzip.decompress((root/'raw/source_summary_points.json.gz').read_bytes()))
        assert len(rows)==2560 and {r['alpha'] for r in rows}==set(S.ALPHAS)
