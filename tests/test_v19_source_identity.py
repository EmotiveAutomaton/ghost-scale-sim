import gzip
import json
import shutil
import math
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import source_identity as S, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_reachable_retrospective import fixture,law


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('sparse',[False,True])
def test_direct_scalar_source_marginalization_and_every_future_coordinate(mode,sparse):
    spec=fixture();st=Q.prepare(spec,[.2,.3,.4,.1]);w=np.array([.6,.1,.2,.1]);p=law(mode)
    if sparse:w=np.array([0.,.5,0.,.5])
    ids=[]
    for t,c in ((1,0),(2,1)):
        old=next(e for e in range(8) if sum(w[h]*p[s,c,e] for h,s in enumerate(st['past'][t-1]))>0)
        ids.append([t,c,old])
    raw=S.evaluate(w,st,p,ids);summary=S.summarize(raw)
    for ai,a in enumerate(S.ALPHAS):
        expected={k:0. for k in summary}
        for e in range(8):
            numer=[np.array([w[h]*(a*(e==old)+(1-a)*p[st['past'][t-1,h],c,e]) for h in range(len(w))]) for t,c,old in ids]
            mass=[math.fsum(n) for n in numer];den=math.fsum(mass)/len(ids)
            assert raw['report_probability'][ai,e]==pytest.approx(den,abs=2e-15)
            if den==0:
                assert not raw['possible'][ai,e];expected['impossible_reports']+=1;continue
            source=np.array(mass)/sum(mass);np.testing.assert_allclose(raw['source_posterior'][ai,:,e],source,atol=2e-15,rtol=0)
            exact=sum(numer)/(len(ids)*den);compatible=[n/m for n,m in zip(numer,mass) if m>0]
            uniform=sum(compatible)/len(compatible)
            expected['expected_source_entropy']+=den*(-sum(q*math.log(q) for q in source if q>0))
            expected['expected_incompatible_source_fraction']+=den*(1-len(compatible)/len(ids))
            # Saved coefficients are sufficient to rebuild the uniform posterior.
            reconstructed=np.zeros(len(w))
            for s,n in enumerate(numer):reconstructed+=raw['uniform_coefficients'][ai,s,e]*n
            np.testing.assert_allclose(reconstructed,uniform,atol=2e-15,rtol=0)
            for arm,rival in [('uniform',uniform),('recent',numer[-1]/mass[-1] if mass[-1]>0 else None)]:
                if rival is None:
                    expected[arm+'_support_failure_mass']+=den
                    assert np.isnan(raw[arm+'_group_tv'][ai,e]);continue
                expected[arm+'_supported_mass']+=den
                gd=np.array([sum(exact[h]-rival[h] for h,g in enumerate(spec['membership']) if g==j) for j in range(3)])
                forecasts=np.array([sum((exact[h]-rival[h])*p[st['future'][h,t]] for h in range(len(w))) for t in range(3)])
                values=dict(group_tv=.5*abs(gd).sum(),max_future_difference=abs(forecasts).max(),mean_future_tv=.5*abs(forecasts).sum(-1).mean())
                for k,v in values.items():
                    assert raw[arm+'_'+k][ai,e]==pytest.approx(v,abs=3e-15)
                    expected[arm+'_supported_error_mass_'+k]+=den*v
                    expected[arm+'_max_supported_'+k]=max(expected[arm+'_max_supported_'+k],v)
        for k,v in expected.items():assert summary[k][ai]==pytest.approx(v,abs=3e-15)


def test_identity_permutation_equal_likelihood_and_single_source():
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([.6,.1,.2,.1]);p=law('random');ids=[[1,0,0],[2,1,1]]
    a=S.evaluate(w,st,p,ids);b=S.evaluate(w,st,p,ids[::-1])
    for k in S.summarize(a):np.testing.assert_allclose(S.summarize(a)[k],S.summarize(b)[k],atol=3e-15,rtol=0)
    one=S.evaluate(w,st,p,ids[:1]);assert np.nanmax(one['uniform_group_tv'])<1e-14 and np.nanmax(one['recent_group_tv'])<1e-14
    # Same likelihood despite distinct identities, including both same old endpoints.
    p[:]=1/8;same=S.evaluate(w,st,p,[[1,0,0],[2,0,0]])
    assert np.nanmax(same['uniform_group_tv'])<1e-14
    np.testing.assert_allclose(same['source_posterior'][:4],.5,atol=1e-15,rtol=0)
    assert all(S.controls().values())


@pytest.mark.parametrize('bad',['weight','weight_nan','law','law_nan','empty','fractional','time','context','endpoint','duplicate','unsupported'])
def test_malformed_inputs_rejected(bad):
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([.6,.1,.2,.1]);p=law('random');ids=[[1,0,0],[2,1,1]]
    if bad=='weight':w[0]=-.1
    if bad=='weight_nan':w[0]=np.nan
    if bad=='law':p[0,0,0]=-.1
    if bad=='law_nan':p[0,0,0]=np.nan
    if bad=='empty':ids=[]
    if bad=='fractional':ids[0][0]=1.1
    if bad=='time':ids[0][0]=3
    if bad=='context':ids[0][1]=4
    if bad=='endpoint':ids[0][2]=8
    if bad=='duplicate':ids[1]=ids[0]
    if bad=='unsupported':p=law('zeros')
    with pytest.raises(ValueError):S.evaluate(w,st,p,ids)


def complete_source_fixture(tmp_path):
    from test_v19_sufficient_review import complete_fixture
    prior,_=complete_fixture(tmp_path/'parent');root=tmp_path/'identity'
    shutil.copytree(prior/'inputs/parent',root/'inputs')
    write(root/'inputs/PARENT_REVIEW.json',dict(numerical_acceptance=True))
    (root/'inputs/bindings').mkdir()
    for evidence in ('aware','omitted'):
        rows=json.loads(gzip.decompress((root/f'inputs/{evidence}/raw/1-forecasts_points.json.gz').read_bytes()));sources=[]
        for i,r in enumerate(rows):
            r['report_sources']=1 if r['duplicates'] else 2
            sources.append([i,1,0,0,'a'])
            if not r['duplicates']:sources.append([i,2,1,1,'b'])
        write(root/f'inputs/bindings/1-{evidence}-4-2-bindings.json',dict(rows=rows,sources=sources))
    design=dict(lineages=[1],draws=[1,2],batch_rows=32,alphas=list(S.ALPHAS),input_files={p.relative_to(root/'inputs').as_posix():file_digest(p) for p in (root/'inputs').rglob('*') if p.is_file()})
    return root,dict(design=design)


@pytest.mark.parametrize('corrupt',[None,'input_hash','pair','source_count','map','roster','forecast'])
def test_complete_native_parent_integration(tmp_path,corrupt):
    root,plan=complete_source_fixture(tmp_path)
    if corrupt:
        path=root/'inputs/bindings/1-omitted-4-2-bindings.json';d=read(path)
        if corrupt in ('input_hash','pair'):d['sources'][0][2]=2
        if corrupt=='source_count':d['rows'][0]['report_sources']+=1
        if corrupt=='map':d['rows'][0]['joint_row']=1
        if corrupt=='roster':d['rows'][0]['draw']=99
        if corrupt=='forecast':d['rows'][0]['forecast'][0][0]+=.01
        write(path,d,immutable=False)
        if corrupt!='input_hash':plan['design']['input_files'][path.relative_to(root/'inputs').as_posix()]=file_digest(path)
        with pytest.raises(ValueError):S.run(root,plan,lambda **kw:None)
    else:
        r=S.run(root,plan,lambda **kw:None)
        assert r['posterior_rows']==512 and r['sources']==768 and r['report_queries']==20480
        assert all(r['controls'].values()) and not r['numerical_acceptance']
        points=json.loads(gzip.decompress((root/'raw/source_identity_summary_points.json.gz').read_bytes()))
        assert len(points)==2560 and len(list((root/'raw').glob('*.npz')))==16
        assert all(abs(row['uniform_supported_mass']-1)<1e-12 for row in points)
