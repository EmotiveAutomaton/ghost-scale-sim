import gzip
import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import source_prior as S, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from test_v19_reachable_retrospective import fixture,law
from test_v19_source_identity import complete_source_fixture


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('sparse',[False,True])
def test_scalar_bayes_and_one_hot_proper_loss(mode,sparse):
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([.6,.1,.2,.1]) if not sparse else np.array([0,.5,0,.5]);p=law(mode);ids=[]
    for t,c in ((1,0),(2,1)):
        old=next(e for e in range(8) if sum(w[h]*p[s,c,e] for h,s in enumerate(st['past'][t-1]))>0);ids.append([t,c,old])
    raw=S.evaluate(w,st,p,ids);summary=S.summarize(raw);priors=[[.5,.5],[1/3,2/3],[2/3,1/3]]
    np.testing.assert_allclose(raw['source_prior'],priors,atol=1e-15,rtol=0)
    for ai,a in enumerate(S.ALPHAS):
        expected={n:{k:0. for k in ('group_tv','max_future_difference','mean_future_tv','squared_forecast_difference','brier_regret')} for n in S.PRIORS[1:]}
        for e in range(8):
            numer=[np.array([w[h]*(a*(e==old)+(1-a)*p[st['past'][t-1,h],c,e]) for h in range(len(w))]) for t,c,old in ids]
            mass=[math.fsum(x) for x in numer];post=[];forecasts=[]
            for i,prior in enumerate(priors):
                den=math.fsum(x*y for x,y in zip(prior,mass))
                assert raw['report_probability'][i,ai,e]==pytest.approx(den,abs=3e-15)
                if not den:
                    assert not raw['possible'][i,ai,e];continue
                v=sum(x*n for x,n in zip(prior,numer))/den;post.append(v)
                np.testing.assert_allclose(raw['source_posterior'][i,ai,:,e],[x*y/den for x,y in zip(prior,mass)],atol=3e-15,rtol=0)
                forecasts.append(np.array([sum(v[h]*p[st['future'][h,t]] for h in range(len(w))) for t in range(st['future'].shape[1])]))
            if not post:
                for name in expected:
                    for key in expected[name]:assert np.isnan(raw[name+'_'+key][ai,e])
                continue
            for i,name in enumerate(S.PRIORS[1:],1):
                delta=forecasts[0]-forecasts[i];gd=[sum(post[0][h]-post[i][h] for h,g in enumerate(fixture()['membership']) if g==j) for j in range(3)]
                before=after=0.
                for t in range(len(delta)):
                    for c in range(4):
                        for f in range(8):
                            one=np.eye(8)[f];pr=forecasts[i][t,c,f]
                            before+=pr*sum((forecasts[0][t,c]-one)**2)/(len(delta)*4)
                            after+=pr*sum((forecasts[i][t,c]-one)**2)/(len(delta)*4)
                values=dict(group_tv=.5*sum(abs(v) for v in gd),max_future_difference=abs(delta).max(),mean_future_tv=.5*abs(delta).sum(-1).mean(),squared_forecast_difference=(delta*delta).sum(-1).mean(),brier_regret=before-after)
                for key,value in values.items():
                    assert raw[name+'_'+key][ai,e]==pytest.approx(value,abs=4e-15)
                    expected[name][key]+=raw['report_probability'][i,ai,e]*value
        for name in expected:
            for key,value in expected[name].items():assert summary[name+'_expected_'+key][ai]==pytest.approx(value,abs=4e-15)


def test_permutation_single_source_equal_likelihood_and_controls():
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([.6,.1,.2,.1]);p=law('random');ids=[[1,0,0],[2,1,1]]
    a=S.summarize(S.evaluate(w,st,p,ids));b=S.summarize(S.evaluate(w,st,p,ids[::-1]))
    for k in a:np.testing.assert_allclose(a[k],b[k],atol=3e-15,rtol=0)
    one=S.summarize(S.evaluate(w,st,p,ids[:1]));assert one['recency_expected_group_tv'].max()<1e-14
    p[:]=1/8;same=S.summarize(S.evaluate(w,st,p,[[1,0,0],[2,0,0]]));assert same['early_expected_group_tv'].max()<1e-14
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
        assert r['posterior_rows']==512 and r['sources']==768 and r['report_queries']==40960
        assert all(r['controls'].values()) and not r['numerical_acceptance']
        rows=json.loads(gzip.decompress((root/'raw/source_prior_summary_points.json.gz').read_bytes()))
        assert len(rows)==2560 and len(list((root/'raw').glob('*.npz')))==16
        assert all(abs(r['recency_expected_squared_forecast_difference']-r['recency_expected_brier_regret'])<1e-12 for r in rows)
