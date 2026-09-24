import gzip
import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import source_disclosure as S, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from test_v19_reachable_retrospective import fixture,law
from test_v19_source_identity import complete_source_fixture


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('sparse',[False,True])
def test_direct_scalar_bayes_proper_loss_and_every_future_coordinate(mode,sparse):
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([.6,.1,.2,.1]) if not sparse else np.array([0,.5,0,.5]);p=law(mode)
    ids=[]
    for t,c in ((1,0),(2,1)):
        old=next(e for e in range(8) if sum(w[h]*p[s,c,e] for h,s in enumerate(st['past'][t-1]))>0)
        ids.append([t,c,old])
    raw=S.evaluate(w,st,p,ids);summary=S.summarize(raw)
    for ai,a in enumerate(S.ALPHAS):
        expectation={k:0. for k in ('squared_forecast_change','brier_improvement','group_tv','max_future_difference','mean_future_tv')};entropy=0.
        for e in range(8):
            numer=[np.array([w[h]*(a*(e==old)+(1-a)*p[st['past'][t-1,h],c,e]) for h in range(len(w))]) for t,c,old in ids]
            mass=[math.fsum(n) for n in numer];total=math.fsum(mass)/len(ids)
            assert raw['report_probability'][ai,e]==pytest.approx(total,abs=3e-15)
            if total==0:
                assert not raw['possible'][ai,:,e].any();continue
            unknown=sum(numer)/(len(ids)*total)
            source=[v/sum(mass) for v in mass];entropy+=total*(-sum(v*math.log(v) for v in source if v))
            fq=np.array([sum(unknown[h]*p[st['future'][h,t]] for h in range(len(w))) for t in range(st['future'].shape[1])])
            for s,(n,m) in enumerate(zip(numer,mass)):
                if not m:
                    assert not raw['possible'][ai,s,e]
                    for k in expectation:assert np.isnan(raw[k][ai,s,e])
                    continue
                known=n/m
                truth=np.array([sum(known[h]*p[st['future'][h,t]] for h in range(len(w))) for t in range(st['future'].shape[1])])
                diff=truth-fq
                # Compute expected proper loss directly by summing every possible
                # future endpoint and its full one-hot outcome vector.
                before=after=0.
                for t in range(len(truth)):
                    for c in range(4):
                        for f in range(8):
                            one=np.eye(8)[f];pr=truth[t,c,f]
                            before+=pr*sum((fq[t,c]-one)**2)/(len(truth)*4)
                            after+=pr*sum((truth[t,c]-one)**2)/(len(truth)*4)
                gd=[sum(known[h]-unknown[h] for h,g in enumerate(fixture()['membership']) if g==j) for j in range(3)]
                values=dict(squared_forecast_change=(diff**2).sum(-1).mean(),brier_improvement=before-after,group_tv=.5*sum(abs(v) for v in gd),max_future_difference=abs(diff).max(),mean_future_tv=.5*abs(diff).sum(-1).mean())
                for k,v in values.items():
                    assert raw[k][ai,s,e]==pytest.approx(v,abs=4e-15)
                    expectation[k]+=m/len(ids)*v
        for k,v in expectation.items():assert summary['expected_'+k][ai]==pytest.approx(v,abs=4e-15)
        assert summary['expected_source_entropy_reduction'][ai]==pytest.approx(entropy,abs=4e-15)


def test_lossless_future_time_multiplicity_and_identities():
    spec=fixture();spec['length']=20;spec['signatures']=[s+[s[-1]]*16 for s in spec['signatures']]
    st=Q.prepare(spec,[.2,.3,.4,.1]);p=law('random');w=np.array([.6,.1,.2,.1]);ids=[[1,0,0],[2,1,1]]
    raw=S.evaluate(w,st,p,ids)
    assert raw['future_time_multiplicity'].sum()==19 and len(raw['future_time_multiplicity'])==2
    operator,inverse,counts=S.future_design(st,p)
    reconstructed=(np.cumsum(operator.toarray().reshape(4,-1,16),axis=1)@p.reshape(16,32)).reshape(4,-1,4,8)
    np.testing.assert_array_equal(reconstructed[:,inverse],p[st['future']])
    a=S.summarize(raw);b=S.summarize(S.evaluate(w,st,p,ids[::-1]))
    for k in a:np.testing.assert_allclose(a[k],b[k],atol=3e-15,rtol=0)
    single=S.summarize(S.evaluate(w,st,p,ids[:1]));assert single['expected_squared_forecast_change'].max()<1e-14
    assert single['expected_source_entropy_reduction'].max()==0
    p[:]=1/8;same=S.summarize(S.evaluate(w,st,p,[[1,0,0],[2,0,0]]))
    assert same['expected_squared_forecast_change'].max()<1e-14
    np.testing.assert_allclose(same['expected_source_entropy_reduction'],math.log(2),atol=3e-15,rtol=0)
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
        assert r['posterior_rows']==512 and r['sources']==768 and r['source_report_queries']==30720
        assert all(r['controls'].values()) and not r['numerical_acceptance']
        rows=json.loads(gzip.decompress((root/'raw/source_disclosure_summary_points.json.gz').read_bytes()))
        assert len(rows)==2560 and len(list((root/'raw').glob('*.npz')))==16
        assert all(abs(r['expected_squared_forecast_change']-r['expected_brier_improvement'])<1e-12 for r in rows)
