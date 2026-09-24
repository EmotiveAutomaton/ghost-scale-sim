import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import aggregate_precision as S
from test_v19_aggregate_report_state import inputs


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('sparse',[False,True])
def test_scalar_bayes_quantization_and_proper_loss(mode,sparse):
    st,w,law,ids=inputs(mode,sparse);raw=S.evaluate(w,st,law,ids);summary=S.summarize(raw)
    for vi,(dtype,repair) in enumerate(S.VARIANTS):
        q=np.array([float(x) for x in raw['group_mass'].astype(dtype)])
        j=np.array([[float(x) for x in row] for row in raw['joint_report_mass'].astype(dtype)])
        if repair=='mass-preserving':
            q/=math.fsum(q)
            j=np.array([row*q[g]/math.fsum(row) if math.fsum(row)>0 else row for g,row in enumerate(j)])
        for ai,a in enumerate(S.ALPHAS):
            for e in range(8):
                num=np.array([a*raw['copy_counts'][e]/len(ids)*q[g]+(1-a)*j[g,e] for g in range(len(q))])
                mass=math.fsum(num)
                assert raw['approximate_report_probability'][vi,ai,e]==pytest.approx(mass,abs=3e-16)
                if not raw['defined'][vi,ai,e]:continue
                truth=np.array([math.fsum(w[h]*(a*int(old==e)+(1-a)*law[st['past'][t-1,h],c,e])/len(ids) for t,c,old in ids) for h in range(len(w))]);truth/=math.fsum(truth)
                squared=[];maximum=[]
                for t in range(st['future'].shape[1]):
                    for c in range(4):
                        target=sum(truth[h]*law[st['future'][h,t],c] for h in range(len(w)))
                        pred=sum(num[g]/mass*law[st['signatures'][g,t],c] for g in range(len(q)))
                        regret=math.fsum(target[e2]*(sum((pred-np.eye(8)[e2])**2)-sum((target-np.eye(8)[e2])**2)) for e2 in range(8))
                        squared.append(regret);maximum.append(max(abs(pred-target)))
                assert raw['future_squared_error'][vi,ai,e]==pytest.approx(np.mean(squared),abs=1e-15)
                assert raw['max_future_probability_error'][vi,ai,e]==pytest.approx(max(maximum),abs=1e-15)
        assert raw['state_bytes'][vi]==9*len(q)*np.dtype(dtype).itemsize+32
        assert (raw['group_mass'] if dtype=='float64' else raw['stored_group_'+dtype]).dtype==np.dtype(dtype)
    np.testing.assert_allclose(sum(summary[f'stratum_{s}_mass'] for s in range(3)),1,rtol=0,atol=1e-14)
    np.testing.assert_allclose(sum(summary[f'stratum_{s}_squared_error_sum'] for s in range(3)),summary['defined_weighted_future_squared_error'],rtol=0,atol=1e-14)


def test_underflow_is_preserved_and_positive_empty_row_fails():
    tiny=np.float16(6e-8)
    q=np.array([1,tiny],dtype='float16');j=np.array([[.125]*8,[float(tiny)/8]*8],dtype='float16')
    rq,rj,failed=S.reconstruct(q,j,'mass-preserving')
    assert failed and rq[1]>0 and not rj[1].any()
    q=np.array([1,1e-9],dtype='float16')
    rq,rj,failed=S.reconstruct(q,j,'mass-preserving')
    assert not failed and rq[1]==0 and not rj[1].any()


def test_support_lost_is_not_zero_error():
    st,w,law,ids=inputs('uniform');law[:,:,:]=0;law[:,:,0]=1-1e-9;law[:,:,1]=1e-9;ids[:,2]=0
    raw=S.evaluate(w,st,law,ids);s=S.summarize(raw)
    assert raw['reference_possible'][0,1] and not raw['approximate_possible'][4,0,1]
    assert np.isnan(raw['future_squared_error'][4,0,1])
    assert s['unusable_report_probability'][4,0]>0
    assert raw['report_strata'][0,1]==0
    assert all(S.controls().values())


def test_source_endpoint_group_permutations():
    st,w,law,ids=inputs();a=S.evaluate(w,st,law,ids);b=S.evaluate(w,st,law,ids[::-1])
    for k in S.summarize(a):np.testing.assert_allclose(S.summarize(a)[k],S.summarize(b)[k],atol=1e-14,rtol=0)
    law=law[:,:,::-1];ids[:,2]=7-ids[:,2];b=S.evaluate(w,st,law,ids)
    np.testing.assert_allclose(a['future_squared_error'],b['future_squared_error'][:,:,::-1],atol=1e-14,rtol=0)
    q,j=a['group_mass'],a['joint_report_mass'];qp,jp,f=S.reconstruct(q[::-1].astype('float16'),j[::-1].astype('float16'),'mass-preserving')
    p,m,post=S.query(qp,jp,a['copy_counts']);np.testing.assert_allclose(p,a['approximate_report_probability'][5],atol=1e-14,rtol=0)


@pytest.mark.parametrize('field',['defined','reference_possible','approximate_possible','report_strata','future_squared_error'])
def test_masks_and_strata_corruption_rejected(field):
    st,w,law,ids=inputs('zeros');raw=S.evaluate(w,st,law,ids)
    if raw[field].dtype==bool:raw[field].flat[0]=not raw[field].flat[0]
    else:raw[field].flat[0]=99
    with pytest.raises(ValueError):S.summarize(raw)


def test_complete_native_integration(tmp_path):
    from test_v19_source_identity import complete_source_fixture
    import gzip,json
    root,plan=complete_source_fixture(tmp_path)
    plan['design'].update(report_state='precision-uniform-source-joint-group-endpoint',storage_dtypes=list(S.DTYPES),reconstruction_modes=list(S.MODES))
    result=S.run(root,plan,lambda **kw:None)
    assert result['posterior_rows']==512 and result['sources']==768
    assert all(result['controls'].values()) and not result['numerical_acceptance']
    rows=json.loads(gzip.decompress((root/'raw/aggregate_precision_summary_points.json.gz').read_bytes()))
    assert len(rows)==15360 and len(list((root/'raw').glob('*.npz')))==16
    assert {(r['storage_dtype'],r['reconstruction']) for r in rows}==set(S.VARIANTS)
