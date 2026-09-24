import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import source_interval as I, retrospective_source as S, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import write, read, file_digest
from test_v19_reachable_retrospective import fixture, law


@pytest.mark.parametrize('mode', ['random', 'uniform', 'zeros', 'disjoint'])
@pytest.mark.parametrize('sparse', [False, True])
def test_direct_bayes_all_coordinates_and_interior_alphas(mode, sparse):
    spec=fixture(); st=Q.prepare(spec,[.2,.3,.4,.1]); p=law(mode)
    W=np.array([[.6,.1,.2,.1],[0,.5,0,.5]])
    if sparse: W[0]=[1,0,0,0]
    ids=[]
    for row,w in enumerate(W):
        for t in (1,2):
            for context in range(4):
                for old in range(8):
                    if sum(w[h]*p[s,context,old] for h,s in enumerate(st['past'][t-1]))>0: ids.append([row,t,context,old])
    raw=S.evaluate(W,st,p,ids); result=I.factors(raw)
    future=p[st['future']].reshape(len(st['mapping']),-1)
    for n,(row,t,c,old) in enumerate(ids):
        prior=W[row]; likelihood=p[st['past'][t-1],c]
        original=prior@future
        independent=prior*likelihood[:,old]; independent/=independent.sum()
        delta=independent@future-original
        for j,(lo,hi) in enumerate(I.INTERVALS):
            for endpoint in range(8):
                forecasts=[]; probs=[]
                for alpha in np.linspace(lo,hi,9):
                    l=(1-alpha)*likelihood[:,endpoint]+alpha*(endpoint==old)
                    probability=prior@l; probs.append(probability)
                    if probability>0:forecasts.append(((prior*l)/probability)@future)
                assert result['report_possible'][n,j,endpoint]==bool(forecasts)
                np.testing.assert_allclose(result['report_probability_lower'][n,j,endpoint],min(probs),atol=1e-14,rtol=0)
                np.testing.assert_allclose(result['report_probability_upper'][n,j,endpoint],max(probs),atol=1e-14,rtol=0)
                if not forecasts: continue
                if endpoint==old:
                    candidates=np.array([original+result['independent_weight_lower'][n,j]*delta,original+result['independent_weight_upper'][n,j]*delta])
                    np.testing.assert_allclose(np.min(forecasts,axis=0),candidates.min(0),atol=1e-14,rtol=0)
                    np.testing.assert_allclose(np.max(forecasts,axis=0),candidates.max(0),atol=1e-14,rtol=0)
                    width=np.max(np.ptp(candidates,axis=0))
                    np.testing.assert_allclose(result['old_endpoint_width_max_future_difference'][n,j],width,atol=1e-14,rtol=0)
                    midpoint=forecasts[4]
                    radius=max(np.max(abs(f-midpoint)) for f in forecasts)
                    np.testing.assert_allclose(result['old_endpoint_midpoint_worst_max_future_difference'][n,j],radius,atol=1e-14,rtol=0)
                else:
                    np.testing.assert_allclose(np.array(forecasts),np.repeat([forecasts[0]],len(forecasts),axis=0),atol=1e-14,rtol=0)


def example():
    return dict(source_rows=np.array([[0,1,0,0]]),independent_report_probability=np.array([[.2,.8,0,0,0,0,0,0]]),independent_vs_copy_group_tv=np.array([.8]),independent_vs_copy_max_future_difference=np.array([.4]),independent_vs_copy_mean_future_tv=np.array([.6]))


def test_degenerate_intervals_support_and_endpoint_decisions():
    raw=example();f=I.factors(raw,((1,1),(0,0),(.5,.5)));m=I.measures(f)
    assert f['report_possible'][0,0].tolist()==[True,False,False,False,False,False,False,False]
    assert (f['old_endpoint_width_group_tv']==0).all()
    f=I.factors(raw);m=I.measures(f)
    assert m['upper_point_unsupported_fraction'][0,0]==.5
    assert m['impossible_endpoints'][0,0]==6
    assert m['equal_supported_endpoint_old_endpoint_width_max_future_difference'][0,0]==.2
    assert (f['midpoint_possible']==f['report_possible']).all()
    assert all(I.controls().values())


@pytest.mark.parametrize('bad', [[], [0,1], [(1,0)], [(-.1,.5)], [(0,1.1)], [(float('nan'),1)], [(0,float('inf'))]])
def test_invalid_intervals_refused(bad):
    with pytest.raises(ValueError):I.factors(example(),bad)


@pytest.mark.parametrize('bad', ['negative','mass','nan','old_zero','bad_id','float_id','distance'])
def test_invalid_source_factors_refused(bad):
    r=example()
    if bad=='negative':r['independent_report_probability'][0,2]=-.1
    elif bad=='mass':r['independent_report_probability'][0,0]=.3
    elif bad=='nan':r['independent_report_probability'][0,0]=float('nan')
    elif bad=='old_zero':r['source_rows'][0,3]=2
    elif bad=='bad_id':r['source_rows'][0,3]=8
    elif bad=='float_id':r['source_rows']=r['source_rows'].astype(float)
    else:r['independent_vs_copy_max_future_difference'][0]=1.5
    with pytest.raises(ValueError):I.factors(r)


def complete_fixture(tmp_path):
    from test_v19_source_review import complete_fixture as source_fixture
    source_fixture(tmp_path)
    source=tmp_path/'source';p=source/'PLAN.json'
    write(source/'COMPLETE.json',dict(plan_sha256=file_digest(p),files={f.relative_to(source).as_posix():file_digest(f) for f in (source/'raw').glob('*')}))
    write(source/'FINAL_REVIEW.json',dict(numerical_acceptance=True))
    root=tmp_path/'interval';(root/'inputs').mkdir(parents=True);shutil.copytree(source,root/'inputs/source')
    cfg=dict(intervals=[list(v) for v in I.INTERVALS],parent_plan_sha256=file_digest(p),input_files={f.relative_to(root/'inputs').as_posix():file_digest(f) for f in (root/'inputs').rglob('*') if f.is_file()})
    return root,dict(design=cfg)


def test_complete_producer_integration(tmp_path):
    root,p=complete_fixture(tmp_path);r=I.run(root,p,lambda **kw:None)
    assert r['posterior_rows']==512 and r['sources']==768 and r['rows']==2048 and not r['numerical_acceptance']
    assert r['source_interval_endpoint_queries']==24576
    assert len(list((root/'raw').glob('*.npz')))==12


@pytest.mark.parametrize('bad',['digest','missing','extra','binding','denominator','parent_plan','not_accepted'])
def test_complete_corruption_refused(tmp_path,bad):
    root,p=complete_fixture(tmp_path);source=root/'inputs/source'
    if bad=='digest':(source/'SUMMARY.json').write_text('{}')
    else:
        p['design']['input_files']={}
        if bad=='missing':next((source/'raw').glob('*.npz')).unlink()
        elif bad=='extra':np.savez(source/'raw/extra_points.npz',x=[1])
        elif bad=='parent_plan':p['design']['parent_plan_sha256']='wrong'
        elif bad=='not_accepted':(source/'FINAL_REVIEW.json').write_text('{"numerical_acceptance":false}')
        else:
            path=next((source/'evaluator').glob('*-bindings.json'));record=read(path)
            if bad=='binding':record['sources'][0][3]=(record['sources'][0][3]+1)%8
            else:record['rows'][0]['report_sources']+=1
            path.write_text(json.dumps(record))
    with pytest.raises((ValueError,KeyError)):I.run(root,p,lambda **kw:None)
