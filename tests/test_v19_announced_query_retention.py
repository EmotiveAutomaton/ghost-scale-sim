import gzip
import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import announced_query_retention as S, retention_query as B, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from test_v19_reachable_retrospective import fixture, law
from test_v19_source_identity import complete_source_fixture


def selected(ids, cp):
    base = B.retention(np.asarray(ids), cp).tolist()
    for ctx in range(4):
        matching = sorted((t for t,c,e in ids if c == ctx), reverse=True)
        other = sorted((t for t,c,e in ids if c != ctx), reverse=True)
        for cutoff in (cp//2, 3*cp//4):
            count = sum(t > cutoff for t,c,e in ids)
            times = set((matching+other)[:count])
            base.append([t in times for t,c,e in ids])
    return np.asarray(base)


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('sparse',[False,True])
def test_independent_group_bayes_and_explicit_proper_losses(mode,sparse):
    st=Q.prepare(fixture(),[.2,.3,.4,.1]); w=np.array([0,.5,0,.5] if sparse else [.6,.1,.2,.1]); p=law(mode)
    ids=[]
    for t,c in ((1,0),(2,1)):
        e=next(e for e in range(8) if sum(w[h]*p[k,c,e] for h,k in enumerate(st['past'][t-1]))>0)
        ids.append([t,c,e])
    raw=S.evaluate(w,st,p,ids); summary=S.summarize(raw); masks=selected(ids,2)
    np.testing.assert_array_equal(raw['retained'],masks)
    group=np.bincount(st['mapping'],weights=w,minlength=len(st['signatures']))
    forecasts={}
    for wi,keep in enumerate(masks):
        for ai,alpha in enumerate(S.ALPHAS):
            for e in range(8):
                numer=np.zeros(len(group))
                for si,(t,c,old) in enumerate(ids):
                    for h,g in enumerate(st['mapping']):
                        likelihood=p[st['past'][t-1,h],c,e] if keep[si] else math.fsum(p[:,c,e])/16
                        numer[g]+=w[h]*(alpha*(old==e)+(1-alpha)*likelihood)/len(ids)
                den=math.fsum(numer)
                assert raw['report_probability'][wi,ai,e]==pytest.approx(den,abs=4e-15)
                assert raw['possible'][wi,ai,e]==(den>0)
                if not den:continue
                post=numer/den
                forecast=np.array([sum(post[g]*p[st['signatures'][g,t]] for g in range(len(group))) for t in range(st['future'].shape[1])])
                forecasts[wi,ai,e]=forecast
                if (0,ai,e) not in forecasts:continue
                truth=forecasts[0,ai,e]
                for ctx in range(4):
                    if wi>=5 and ctx!=(wi-5)//2:
                        assert np.isnan(raw['query_squared_regret'][ctx,wi,ai,e])
                        assert np.isnan(raw['query_max_future_probability_error'][ctx,wi,ai,e])
                        continue
                    proper=sum(truth[t,ctx,y]*(sum((forecast[t,ctx]-np.eye(8)[y])**2)-sum((truth[t,ctx]-np.eye(8)[y])**2)) for t in range(len(truth)) for y in range(8))/len(truth)
                    assert raw['query_squared_regret'][ctx,wi,ai,e]==pytest.approx(proper,abs=4e-15)
                    assert raw['query_max_future_probability_error'][ctx,wi,ai,e]==pytest.approx(abs(forecast[:,ctx]-truth[:,ctx]).max(),abs=4e-15)
            if wi>=5:
                ctx=(wi-5)//2; label=S.WINDOWS[wi]
                expected=math.fsum(raw['report_probability'][0,ai,e]*raw['query_squared_regret'][ctx,wi,ai,e] for e in range(8) if raw['comparable'][wi,ai,e])
                assert summary[label+'_expected_squared_regret'][ai]==pytest.approx(expected,abs=4e-15)
    # The entire accepted reusable-state comparison must reproduce.
    inherited=B.summarize(B.evaluate(w,st,p,ids))
    for k,v in inherited.items():np.testing.assert_allclose(summary[k],v,rtol=0,atol=4e-15)
    assert len(summary)==174 and raw['shared_prior_float64_count']==32
    for offset,size in enumerate(('half','quarter')):
        for field in ('compact_float64_count','compact_int32_count'):
            expected=sum(raw[field][5+2*c+offset] for c in range(4))
            np.testing.assert_array_equal(summary['four_announced_'+size+'_'+field],np.full(5,expected))


@pytest.mark.parametrize('cp',[1,2,3,4,8,17,32])
@pytest.mark.parametrize('density',['all','sparse','single_context'])
def test_selection_is_matched_count_outcome_blind_and_order_covariant(cp,density):
    ids=np.array([[t,0 if density=='single_context' else t%4,t%8] for t in range(1,cp+1) if density!='sparse' or t%4!=0])
    keep=S.retention(ids,cp);np.testing.assert_array_equal(keep,selected(ids,cp))
    changed=ids.copy();changed[:,2]=7-changed[:,2]
    np.testing.assert_array_equal(S.retention(changed,cp),keep)
    order=np.arange(len(ids))[::-1]
    np.testing.assert_array_equal(S.retention(ids[order],cp),keep[:,order])
    for c in range(4):
        np.testing.assert_array_equal(keep[5+2*c:7+2*c].sum(-1),keep[1:3].sum(-1))
        if not any(ids[:,1]==c):np.testing.assert_array_equal(keep[5+2*c:7+2*c],keep[1:3])


def test_context_label_covariance_and_known_answer_controls():
    spec=dict(hypotheses=[['none',0,i] for i in range(16)],length=8,checkpoint=4,signatures=[[i]*5 for i in range(16)],membership=list(range(16)))
    w=np.arange(1,17,dtype=float);w/=w.sum();st=Q.prepare(spec,w);p=law('random');ids=np.array([[t,t-1,t-1] for t in range(1,5)])
    a=S.summarize(S.evaluate(w,st,p,ids));order=np.array([2,0,3,1]);changed=ids.copy();changed[:,1]=np.argsort(order)[ids[:,1]]
    b=S.summarize(S.evaluate(w,st,p[:,order],changed))
    for c in range(4):
        for size in ('half','quarter'):
            for field in ('expected_squared_regret','expected_max_probability_error','compact_float64_count','compact_int32_count'):
                np.testing.assert_allclose(b[f'announced{c}_{size}_{field}'],a[f'announced{order[c]}_{size}_{field}'],atol=4e-15,rtol=0)
    assert all(S.controls().values())


@pytest.mark.parametrize('bad',['weight','weight_nan','law','law_nan','empty','fractional','time','context','endpoint','duplicate','unsupported'])
def test_invalid_inputs_rejected(bad):
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


@pytest.mark.parametrize('bad',['remainder','histogram','shape','square','maximum','mask','count','applicability','universal'])
def test_invalid_raw_masks_and_remainder_rejected(bad):
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);raw=S.evaluate([.6,.1,.2,.1],st,law('random'),[[1,0,0],[2,1,1]])
    if bad=='remainder':raw['forgotten_mass'][5]=0
    if bad=='histogram':raw['forgotten_context_counts'][5,0]+=1
    if bad=='shape':raw['query_squared_regret']=raw['query_squared_regret'][:3]
    if bad=='square':raw['query_squared_regret'][0,5,0,0]+=.1
    if bad=='maximum':raw['query_max_future_probability_error'][0,5,0,0]+=1
    if bad=='mask':raw['query_squared_regret'][0,5,0,0]=np.nan
    if bad=='count':raw['retained_sources'][5]+=1
    if bad=='applicability':raw['applicable_query'][1,5]=True
    if bad=='universal':raw['future_squared_regret'][5]=0
    with pytest.raises(ValueError):S.summarize(raw)


@pytest.mark.parametrize('corrupt',[None,'input_hash','pair','source_count','map','roster','forecast','selection','contexts'])
def test_native_bound_population_and_frozen_design(tmp_path,corrupt):
    root,plan=complete_source_fixture(tmp_path);plan['design']['selection_rule']='announced-context-first-then-recency';plan['design']['query_contexts']=list(S.CONTEXTS)
    if corrupt:
        if corrupt in ('selection','contexts'):
            plan['design']['selection_rule' if corrupt=='selection' else 'query_contexts']='wrong'
        else:
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
        rows=json.loads(gzip.decompress((root/'raw/announced_query_summary_points.json.gz').read_bytes()))
        assert len(rows)==2560 and all(abs(r['full_expected_squared_regret'])<1e-12 for r in rows)
