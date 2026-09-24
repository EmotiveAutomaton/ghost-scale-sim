import gzip
import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import retrospective_forgetting as S, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from test_v19_reachable_retrospective import fixture, law
from test_v19_source_identity import complete_source_fixture


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('sparse',[False,True])
def test_scalar_bayes_compact_joint_state_and_proper_loss(mode,sparse):
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([0,.5,0,.5] if sparse else [.6,.1,.2,.1]);p=law(mode)
    ids=[]
    for t,c in ((1,0),(2,1)):
        old=next(e for e in range(8) if sum(w[h]*p[k,c,e] for h,k in enumerate(st['past'][t-1]))>0)
        ids.append([t,c,old])
    raw=S.evaluate(w,st,p,ids);summary=S.summarize(raw);G=len(st['signatures']);q=np.bincount(st['mapping'],weights=w,minlength=G)
    forecasts={}
    for wi,label in enumerate(S.WINDOWS):
        kept=[True if wi==0 else t>1 if wi==1 else t>1 if wi==2 else False for t,c,e in ids]
        assert raw['retained'][wi].tolist()==kept
        # Independent materialization of all group/past-state tables.
        joint=[]
        for t,c,e in ids:
            J=np.zeros((G,16))
            for h,m in enumerate(st['past'][t-1]):J[st['mapping'][h],m]+=w[h]
            joint.append(J)
        cells=[len({(int(g),int(m)) for g,m in zip(st['mapping'],st['past'][t-1])}) for t,c,e in ids]
        mixed=[]
        for t,c,e in ids:
            sets=[{int(m) for g,m in zip(st['mapping'],st['past'][t-1]) if g==i} for i in range(G)]
            mixed.append(sum(len(x) for x in sets if len(x)>1))
        assert raw['compact_float64_count'][wi]==G+sum(m for m,k in zip(mixed,kept) if k)
        assert raw['compact_int32_count'][wi]==12+3*sum(kept)+2*sum(m for m,k in zip(cells,kept) if k)
        for ai,a in enumerate(S.ALPHAS):
            for e in range(8):
                numer=np.zeros(G)
                for si,(t,c,old) in enumerate(ids):
                    for g in range(G):
                        factor=sum(joint[si][g,k]*p[k,c,e] for k in range(16)) if kept[si] else q[g]*sum(p[k,c,e] for k in range(16))/16
                        numer[g]+=(a*(old==e)*q[g]+(1-a)*factor)/len(ids)
                den=math.fsum(numer);assert raw['report_probability'][wi,ai,e]==pytest.approx(den,abs=4e-15)
                if not den:
                    assert not raw['possible'][wi,ai,e];continue
                post=numer/den
                future=np.array([sum(post[g]*p[st['signatures'][g,t]] for g in range(G)) for t in range(st['future'].shape[1])])
                forecasts[wi,ai,e]=(future,post)
                if (0,ai,e) not in forecasts:continue
                truth,trueq=forecasts[0,ai,e];delta=future-truth
                assert raw['max_future_probability_error'][wi,ai,e]==pytest.approx(abs(delta).max(),abs=4e-15)
                assert raw['updated_group_total_variation'][wi,ai,e]==pytest.approx(.5*abs(post-trueq).sum(),abs=4e-15)
                regret=0.
                for t in range(len(truth)):
                    for c in range(4):
                        for y in range(8):
                            one=np.eye(8)[y]
                            regret+=truth[t,c,y]*(sum((future[t,c]-one)**2)-sum((truth[t,c]-one)**2))/(len(truth)*4)
                assert raw['future_squared_regret'][wi,ai,e]==pytest.approx(regret,abs=4e-15)
            expected=math.fsum(raw['report_probability'][0,ai,e]*raw['future_squared_regret'][wi,ai,e] for e in range(8) if raw['comparable'][wi,ai,e])
            assert summary[label+'_expected_squared_regret'][ai]==pytest.approx(expected,abs=4e-15)


def test_permutation_mass_controls_and_deleted_remainder():
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([.6,.1,.2,.1]);p=law('random');ids=[[1,0,0],[2,1,1]]
    raw=S.evaluate(w,st,p,ids);a=S.summarize(raw);b=S.summarize(S.evaluate(w,st,p,ids[::-1]))
    for k in a:np.testing.assert_allclose(a[k],b[k],rtol=0,atol=4e-15)
    assert all(S.controls().values())
    raw['forgotten_mass'][1]=0
    with pytest.raises(ValueError,match='remainder mass'):S.summarize(raw)
    raw=S.evaluate(w,st,p,ids);raw['forgotten_context_counts'][1,0]+=1
    with pytest.raises(ValueError,match='remainder histogram'):S.summarize(raw)


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
        assert r['posterior_rows']==512 and r['sources']==768 and r['report_queries']==20480
        assert all(r['controls'].values()) and not r['numerical_acceptance']
        rows=json.loads(gzip.decompress((root/'raw/retrospective_forgetting_summary_points.json.gz').read_bytes()))
        assert len(rows)==2560 and len(list((root/'raw').glob('*.npz')))==16
        assert all(abs(r['full_expected_squared_regret'])<1e-12 for r in rows)
