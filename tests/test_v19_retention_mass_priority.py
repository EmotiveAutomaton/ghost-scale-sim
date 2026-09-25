import gzip
import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import retention_mass_priority as S, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from test_v19_reachable_retrospective import fixture, law
from test_v19_source_identity import complete_source_fixture


@pytest.mark.parametrize('prior',['uniform','recency','early'])
@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('sparse',[False,True])
def test_scalar_bayes_compact_joint_state_and_proper_loss(mode,sparse,prior):
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([0,.5,0,.5] if sparse else [.6,.1,.2,.1]);p=law(mode)
    ids=[]
    for t,c in ((1,0),(2,1)):
        old=next(e for e in range(8) if sum(w[h]*p[k,c,e] for h,k in enumerate(st['past'][t-1]))>0)
        ids.append([t,c,old])
    raw=S.evaluate(w,st,p,ids,source_prior=prior);summary=S.summarize(raw);G=len(st['signatures']);q=np.bincount(st['mapping'],weights=w,minlength=G)
    times=[row[0] for row in ids]
    unscaled=[1 if prior=='uniform' else t if prior=='recency' else 1/t for t in times]
    probabilities=[v/math.fsum(unscaled) for v in unscaled]
    np.testing.assert_allclose(raw['source_probability'],probabilities,atol=4e-15,rtol=0)
    forecasts={}
    for wi,label in enumerate(S.WINDOWS):
        kept=([True,True] if wi==0 else [True,False] if wi>=5 and prior=='early' else [False,True])
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
        assert raw['compact_float64_count'][wi]==13+G+sum(m for m,k in zip(mixed,kept) if k)
        assert raw['compact_int32_count'][wi]==3*sum(kept)+2*sum(m for m,k in zip(cells,kept) if k)
        for ai,a in enumerate(S.ALPHAS):
            for e in range(8):
                numer=np.zeros(G)
                for si,(t,c,old) in enumerate(ids):
                    for g in range(G):
                        factor=sum(joint[si][g,k]*p[k,c,e] for k in range(16)) if kept[si] else q[g]*sum(p[k,c,e] for k in range(16))/16
                        numer[g]+=probabilities[si]*(a*(old==e)*q[g]+(1-a)*factor)
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


@pytest.mark.parametrize('prior', S.PRIORS)
def test_fixed_weight_ranking_ties_counts_and_permutations(prior):
    ids=np.array([[t,t%4,t%8] for t in (1,2,4,7,8)])
    keep=S.retention(ids,8,prior)
    weights=[1 if prior=='uniform' else t if prior=='recency' else 1/t for t in ids[:,0]]
    ordered=sorted(range(len(ids)),key=lambda i:(-weights[i],-int(ids[i,0])))
    for wi,ref in ((5,1),(6,2)):
        assert set(np.flatnonzero(keep[wi]))==set(ordered[:int(keep[ref].sum())])
    changed=ids[::-1].copy();changed[:,1]=3-changed[:,1];changed[:,2]=7-changed[:,2]
    assert np.array_equal(S.retention(changed,8,prior),keep[:,::-1])
    if prior!='early':np.testing.assert_array_equal(keep[5:7],keep[1:3])


@pytest.mark.parametrize('prior', S.PRIORS)
def test_weighted_remainder_full_copy_and_constant_identities(prior):
    spec=dict(hypotheses=[['none',0,i] for i in range(16)],length=8,checkpoint=8,signatures=[[i] for i in range(16)],membership=list(range(16)))
    w=np.arange(1,17,dtype=float);w/=w.sum();st=Q.prepare(spec,w)
    ids=np.array([[t,t%4,t%8] for t in range(1,9)])
    for mode in ('random','uniform'):
        raw=S.evaluate(w,st,law(mode),ids,source_prior=prior);summary=S.summarize(raw)
        for wi,name in enumerate(S.WINDOWS):
            assert raw['forgotten_mass'][wi]==pytest.approx(sum(raw['source_probability'][i] for i in range(8) if not raw['retained'][wi,i]),abs=4e-15)
            assert abs(summary[name+'_expected_squared_regret'][-1])<1e-12
            if mode=='uniform':assert max(abs(summary[name+'_expected_squared_regret']))<1e-12
        assert max(abs(summary['full_expected_squared_regret']))==0
    raw['forgotten_context_mass'][5,0]+=.1
    with pytest.raises(ValueError,match='remainder histogram'):S.summarize(raw)
    raw=S.evaluate(w,st,law('random'),ids,source_prior=prior);raw['forgotten_mass'][5]=0
    with pytest.raises(ValueError,match='remainder mass'):S.summarize(raw)
    assert all(S.controls().values())


@pytest.mark.parametrize('corrupt',[None,'input_hash','pair','source_count','map','roster','forecast'])
def test_complete_native_parent_integration(tmp_path,corrupt):
    root,plan=complete_source_fixture(tmp_path)
    plan['design'].update(selection_rule='highest-source-weight-then-recent-time',source_priors=list(S.PRIORS))
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
        result=S.run(root,plan,lambda **kw:None)
        assert result['posterior_rows']==512 and result['sources']==768
        assert all(result['controls'].values()) and not result['numerical_acceptance']
        rows=json.loads(gzip.decompress((root/'raw/retention_mass_priority_summary_points.json.gz').read_bytes()))
        assert len(rows)==7680 and len(list((root/'raw').glob('*.npz')))==16
        assert all(abs(r['full_expected_squared_regret'])<1e-12 for r in rows)
