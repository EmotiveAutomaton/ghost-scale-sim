import gzip
import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import retention_source_prior as S, reachable_retrospective as Q
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
        kept=([True,True] if wi==0 else [False,True])
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


def test_permutation_mass_controls_and_deleted_remainder():
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([.6,.1,.2,.1]);p=law('random');ids=[[1,0,0],[2,1,1]]
    raw=S.evaluate(w,st,p,ids);a=S.summarize(raw);b=S.summarize(S.evaluate(w,st,p,ids[::-1]))
    for k in a:np.testing.assert_allclose(a[k],b[k],rtol=0,atol=4e-15)
    assert all(S.controls().values())
    raw['forgotten_mass'][1]=0
    with pytest.raises(ValueError,match='remainder mass'):S.summarize(raw)
    raw=S.evaluate(w,st,p,ids);raw['forgotten_context_mass'][1,0]+=.1
    with pytest.raises(ValueError,match='remainder histogram'):S.summarize(raw)


def test_outcome_blind_matched_counts_midpoint_ranks_and_permutation():
    ids=np.array([[1,0,0],[2,1,1],[3,1,2],[4,2,3],[5,1,4],[6,2,5],[7,1,6],[8,3,7]])
    keep=S.retention(ids,8)
    assert np.array_equal(keep.sum(-1),[8,4,2,4,2])
    assert np.flatnonzero(keep[3]).tolist()==[1,3,5,7]
    assert np.flatnonzero(keep[4]).tolist()==[2,6]
    changed=ids.copy();changed[:,2]=7-changed[:,2];changed[:,1]=3-changed[:,1]
    assert np.array_equal(S.retention(changed,8),keep)
    order=np.array([6,1,7,3,4,2,5,0])
    assert np.array_equal(S.retention(ids[order],8),keep[:,order])


def test_all_counts_unique_and_nonuniform_times():
    for n in range(1,33):
        ids=np.array([[t*t+1,2,t%8] for t in range(n)])
        for count in range(n+1):
            mask=S.spaced_mask(ids,count)
            assert mask.sum()==count
            expected=[] if not count else [math.floor((j+.5)*n/count) for j in range(count)]
            assert np.flatnonzero(mask).tolist()==expected
    with pytest.raises(ValueError):S.spaced_mask(ids,-1)
    with pytest.raises(ValueError):S.spaced_mask(ids,len(ids)+1)


def test_equal_selected_roster_gives_identical_forecasts():
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([.6,.1,.2,.1]);p=law('random')
    raw=S.evaluate(w,st,p,[[1,2,0],[2,2,1]])
    for key in ('report_probability','future_squared_regret','max_future_probability_error','updated_group_total_variation','compact_float64_count','compact_int32_count'):
        np.testing.assert_array_equal(raw[key][1:3],raw[key][3:5])


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
    plan['design']['selection_rule']='midpoint-sorted-time-ranks'
    plan['design']['source_priors']=list(S.PRIORS)
    from ghostscale.validation.soundingline.v19 import time_retention as U
    uniform_root,uniform_plan=complete_source_fixture(tmp_path/'uniform-original')
    uniform_plan['design']['selection_rule']='midpoint-sorted-time-ranks'
    U.run(uniform_root,uniform_plan,lambda **kw:None)
    import shutil
    shutil.copytree(uniform_root/'raw',root/'inputs/uniform/raw')
    write(root/'inputs/uniform/FINAL_REVIEW.json',dict(numerical_acceptance=True))
    write(root/'inputs/uniform/PLAN.json',uniform_plan)
    for p in (root/'inputs/uniform').rglob('*'):
        if p.is_file():plan['design']['input_files'][p.relative_to(root/'inputs').as_posix()]=file_digest(p)
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
        assert r['posterior_rows']==512 and r['sources']==768 and r['report_queries']==61440
        assert all(r['controls'].values()) and not r['numerical_acceptance']
        rows=json.loads(gzip.decompress((root/'raw/retention_source_prior_summary_points.json.gz').read_bytes()))
        assert len(rows)==7680 and len(list((root/'raw').glob('*.npz')))==16
        assert all(abs(r['full_expected_squared_regret'])<1e-12 for r in rows)


@pytest.mark.parametrize('prior', S.PRIORS)
def test_weighted_remainder_source_permutation_and_uniform_reduction(prior):
    from ghostscale.validation.soundingline.v19 import time_retention as U
    spec=dict(hypotheses=[['none',0,i] for i in range(16)],length=8,checkpoint=8,signatures=[[i] for i in range(16)],membership=list(range(16)))
    w=np.arange(1,17,dtype=float);w/=w.sum();st=Q.prepare(spec,w);p=law('random')
    ids=np.array([[1,0,0],[2,1,1],[4,2,2],[7,3,3],[8,0,4]])
    a=S.evaluate(w,st,p,ids,source_prior=prior);b=S.evaluate(w,st,p,ids[::-1],source_prior=prior)
    for key in S.summarize(a):np.testing.assert_allclose(S.summarize(a)[key],S.summarize(b)[key],rtol=0,atol=4e-15)
    for wi in range(5):
        expected=sum(a['source_probability'][i] for i in range(len(ids)) if not a['retained'][wi,i])
        assert a['forgotten_mass'][wi]==pytest.approx(expected,abs=4e-15)
        for ctx in range(4):
            assert a['forgotten_context_mass'][wi,ctx]==pytest.approx(sum(a['source_probability'][i] for i in range(len(ids)) if not a['retained'][wi,i] and ids[i,1]==ctx),abs=4e-15)
    if prior=='uniform':
        base=U.evaluate(w,st,p,ids)
        for key in ('report_probability','possible','comparable','future_squared_regret','max_future_probability_error','updated_group_total_variation','retained','retained_sources','forgotten_mass'):
            np.testing.assert_allclose(a[key],base[key],rtol=0,atol=4e-15)
        np.testing.assert_array_equal(a['compact_float64_count'],base['compact_float64_count']+13)
        np.testing.assert_array_equal(a['compact_int32_count'],base['compact_int32_count']-12)
    a['forgotten_context_mass'][1,0]+=.05
    with pytest.raises(ValueError,match='remainder histogram'):S.summarize(a)


def test_weighted_full_copy_constant_identities_and_selection_blindness():
    spec=dict(hypotheses=[['none',0,i] for i in range(16)],length=8,checkpoint=8,signatures=[[i] for i in range(16)],membership=list(range(16)))
    w=np.full(16,1/16);st=Q.prepare(spec,w);ids=np.array([[t,t%4,t%8] for t in range(1,9)])
    for prior in S.PRIORS:
        for p in (law('random'),law('uniform')):
            raw=S.evaluate(w,st,p,ids,source_prior=prior);m=S.summarize(raw)
            assert np.max(abs(m['full_expected_squared_regret']))<1e-12
            for name in S.WINDOWS:assert abs(m[name+'_expected_squared_regret'][-1])<1e-12
        assert np.max(abs(m['spaced_half_expected_squared_regret']))<1e-12
        np.testing.assert_array_equal(raw['retained'],S.retention(ids,8))
    with pytest.raises(ValueError,match='source prior'):S.evaluate(w,st,p,ids,source_prior='unfrozen')


def test_reused_uniform_equals_complete_recalculation_and_rejects_identity():
    from ghostscale.validation.soundingline.v19 import time_retention as U
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);w=np.array([.6,.1,.2,.1]);p=law('random');ids=np.array([[1,0,0],[2,1,1]])
    inherited=U.evaluate(w,st,p,ids);converted=S.reuse_uniform(inherited,ids);recomputed=S.evaluate(w,st,p,ids)
    assert set(converted)==set(recomputed)
    for k in converted:np.testing.assert_allclose(converted[k],recomputed[k],rtol=0,atol=4e-15)
    ids[0,2]=2
    with pytest.raises(ValueError,match='uniform source identity'):S.reuse_uniform(inherited,ids)
