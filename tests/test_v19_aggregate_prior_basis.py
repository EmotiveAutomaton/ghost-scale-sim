import gzip
import json
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import aggregate_prior_basis as S
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from test_v19_aggregate_report_state import inputs
from test_v19_source_identity import complete_source_fixture


@pytest.mark.parametrize('mode', ['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('sparse', [False,True])
def test_scalar_source_bayes_and_posterior_rival_every_coordinate(mode,sparse):
    st,w,law,ids=inputs(mode,sparse)
    raw=S.evaluate(w,st,law,ids);summary=S.summarize(raw)
    q=raw['group_mass'];probs,mask,post=S.update(q,raw['joint_basis_report_mass'],raw['basis_copy_probability'])
    times=[int(x[0]) for x in ids]
    pi=np.array([[1/len(times) for t in times], [t/sum(times) for t in times], [(1/t)/math.fsum(1/v for v in times) for t in times]])
    np.testing.assert_allclose(pi,raw['basis_source_probability'],atol=2e-15,rtol=0)
    joint=np.zeros_like(raw['joint_basis_report_mass'])
    for k in range(3):
        for e in range(8):
            for h,g in enumerate(st['mapping']):
                joint[k,e,g]+=math.fsum(w[h]*pi[k,s]*law[st['past'][t-1,h],c,e] for s,(t,c,old) in enumerate(ids))
    np.testing.assert_allclose(joint,raw['joint_basis_report_mass'],atol=2e-15,rtol=0)
    for ai,alpha in enumerate(S.ALPHAS):
        for e in range(8):
            vertex=[]
            for k in range(3):
                numer=np.array([math.fsum(w[h]*pi[k,s]*(alpha*int(old==e)+(1-alpha)*law[st['past'][t-1,h],c,e]) for s,(t,c,old) in enumerate(ids)) for h in range(len(w))])
                den=math.fsum(numer);vertex.append(numer/den if den else np.zeros_like(w))
            for mi,mix in enumerate(S.MIXTURES):
                prior=[math.fsum(mix[k]*pi[k,s] for k in range(3)) for s in range(len(ids))]
                numer=np.array([math.fsum(w[h]*prior[s]*(alpha*int(old==e)+(1-alpha)*law[st['past'][t-1,h],c,e]) for s,(t,c,old) in enumerate(ids)) for h in range(len(w))])
                den=math.fsum(numer)
                assert probs[mi,ai,e]==pytest.approx(den,abs=2e-15)
                assert mask[mi,ai,e]==(den>0)
                if not den:
                    assert np.isnan(raw['posterior_mixture_squared_regret'][mi,ai,e]);continue
                exact=numer/den;rival=sum(mix[k]*vertex[k] for k in range(3))
                errors=[];rivals=[];tv=.5*sum(abs(post[mi,ai,e,g]-sum(exact[h] for h,gg in enumerate(st['mapping']) if gg==g)) for g in range(len(q)))
                for t in range(st['future'].shape[1]):
                    for context in range(4):
                        truth=sum(exact[h]*law[st['future'][h,t],context] for h in range(len(w)))
                        actual=sum(post[mi,ai,e,g]*law[st['signatures'][g,t],context] for g in range(len(q)))
                        wrong=sum(rival[h]*law[st['future'][h,t],context] for h in range(len(w)))
                        np.testing.assert_allclose(actual,truth,atol=2e-15,rtol=0)
                        errors.append(actual-truth);rivals.append(wrong-truth)
                        proper=math.fsum(truth[y]*(sum((wrong-np.eye(8)[y])**2)-sum((truth-np.eye(8)[y])**2)) for y in range(8))
                        assert proper==pytest.approx(sum((wrong-truth)**2),abs=2e-15)
                assert raw['updated_group_total_variation'][mi,ai,e]==pytest.approx(tv,abs=2e-15)
                assert raw['posterior_mixture_squared_regret'][mi,ai,e]==pytest.approx(np.mean(np.sum(np.array(rivals)**2,-1)),abs=2e-15)
                assert raw['posterior_mixture_max_future_error'][mi,ai,e]==pytest.approx(np.max(abs(np.array(rivals))),abs=2e-15)
    assert np.all(summary['basis_state_bytes']==8*(25*len(q)+24))
    assert np.all(summary['single_prior_state_bytes']==8*(9*len(q)+8))
    assert all(S.controls().values())


def test_source_group_endpoint_and_basis_permutations():
    st,w,law,ids=inputs();raw=S.evaluate(w,st,law,ids)
    permuted=S.evaluate(w,st,law,ids[::-1])
    for n in raw:
        if n not in ('source_rows','basis_source_probability'):
            np.testing.assert_allclose(raw[n],permuted[n],atol=2e-15,rtol=0)
    q,j,c=raw['group_mass'],raw['joint_basis_report_mass'],raw['basis_copy_probability']
    p,m,z=S.update(q,j,c)
    pp,mm,zz=S.update(q[::-1],j[:,:,::-1],c)
    np.testing.assert_allclose(pp,p,atol=2e-15,rtol=0);np.testing.assert_allclose(zz,z[:,:,:,::-1],atol=2e-15,rtol=0)
    pp,mm,zz=S.update(q,j[:,::-1],c[:,::-1])
    np.testing.assert_allclose(pp,p[:,:,::-1],atol=2e-15,rtol=0);np.testing.assert_allclose(zz,z[:,:,::-1],atol=2e-15,rtol=0)
    pp,mm,zz=S.update(q,j[::-1],c[::-1],S.MIXTURES[:,::-1])
    np.testing.assert_allclose(pp,p,atol=2e-15,rtol=0);np.testing.assert_allclose(zz,z,atol=2e-15,rtol=0)


@pytest.mark.parametrize('bad',[[-1,1,1],[1,1,1],[0,0,0],[float('nan'),0,1]])
def test_invalid_convex_mixtures_rejected(bad):
    st,w,law,ids=inputs();raw=S.evaluate(w,st,law,ids)
    with pytest.raises(ValueError):S.update(raw['group_mass'],raw['joint_basis_report_mass'],raw['basis_copy_probability'],[bad])


@pytest.mark.parametrize('field',['group_mass','joint_basis_report_mass','basis_copy_probability','mixture_weights','basis_source_probability','possible','report_probability','reference_probability','basis_float64_count','single_prior_float64_count','max_future_probability_error','future_squared_error','updated_group_total_variation','posterior_mixture_max_future_error','posterior_mixture_squared_regret'])
def test_corrupt_state_support_and_error_masks_rejected(field):
    st,w,law,ids=inputs('zeros');raw=S.evaluate(w,st,law,ids)
    if field=='possible':raw[field].flat[0]=not raw[field].flat[0]
    elif field.endswith('error') or field.endswith('regret') or field=='updated_group_total_variation':raw[field].flat[0]=0
    elif field=='reference_probability':raw[field].flat[0]=np.nan
    else:raw[field].flat[0]+=1
    with pytest.raises(ValueError):S.summarize(raw)


@pytest.mark.parametrize('corrupt',[None,'input_hash','pair','source_count','map','roster','forecast','design','priors','mixtures'])
def test_complete_native_parent_integration(tmp_path,corrupt):
    root,plan=complete_source_fixture(tmp_path)
    plan['design'].update(report_state='three-source-prior-basis-seven-mixtures',source_priors=list(S.PRIORS),mixtures=S.MIXTURES.tolist())
    if corrupt:
        path=root/'inputs/bindings/1-omitted-4-2-bindings.json';d=read(path)
        if corrupt in ('input_hash','pair'):d['sources'][0][2]=2
        if corrupt=='source_count':d['rows'][0]['report_sources']+=1
        if corrupt=='map':d['rows'][0]['joint_row']=1
        if corrupt=='roster':d['rows'][0]['draw']=99
        if corrupt=='forecast':d['rows'][0]['forecast'][0][0]+=.01
        if corrupt=='design':plan['design']['report_state']='arbitrary-prior'
        if corrupt=='priors':plan['design']['source_priors'].reverse()
        if corrupt=='mixtures':plan['design']['mixtures'][0]=[.5,.5,0]
        write(path,d,immutable=False)
        if corrupt!='input_hash':plan['design']['input_files'][path.relative_to(root/'inputs').as_posix()]=file_digest(path)
        with pytest.raises(ValueError):S.run(root,plan,lambda **kw:None)
    else:
        result=S.run(root,plan,lambda **kw:None)
        assert (result['posterior_rows'],result['sources'],result['report_queries'])==(512,768,143360)
        rows=json.loads(gzip.decompress((root/'raw/aggregate_prior_basis_summary_points.json.gz').read_bytes()))
        assert len(rows)==17920 and set(r['mixture'] for r in rows)==set(range(7))
        assert all(result['controls'].values()) and not result['numerical_acceptance']
