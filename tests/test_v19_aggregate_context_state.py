import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import aggregate_context_state as S
from test_v19_aggregate_report_state import inputs


@pytest.mark.parametrize('mode', ['random', 'uniform', 'zeros', 'disjoint'])
@pytest.mark.parametrize('sparse', [False, True])
def test_scalar_context_bayes_and_proper_loss(mode, sparse):
    st, w, law, ids = inputs(mode, sparse)
    raw = S.evaluate(w, st, law, ids); summary = S.summarize(raw)
    probability, possible, posterior = S.update(raw['group_mass'], raw['joint_context_report_mass'], raw['context_copy_counts'])
    for ai, alpha in enumerate(S.ALPHAS):
        futures = {}; endpoint_numer = np.zeros((8, len(w)))
        for c in range(4):
            for e in range(8):
                numer = np.array([math.fsum(w[h]*(alpha*int(old == e)+(1-alpha)*law[st['past'][t-1,h],c,e])/len(ids) for t,ctx,old in ids if ctx == c) for h in range(len(w))])
                den = math.fsum(numer); endpoint_numer[e] += numer
                assert probability[ai,c,e] == pytest.approx(den,abs=2e-15)
                assert possible[ai,c,e] == (den > 0)
                if not den: continue
                truth = np.array([[sum(numer[h]/den*law[st['future'][h,t],ctx] for h in range(len(w))) for ctx in range(4)] for t in range(st['future'].shape[1])])
                compact = np.array([[sum(posterior[ai,c,e,g]*law[st['signatures'][g,t],ctx] for g in range(len(st['signatures']))) for ctx in range(4)] for t in range(st['future'].shape[1])])
                np.testing.assert_allclose(truth,compact,atol=2e-15,rtol=0);futures[c,e]=truth
        gain = 0.
        for (c,e), truth in futures.items():
            weights = endpoint_numer[e]/endpoint_numer[e].sum()
            pooled = np.array([[sum(weights[h]*law[st['future'][h,t],ctx] for h in range(len(w))) for ctx in range(4)] for t in range(st['future'].shape[1])])
            # Explicit expected one-hot categorical loss, independent of distance.
            excess = np.zeros(truth.shape[:-1])
            for y in range(8):
                target = np.eye(8)[y]
                excess += truth[...,y]*(((pooled-target)**2).sum(-1)-((truth-target)**2).sum(-1))
            assert raw['endpoint_only_squared_regret'][ai,c,e] == pytest.approx(excess.mean(),abs=2e-15)
            gain += probability[ai,c,e]*excess.mean()
        assert summary['expected_endpoint_only_squared_regret'][ai] == pytest.approx(gain,abs=2e-15)
    assert all(S.controls().values())


def test_source_context_endpoint_permutations_and_missing_context():
    st,w,law,ids=inputs();a=S.evaluate(w,st,law,ids)
    b=S.evaluate(w,st,law,ids[::-1])
    for key in a:
        if key != 'source_rows':np.testing.assert_allclose(a[key],b[key],atol=2e-15,rtol=0,equal_nan=True)
    changed=ids.copy();changed[:,1]=3-ids[:,1];changed[:,2]=7-ids[:,2]
    b=S.evaluate(w,st,law[:,::-1,::-1],changed)
    np.testing.assert_allclose(a['report_probability'][:,::-1,::-1],b['report_probability'],atol=2e-15,rtol=0)
    np.testing.assert_allclose(a['joint_context_report_mass'][:,::-1,::-1],b['joint_context_report_mass'],atol=2e-15,rtol=0)
    assert not a['possible'][:,2:].any()
    assert np.isnan(a['endpoint_only_squared_regret'][:,2:]).all()


@pytest.mark.parametrize('key',['group_mass','joint_context_report_mass','context_copy_counts','source_rows','report_probability','possible','endpoint_report_probability','max_future_probability_error','future_squared_error','updated_group_total_variation','endpoint_only_squared_regret','context_float64_count','context_int32_count','endpoint_float64_count','endpoint_int32_count'])
def test_corrupted_table_histogram_mask_storage(key):
    st,w,law,ids=inputs('zeros');raw=S.evaluate(w,st,law,ids)
    if key=='possible':raw[key].flat[0]=not raw[key].flat[0]
    elif key=='source_rows':raw[key][0,1]=3
    elif key in ('max_future_probability_error','future_squared_error','updated_group_total_variation','endpoint_only_squared_regret'):raw[key][0,3,0]=0
    else:raw[key].flat[0]+=1
    with pytest.raises(ValueError):S.summarize(raw)


def test_complete_handler(tmp_path):
    from test_v19_source_identity import complete_source_fixture
    from ghostscale.validation.soundingline.v18_3.io import read
    root,plan=complete_source_fixture(tmp_path)
    plan['design']['report_state']='fixed-uniform-source-joint-group-context-endpoint'
    result=S.run(root,plan,lambda **kw:None)
    assert result['posterior_rows']==512 and result['report_queries']==81920
    assert result['rows']==2560 and all(read(root/'CONTROLS.json').values())
    assert not result['numerical_acceptance']
