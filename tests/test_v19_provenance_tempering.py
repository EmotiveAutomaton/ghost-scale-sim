import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import provenance_tempering as P, unknown_change as U


def test_required_known_answer_and_placebo_controls():
    assert all(P.controls().values())


def test_source_time_and_query_time_scalar_reference():
    table=np.arange(1,513,dtype=float).reshape(16,4,8);table/=table.sum(-1,keepdims=True)
    obs=[dict(step=i,source_step=i,source_id=str(i),context=i%4,endpoint=i%8) for i in range(1,33)]
    obs[16]=dict(obs[15],step=17)
    out,hs,prior=P.filter_powers(table,obs,32,'purpose',[16,17,32])
    for step in (16,17,32):
        unique={r['source_id']:r for r in obs[:step]}
        for power in P.POWERS:
            weight=prior.copy()
            for j,(kind,t,m) in enumerate(hs):
                for r in unique.values():
                    mm=int(U.FLIPS[kind][m]) if kind!='none' and r['source_step']>t else m
                    weight[j]*=table[mm,r['context'],r['endpoint']]**power
            weight/=weight.sum();posterior=np.zeros(16)
            for w,(kind,t,m) in zip(weight,hs):
                mm=int(U.FLIPS[kind][m]) if kind!='none' and step>t else m
                posterior[mm]+=w
            assert np.allclose(weight,out[step,power][1],rtol=0,atol=2e-14)
            assert np.allclose(posterior,out[step,power][0],rtol=0,atol=2e-14)


def test_conflicting_and_misordered_sources_rejected():
    table=np.full((16,4,8),1/8);a=dict(step=1,source_step=1,source_id='a',context=0,endpoint=0)
    with pytest.raises(ValueError,match='conflict'):P.filter_powers(table,[a,dict(a,step=2,endpoint=1)],32,'purpose',[2])
    with pytest.raises(ValueError,match='nonconsecutive'):P.filter_powers(table,[dict(a,step=2)],32,'purpose',[2])
