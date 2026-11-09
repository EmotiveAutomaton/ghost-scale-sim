from fractions import Fraction
from datetime import datetime,timezone
import pytest
from ghostscale.validation.soundingline.v27 import recheck,contract


def test_exact_symmetric_posterior_does_not_discard_real_bias():
    strengths=(.25,.5,.75,1.)
    assert isinstance(recheck.exact_unbiased_harm(32,32,strengths),Fraction)
    assert recheck.no_boost_before_end([1,0]*32,strengths,1.)
    assert not recheck.no_boost_before_end([1]*32+[0]*32,strengths,.3)
    assert recheck.exact_unbiased_harm(7,10,strengths)==recheck.exact_unbiased_harm(10,7,strengths)
    assert recheck.exact_balanced_bias([1,0]*32,strengths,1.,.05)
    assert not recheck.exact_balanced_bias([1,0]*32,strengths,.7,.05)


def test_additional_healing_is_family_number_and_owner_bound():
    a=contract.acceptance('2026-11-07T18:18:53+00:00','packet');t=datetime(2026,11,9,14,tzinfo=timezone.utc)
    p=dict(family='K5',kind='healing',lane='healing',pool='followup_healing',healing_number=3,
        cpu_cap_seconds=100,projected_wall_seconds=200,pass_caps=dict(original=30,replay=30,review=30),
        numerical_threads=1,priority='below-normal',design_check_sha256='d',controls_sha256='c',
        protocol_sha256='p',source_manifest_sha256='s',blocks=16)
    with pytest.raises(ValueError,match='healing limit'):contract.admission(p,a,t,{})
    p=p|dict(healing_authorization='owner-20261109-one-extra-per-family')
    assert contract.admission(p,a,t,{})
    for patch in (dict(family='K1'),dict(healing_number=4),dict(healing_authorization='unapproved')):
        with pytest.raises(ValueError,match='healing limit'):contract.admission(p|patch,a,t,{})


def test_retained_balanced_history_has_zero_signed_bias_and_no_capture():
    items=[1,1,0,0,1,1,0,1,0,1,0,1,0,1,1,0,0,1,1,1,1,0,1,0,0,1,0,1,1,0,1,1,
           1,0,1,1,0,0,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0]
    s=dict(strengths=[.25,.5,.75,1.],theta=.7,delta=.05,switch=16,warning=.95,records=64,
           x=.5,tau=.5,zeta=0.,context_switch=False,sources=1,bias=.1)
    raw=[dict(source='M',truth=[0],items=[items],warning=[0],retrieval=[[1]*64])]
    assert recheck.exact_balanced_bias(items,s['strengths'],s['theta'],s['delta'])
    for independent in (False,True):
        result=recheck.analyze(raw,s,independent)
        assert result['M',0,'K_capture']==[.5]
        assert result['M',0,'K_captured']==[0.]


def test_legacy_probe_seed_identity_is_distinct_from_global_setting_number():
    from ghostscale.validation.soundingline.v27.study import global_setting_index
    plan=dict(settings=[{},{}],setting_indices=[0,9],seed_setting_indices=[0,1])
    assert global_setting_index(plan,1)==1
    del plan['seed_setting_indices']
    assert global_setting_index(plan,1)==9
