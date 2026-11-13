"""Prospective known answers and independent arithmetic for V29."""
import math
import numpy as np
from ghostscale.validation.soundingline.v29 import common,ledger,ledger_reference,q2,q2_reference,x2,g1b,c1

def test_sustained_crossing_keeps_reversals_and_censoring():
    assert common.sustained(np.array([[4,2,4,4],[4,4,4,2]])).tolist()==[3,5]

def test_class_means_are_not_individual_aliases():
    classes=np.array([0,0,1])
    out=common.class_evidence(np.array([[0.,-10.,0.]]),classes)
    assert out[0,0]<out[0,1]
    assert np.isclose(out[0,0],math.log((1+math.exp(-10))/2))

def test_approved_ledger_counterexample():
    x=ledger.construction_checks()
    assert abs(x['named_counterexample']['live_sell']-.18242552380635635)<1e-14
    assert x['named_counterexample']['rule_sell']==.02
    assert x['corrected_identity'] and x['buying_at_cap_follows_money_only']

def test_empty_frozen_mass_is_zero_probability():
    assert ledger_reference.lse(np.array([],dtype=float))==-np.inf
    s=dict(part='X2',truth=[4,4,5],environment='scarce',regime='LOW',buy_share=.1,start=0,steps=32,life=32,worlds=8,blocks=8)
    raw=ledger.generate(s,'excluded-test')
    a=x2.infer(raw,s)
    _,ll,_=ledger_reference.likelihood(raw,s,True)
    assert np.allclose(a['final_likelihood'],ll,atol=1e-9,rtol=0)

def test_q2_known_answers(tmp_path):
    assert q2.controls(tmp_path)['passed']

def test_q2_watching_is_not_an_owned_action():
    s=dict(part='test',truth=0,order='W',length=50,worlds=40,blocks=8)
    raw=q2.generate(s,'excluded-test')
    assert raw['actions'].sum()==0
    a=q2.infer(raw,s);b=q2_reference.infer(raw,s)
    assert np.array_equal(a['final_likelihood'],np.zeros((40,8)))
    for key in a:assert np.allclose(a[key],b[key],atol=1e-9,rtol=0)

def test_q2_long_extreme_path_is_numerically_stable():
    s=dict(part='test',truth=7,order='UCU',length=300,worlds=8,blocks=8)
    raw=q2.generate(s,'excluded-test')
    a=q2.infer(raw,s);b=q2_reference.infer(raw,s)
    for key in a:assert np.allclose(a[key],b[key],atol=1e-8,rtol=0)

def test_cross_context_known_answers(tmp_path):
    assert g1b.controls(tmp_path)['passed']

def test_trigger_table_known_answers(tmp_path):
    assert c1.controls(tmp_path)['passed']
