"""Known answers and separation checks for the October 18 directed studies."""
from copy import deepcopy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import directed_probes as n16,probed_falsifier as n16b,lean_taste as n17,play_information as n18

@pytest.mark.parametrize('model',[n16,n16b,n17,n18])
def test_known_answers(model):
    assert model.controls()['passed']

@pytest.mark.parametrize('source',list(n16.SOURCES))
def test_n16_reconstructs_all_policy_propensities_and_scores(source):
    v=[v for v in n16.variants() if v['source']==source][-1]
    p,t,e,m=n16.case(v,'engineering',15,1,{},{} )
    check=n16.check(v,p,t,e,{},{} )
    assert all(abs(m[k]-check[k])<1e-10 for k in m)
    bad=deepcopy(p);bad['traces'][0]['steps'][0]['selection_probabilities'][0][0]=.123
    with pytest.raises(ValueError,match='selection propensity'):n16.check(v,bad,t,e,{},{} )

def test_n16_metric_membership_stays_fixed_when_random_menu_omits_types():
    v=n16.variants()[0];names=[]
    for c in range(6):names.append(set(n16.case(v,'engineering-menus',0,c,{},{} )[3]))
    assert all(x==names[0] for x in names)

def test_n16_nontiming_policy_law_does_not_expose_timing():
    v=next(v for v in n16.variants() if v['source']=='N9' and not v['original']['settings']['timing'])
    p,_,_,_=n16.case(v,'engineering-no-timing',0,0,{},{} )
    _,laws=n16.state('N9',p)
    assert laws.shape[-1]==2 and all(row['outcome'] in (0,1) for t in p['traces'] for row in t['steps'])

def test_n16b_equal_input_null_remains_exact_and_reconstructs():
    v=n16b.variants()[1];p,t,e,m=n16b.case(v,'engineering',15,0,{},{} )
    checked=n16b.check(v,p,t,e,{},{} )
    assert all(abs(m[k]-checked[k])<1e-10 for k in m)
    assert all(abs(m[f'k{k}|primary'])<1e-12 for k in (2,4,8))
    for k in ('2','4','8'):assert np.allclose(e[k]['records'],e[k]['goal'],atol=1e-13,rtol=0)

def test_n17_fit_cannot_read_heldout_answers_or_hidden_mechanism():
    fixed=dict(config=dict(location=-1.,spread=1.,temperature=1.))
    v=n17.variants()[8];p,t,e,m=n17.case(v,'engineering',0,0,fixed,{})
    changed=deepcopy(t);changed['heldout_y']=[(y+1)%6 for y in t['heldout_y']];changed['mechanism']=1-t['mechanism']
    alternate,_=n17.evaluate(p,changed)
    assert e['fitted']==alternate['fitted']
    assert 'heldout_y' not in p and 'parameters' not in p
    checked=n17.check(v,p,t,e,fixed,{})
    assert all(abs(m[k]-checked[k])<1e-10 for k in m)

def test_n18_repeated_records_do_not_create_new_laws():
    v=n18.variants()[0];a=n18.case(v,'engineering',0,0,{},{});b=n18.case(v,'engineering',0,1,{},{})
    assert a==b
    assert a!=n18.case(v,'engineering',1,0,{},{})

def test_directed_family_caps_and_contract_registration():
    from ghostscale.validation.soundingline.v23 import extension_contract as contract,directed_common as common
    assert sum(common.CAPS.values())==19.5*3600
    for name in common.FAMILIES:assert name in contract.CONDITIONAL_MODULES
