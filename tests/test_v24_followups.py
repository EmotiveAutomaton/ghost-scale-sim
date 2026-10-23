import copy
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v24 import theory_pilot as theory, gallery_pilot as gallery, n16_followup


@pytest.mark.parametrize('family',list(theory.MODULES))
def test_every_theory_setting_has_independent_prefix_reconstruction(family):
    cfg=theory.configuration(family)
    for setting in cfg['settings']:
        for mechanism in cfg['mechanisms']:
            public,truth=theory.generate(family,setting,mechanism,'v24-engineering-excluded-complete-roster')
            assert 'mechanism' not in public
            a=theory.measure(family,public,truth);b=theory.measure(family,public,truth,True)
            theory.close(a,b)
            # Each prefix is one shared trajectory, never a newly seeded shorter run.
            first=a[0]['posterior']
            expected=theory.PRIORS[0] if family=='CONTEXT_CONVERGENCE' else [1/len(cfg['mechanisms'])]*len(cfg['mechanisms'])
            assert np.allclose(first,expected)


def test_theory_observation_aliases_remain_aliases():
    family='NEEDS_DRIVES';setting=dict(satiated='none',consequence_swap=False,informative=False)
    p,t=theory.generate(family,setting,'need_a','v24-engineering-excluded-alias')
    assert np.allclose(theory.measure(family,p,t)[-1]['posterior'],[.25]*4)
    family='HIGHER_GOAL_EDITING';setting=dict(conflict=False,higher_goal_observable=True,role_permuted=False)
    p,t=theory.generate(family,setting,'higher_goal','v24-engineering-excluded-alias')
    posterior=theory.measure(family,p,t)[-1]['posterior']
    assert posterior[0]==pytest.approx(posterior[2],abs=1e-12)


def test_gallery_scalar_checks_all_declared_geometries():
    for setting in gallery.settings():
        p=gallery.generate(setting,'v24-engineering-excluded-gallery-grid')
        a=gallery.metrics(p);b=gallery.metrics(p,True);theory.close(a,b)
        assert a['uniform_label_reference']==pytest.approx(1/setting['writers'],abs=1e-12)
        assert a['identity_excluding_reference']==pytest.approx((1-a['accuracy'])/(setting['writers']-1),abs=1e-12)
        if setting['condition']=='same_sample_overlap':assert a['accuracy']==1


def test_gallery_ties_and_nonfinite_rejection():
    p=dict(setting=dict(writers=9,features=1),a=[[0.]]*9,b=[[0.]]*9,groups=[i%3 for i in range(9)])
    assert np.allclose(gallery.weights(p),np.full((9,9),1/9))
    p['a'][0]=[float('nan')]
    with pytest.raises(ValueError,match='nonfinite'):gallery.metrics(p)


def test_retained_omission_bound_and_independent_ratio():
    mass={'correct':[.7,.3],'high-reliability':[.8,.2],'omit-0':[0.,1.],'omit-1':[1.,0.]}
    evidence=dict(posteriors=[dict(scenario=s,policy='P'+str(p),mass=[v]*6) for s,v in mass.items() for p in range(1,6)])
    a,_=n16_followup.calculate(evidence,dict(mechanism=0));b,_=n16_followup.calculate(evidence,dict(mechanism=0),True)
    theory.close(a,b)
    assert all(v==pytest.approx(math.log(.1)) for v in a[24:36])
    assert a[:12]==pytest.approx([0.]*12)


def test_independent_review_rejects_changed_value_and_grid():
    with pytest.raises(ValueError,match='numerical'):theory.close({'x':[.5]},{'x':[.6]})
    with pytest.raises(ValueError,match='incomplete'):theory.summarize([],dict(family='SETPOINTS',configuration=theory.configuration('SETPOINTS')))
    with pytest.raises(ValueError,match='incomplete'):n16_followup.summarize([])
