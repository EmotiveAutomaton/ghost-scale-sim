from copy import deepcopy
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import purchase_calibration as C
from ghostscale.validation.soundingline.v18_4 import distinct_families as D

def test_live_placebo_and_exact_threshold_ties():
    assert all(C.controls().values())
    x=C.threshold(list(range(128)))
    assert x['threshold']==115 and x['strict_exceedances']==12 and x['ties']==1
    with pytest.raises(ValueError):C.threshold([float('nan')]*128)

def test_independent_prefix_sensor_copy_and_corruption():
    p,_=D.make_case(-19631,2,'in-family','interleaved',1);p['prior_mode']='duplicate-supplied'
    r=C.sensor(p);assert C.verify(p,r)
    copy=deepcopy(p);copy['history']=[o for o in copy['history'] for _ in range(3)]
    assert C.sensor(copy)==r
    bad=deepcopy(r);bad['predictions'][9][0]+=.01
    with pytest.raises(ValueError):C.verify(p,bad)
    copy['history'][1]=dict(copy['history'][1],artifact=999)
    with pytest.raises(ValueError):C.sensor(copy)

def test_future_noninterference_and_final_observation_not_a_trigger():
    p,_=D.make_case(-19632,8,'in-family','old-first',1);p['prior_mode']='equal-class'
    r=C.sensor(p);changed=deepcopy(p)
    changed['history'][16:]=[dict(o,program=[],artifact=0) for o in changed['history'][16:]]
    q=C.sensor(changed)
    assert np.array_equal(r['predictions'][:17],q['predictions'][:17])
    assert r['values']['raw'][:9]==q['values']['raw'][:9]
    a=[0.]*32;a[31]=100.
    assert C.rolling(a,[0.]*32)['maxima']['raw']==0
    a[30]=8.;assert C.rolling(a,[0.]*32)['maxima']['raw']==2.
