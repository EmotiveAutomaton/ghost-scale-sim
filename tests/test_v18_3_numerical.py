import numpy as np
from ghostscale.validation.soundingline.v18_3 import calibration as C,numerical_audit as N,provenance as P,source_uptake as U


def test_reliability_equivalent_forecasts_and_bin_boundaries():
    truth=np.array([[1.,0.]])
    a=C.reliability(np.array([[.5,.5]]),truth)
    b=C.reliability(np.array([[.5+1e-13,.5-1e-13]]),truth)
    assert a['expected_absolute_calibration_gap']==b['expected_absolute_calibration_gap']==0
    a=C.reliability(np.array([[.9,.1]]),truth);b=C.reliability(np.array([[.9-1e-13,.1+1e-13]]),truth)
    assert a['bins'][0]['bin']==b['bins'][0]['bin']==9
    assert abs(a['expected_absolute_calibration_gap']-b['expected_absolute_calibration_gap'])<1e-10


def test_shared_native_tie_coin_is_invariant_to_numerical_noise():
    unit=P.unit(18909,mode='shared-error',split='pilot')
    unit['rows']=[dict(method='a',instrument='valid',posterior=[.5,.5],after_correction=[[.5,.5]]),
        dict(method='b',instrument='valid',posterior=[.5-1e-13,.5+1e-13],after_correction=[[.5+1e-13,.5-1e-13]])]
    rows=U.apply(unit)['rows']
    assert rows[0]['initial']==rows[1]['initial'] and rows[0]['corrected']==rows[1]['corrected']


def test_compression_tie_range_known_uninformative_old_question():
    ph=np.ones(8)/8;old=np.ones((8,2))/2;new=np.array([[1.,0.] if i<4 else [0.,1.] for i in range(8)])
    result=N.tied_codebooks(ph,old,new,2)
    assert result['numerically_optimal_codes']==result['candidates']==127
    assert abs(result['best_new_loss_among_ties'])<1e-12
    assert abs(result['worst_new_loss_among_ties']-np.log(2))<1e-12
