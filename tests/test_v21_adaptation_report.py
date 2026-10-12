import numpy as np
import pytest
from runners.report_v21_adaptation import interval, nested
from runners.report_v21_support import interval as reference_interval


def test_worlds_not_fits_and_missing_fit_rejected():
    rows=[dict(lineage=w,condition='shifted',reader='modular',k=0,fit=f,
               metrics={'log_loss':v}) for w,f,v in [(1,0,0),(1,1,2),(2,0,4),(2,1,8)]]
    result=nested(rows)
    assert [r['metrics']['log_loss'] for r in result]==[1,6]
    with pytest.raises(AssertionError,match='distinct nested fits'):nested(rows[:-1])
    with pytest.raises(AssertionError,match='distinct nested fits'):nested(rows+[rows[0]])


def test_bootstrap_preserves_registered_resampling_and_infinities():
    values=[-2,3,0,5,7,2,1,9]
    actual=interval(values);expected=reference_interval(values)
    for key in ('mean','lower','upper'):assert actual[key]==expected[key]
    assert interval([1,np.inf])['mean'] is None
