import pytest
from runners.report_v21_group import contrast, fit_digest


def test_contrast_pairs_worlds_not_input_order():
    rows=[dict(world=w,reader=r,k=-1,metrics={'log_loss':v}) for w,r,v in
          [(4,'a',1.),(2,'b',10.),(2,'a',12.),(4,'b',0.)]]
    result=contrast(rows,'a','b','log_loss')
    assert result['mean']==1.5
    assert result['lower']==1 and result['upper']==2
    with pytest.raises(StopIteration): contrast(rows[:-1],'a','b','log_loss')


def test_only_fit_timing_is_excluded_from_identity():
    assert fit_digest({'fit_cpu_seconds':1,'early_stopped':False})=={'early_stopped':False}
    assert fit_digest({'early_stopped':False})!=fit_digest({'early_stopped':True})
