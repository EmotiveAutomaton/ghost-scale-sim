from ghostscale.validation.soundingline.v27 import study,reporting

def test_zero_support_is_negative_infinity_not_missing_or_clipped():
    rows=[dict(setting=0,block=i,data=dict(measures=[dict(metric='misread_gain',labels={'reader':'contrasts'},statistic='extended_mean',values=values)])) for i,values in enumerate(([None,-2.],[-3.,-4.]))]
    result=study.aggregate(rows)[0]
    assert result['estimate_negative_infinity'] and result['lower_negative_infinity']
    assert result['interval95'][1]<-.05 and not result['interval_unavailable']
    summary=dict(family='K7',settings=[dict(law='M3',fake=0)],cells=[result])
    assert reporting.annotate(summary)[0]['verdict']=='met'
