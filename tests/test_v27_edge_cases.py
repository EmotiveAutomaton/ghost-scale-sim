from ghostscale.validation.soundingline.v27 import study,reporting

def test_zero_support_is_negative_infinity_not_missing_or_clipped():
    rows=[dict(setting=0,block=i,data=dict(measures=[dict(metric='misread_gain',labels={'reader':'contrasts'},statistic='extended_mean',values=values)])) for i,values in enumerate(([None,-2.],[-3.,-4.]))]
    result=study.aggregate(rows)[0]
    assert result['estimate_negative_infinity'] and result['lower_negative_infinity']
    assert result['interval95'][1]<-.05 and not result['interval_unavailable']
    summary=dict(family='K7',settings=[dict(law='M3',fake=0)],cells=[result])
    assert reporting.annotate(summary)[0]['verdict']=='met'


def test_censored_ratio_retains_finite_identification_bound():
    from ghostscale.validation.soundingline.v27.censoring import ratio_bounds
    b=ratio_bounds([[[None,100.]],[[None,100.]]],100,['known'])
    assert b['identified_estimate_bounds']==[1.01,None]
    assert b['interval95']==[1.01,None]
    row=dict(setting=0,metric='overshoot_ratio',labels={},statistic='median_ratio',**b)
    summary=dict(family='K3',settings=[dict(beta=1)],cells=[row])
    assert reporting.annotate(summary,[row])[0]['verdict']=='unresolved at this sample size'
    both=ratio_bounds([[[None,None]],[[None,None]]],100,['both'])
    assert both['identified_estimate_bounds']==[0.,None]


def test_positive_primary_can_expand_when_another_primary_is_negative(tmp_path):
    from ghostscale.validation.soundingline.v27 import continuation
    from ghostscale.validation.soundingline.v27.common import write
    packet=tmp_path/'primary'
    write(packet/'LOCAL_LANDING.json',dict(verdict='not met'))
    write(packet/'PLAN.json',dict(family='K7',card='K7'))
    base=dict(statistic='mean',estimate_censored=False,lower_censored=False,upper_censored=False,interval_unavailable=False)
    write(packet/'original/SUMMARY.json',dict(family='K7',settings=[dict(law='M1',fake=0),dict(law='M2',fake=0)],cells=[
        base|dict(setting=0,metric='marker_ratio',labels={},estimate=2.,interval95=[1.6,2.4]),
        base|dict(setting=1,metric='misread_gain',labels={},estimate=0.,interval95=[-.01,.01])]))
    assert continuation.eligible_second_generation(tmp_path,packet)
