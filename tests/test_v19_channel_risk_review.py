from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v19 import channel_risk_review as R, channel_realized_risk as M
from test_v19_randomized_channel_regret import fixture


def test_known_controls(): assert all(R.controls().values())


@pytest.mark.parametrize('counts',M.MIXTURES)
def test_complete_independent_reconstruction(counts):
    library=fixture();result=M.solve(library,counts)
    metrics,count=R.audit(library,counts,result)
    assert count>0 and len(metrics)==3


@pytest.mark.parametrize('fault',['weight','score','tie','threshold','survivor','charge','reference'])
def test_corrupt_record_rejected(fault):
    library=fixture();result=M.solve(library,(1,1,2))
    if fault=='weight':result['lottery_sets'][0]['candidates'][0][2]+=1
    elif fault=='score':result['policy_score_numerators'][0][0]+=1
    elif fault=='tie':result['lottery_sets'][0]['optimal_indices']=[]
    elif fault=='threshold':result['risk_levels'][0]['threshold']=[-1,1]
    elif fault=='survivor':result['lottery_sets'][0]['surviving_policies']=[]
    elif fault=='charge':result['policy_used_bytes'][0][0]+=1
    else:result['unrestricted_optimal_score_numerators'][0]+=1
    with pytest.raises(AssertionError):R.audit(library,(1,1,2),result)


def test_single_prior_identical_mass_null():
    library=fixture()
    for r in library['candidates']:r['masses']=[[1,2]]*3
    values,_=R.audit(library,(4,0,0),M.solve(library,(4,0,0)))
    assert all(r['worst_expected_regret']==r['maximum_realized_coverage_loss']==0 for r in values)
