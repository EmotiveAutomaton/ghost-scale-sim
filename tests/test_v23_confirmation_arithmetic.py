import math
import pytest
from ghostscale.validation.soundingline.v23 import confirmation_arithmetic as a,registry,evidence
from ghostscale.validation.soundingline.v23.common import write,digest

def test_arithmetic_uses_real_support_lowest_threshold_and_no_confirmation(tmp_path):
    cards={};entries=[];dispositions={f:'RESOURCE_DEFERRED' for f in registry.ORDER}
    for family,mu in [('N5',.949),('R1',.031)]:
        variant=registry.variants(family)[0]['id'];path=tmp_path/(family+'.json')
        write(path,dict(curve=[dict(variant=dict(id=variant),raw_mean=mu)]));cards[family]=path
        entries.append(dict(family=family,variant=variant,shrunk_mean=mu,machine_card_sha256=digest(path)))
        dispositions[family]='NOMINATED'
    order=dict(state='DISCOVERY_FROZEN',dispositions=dispositions,nominations=entries)
    result=a.calculate(order,cards)
    assert result['reserved_worlds_drawn']==0
    for j,row in enumerate(result['families'],1):
        assert row['minimum_log_threshold']==pytest.approx(math.log(10*(j+1)))
        v=evidence.betting_evalue([row['discovery_raw_mean']]*row['worlds'],row['support']['lower_bound'],independent_balanced_blocks=True)
        assert row['expected_log_e_upper_bound_if_discovery_mean_transfers']==pytest.approx(v['log_e'])
        assert row['state']=='CONFIRMATION_INFEASIBLE_AS_FROZEN'
    assert result['families'][1]['worlds']==256
    write(cards['R1'],{},immutable=False)
    with pytest.raises(ValueError,match='card changed'):a.calculate(order,cards)
