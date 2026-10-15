import math
import pytest
from ghostscale.validation.soundingline.v23 import support, registry, repairs


@pytest.mark.parametrize('family',['R1','R2','R3','R4'])
def test_repair_support_covers_extreme_engineering_cases(family):
    bound=support.repair_bound(family)
    assert math.isfinite(bound['lower_bound']) and bound['lower_bound']<0
    assert not bound['empirical_floor']
    variants=registry.variants(family)
    for v in (variants[0], variants[-1]):
        p,t=repairs.sample(v,'support-control',15,0)
        predictions,_=repairs.evaluate(v,p,t)
        mechanism=registry.PRIMARY[family][1]
        for probability in predictions[mechanism].ravel():
            assert math.log(probability)>=bound['lower_bound']
    import json
    support.verify(json.loads(json.dumps(dict(family=family,confirmation_support=bound))))
    with pytest.raises(ValueError,match='support proof'):support.verify(dict(family=family))
