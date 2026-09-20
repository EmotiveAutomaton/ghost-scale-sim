import copy
import numpy as np
from ghostscale.validation.soundingline.v18_4 import compression as P
from ghostscale.validation.soundingline.v18_3 import compression as C


def test_uniform_null_all_codes_same_predictive_loss():
    histories=list(__import__('itertools').product((0,1),repeat=3));ph=np.ones(8)/8
    predictions=[np.ones((8,4))/4]*3
    rows=P.select(histories,ph,predictions,0)
    assert all(abs(r['training_losses'][0]-np.log(4))<1e-12 for r in rows)
    assert next(r for r in rows if r['cardinality']==2)['old_optimum_ties']==127


def test_single_and_full_codes_equivalent_and_reference_verified():
    unit=P.unit(1804,cell=7,tilt=1.)
    assert P.verify(unit)
    for k in (1,8):
        losses=[r['new_loss'] for r in unit['rows'] if r['cardinality']==k]
        assert max(losses)-min(losses)<1e-12
    assert min(r['new_loss'] for r in unit['rows'] if r['cardinality']==8)<=min(r['new_loss'] for r in unit['rows'] if r['cardinality']==2)+1e-12


def test_selector_is_future_blind_and_corruption_detected():
    unit=P.unit(1805,cell=0)
    before=[r['code'] for r in P.select(unit['histories'],np.array(unit['history_probabilities']),[np.array(p) for p in unit['training_predictions']],unit['index'])]
    changed=copy.deepcopy(unit);changed['future_predictions'][0]=np.ones((8,16)).tolist()
    after=[r['code'] for r in P.select(changed['histories'],np.array(changed['history_probabilities']),[np.array(p) for p in changed['training_predictions']],changed['index'])]
    assert before==after
    import pytest
    with pytest.raises(ValueError):P.verify(changed)
