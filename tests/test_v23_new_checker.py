from copy import deepcopy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import registry,dispatch,checker_new


@pytest.mark.parametrize('family',['N2','N3','N6','N8'])
def test_new_independent_posteriors(family):
    variants=registry.variants(family)
    for v in (variants[0],variants[len(variants)//2],variants[-1]):
        v=deepcopy(v)
        # A deterministic engineering slice; full production counts enter pilots.
        v['settings']['records']=2
        p,t=dispatch.sample(v,'independent-n-control',7,2)
        prediction,detail=dispatch.evaluate(v,p,t)
        check=getattr(checker_new,'verify_'+family.lower())
        assert check(v,p,t,prediction,detail)<1e-11
        broken=deepcopy(prediction);broken['law-marginal']=np.zeros_like(broken['law-marginal'])
        with pytest.raises(ValueError,match='mismatch'):check(v,p,t,broken,detail)
