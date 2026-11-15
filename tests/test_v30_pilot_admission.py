import numpy as np
from ghostscale.validation.soundingline.v30 import l,l_pair
def test_activation_pilots_exclude_fixed_fast_and_exact_alias():
    s=dict(part="filter",truth=0);different=np.zeros((40,12,3));different[:,:,1]=1;different[:,:,2]=2
    for module in (l,l_pair):
        assert module.pilot_exclusion(s,dict(separation=np.full(40,10),_ll=different))
        assert module.pilot_exclusion(s,dict(separation=np.full(40,11),_ll=np.zeros_like(different)))
        assert module.pilot_exclusion(s,dict(separation=np.full(40,11),_ll=different)) is None
        assert module.pilot_exclusion(dict(part="correlation"),{}) is None
