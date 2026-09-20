"""The initial R1 catalog is a relabeling, not two distinct predictive laws."""
import numpy as np
from ghostscale.validation.soundingline.v18_4 import family_expansion as F


def test_coupling_catalog_is_uniform_prior_predictive_alias():
    W=F.W
    permutation=[W.STATES.index((0 if s[0]==0 else 3-s[0],*s[1:])) for s in W.STATES]
    for cell in range(0,16,2):
        a,b,_=F.catalog(W.make_world(cell,18047000))
        for s,j in zip(W.STATES,permutation):
            assert W.repertoire(a,s)==W.repertoire(b,W.STATES[j])
        for query in (*W.QUERIES,*W.FUTURES):
            assert np.array_equal(W.matrix(a,query),W.matrix(b,query)[permutation])
        public,_=F.make_case(0,cell,'new-rule','interleaved',1,16)
        fixed=F.read_stream(public,'fixed');mixed=F.read_stream(public,'mixture')
        for left,right in zip(fixed['trace']+fixed['final'],mixed['trace']+mixed['final']):
            assert np.allclose(left['prediction'],right['prediction'],atol=1e-14,rtol=0)
