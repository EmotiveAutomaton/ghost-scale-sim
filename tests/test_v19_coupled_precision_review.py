from fractions import Fraction as F
import copy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import coupled_precision_review as R
from ghostscale.validation.soundingline.v19 import coupled_precision as P
from ghostscale.validation.soundingline.v19 import precision_stability as B


def test_review_controls(): assert all(R.controls().values())


def test_independent_partition_known_interior_extremum():
    r=R.reconstruct_partition([[(0,4),(4,-4)],[(3,0)]],[0,1],[0,1])
    assert [x['weight'] for x in r['boundaries']]==[0,F(1,4),F(1,2),F(3,4),1]
    assert r['recorded_maximum_regrets']==[1,1]
    assert r['boundaries'][2]['objectives']==[2,3]


def test_complete_reconstruction_and_corrupt_bound_rejected():
    a=np.array([[3.,2.,1.],[2.,1.,0.],[4.,3.,1.],[1.,.5,0.]])
    b=a+np.array([[1.,.5,.25],[.25,1.,.5],[.5,.25,1.],[1.,.25,.5]])
    choices=[[0]]*3;box=B.evaluate(a,b,choices)['summaries']
    assert R.reconstruct(a,b,choices,box)==P.evaluate(a,b,choices,box)
    bad=copy.deepcopy(box);bad[0]['recorded_regret_upper_numerators']=[-1]
    with pytest.raises(AssertionError):R.reconstruct(a,b,choices,bad)


def test_translation_order_and_persistent_tie():
    lines=[[(0,4),(4,-4)],[(3,0)],[(3,0)]]
    r=R.reconstruct_partition(lines,[0,1,2],[0,1,2])
    shifted=R.reconstruct_partition([[(a+11,b-7) for a,b in row] for row in lines],[0,1,2],[0,1,2])
    for key in ('segments','recorded_maximum_regrets','possible_optima','guaranteed_optima'):assert r[key]==shifted[key]
    assert r['segments'][0]['choices']==[1,2]
