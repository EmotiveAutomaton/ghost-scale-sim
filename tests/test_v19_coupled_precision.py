from fractions import Fraction as F
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import coupled_precision as M
from ghostscale.validation.soundingline.v19 import precision_stability as B


def test_controls():assert all(M.controls().values())


def test_interior_optimum_and_maximum_regret():
    r=M.solve([[(0,4),(4,-4)],[(3,0)]],[0,1],[0,1])
    assert [x['weight'] for x in r['boundaries']]==[F(0),F(1,4),F(1,2),F(3,4),F(1)]
    assert [x['choices'] for x in r['segments']]==[[1],[0],[0],[1]]
    assert r['guaranteed_optima']==[] and r['recorded_maximum_regrets']==[1,1]


def test_identity_ties_and_infeasible_choice():
    r=M.solve([[(2,0),(1,0)],[(2,0),(0,0)]],[0,1],[0])
    assert r['guaranteed_optima']==[0,1] and r['recorded_maximum_regrets']==[0]
    with pytest.raises(ValueError):M.solve([[(0,0)]],[0],[1])


def test_common_affine_translation_and_order():
    lines=[[(0,4),(4,-4)],[(3,0)],[(4,1)]]
    r=M.solve(lines,[0,1,2],[0,1]);translated=[[(a+7,b-3) for a,b in row] for row in lines]
    s=M.solve(translated,[0,1,2],[0,1])
    for name in ('segments','possible_optima','guaranteed_optima','recorded_maximum_regrets'):assert r[name]==s[name]
    rev=M.solve(lines,[2,1,0],[1,0]);assert rev['recorded_maximum_regrets']==r['recorded_maximum_regrets'][::-1]


def test_complete_library_inside_box_and_context_permutation():
    a=np.array([[3.,2.,1.],[2.,1.,0.],[4.,3.,1.],[1.,.5,0.]])
    b=a+np.array([[1.,.5,.25],[.25,1.,.5],[.5,.25,1.],[1.,.25,.5]])
    chosen=[[0]]*3;box=B.evaluate(a,b,chosen)
    r=M.evaluate(a,b,chosen,box['summaries']);s=M.evaluate(a[::-1],b[::-1],chosen)
    assert len(r['lines'])==81 and r['charges']==list(box['charges'])
    assert [x['recorded_maximum_regrets'] for x in r['summaries']]==[x['recorded_maximum_regrets'] for x in s['summaries']]
    assert all(len(x['possible_optima'])<=len(x['box_not_ruled_out']) for x in r['summaries'])
    assert all(F(*x['recorded_maximum_regrets'][0])<=F(*x['box_regret_upper_bounds'][0]) for x in r['summaries'])
    with pytest.raises(ValueError):M.evaluate(a,np.full((4,3),np.nan),chosen)
