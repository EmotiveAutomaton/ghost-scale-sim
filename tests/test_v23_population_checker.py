from copy import deepcopy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import makers,registry,checker_population as checker


@pytest.mark.parametrize('structure',['id','alias','drift-.01','drift-.05'])
def test_independent_joint_population_recursion_and_mutation_rejection(structure):
    v=deepcopy(registry.variants('N1')[0]);v['settings'].update(makers=3,records=4,structure=structure)
    public,truth=makers.n1_world(v,'independent-population-control',7)
    pred,detail=makers.n1_infer(v,public)
    menu=makers.cultures(v['settings']['concentration'])
    assert checker.verify_n1(v,public,truth,pred,detail,menu)<1e-10
    broken=deepcopy(detail);broken['culture_posterior']=np.full(32,1/32)
    with pytest.raises(ValueError,match='mismatch'):checker.verify_n1(v,public,truth,pred,broken,menu)


def test_one_type_population_and_no_personal_records_have_known_answers():
    v=deepcopy(registry.variants('N1')[0]);v['settings'].update(makers=2,records=0)
    p={'menus':[[],[]],'choices':[[],[]],
       'own':np.array([[[1,0,0,0],[0,0,0,0],[0,0,0,0]]]*2),
       'other':np.array([[[1,0,0,0],[0,0,0,0],[0,0,0,0]]]*2)}
    menu=np.zeros((32,200));menu[:,0]=1
    pred,detail=checker.reconstruct(v,p,menu)
    expected=np.mean([np.array([np.exp(b),1,1])/(np.exp(b)+2) for b in (1,2,4)],axis=0)
    assert np.allclose(pred['law-marginal'],[expected,expected],atol=1e-12)
    assert np.allclose(pred['law-marginal'],pred['population-only'],atol=1e-12,rtol=0)
    assert np.allclose(detail['culture_posterior'],1/32)
    assert np.allclose(detail['readers']['law-marginal']['recipe_presence'][:,0],[1,1],atol=1e-12,rtol=0)
