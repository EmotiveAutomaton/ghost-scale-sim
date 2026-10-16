from copy import deepcopy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import registry,dispatch,checker_recipes


@pytest.mark.parametrize('family',['N4','N7'])
def test_recipe_readers_independent_and_sidecar_mutation_rejected(family):
    indices=(0,-1) if family=='N7' else (0,16,80,160,171)
    for index in indices:
        v=deepcopy(registry.variants(family)[index]);v['settings']['records']=2
        if family=='N4' and not v['settings']['extension']:v['settings']['recipes']=1
        p,t=dispatch.sample(v,'engineering-recipes',3,0)
        pred,detail=dispatch.evaluate(v,p,t)
        check=getattr(checker_recipes,'verify_'+family.lower())
        assert check(v,p,t,pred,detail)<1e-11
        bad=deepcopy(pred);bad['law-marginal']=np.zeros_like(pred['law-marginal'])
        with pytest.raises(ValueError,match='mismatch'):check(v,p,t,bad,detail)


def test_independent_dictionary_matches_finite_contract():
    from ghostscale.validation.soundingline.v23.makers import recipe_library
    for drives,exactly,seeking in [(4,None,'dimension'),(6,1,'vigor'),(9,2,'dimension')]:
        recipes,weights=checker_recipes.dictionary(drives,exactly,seeking)
        expected,reference,_=recipe_library(drives,exactly=exactly,seeking=seeking)
        assert tuple(recipes)==expected
        np.testing.assert_array_equal(weights,reference)


def test_vector_statement_inference_preserves_scalar_model():
    from ghostscale.validation.soundingline.v23 import statements
    for index in (0,59,127):
        v=deepcopy(registry.variants('N7')[index]);v['settings']['records']=8
        p,t=dispatch.sample(v,'engineering-statement-vector',7,3)
        shift=np.linspace(-.2,.2,10)
        for mode in ('law-marginal','choices-only','statements-only','both-naive','pooled-correction','training-choices'):
            a,ad=statements.infer(v,p,mode,shift);b,bd=statements.infer_scalar(v,p,mode,shift)
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
            np.testing.assert_allclose(ad['expected_weights'],bd['expected_weights'],atol=1e-12,rtol=0)


def test_vector_value_inference_preserves_scalar_likelihood():
    from ghostscale.validation.soundingline.v23 import values
    for index in (0,16,80):
        v=deepcopy(registry.variants('N4')[index]);v['settings'].update(records=3,recipes=1)
        p,t=values.sample(v,'engineering-value-vector',15,2)
        for kwargs in ({},{'linear':True},{'seeking':'dimension'},{'seeking':'vigor'},{'known_law':15}):
            a,ad=values.infer(v,p,**kwargs);b,bd=values.infer_scalar(v,p,**kwargs)
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
            np.testing.assert_allclose(ad['recipe_presence'],bd['recipe_presence'],atol=1e-12,rtol=0)


def test_vector_statement_checker_matches_independent_scalar_masses():
    from ghostscale.validation.soundingline.v23 import statements
    for index in (0,59,127):
        v=deepcopy(registry.variants('N7')[index]);v['settings']['records']=3
        p,_=statements.sample(v,'engineering-independent-vector',15,2)
        for mode in ('law-marginal','choices-only','statements-only','both-naive','pooled-correction','training-choices'):
            a,ad=checker_recipes.n7(v,p,mode,np.linspace(-.2,.2,10))
            b,bd=checker_recipes.n7_scalar(v,p,mode,np.linspace(-.2,.2,10))
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
            np.testing.assert_allclose(ad['expected_weights'],bd['expected_weights'],atol=1e-12,rtol=0)
    a=dict(statement_sd=.25,shift=.5,shift_kind='shared',aspiration=.3,records=0)
    b=dict(a,records=32)
    assert statements.shift_training_identity(a)==statements.shift_training_identity(b)
    assert statements.shift_training_identity(a)!=statements.shift_training_identity(dict(a,shift=1.))
def test_n4_population_fit_is_shared_across_depth_and_menu_design():
    from ghostscale.validation.soundingline.v23 import values,checker_recipes
    import numpy as np
    first,rows=values.pooling_training('depth-eight-random',4,1,'conjunctive','dimension')
    second,again=values.pooling_training('depth-512-separating',4,1,'conjunctive','dimension')
    np.testing.assert_array_equal(first,second)
    assert rows is again and len(rows)==32
    assert checker_recipes.verify_pooling_training(4,1,'conjunctive','dimension')
