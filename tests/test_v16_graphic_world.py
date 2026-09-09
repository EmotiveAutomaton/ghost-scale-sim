from itertools import product
import random
import pytest
from ghostscale.validation.soundingline.v16.graphic_world import execute,learn,construct,expand
from ghostscale.validation.soundingline.v16.graphic_reference import interpret

def test_larger_graphic_world_independent_physics_and_boundaries():
    for program in product(range(32),repeat=2):
        for initial in [0,32769]:
            assert execute(program,initial=initial)==interpret(program,initial=initial)
    rng=random.Random(88217)
    for _ in range(1000):
        program=[rng.randrange(-1,34) for _ in range(rng.randrange(9))]
        initial=rng.randrange(65536)
        assert execute(program,initial=initial)==interpret(program,initial=initial)
    assert execute([15,31])["artifact"]==0
    assert not execute([0]*7)["legal"]
    assert execute([0]*7)["primitive_cost"]==6
    assert not execute([True])["legal"]
    with pytest.raises(ValueError):
        execute([],initial=65536)

def test_large_graphic_acquisition_changes_new_composition_at_counted_budget():
    training=[{"program":[0,1],"target":3,"feedback":True} for _ in range(7)]
    second=[{"program":[4,5],"target":48,"feedback":True} for _ in range(7)]
    first=learn(training)
    other=learn(second)
    assert first=={"library":[[0,1]],"definition_cost":2,"processing_primitives":14}
    library=first["library"]+other["library"]
    trained=construct(51,library,budget=256)
    primitive=construct(51,[],budget=256)
    assert not trained["search_timeout"] and primitive["search_timeout"]
    assert interpret(trained["program"])["artifact"]==51
    assert trained["program"]==expand(trained["tokens"],library)
    for result,lib in [(trained,library),(primitive,[])]:
        cost=0
        for attempt in result["attempts"]:
            reference=interpret(expand(attempt["tokens"],lib))
            assert reference==attempt["execution"]
            cost+=reference["primitive_cost"]
        assert cost==result["search_primitives"]<=256
    assert learn([{**item,"feedback":False} for item in training])["library"]==[]
    assert learn([{**item,"target":0} for item in training])["library"]==[]

def test_large_graphic_behavior_recoding_does_not_identify_order():
    assert execute([0,1,4,5])["artifact"]==execute([1,0,5,4])["artifact"]
    assert execute([0,16,0])["artifact"]==execute([0])["artifact"]
    assert execute(expand(["m0",4],[[0,1]]))==execute([0,1,4])

