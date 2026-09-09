from collections import defaultdict
from itertools import product
import math
import pytest
from ghostscale.validation.soundingline.v16.mechanism_models import board_prior,deterministic,distribution
from ghostscale.validation.soundingline.v16.reconstruction import RELIABILITIES,TOPIC_PROBABILITIES
from ghostscale.validation.soundingline.v16.learning import learn
from ghostscale.validation.soundingline.v16.reference import interpret
from ghostscale.validation.soundingline.v16.assembly_reference import interpret as assembly_interpret


def test_habit_prior_retains_order_and_matches_ordered_acquisition():
    exact=defaultdict(float)
    for reliability,topic,direction in product(RELIABILITIES,TOPIC_PROBABILITIES,range(2)):
        probabilities=[reliability*(topic if direction==0 else 1-topic),reliability*(1-topic if direction==0 else topic),1-reliability]
        for sequence in product(range(3),repeat=4):
            attempts=[[(0,1),(2,3),(0,)][kind] for kind in sequence]
            targets=[12 if kind==1 else 3 for kind in sequence]
            library=learn(attempts,targets).library
            exact[library]+=math.prod(probabilities[kind] for kind in sequence)/18
    libraries,masses=board_prior(4)
    for library,mass in zip(libraries,masses):
        assert abs(mass-exact[library])<1e-10
    larger,_=board_prior(8)
    assert ((0,1),(2,3)) in larger and ((2,3),(0,1)) in larger


def test_habit_is_actual_prior_execution_not_a_different_outcome_label():
    world={"family":"W1"}
    bounded=deterministic(world,((0,1),),12,"bounded",budget=4096,max_steps=3)
    habit=deterministic(world,((0,1),),12,"habit",budget=4096,max_steps=3)
    assert bounded["task_success"]
    assert not habit["task_success"] and habit["habit_prefix"]==[0,1]
    assert interpret(habit["program"])["artifact"]==habit["execution"]["artifact"]
    no_skill_bounded=deterministic(world,(),12,"bounded",budget=4096,max_steps=3)
    no_skill_habit=deterministic(world,(),12,"habit",budget=4096,max_steps=3)
    assert no_skill_bounded["program"]==no_skill_habit["program"]


def test_bounded_and_habit_artifact_distributions_are_real_executions():
    cases=[({"family":"W1"},((0,1),),[3,12,7],3),
           ({"family":"W2","parents":[-1,0,0],"defaults":[0,0,0]},((0,1),),[(0,0,0),(1,0,-1)],4)]
    for world,library,goals,max_steps in cases:
        for family,goal,budget in product(["bounded","habit"],goals,[8,128]):
            result=deterministic(world,library,goal,family,budget=budget,max_steps=max_steps)
            checked=interpret(result["program"]) if world["family"]=="W1" else assembly_interpret(world,result["program"])
            assert checked["legal"] and checked["artifact"]==result["execution"]["artifact"]
            if world["family"]=="W2":
                assert checked["successfully_stopped"]
            support,probabilities,programs,weights=distribution(world,library,goal,family,budget=budget,max_steps=max_steps)
            state=result["execution"]["state"]
            state=tuple(state) if isinstance(state,list) else state
            assert probabilities[support.index(state)]==1 and sum(probabilities)==1
            assert list(programs[0])==result["program"]


def test_option_generator_learns_from_public_executed_generic_exploration():
    world={"family":"W1","option_training":[[0,1],[2,3]]*8}
    result=deterministic(world,(),7,"options",budget=4096,max_steps=3)
    assert result["task_success"]
    assert interpret(result["program"])["artifact"]==7
    support,probabilities,programs,weights=distribution(world,(),7,"options",budget=4096,max_steps=3)
    assert probabilities[support.index(7)]==1
    with pytest.raises(ValueError,match="exploration"):
        deterministic({"family":"W1"},(),7,"options")
