from dataclasses import replace
from itertools import product
from ghostscale.validation.soundingline.v16.opportunity_gates import run
from ghostscale.validation.soundingline.v16.opportunity import (
    CAUSES,COSTS,LAPSES,PROBES,response_probability,make,decide,enact)


def test_causal_opportunity_gates():
    result = run()
    assert result["instrument_state"] == "valid", result


def test_response_table_matches_independent_scalar_actions():
    for cause,cost,lapse,probe in product(CAUSES,COSTS,LAPSES,PROBES):
        actual = cause not in {"physical","false-affordance"}
        knows = cause not in {"physical","knowledge"}
        considers = cause != "consideration"
        goal = int(cause != "purpose")
        budget = 1 if cause == "search" else 2
        reward = 1
        if probe == "reminder":
            considers = True
        if probe == "demonstration":
            knows,considers = actual,True
        if probe == "tool":
            actual,knows,considers = True,True,True
        if probe == "retarget":
            goal = 1
        if probe == "search":
            budget = 2
        if probe == "higher-reward":
            reward = 2
        intended = int(knows and considers and budget >= 2 and goal == 1 and reward > cost)
        error = lapse if probe != "baseline" else 0
        expected = ((1-error)*intended + error*(1-intended)) if actual else 0
        assert abs(response_probability(cause,cost,lapse,probe)-expected) < 1e-10


def test_search_effort_is_distinct_from_considered_options():
    maker = make("search")
    choice,visited = decide(maker)
    assert maker.considered == (0,1)
    assert visited == [0]
    assert choice == 0
    assert decide(replace(maker,search_budget=2))[0] == 1
