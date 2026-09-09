import numpy as np
from ghostscale.validation.soundingline.v16.options import discover,expand,extrema,observed_transitions,plan
from ghostscale.validation.soundingline.v16.reference import interpret


def test_options_use_only_observed_edges_and_terminate():
    traces=[[0,1],[1,0],[0,2],[2,0]]*4
    observed=observed_transitions(traces)
    learned=discover(observed)
    assert learned["observed_state_fraction"]<1
    for option in learned["options"]:
        for start in map(int,option["initiation"]):
            expansion=expand(option,start,learned["edges"])
            assert expansion["terminated"]
            assert interpret(expansion["program"],start=start)["artifact"]==option["target"]
        assert expand(option,15,learned["edges"])["unsupported"]
    result=plan(7,options=learned,primitive_budget=4096)
    assert interpret(result["program"])["artifact"]==7


def test_empty_and_disconnected_observation_graph_are_not_complete_worlds():
    assert discover([])["options"]==[]
    observed=[{"before":0,"action":0,"after":1,"legal":True,"primitive_cost":1},
              {"before":4,"action":1,"after":6,"legal":True,"primitive_cost":1}]
    learned=discover(observed)
    assert learned["component_count"]==2
    for option in learned["options"]:
        for state in learned["observed_nodes"]:
            expanded=expand(option,state,learned["edges"])
            if expanded["terminated"]:
                assert (state in [0,1])==(option["target"] in [0,1])


def test_degenerate_eigenspace_extrema_are_basis_and_sign_invariant():
    basis=np.array([[1,0],[0,1],[-1,0],[0,-1]],dtype=float)/np.sqrt(2)
    angle=0.371
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    original=basis@basis.T
    changed=(basis@rotation)@(basis@rotation).T
    assert np.max(np.abs(original-changed))<1e-10
    assert extrema(original)==extrema(changed)==[0,2]
    assert extrema(original)==extrema((-basis)@(-basis).T)
