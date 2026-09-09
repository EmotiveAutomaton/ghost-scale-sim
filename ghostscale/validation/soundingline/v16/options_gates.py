"""Live, null, disconnected-graph and eigenbasis controls for native options."""
import numpy as np
from .options import discover,observed_transitions,expand,extrema,plan
from .reference import interpret


def run():
    learned=discover(observed_transitions([[0,1],[1,0],[0,2],[2,0]]*4))
    valid=True
    for option in learned["options"]:
        for start in map(int,option["initiation"]):
            expansion=expand(option,start,learned["edges"])
            valid &= expansion["terminated"] and interpret(expansion["program"],start=start)["artifact"]==option["target"]
    basis=np.array([[1,0],[0,1],[-1,0],[0,-1]],dtype=float)/np.sqrt(2)
    rotation=np.array([[0.6,-0.8],[0.8,0.6]])
    empty=discover([])
    cases={"options-observed-edges":{"incomplete_graph_is_reported":learned["observed_state_fraction"]<1,
                   "empty_graph_adds_no_skills":empty["options"]==[],
                   "empty_options_equal_primitive":plan(7,options=empty)==plan(7)},
           "options-termination":{"all_learned_policies_terminate":valid,
                   "unobserved_start_is_unsupported":all(expand(option,15,learned["edges"])["unsupported"] for option in learned["options"])},
           "options-degeneracy":{"rotated_basis_has_same_extrema":extrema(basis@basis.T)==extrema((basis@rotation)@(basis@rotation).T),
                   "sign_cannot_be_identity":extrema(basis@basis.T)==extrema((-basis)@(-basis).T)}}
    gates=[{"id":name,"checks":checks,"instrument_state":"valid" if all(checks.values()) else "failed",
            "evidence_scope":"fixture"} for name,checks in cases.items()]
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}
