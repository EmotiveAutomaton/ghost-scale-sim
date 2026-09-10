"""M03 known audience/history independence and actual construction controls."""
import copy
import json
from .audience import public_reader
from .selection_gates import fixture as selection_fixture
from .graphic_reference import interpret

def fixture(policy,*,phase=2):
    base=selection_fixture()
    process=policy in {"rehearsal-only","both","direct-both"}
    audience=policy in {"audience-only","both","direct-both"}
    rehearsals=[{"artifact":interpret([1,0,3,4,5])["artifact"],"first_action":1} for _ in range(2)]
    return {"schema_version":"v16.audience.1","task_id":"known-audience","phase":phase,"world":base["world"],
        "history":base["history"],"retention":base["retention"],"produced_count":base["produced_count"],
        "audience":1 if audience and phase==2 else None,"rehearsals":rehearsals if process and phase==2 else [],"target_topic":1}

def run():
    none=public_reader(json.dumps(fixture("none")).encode(),"none")
    audience=public_reader(json.dumps(fixture("audience-only")).encode(),"audience-only")
    both=public_reader(json.dumps(fixture("both")).encode(),"both")
    direct=public_reader(json.dumps(fixture("direct-both")).encode(),"direct-both")
    initial=fixture("both",phase=1)
    before=copy.deepcopy(initial)
    request=public_reader(json.dumps(initial).encode(),"both")
    work=interpret(both["new_composition"])
    checks={
        "audience-history-independence":{
            "audience_does_not_rewrite_old_core_inference":all(abs(a-b)<1e-12 for key in ["historical_core","historical_vector"]
                                                             for a,b in zip(none[key],audience[key])),
            "rehearsal_can_update_core":both["historical_core"][1]>0.8,
            "all_colliding_old_routes_retain_support":all(value>0 for value in both["historical_vector"]),
            "original_observation_unchanged":initial==before},
        "audience-paid-queries":{"requests_precede_answers":request=={"queries":{"audience":True,"rehearsals":2}},
            "direct_rival_receives_same_information":all(abs(a-b)<1e-12 for key in ["historical_core","historical_vector","audience","future_core","future_style"]
                                                        for a,b in zip(both[key],direct[key]))},
        "audience-construction":{"audience_changes_real_program":none["new_composition"]!=audience["new_composition"],
            "audience_one_target_executes":work["legal"] and work["artifact"]==sum(2**cell for cell in [0,1,3,6,7]),
            "five_primitive_budget":work["primitive_cost"]==5,
            "direct_rival_constructs_same_work":both["new_composition"]==direct["new_composition"]}}
    gates=[{"id":name,"checks":values,"instrument_state":"valid" if all(values.values()) else "failed","evidence_scope":"fixture"}
           for name,values in checks.items()]
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}
