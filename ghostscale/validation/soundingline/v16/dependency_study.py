"""S03 two committed phases with paid observations and actual dependency repairs."""
import copy
import uuid
from .records import read,write,digest,now
from .dependency_monitor import DESIGN,POLICIES,EXTENSION,prepare,conflict
from .assembly_reference import interpret
from .estimands import Estimand,paired_summary


def execute_unit(root,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery"):
    uid=digest([namespace,condition["id"],index])[:24]
    destination=root/"units"/f"{uid}_points.json"
    if destination.exists():
        row=read(destination)
        if row["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("dependency unit differs from source lock")
        return row
    private=prepare(namespace,condition,index,constructors)
    public_path=root/"public"/f"{uid}.json"
    opaque=read(public_path)["task_id"] if public_path.exists() else uuid.uuid4().hex
    state=private["current_execution"]["state"]
    public={"schema_version":"v16.dependency-monitor.1","task_id":opaque,"world":private["world"],
            "training":private["training"],"goal":private["goal"],"visible_parts":[i for i,value in enumerate(state) if value>=0],
            "phase":1,"preparatory_program":[],"inspected_state":None,"search_budget":condition["budget"],"old_routine_reused":True}
    public_hash=write(public_path,public)
    first={name:reader.request(EXTENSION,public,policy=name) for name in POLICIES}
    first_path=root/"predictions"/f"{uid}-phase-1.json"
    submitted=read(first_path)["submitted_at"] if first_path.exists() else now()
    first_hash=write(first_path,{"submitted_at":submitted,"public_hash":public_hash,"arms":first})
    preparations={};second_inputs={}
    for name,choice in first.items():
        if 9 in choice["preparatory_program"]:
            raise ValueError("preparatory phase must not stop before inspection/repair")
        result=interpret(private["world"],choice["preparatory_program"],initial=state)
        preparations[name]=result
        second_inputs[name]={**copy.deepcopy(public),"phase":2,"preparatory_program":choice["preparatory_program"],
            "visible_parts":[i for i,value in enumerate(result["state"]) if value>=0],
            "inspected_state":result["state"] if choice["request_inspection"] else None}
    second={name:reader.request(EXTENSION,observation,policy=name) for name,observation in second_inputs.items()}
    second_path=root/"predictions"/f"{uid}.json"
    repaired_at=read(second_path)["submitted_at"] if second_path.exists() else now()
    prediction_hash=write(second_path,{"submitted_at":repaired_at,"phase_one_hash":first_hash,
                                       "public_inputs":second_inputs,"arms":second})
    arms={}
    for name,prediction in second.items():
        start=preparations[name]["state"]
        execution=interpret(private["world"],prediction["repair"]["program"],initial=start)
        actual_conflict=conflict(start,private["goal"])
        detected=prediction["detected_conflict"]
        collateral=sum(3<=item["action"]<=5 and private["goal"][item["action"]-3]>=0 and item["legal"]
                       for item in execution["trace"])
        success=execution["legal"] and execution["successfully_stopped"] and not prediction["repair"]["search_timeout"] and execution["state"]==private["goal"]
        arms[name]={"preparation":preparations[name],"repair_execution":execution,"outcomes":{
            "success":float(success),"detected_conflict":float(detected),"actual_conflict":float(actual_conflict),
            "false_alarm":float(detected and not actual_conflict),"missed_conflict":float(actual_conflict and not detected),
            "actions_before_monitor_decision":float(preparations[name]["primitive_cost"]),
            "collateral_removals":float(collateral),"collateral_saving":float(-collateral),
            "inspection_queries":float(first[name]["request_inspection"]),
            "preparation_primitives":float(preparations[name]["primitive_cost"]),"repair_primitives":float(execution["primitive_cost"]),
            "old_execution_primitives":float(private["current_execution"]["primitive_cost"]),
            "training_primitives":float(sum(item["execution"]["primitive_cost"] for item in private["training"])),
            "routine_definition_cost":float(len(private["routine"])),"old_goal_error_change":float(private["old_goal_error_change"]),
            "search_evaluations":float(prediction["repair"]["successor_evaluations"]),
            "checking":float(first[name]["costs"]["checking"]+prediction["costs"]["checking"]),
            "simulated_primitives":float(prediction["costs"]["simulation_primitives"])}}
    private_hash=write(root/"private"/f"{uid}.json",private)
    row={"unit_id":uid,"card_id":"S03","condition":condition["id"],"condition_spec":condition,
         "constructor_id":private["constructor_id"],"maker_history_id":digest([namespace,condition["kind"],index])[:24],
         "lineage":namespace,"evidence_scope":scope,"seed_components":{"index":index,"constructors":constructors},
         "packet_hash":packet["packet_hash"],"public_hash":public_hash,"private_hash":private_hash,"prediction_hash":prediction_hash,
         "arms":arms,"failures":[],"prediction_submitted_at":submitted,"scored_at":now()}
    write(destination,row)
    return row


def summarize(rows):
    conditions={}
    for condition in DESIGN["conditions"]:
        selected=[row for row in rows if row["condition"]==condition["id"]]
        specs=[Estimand(f"S03-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
               for a,b,metric,units,bar in DESIGN["primary"]]
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in specs],
            "arms":{name:{metric:sum(row["arms"][name]["outcomes"][metric] for row in selected)/len(selected)
                          for metric in selected[0]["arms"][name]["outcomes"]} for name in POLICIES}}
    return {"card_id":"S03","conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
            "instrument_state":"valid","independent_reaggregation":"pending","scope_limit":DESIGN["memory_scope"]}
