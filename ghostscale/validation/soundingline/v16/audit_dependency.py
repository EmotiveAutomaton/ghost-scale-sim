"""Independent S03 learning, information access, physical repairs and statistics."""
from collections import Counter
import hashlib
import json
from .assembly_reference import interpret
from .audit_statistics import verify


def load(path,expected):
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError("dependency raw commitment mismatch")
    return json.loads(payload)


def conflict(state,goal):
    return any(value>=0 and value!=goal[index] for index,value in enumerate(state))


def audit_unit(root,row):
    uid=row["unit_id"]
    public=load(root/"public"/f"{uid}.json",row["public_hash"])
    private=load(root/"private"/f"{uid}.json",row["private_hash"])
    final=load(root/"predictions"/f"{uid}.json",row["prediction_hash"])
    first=load(root/"predictions"/f"{uid}-phase-1.json",final["phase_one_hash"])
    if first["public_hash"]!=row["public_hash"]:
        raise ValueError("first decision used another public observation")
    world,goal=private["world"],private["goal"]
    if public["training"]!=private["training"] or public["world"]!=world or public["goal"]!=goal:
        raise ValueError("permitted own training/world/goal differs from execution contract")
    counts=Counter()
    training_cost=0
    for example in private["training"]:
        result=interpret(world,example["program"])
        expected=result["legal"] and result["successfully_stopped"] and result["state"]==example["target"]
        if result!=example["execution"] or example["feedback"]!=expected:
            raise ValueError("old routine acquisition did not execute as recorded")
        training_cost+=result["primitive_cost"]
        if expected:
            counts[tuple(example["program"])]+=1
    available=[program for program,count in counts.items() if count>=3]
    routine=list(min(available,key=lambda program:(-counts[program],program)))[:-1] if available else []
    if routine!=private["routine"]:
        raise ValueError("old procedure was not acquired from successful repeated trials")
    current=interpret(world,routine)
    if current!=private["current_execution"]:
        raise ValueError("old routine was not actually executed")
    improvement=sum(a!=b for a,b in zip([-1,-1,-1],goal))-sum(a!=b for a,b in zip(current["state"],goal))
    if private["old_goal_error_change"]!=improvement:
        raise ValueError("initial usefulness was not measured from actual goal error")
    for name,arm in row["arms"].items():
        first_choice=first["arms"][name]
        if 9 in first_choice["preparatory_program"]:
            raise ValueError("object stopped before repair opportunity")
        preparation=interpret(world,first_choice["preparatory_program"],initial=current["state"])
        if preparation!=arm["preparation"]:
            raise ValueError("preparatory actions differ from committed choice")
        visible=final["public_inputs"][name]
        expected_visible_parts=[index for index,value in enumerate(preparation["state"]) if value>=0]
        supplied=preparation["state"] if first_choice["request_inspection"] else None
        if visible!={**public,"phase":2,"preparatory_program":first_choice["preparatory_program"],
                     "visible_parts":expected_visible_parts,"inspected_state":supplied}:
            raise ValueError("monitor received unpurchased or incorrect current-state evidence")
        prediction=final["arms"][name]
        if supplied is not None:
            expected_assumed=list(supplied)
            simulated=0
        elif name in {"self-model","direct-simulation"}:
            expected_assumed=interpret(world,routine+first_choice["preparatory_program"])["state"]
            simulated=len(routine)+len(first_choice["preparatory_program"])
        else:
            expected_assumed=[world["defaults"][part] if part in expected_visible_parts else -1 for part in range(3)]
            simulated=0
        detected=conflict(expected_assumed,goal)
        if prediction["assumed_state"]!=expected_assumed or prediction["detected_conflict"]!=detected:
            raise ValueError("monitor used information beyond its declared route")
        repair=prediction["repair"]
        for attempt in repair["attempts"]:
            step=interpret(world,[attempt["action"]],initial=attempt["before"])
            if (step["state"],step["legal"],step["stopped"])!=(attempt["after"],attempt["legal"],attempt["stopped"]):
                raise ValueError("repair planner's primitive checking trace differs")
        if len(repair["attempts"])!=repair["successor_evaluations"] or len(repair["attempts"])>public["search_budget"]:
            raise ValueError("repair planner exceeded its counted work budget")
        execution=interpret(world,repair["program"],initial=preparation["state"])
        if execution!=arm["repair_execution"]:
            raise ValueError("repair did not physically execute as submitted")
        actual_conflict=conflict(preparation["state"],goal)
        collateral=sum(3<=step["action"]<=5 and goal[step["action"]-3]>=0 and step["legal"] for step in execution["trace"])
        success=execution["legal"] and execution["successfully_stopped"] and not repair["search_timeout"] and execution["state"]==goal
        expected={"success":float(success),"detected_conflict":float(detected),"actual_conflict":float(actual_conflict),
            "false_alarm":float(detected and not actual_conflict),"missed_conflict":float(actual_conflict and not detected),
            "actions_before_monitor_decision":float(preparation["primitive_cost"]),
            "collateral_removals":float(collateral),"collateral_saving":float(-collateral),
            "inspection_queries":float(first_choice["request_inspection"]),"preparation_primitives":float(preparation["primitive_cost"]),
            "repair_primitives":float(execution["primitive_cost"]),"old_execution_primitives":float(current["primitive_cost"]),
            "training_primitives":float(training_cost),"routine_definition_cost":float(len(routine)),
            "old_goal_error_change":float(improvement),"search_evaluations":float(repair["successor_evaluations"]),
            "checking":6.0,"simulated_primitives":float(simulated)}
        if expected!=arm["outcomes"]:
            raise ValueError("independent dependency outcome mismatch")
    # Same evidence and ordinary forward simulation are a mandatory serious rival.
    if final["arms"]["self-model"]!=final["arms"]["direct-simulation"]:
        raise ValueError("equally informed forward-monitor identity failed")


def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    count=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
            "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,"full_rollout_replay":False}
