"""Independent V03 paid-access joins, real interventions and all outcome reductions."""
import hashlib
import json
import math
import random
from .audit_graphic_maker import seed
from .audit_tradeoffs import preparation
from .assembly_reference import interpret
from .preference_probe_reference import model,choice,prediction,CAUSES
from .audit_statistics import verify

def load(path,expected):
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError("preference-probe commitment mismatch")
    return json.loads(payload)

def check_event(event,world,base,law,cause,query,routine,old,rng):
    expected=model(world,base,law,cause,query,routine,old)
    for key in ["context","intrinsic","audience_coefficient","menu"]:
        if event[key]!=expected[key]:
            raise ValueError("actual probe intervention or bounded opportunity differs")
    for key in ["utilities","probabilities"]:
        if len(event[key])!=len(expected[key]) or any(abs(a-b)>1e-12 for a,b in zip(event[key],expected[key])):
            raise ValueError("probe utility or action law differs independently")
    point=rng.random();total=0;selected=len(expected["probabilities"])-1
    for index,probability in enumerate(expected["probabilities"]):
        total+=probability
        if point<total:
            selected=index;break
    execution=interpret(world,expected["menu"]["options"][selected]["program"],initial=base["initial"])
    if selected!=event["choice"] or execution!=event["execution"]:
        raise ValueError("probe or future did not execute the actual sampled program")
    return execution

def compare_choice(actual,expected):
    if actual["query"]!=expected["query"] or actual["costs"]!=expected["costs"]:
        raise ValueError("inquiry target, paid query or counted work differs independently")
    for key in ["information_gain","net_information"]:
        if (actual[key] is None)!=(expected[key] is None) or (actual[key] is not None and abs(actual[key]-expected[key])>1e-11):
            raise ValueError("direct mutual information differs from expected entropy reduction")

def audit_unit(root,row):
    uid=row["unit_id"]
    public=load(root/"public"/f"{uid}.json",row["public_hash"])
    private=load(root/"private"/f"{uid}.json",row["private_hash"])
    original=load(root/"private"/f"{uid}-original.json",private["original_hash"])
    final=load(root/"predictions"/f"{uid}.json",row["prediction_hash"])
    first=load(root/"predictions"/f"{uid}-phase-1.json",final["phase_one_hash"])
    if first["public_hash"]!=row["public_hash"] or first["submitted_at"]>final["submitted_at"] or final["submitted_at"]!=row["prediction_submitted_at"] or final["submitted_at"]>row["scored_at"]:
        raise ValueError("probe and future prediction commitments are not ordered")
    namespace,index=row["lineage"],row["seed_components"]["index"];condition=row["condition_spec"]
    prep=preparation(namespace,condition,index,row["seed_components"]["constructors"])
    if prep!=original["acquired"] or original["cause"]!=condition["cause"]:
        raise ValueError("probe acquisition or rival assignment differs")
    rng=random.Random(seed(namespace,"tradeoff-law",prep["constructor_id"]))
    law={"low":rng.uniform(0.15,0.30),"high":rng.uniform(0.70,0.85),"beta":rng.uniform(4.0,10.0)}
    if law!=original["law"] or len(original["history"])!=8:
        raise ValueError("probe constructor or original history differs")
    world=prep["world"];old=prep["current_execution"]["state"]
    rng=random.Random(seed(namespace,"probe-task",index))
    base={"date":8,"initial":old,"required_part":2,"orientation_standard":list(world["defaults"]),
        "price":rng.uniform(0.03,0.10),"max_steps":6,"search_budget":condition["search_budget"],
        "consideration":"all","retention":"random","produced_count":1}
    history=[];programs=[]
    for date,event in enumerate(original["history"]):
        result=check_event(event,world,{**base,"date":date},law,condition["cause"],"history",prep["routine"],old,
            random.Random(seed(namespace,"probe-forced-history",index,date)))
        history.append({"context":event["context"],"state":result["state"]})
        programs.append(event["menu"]["options"][event["choice"]]["program"])
        if programs[-1]!=[9] or result["state"]!=old:
            raise ValueError("original public histories are not actually indistinguishable")
    fees={name:condition["query_fee"]*multiplier for name,multiplier in [("open",1),("time",1.5),("neutral",1.5),("all",2)]}
    expected_public={"schema_version":"v16.preference-probe.1","task_id":public["task_id"],"phase":1,"world":world,"law":law,
        "training":[{key:trial[key] for key in ["date","program","target","feedback"]} for trial in prep["training"]],
        "history":history,"task_context":base,"query_fees":fees,"answer":None}
    if public!=expected_public:
        raise ValueError("unallowed or incorrect original public probe evidence")
    choices={policy:choice(public,policy) for policy in first["arms"]}
    for policy,expected in choices.items():
        compare_choice(first["arms"][policy],expected)
    needed={value["query"] for value in choices.values() if value["query"] is not None}
    if needed!=set(private["probes"]):
        raise ValueError("requested probe production missing or extra")
    for query,event in private["probes"].items():
        check_event(event,world,base,law,condition["cause"],query,prep["routine"],old,
            random.Random(seed(namespace,"preference-probe",condition["id"],index,query)))
    future=private["future"]
    result=check_event(future,world,base,law,condition["cause"],"future",prep["routine"],old,
        random.Random(seed(namespace,"probe-future",condition["id"],index)))
    for policy,arm in row["arms"].items():
        query=choices[policy]["query"]
        expected_input={**public,"phase":2,"answer":{"query":query,"state":private["probes"][query]["execution"]["state"]} if query else None}
        if final["public_inputs"][policy]!=expected_input:
            raise ValueError("prediction received unpurchased probe evidence")
        expected=prediction(expected_input,policy);actual=final["arms"][policy]
        for key in ["model_mismatch","states","historical_programs","reproduction_program","costs"]:
            if actual[key]!=expected[key]:
                raise ValueError("independent probe prediction support, programs or work differs")
        for key in ["probabilities","cause_posterior"]:
            if len(actual[key])!=len(expected[key]) or any(abs(a-b)>1e-11 for a,b in zip(actual[key],expected[key])):
                raise ValueError("independent probe posterior differs")
        if abs(actual["cause_entropy"]-expected["cause_entropy"])>1e-11:
            raise ValueError("residual rival ambiguity differs")
        reproduction=interpret(world,expected["reproduction_program"])
        if arm["reproduction_execution"]!=reproduction:
            raise ValueError("artifact reconstruction was not actually executed")
        position=expected["states"].index(result["state"])
        outcomes={"future_choice_log_score":math.log(expected["probabilities"][position]),
            "future_choice_brier":sum((p-float(i==position))**2 for i,p in enumerate(expected["probabilities"])),
            "cause_log_score":math.log(expected["cause_posterior"][CAUSES.index(condition["cause"])]),
            "cause_entropy":expected["cause_entropy"],
            "artifact_reproduction_success":float(reproduction["successfully_stopped"] and reproduction["state"]==old),
            "historical_program_success":float(expected["historical_programs"]==programs),
            "query_count":float(query is not None),"query_fee":fees[query] if query else 0.0,
            "probe_production_primitives":float(private["probes"][query]["execution"]["primitive_cost"] if query else 0),
            "probe_search_primitives":float(private["probes"][query]["menu"]["successor_evaluations"] if query else 0),
            "reproduction_primitives":float(reproduction["primitive_cost"]),
            "training_primitives":float(sum(trial["execution"]["primitive_cost"] for trial in prep["training"])),
            "old_execution_primitives":float(prep["current_execution"]["primitive_cost"]),
            "routine_definition_primitives":float(len(prep["routine"])),
            "history_production_primitives":float(sum(event["execution"]["primitive_cost"] for event in original["history"])),
            "history_search_primitives":float(sum(event["menu"]["successor_evaluations"] for event in original["history"])),
            "future_production_primitives":float(result["primitive_cost"]),"future_search_primitives":float(future["menu"]["successor_evaluations"])}
        outcomes.update({f"reader_{key}":float(choices[policy]["costs"].get(key,0)+expected["costs"].get(key,0))
            for key in set(choices[policy]["costs"])|set(expected["costs"])})
        if set(outcomes)!=set(arm["outcomes"]) or any(abs(value-arm["outcomes"][key])>1e-10 for key,value in outcomes.items()):
            raise ValueError("independent probe scores or access costs differ")
    if final["arms"]["all"]!=final["arms"]["direct-all"]:
        raise ValueError("same-information direct probe identity failed")

def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    count=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
        "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,"full_rollout_replay":False}
