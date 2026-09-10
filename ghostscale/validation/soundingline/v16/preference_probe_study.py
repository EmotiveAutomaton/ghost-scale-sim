"""V03 indistinguishable original histories, paid requests, probes and fresh outcomes."""
import copy
import math
import random
import uuid
from .records import seed_for,digest,read,write,now
from .dependency_monitor import prepare
from .tradeoffs_study import law_for
from .tradeoffs_world import public_training
from .preference_probe_world import produce,CAUSES
from .preference_probe import DESIGN,POLICIES,EXTENSION
from .assembly_reference import interpret
from .estimands import Estimand,paired_summary

def task_context(namespace,index,condition,acquired):
    rng=random.Random(seed_for(namespace,"probe-task",index))
    return {"date":8,"initial":acquired["current_execution"]["state"],"required_part":2,
        "orientation_standard":list(acquired["world"]["defaults"]),"price":rng.uniform(0.03,0.10),
        "max_steps":6,"search_budget":condition["search_budget"],"consideration":"all","retention":"random","produced_count":1}

def execute_unit(root,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery"):
    uid=digest([namespace,condition["id"],index])[:24];destination=root/"units"/f"{uid}_points.json"
    if destination.exists():
        row=read(destination)
        if row["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("preference probe source identity differs")
        return row
    acquired=prepare(namespace,{"kind":condition["kind"]},index,constructors)
    world=acquired["world"];old=acquired["current_execution"]["state"]
    law=law_for(namespace,acquired["constructor_id"])
    base=task_context(namespace,index,condition,acquired)
    history=[produce(world,{**base,"date":date},law,condition["cause"],"history",acquired["routine"],old,
                     random.Random(seed_for(namespace,"probe-forced-history",index,date))) for date in range(8)]
    original_hash=write(root/"private"/f"{uid}-original.json",{"acquired":acquired,"cause":condition["cause"],"law":law,"history":history})
    path=root/"public"/f"{uid}.json"
    task_id=read(path)["task_id"] if path.exists() else uuid.uuid4().hex
    public={"schema_version":"v16.preference-probe.1","task_id":task_id,"phase":1,"world":world,"law":law,
        "training":public_training(acquired["training"]),"history":[{"context":event["context"],"state":event["execution"]["state"]} for event in history],
        "task_context":base,"query_fees":{name:condition["query_fee"]*multiplier for name,multiplier in [("open",1),("time",1.5),("neutral",1.5),("all",2)]},"answer":None}
    public_hash=write(path,public)
    first={policy:reader.request(EXTENSION,public,policy=policy) for policy in POLICIES}
    first_path=root/"predictions"/f"{uid}-phase-1.json"
    requested_at=read(first_path)["submitted_at"] if first_path.exists() else now()
    first_hash=write(first_path,{"submitted_at":requested_at,"public_hash":public_hash,"arms":first})
    queries=sorted({choice["query"] for choice in first.values() if choice["query"] is not None})
    # Unique requested interventions execute once, yoked across equal-query arms.
    probes={query:produce(world,base,law,condition["cause"],query,acquired["routine"],old,
            random.Random(seed_for(namespace,"preference-probe",condition["id"],index,query))) for query in queries}
    inputs={}
    for policy,choice in first.items():
        query=choice["query"]
        inputs[policy]={**copy.deepcopy(public),"phase":2,"answer":
            {"query":query,"state":probes[query]["execution"]["state"]} if query is not None else None}
    predictions={policy:reader.request(EXTENSION,visible,policy=policy) for policy,visible in inputs.items()}
    if any(prediction["model_mismatch"] for prediction in predictions.values()):
        raise ValueError("known probe planner lost actual support")
    prediction_path=root/"predictions"/f"{uid}.json"
    submitted=read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash=write(prediction_path,{"phase_one_hash":first_hash,"submitted_at":submitted,"public_inputs":inputs,"arms":predictions})
    future=produce(world,base,law,condition["cause"],"future",acquired["routine"],old,
        random.Random(seed_for(namespace,"probe-future",condition["id"],index)))
    private_hash=write(root/"private"/f"{uid}.json",{"original_hash":original_hash,"probes":probes,"future":future})
    training=sum(trial["execution"]["primitive_cost"] for trial in acquired["training"])
    programs=[event["menu"]["options"][event["choice"]]["program"] for event in history]
    arms={}
    for policy,prediction in predictions.items():
        query=first[policy]["query"]
        reproduction=interpret(world,prediction["reproduction_program"])
        position=prediction["states"].index(future["execution"]["state"])
        outcomes={"future_choice_log_score":math.log(prediction["probabilities"][position]),
            "future_choice_brier":sum((p-float(i==position))**2 for i,p in enumerate(prediction["probabilities"])),
            "cause_log_score":math.log(prediction["cause_posterior"][CAUSES.index(condition["cause"])]),
            "cause_entropy":prediction["cause_entropy"],
            "artifact_reproduction_success":float(reproduction["successfully_stopped"] and reproduction["state"]==old),
            "historical_program_success":float(prediction["historical_programs"]==programs),
            "query_count":float(query is not None),"query_fee":public["query_fees"][query] if query else 0.0,
            "probe_production_primitives":float(probes[query]["execution"]["primitive_cost"] if query else 0),
            "probe_search_primitives":float(probes[query]["menu"]["successor_evaluations"] if query else 0),
            "reproduction_primitives":float(reproduction["primitive_cost"]),
            "training_primitives":float(training),"old_execution_primitives":float(acquired["current_execution"]["primitive_cost"]),
            "routine_definition_primitives":float(len(acquired["routine"])),
            "history_production_primitives":float(sum(event["execution"]["primitive_cost"] for event in history)),
            "history_search_primitives":float(sum(event["menu"]["successor_evaluations"] for event in history)),
            "future_production_primitives":float(future["execution"]["primitive_cost"]),
            "future_search_primitives":float(future["menu"]["successor_evaluations"])}
        outcomes.update({f"reader_{key}":float(first[policy]["costs"].get(key,0)+prediction["costs"].get(key,0))
                         for key in set(first[policy]["costs"])|set(prediction["costs"])})
        arms[policy]={"reproduction_execution":reproduction,"outcomes":outcomes}
    row={"unit_id":uid,"card_id":"V03","condition":condition["id"],"condition_spec":condition,
        "constructor_id":acquired["constructor_id"],"maker_history_id":digest([namespace,"maker",condition["id"],index])[:24],
        "lineage":namespace,"seed_components":{"index":index,"constructors":constructors},"evidence_scope":scope,
        "packet_hash":packet["packet_hash"],"public_hash":public_hash,"private_hash":private_hash,"prediction_hash":prediction_hash,
        "prediction_submitted_at":submitted,"scored_at":now(),"arms":arms,"failures":[]}
    write(destination,row)
    return row

def summarize(rows):
    conditions={}
    for condition in DESIGN["conditions"]:
        selected=[row for row in rows if row["condition"]==condition["id"]]
        specifications=[Estimand(f"V03-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
            for a,b,metric,units,bar in DESIGN["primary"]]
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in specifications],
            "arms":{policy:{metric:sum(row["arms"][policy]["outcomes"][metric] for row in selected)/len(selected)
                for metric in selected[0]["arms"][policy]["outcomes"]} for policy in POLICIES}}
    return {"card_id":"V03","conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
        "instrument_state":"valid","independent_reaggregation":"pending","scope_limit":DESIGN["scope"],
        "inquiry_target_boundary":DESIGN["target_comparison_limit"]}
