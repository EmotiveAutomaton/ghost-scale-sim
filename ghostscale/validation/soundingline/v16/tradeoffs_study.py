"""V01/V02 actual acquired preparation, dated choices and withheld continuation."""
import math
import random
import uuid
from .records import seed_for,digest,read,write,now
from .dependency_monitor import prepare
from .tradeoffs_world import batch,public_training
from .tradeoffs import design,POLICIES,EXTENSION
from .estimands import Estimand,paired_summary

def law_for(namespace,constructor_id):
    rng=random.Random(seed_for(namespace,"tradeoff-law",constructor_id))
    return {"low":rng.uniform(0.15,0.30),"high":rng.uniform(0.70,0.85),"beta":rng.uniform(4.0,10.0)}

def context_for(namespace,index,date,condition,world,initial,*,future=False):
    rng=random.Random(seed_for(namespace,"tradeoff-context",index,date))
    required=rng.randrange(3)
    standard=[rng.randrange(2) for _ in range(3)]
    if condition["purpose"]!="varied":
        required=1 if condition["purpose"]=="stable" or date<4 else 2
        standard=[value if condition["purpose"]=="stable" or date<4 else 1-value for value in world["defaults"]]
    price=rng.uniform(0.03,0.15)
    if condition["price_regime"]=="costly":
        price*=3
    steps=condition["steps"] if condition["steps"] is not None else rng.choice([4,6])
    search_budget=rng.choice([64,256])
    consideration=condition["consideration"] if condition["consideration"]!="varied" else rng.choice(["all","familiar"])
    retention=condition["retention"] if condition["retention"]!="varied" else rng.choice(["random","selected"])
    return {"date":date,"initial":list(initial),"required_part":required,"orientation_standard":standard,"price":price,
        "max_steps":steps,"search_budget":search_budget,"consideration":consideration,
        "retention":"random" if future else retention,"produced_count":1 if future else 2}

def execute_unit(root,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery",card="V01"):
    uid=digest([namespace,condition["id"],index])[:24]
    destination=root/"units"/f"{uid}_points.json"
    if destination.exists():
        row=read(destination)
        if row["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("tradeoff packet identity differs")
        return row
    acquired=prepare(namespace,{"kind":condition["kind"]},index,constructors)
    world=acquired["world"]
    old=acquired["current_execution"]["state"]
    law=law_for(namespace,acquired["constructor_id"])
    history=[];initial=list(old)
    for date in range(8):
        context=context_for(namespace,index,date,condition,world,initial)
        event=batch(world,context,law,condition["profile"],acquired["routine"],old,
                    random.Random(seed_for(namespace,"tradeoff-choice",condition["id"],index,date)))
        history.append({"context":context,"batch":event})
        initial=event["candidates"][event["retained_index"]]["execution"]["state"] if card=="V02" else list(old)
    original={"acquired":acquired,"law":law,"profile":condition["profile"],"history":history}
    original_hash=write(root/"private"/f"{uid}-original.json",original)
    path=root/"public"/f"{uid}.json"
    task_id=read(path)["task_id"] if path.exists() else uuid.uuid4().hex
    future_context=context_for(namespace,index,8,condition,world,initial,future=True)
    public={"schema_version":"v16.tradeoffs.1","task_id":task_id,"world":world,"law":law,
        "training":public_training(acquired["training"]),"history":[{"context":event["context"],
            "state":event["batch"]["candidates"][event["batch"]["retained_index"]]["execution"]["state"]} for event in history],
        "future_context":future_context}
    public_hash=write(path,public)
    predictions={policy:reader.request(EXTENSION,public,policy=policy) for policy in POLICIES}
    if any(prediction["model_mismatch"] for prediction in predictions.values()):
        raise ValueError("correctly specified public opportunity unexpectedly lost actual support")
    prediction_path=root/"predictions"/f"{uid}.json"
    submitted=read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash=write(prediction_path,{"public_hash":public_hash,"submitted_at":submitted,"arms":predictions})
    # Every policy committed before the new, unselected executed choice.
    future=batch(world,future_context,law,condition["profile"],acquired["routine"],old,
                 random.Random(seed_for(namespace,"tradeoff-future",condition["id"],index)))
    execution=future["candidates"][0]["execution"]
    private_hash=write(root/"private"/f"{uid}.json",{"original_hash":original_hash,"future":future})
    training_cost=sum(trial["execution"]["primitive_cost"] for trial in acquired["training"])
    original_search=sum(event["batch"]["menu"]["successor_evaluations"] for event in history)
    production=sum(candidate["execution"]["primitive_cost"] for event in history for candidate in event["batch"]["candidates"])
    arms={}
    for policy,prediction in predictions.items():
        position=prediction["states"].index(execution["state"])
        errors=[(probability-float(i==position))**2 for i,probability in enumerate(prediction["probabilities"])]
        outcomes={"future_choice_log_score":math.log(prediction["probabilities"][position]),
            "future_choice_brier":sum(errors),"revision_cost_accuracy":-abs(prediction["expected_revision_primitives"]-execution["primitive_cost"]),
            "future_legal":float(execution["legal"] and execution["successfully_stopped"]),
            "future_revision_primitives":float(execution["primitive_cost"]),
            "original_search_primitives":float(original_search),"original_production_primitives":float(production),
            "future_search_primitives":float(future["menu"]["successor_evaluations"]),"training_primitives":float(training_cost),
            "old_execution_primitives":float(acquired["current_execution"]["primitive_cost"]),"routine_definition_primitives":float(len(acquired["routine"])),
            "future_considered_targets":float(len(future["menu"]["considered_targets"])),"future_found_options":float(len(future["menu"]["options"])),
            "future_search_exhausted":float(future["menu"]["search_exhausted"])}
        outcomes.update({f"reader_{key}":float(value) for key,value in prediction["costs"].items()})
        arms[policy]={"outcomes":outcomes}
    row={"unit_id":uid,"card_id":card,"condition":condition["id"],"condition_spec":condition,
        "constructor_id":acquired["constructor_id"],"maker_history_id":digest([namespace,"maker",condition["id"],index])[:24],
        "lineage":namespace,"seed_components":{"index":index,"constructors":constructors},"evidence_scope":scope,
        "packet_hash":packet["packet_hash"],"public_hash":public_hash,"private_hash":private_hash,"prediction_hash":prediction_hash,
        "prediction_submitted_at":submitted,"scored_at":now(),"arms":arms,"failures":[]}
    write(destination,row)
    return row

def summarize(rows,card="V01"):
    specification=design(card);conditions={}
    for condition in specification["conditions"]:
        selected=[row for row in rows if row["condition"]==condition["id"]]
        contrasts=[Estimand(f"{card}-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
                   for a,b,metric,units,bar in specification["primary"]]
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in contrasts],
            "arms":{policy:{metric:sum(row["arms"][policy]["outcomes"][metric] for row in selected)/len(selected)
                for metric in selected[0]["arms"][policy]["outcomes"]} for policy in POLICIES}}
    return {"card_id":card,"conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
        "instrument_state":"valid","independent_reaggregation":"pending","scope_limit":specification["scope"]}
