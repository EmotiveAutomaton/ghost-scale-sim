"""M03 original history, paid query commitments, and genuinely executed new work."""
import copy
import math
import random
import uuid
from .records import digest,seed_for,read,write,now
from .recognition import constructor,acquire,produce,observe
from .selection_study import make_batch,observation
from .audience import DESIGN,POLICIES,EXTENSION
from .graphic_reference import interpret
from .estimands import Estimand,paired_summary

def execute_unit(root,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery"):
    uid=digest([namespace,condition["id"],index])[:24]
    destination=root/"units"/f"{uid}_points.json"
    if destination.exists():
        row=read(destination)
        if row["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("audience packet source differs")
        return row
    constructor_id=f"constructor-{index%constructors:03d}"
    world=constructor(namespace,constructor_id,condition)
    maker=acquire(namespace,index,0,world)
    audience=random.Random(seed_for(namespace,"audience",index)).randrange(2)
    original={"world":world,"maker":maker,"audience":audience,
        "batches":[make_batch(world,maker,namespace,index,date,condition["count"],condition["retention"],audience) for date in range(3)]}
    original_hash=write(root/"private"/f"{uid}-original.json",original)
    path=root/"public"/f"{uid}.json"
    task_id=read(path)["task_id"] if path.exists() else uuid.uuid4().hex
    public={"schema_version":"v16.audience.1","task_id":task_id,"phase":1,
        "world":{key:world[key] for key in ["permutation","style_reuse","core_reuse"]},
        "history":[observation(batch,process=False,include_rejected=False) for batch in original["batches"]],
        "retention":condition["retention"],"produced_count":condition["count"],"audience":None,"rehearsals":[],"target_topic":1}
    public_hash=write(path,public)
    first={policy:reader.request(EXTENSION,public,policy=policy) for policy in POLICIES}
    first_path=root/"predictions"/f"{uid}-phase-1.json"
    requested_at=read(first_path)["submitted_at"] if first_path.exists() else now()
    first_hash=write(first_path,{"submitted_at":requested_at,"public_hash":public_hash,"arms":first})
    # All requests precede these actual new executions and every answer disclosure.
    rehearsal=[produce(world,maker,random.Random(seed_for(namespace,"audience-rehearsal",index,date)),1) for date in range(2)]
    second_public={}
    for policy,choice in first.items():
        queries=choice["queries"]
        if queries["rehearsals"] not in [0,2] or type(queries["audience"]) is not bool:
            raise ValueError("unregistered paid query")
        second_public[policy]={**copy.deepcopy(public),"phase":2,
            "audience":audience if queries["audience"] else None,
            "rehearsals":[observe(event,process=True) for event in rehearsal] if queries["rehearsals"] else []}
    predictions={policy:reader.request(EXTENSION,visible,policy=policy) for policy,visible in second_public.items()}
    prediction_path=root/"predictions"/f"{uid}.json"
    submitted=read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash=write(prediction_path,{"submitted_at":submitted,"phase_one_hash":first_hash,
                                          "public_inputs":second_public,"arms":predictions})
    future=produce(world,maker,random.Random(seed_for(namespace,"audience-future",index)),1)
    old_orders=[batch["candidates"][batch["retained_indices"][0]]["core_order"] for batch in original["batches"]]
    history_index=sum(bit*(2**(2-position)) for position,bit in enumerate(old_orders))
    p=world["permutation"]
    targets=[sum(1<<cell for cell in p[:2]+[p[3]]+p[4+2*style:6+2*style]) for style in [0,1]]
    training=sum(trial["execution"]["primitive_cost"] for part in maker.values() for trial in part["training"])
    original_cost=sum(event["execution"]["primitive_cost"] for batch in original["batches"] for event in batch["candidates"])
    rehearsal_cost=sum(event["execution"]["primitive_cost"] for event in rehearsal)
    arms={}
    for policy,prediction in predictions.items():
        construction=interpret(prediction["new_composition"])
        legal=construction["legal"] and construction["stopped"]
        queries=first[policy]["queries"]
        arms[policy]={"construction_execution":construction,"outcomes":{
            "audience_success":float(legal and construction["artifact"]==targets[audience]),
            "brief_success":float(legal and construction["artifact"] in targets),"legal":float(legal),
            "historical_core_log_score":math.log(prediction["historical_core"][old_orders[0]]),
            "historical_vector_log_score":math.log(prediction["historical_vector"][history_index]),
            "acquired_core_log_score":math.log(prediction["acquired_core"][maker["core"]["choice"]]),
            "future_core_log_score":math.log(prediction["future_core"][future["core_order"]]),
            "future_style_log_score":math.log(prediction["future_style"][future["decoration"]]),
            "audience_log_score":math.log(prediction["audience"][audience]),
            "audience_queries":float(queries["audience"]),"rehearsal_observations":float(queries["rehearsals"]),
            "rehearsal_production_primitives":float(rehearsal_cost if queries["rehearsals"] else 0),
            "training_primitives":float(training),"library_definition_cost":4.0,
            "original_production_primitives":float(original_cost),
            "future_production_primitives":float(future["execution"]["primitive_cost"]),
            "reader_construction_primitives":float(construction["primitive_cost"]),
            "likelihood_terms":float(prediction["costs"]["likelihood_terms"]),
            "prediction_terms":float(prediction["costs"]["prediction_terms"])}}
    private_hash=write(root/"private"/f"{uid}.json",{"original_hash":original_hash,"rehearsals":rehearsal,"future":future})
    row={"unit_id":uid,"card_id":"M03","condition":condition["id"],"condition_spec":condition,
         "constructor_id":constructor_id,"maker_history_id":digest([namespace,"maker",index])[:24],
         "lineage":namespace,"seed_components":{"index":index,"constructors":constructors},"evidence_scope":scope,
         "packet_hash":packet["packet_hash"],"public_hash":public_hash,"prediction_hash":prediction_hash,"private_hash":private_hash,
         "arms":arms,"failures":[],"prediction_submitted_at":submitted,"scored_at":now()}
    write(destination,row)
    return row

def summarize(rows):
    conditions={}
    for condition in DESIGN["conditions"]:
        selected=[row for row in rows if row["condition"]==condition["id"]]
        comparisons=DESIGN["primary"]+[("audience-only","none","historical_core_log_score","nats_per_event",0.02)]
        specs=[Estimand(f"M03-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
               for a,b,metric,units,bar in comparisons]
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in specs],
            "arms":{policy:{metric:sum(row["arms"][policy]["outcomes"][metric] for row in selected)/len(selected)
                           for metric in selected[0]["arms"][policy]["outcomes"]} for policy in POLICIES}}
    return {"card_id":"M03","conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
            "instrument_state":"valid","independent_reaggregation":"pending","scope_limit":DESIGN["history_target"]}
