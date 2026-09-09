"""M02 production precedes retention; future work follows committed predictions."""
import math
import random
import uuid
from .records import digest,seed_for,read,write,now
from .recognition import constructor,acquire,produce,observe
from .selection import DESIGN,METHODS,EXTENSION,retained
from .estimands import Estimand,paired_summary

def make_batch(world,maker,namespace,index,date,count,retention,audience,*,future=False):
    label="future-batch" if future else "batch"
    events=[produce(world,maker,random.Random(seed_for(namespace,label,index,date,position)),date%2) for position in range(count)]
    kept=retained(events,retention,audience,random.Random(seed_for(namespace,label+"-selection",index,date)))
    return {"candidates":events,"retained_indices":kept}

def observation(batch,*,process,include_rejected):
    events=batch["candidates"];kept=batch["retained_indices"]
    visible=[observe(event,process=process) for event in events]
    return {"released":[visible[i] for i in kept],
            "full_candidates":[{"observation":event,"retained":i in kept} for i,event in enumerate(visible)] if include_rejected else None}

def execute_unit(root,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery"):
    uid=digest([namespace,condition["id"],index])[:24]
    destination=root/"units"/f"{uid}_points.json"
    if destination.exists():
        row=read(destination)
        if row["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("selection source lock changed")
        return row
    constructor_id=f"constructor-{index%constructors:03d}"
    world=constructor(namespace,constructor_id,{"family":"learned-order"})
    maker=acquire(namespace,index,0,world)
    audience=random.Random(seed_for(namespace,"audience",index)).randrange(2)
    batches=[make_batch(world,maker,namespace,index,date,condition["count"],condition["retention"],audience) for date in range(3)]
    public_path=root/"public"/f"{uid}.json"
    task_id=read(public_path)["task_id"] if public_path.exists() else uuid.uuid4().hex
    public={"schema_version":"v16.selection.1","task_id":task_id,
            "world":{key:world[key] for key in ["permutation","style_reuse","core_reuse"]},
            "retention":condition["retention"],"produced_count":condition["count"],"audience":audience,
            "history":[observation(batch,process=condition["process"],include_rejected=condition["rejected"]) for batch in batches]}
    public_hash=write(public_path,public)
    predictions={method:reader.request(EXTENSION,public,method=method) for method in METHODS}
    prediction_path=root/"predictions"/f"{uid}.json"
    submitted=read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash=write(prediction_path,{"submitted_at":submitted,"public_hash":public_hash,"arms":predictions})
    raw_future=produce(world,maker,random.Random(seed_for(namespace,"raw-future",index)),1)
    release_future=make_batch(world,maker,namespace,index,0,condition["count"],condition["retention"],audience,future=True)
    training=sum(trial["execution"]["primitive_cost"] for part in maker.values() for trial in part["training"])
    produced=sum(event["execution"]["primitive_cost"] for batch in batches for event in batch["candidates"])
    future_cost=raw_future["execution"]["primitive_cost"]+sum(event["execution"]["primitive_cost"] for event in release_future["candidates"])
    retained_count=sum(len(batch["retained_indices"]) for batch in batches)
    rejected_count=3*condition["count"]-retained_count
    visible_count=3*condition["count"] if condition["rejected"] else retained_count
    actual_style_one=world["style_reuse"] if maker["decoration"]["choice"]==1 else 1-world["style_reuse"]
    arms={}
    for method,prediction in predictions.items():
        released=release_future["retained_indices"]
        arms[method]={"outcomes":{
            "future_raw_style_log_score":math.log(prediction["future_raw_style"][raw_future["decoration"]]),
            "future_raw_core_log_score":math.log(prediction["future_raw_core"][raw_future["core_order"]]),
            "future_release_style_log_score":sum(math.log(prediction["future_release_style"][release_future["candidates"][i]["decoration"]]) for i in released)/len(released),
            "acquired_style_log_score":math.log(prediction["acquired_style"][maker["decoration"]["choice"]]),
            "acquired_core_log_score":math.log(prediction["acquired_core"][maker["core"]["choice"]]),
            "raw_style_absolute_probability_error":abs(prediction["future_raw_style"][1]-actual_style_one),
            "training_primitives":float(training),"library_definition_cost":4.0,
            "observed_production_primitives":float(produced),"future_production_primitives":float(future_cost),
            "retained_works":float(retained_count),"rejected_works":float(rejected_count),
            "paid_rejected_work_observations":float(rejected_count if condition["rejected"] else 0),
            "process_observations":float(visible_count if condition["process"] else 0),
            "audience_observations":1.0,"likelihood_terms":float(prediction["costs"]["likelihood_terms"]),
            "prediction_terms":float(prediction["costs"]["prediction_terms"])}}
    private_hash=write(root/"private"/f"{uid}.json",{"world":world,"maker":maker,"audience":audience,
                       "batches":batches,"raw_future":raw_future,"release_future":release_future})
    row={"unit_id":uid,"card_id":"M02","condition":condition["id"],"condition_spec":condition,
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
        specs=[Estimand(f"M02-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
               for a,b,metric,units,bar in DESIGN["primary"]]
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in specs],
            "arms":{method:{metric:sum(row["arms"][method]["outcomes"][metric] for row in selected)/len(selected)
                           for metric in selected[0]["arms"][method]["outcomes"]} for method in METHODS}}
    return {"card_id":"M02","conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
            "instrument_state":"valid","independent_reaggregation":"pending","scope_limit":DESIGN["coverage_limit"]}

