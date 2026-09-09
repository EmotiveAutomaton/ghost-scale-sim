"""M01 paired recognition and prospective production on a larger graphic world."""
import math
import random
import uuid
from .recognition import DESIGN,METHODS,EXTENSION,constructor,acquire,produce,observe
from .records import seed_for,digest,read,write,now
from .estimands import Estimand,paired_summary

def execute_unit(root,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery"):
    uid=digest([namespace,condition["id"],index])[:24]
    destination=root/"units"/f"{uid}_points.json"
    if destination.exists():
        row=read(destination)
        if row["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("recognition source identity changed")
        return row
    constructor_id=f"constructor-{index%constructors:03d}"
    world=constructor(namespace,constructor_id,condition)
    makers=[acquire(namespace,index,maker,world) for maker in [0,1]]
    source=random.Random(seed_for(namespace,"source",index)).randrange(2)
    references=[[produce(world,makers[maker],random.Random(seed_for(namespace,"reference",index,maker,date)),0)
                 for date in range(3)] for maker in [0,1]]
    anonymous=[produce(world,makers[source],random.Random(seed_for(namespace,"anonymous",index,date)),
                       int(condition["topic"]=="new-topic"),changed=condition["surface"]=="changed-purpose")
               for date in range(3)]
    path=root/"public"/f"{uid}.json"
    task_id=read(path)["task_id"] if path.exists() else uuid.uuid4().hex
    public={"schema_version":"v16.recognition.1","task_id":task_id,
            "world":{key:world[key] for key in ["permutation","style_reuse","core_reuse"]},
            "references":[[observe(event,process=date<condition["dose"]) for date,event in enumerate(events)]
                          for events in references],
            "anonymous":[observe(event,process=date<condition["dose"]) for date,event in enumerate(anonymous)]}
    public_hash=write(path,public)
    predictions={method:reader.request(EXTENSION,public,method=method) for method in METHODS}
    prediction_path=root/"predictions"/f"{uid}.json"
    submitted=read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash=write(prediction_path,{"submitted_at":submitted,"public_hash":public_hash,"arms":predictions})
    future=produce(world,makers[source],random.Random(seed_for(namespace,"future",index)),1)
    training_primitives=sum(trial["execution"]["primitive_cost"] for maker in makers for part in maker.values() for trial in part["training"])
    definitions=sum(part["compiled"]["definition_cost"] for maker in makers for part in maker.values())
    observed_primitives=sum(event["execution"]["primitive_cost"] for group in references+[anonymous] for event in group)
    arms={}
    for method,prediction in predictions.items():
        arms[method]={"outcomes":{
            "identity_accuracy":float(prediction["identity_choice"]==source),
            "identity_log_score":math.log(prediction["identity"][source]),
            "future_core_log_score":math.log(prediction["future_core"][future["core_order"]]),
            "future_decoration_log_score":math.log(prediction["future_decoration"][future["decoration"]]),
            "historical_core_log_score":math.log(prediction["historical_core"][anonymous[0]["core_order"]]),
            "acquired_core_log_score":math.log(prediction["acquired_core"][makers[source]["core"]["choice"]]),
            "acquired_decoration_log_score":math.log(prediction["acquired_decoration"][makers[source]["decoration"]["choice"]]),
            "training_primitives":float(training_primitives),"library_definition_cost":float(definitions),
            "observed_production_primitives":float(observed_primitives),"future_production_primitives":float(future["execution"]["primitive_cost"]),
            "process_queries":float(3*condition["dose"]),
            "likelihood_terms":float(prediction["costs"]["likelihood_terms"]),
            "prediction_terms":float(prediction["costs"]["prediction_terms"])}}
    private={"world":world,"makers":makers,"source":source,"references":references,"anonymous":anonymous,"future":future}
    private_hash=write(root/"private"/f"{uid}.json",private)
    row={"unit_id":uid,"card_id":"M01","condition":condition["id"],"condition_spec":condition,
         "constructor_id":constructor_id,"maker_history_id":digest([namespace,"maker-pair",index])[:24],
         "lineage":namespace,"seed_components":{"index":index,"constructors":constructors},"evidence_scope":scope,
         "packet_hash":packet["packet_hash"],"public_hash":public_hash,"prediction_hash":prediction_hash,"private_hash":private_hash,
         "arms":arms,"failures":[],"prediction_submitted_at":submitted,"scored_at":now()}
    write(destination,row)
    return row

def summarize(rows):
    conditions={}
    for condition in DESIGN["conditions"]:
        selected=[row for row in rows if row["condition"]==condition["id"]]
        specs=[Estimand(f"M01-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
               for a,b,metric,units,bar in DESIGN["primary"]]
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in specs],
            "arms":{method:{metric:sum(row["arms"][method]["outcomes"][metric] for row in selected)/len(selected)
                           for metric in selected[0]["arms"][method]["outcomes"]} for method in METHODS}}
    return {"card_id":"M01","conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
            "instrument_state":"valid","independent_reaggregation":"pending","scope_limit":DESIGN["coverage_limit"]}

