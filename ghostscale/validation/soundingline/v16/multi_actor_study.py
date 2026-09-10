"""M04 committed views and fresh outcomes for each observable production role."""
import copy
import math
import random
import uuid
from .records import read,write,digest,seed_for,now
from .multi_actor_world import prepare,batch,artifact
from .recognition import produce,observe
from .multi_actor import DESIGN,POLICIES,EXTENSION,ROLES
from .estimands import Estimand,paired_summary

def physical_cost(episode):
    return sum(event["producer"]["execution"]["primitive_cost"]+event["revision_execution"]["primitive_cost"] for event in episode["candidates"])

def execute_unit(root,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery"):
    uid=digest([namespace,condition["id"],index])[:24]
    destination=root/"units"/f"{uid}_points.json"
    if destination.exists():
        row=read(destination)
        if row["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("multi-actor source identity changed")
        return row
    private=prepare(namespace,condition,index,constructors)
    history=[batch(private,namespace,index,"history",date,count=DESIGN["production_count"]) for date in range(3)]
    original_hash=write(root/"private"/f"{uid}-original.json",{"construction":private,"history":history})
    path=root/"public"/f"{uid}.json"
    task_id=read(path)["task_id"] if path.exists() else uuid.uuid4().hex
    public={"schema_version":"v16.multi-actor.1","task_id":task_id,"phase":1,
        "world":{key:private["world"][key] for key in ["permutation","style_reuse","core_reuse"]},
        "brief":private["brief"],"produced_count":DESIGN["production_count"],"history":[artifact(episode) for episode in history],
        "producer_view":None,"revision_view":None,"unselected_view":None,"brief_view":None}
    public_hash=write(path,public)
    first={policy:reader.request(EXTENSION,public,policy=policy) for policy in POLICIES}
    first_path=root/"predictions"/f"{uid}-phase-1.json"
    requested_at=read(first_path)["submitted_at"] if first_path.exists() else now()
    first_hash=write(first_path,{"submitted_at":requested_at,"public_hash":public_hash,"arms":first})
    # Only now perform the two optional new physical probes.
    unselected=batch(private,namespace,index,"probe-unselected",0,count=1,bypass_selection=True)
    flipped=batch(private,namespace,index,"probe-brief",0,count=1,brief=1-private["brief"],bypass_selection=True)
    current=history[0]["candidates"][history[0]["retained_index"]]
    inputs={}
    for policy,request in first.items():
        queries=request["queries"]
        inputs[policy]={**copy.deepcopy(public),"phase":2,
            "producer_view":observe(current["producer"],process=True) if queries["producer"] else None,
            "revision_view":{"before":current["producer"]["execution"]["artifact"],"after":current["artifact"],
                             "program":current["revision_program"]} if queries["revision"] else None,
            "unselected_view":{"artifact":artifact(unselected)} if queries["unselected"] else None,
            "brief_view":{"new_brief":1-private["brief"],"artifact":artifact(flipped)} if queries["brief_flip"] else None}
    predictions={policy:reader.request(EXTENSION,visible,policy=policy) for policy,visible in inputs.items()}
    prediction_path=root/"predictions"/f"{uid}.json"
    submitted=read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash=write(prediction_path,{"submitted_at":submitted,"phase_one_hash":first_hash,"public_inputs":inputs,"arms":predictions})
    future_raw=produce(private["world"],private["producer"],random.Random(seed_for(namespace,"future-producer",index)),private["brief"] if private["shared_brief"] else private["own_purpose"])
    future_release=batch(private,namespace,index,"future-release",0,count=DESIGN["production_count"])
    future_revision=batch(private,namespace,index,"future-revision",0,count=1,bypass_selection=True)
    future_brief=batch(private,namespace,index,"future-brief",0,count=1,brief=1-private["brief"],bypass_selection=True)
    revision_event=future_revision["candidates"][0]
    targets={"producer_core":future_raw["core_order"],"producer_style":future_raw["decoration"],
             "revision":0 if not revision_event["revision_program"] else 1+revision_event["post_style"],
             "release":future_release["candidates"][future_release["retained_index"]]["post_style"],
             "brief":future_brief["candidates"][0]["topic"]}
    revision_index=["none","self","other"].index(private["revision"])
    topology_index=revision_index*4+int(private["selector"])*2+int(private["shared_brief"])
    training=sum(trial["execution"]["primitive_cost"] for maker in [private["producer"],private["editor"]]
                 for part in maker.values() for trial in part["training"])
    original_cost=sum(physical_cost(episode) for episode in history)
    future_cost=future_raw["execution"]["primitive_cost"]+sum(physical_cost(episode) for episode in [future_release,future_revision,future_brief])
    arms={}
    for policy,prediction in predictions.items():
        queries=first[policy]["queries"]
        outcomes={f"future_{role}_log_score":math.log(prediction[role][targets[role]]) for role in ROLES}
        outcomes.update({
            "historical_core_log_score":math.log(prediction["historical_core"][current["producer"]["core_order"]]),
            "topology_log_score":math.log(prediction["topology"][topology_index]),
            "topology_entropy":prediction["topology_entropy"],"compatible_topologies":float(prediction["compatible_topologies"]),
            "training_primitives":float(training),"library_definition_cost":8.0,"original_production_primitives":float(original_cost),
            "future_production_primitives":float(future_cost),"query_count":float(sum(queries.values())),
            "probe_production_primitives":float(physical_cost(unselected)*queries["unselected"]+physical_cost(flipped)*queries["brief_flip"]),
            "retained_works":3.0,"rejected_works":float(3*(DESIGN["production_count"]-1)),
            "likelihood_terms":float(prediction["costs"]["likelihood_terms"]),
            "prediction_terms":float(prediction["costs"]["prediction_terms"]),"entropy_terms":float(prediction["costs"]["entropy_terms"])})
        arms[policy]={"outcomes":outcomes}
    private_hash=write(root/"private"/f"{uid}.json",{"original_hash":original_hash,"unselected_probe":unselected,"brief_probe":flipped,
        "future_raw":future_raw,"future_release":future_release,"future_revision":future_revision,"future_brief":future_brief})
    row={"unit_id":uid,"card_id":"M04","condition":condition["id"],"condition_spec":condition,
         "constructor_id":private["constructor_id"],"maker_history_id":digest([namespace,"producer-editor-pair",index])[:24],
         "lineage":namespace,"seed_components":{"index":index,"constructors":constructors},"evidence_scope":scope,
         "packet_hash":packet["packet_hash"],"public_hash":public_hash,"prediction_hash":prediction_hash,"private_hash":private_hash,
         "arms":arms,"failures":[],"prediction_submitted_at":submitted,"scored_at":now()}
    write(destination,row)
    return row

def summarize(rows):
    conditions={}
    for condition in DESIGN["conditions"]:
        selected=[row for row in rows if row["condition"]==condition["id"]]
        specs=[Estimand(f"M04-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
               for a,b,metric,units,bar in DESIGN["primary"]]
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in specs],
            "arms":{policy:{metric:sum(row["arms"][policy]["outcomes"][metric] for row in selected)/len(selected)
                           for metric in selected[0]["arms"][policy]["outcomes"]} for policy in POLICIES}}
    return {"card_id":"M04","conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
            "instrument_state":"valid","independent_reaggregation":"pending","scope_limit":DESIGN["coverage_limit"]}
