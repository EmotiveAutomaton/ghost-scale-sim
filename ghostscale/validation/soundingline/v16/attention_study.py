"""Prospective K05 acquisition/transfer records, retaining all paired histories."""
import uuid
from .attention_craft import DESIGN,EXTENSION,LEARNING_EXTENSION,prepare,perform
from .records import read,write,digest,now
from .reference import interpret
from .estimands import Estimand,paired_summary


def execute_unit(root,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery"):
    uid=digest([namespace,condition["id"],index])[:24]
    destination=root/"units"/f"{uid}_points.json"
    if destination.exists():
        row=read(destination)
        if row["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("attention unit differs from source lock")
        return row
    prepared=prepare(namespace,condition,index,constructors)
    acquisition_path=root/"private"/f"{uid}-acquisition-design.json"
    acquisition_hash=write(acquisition_path,prepared)
    allocation_path=root/"public"/f"{uid}-allocation.json"
    opaque=read(allocation_path)["task_id"] if allocation_path.exists() else uuid.uuid4().hex
    allocation_hash=write(allocation_path,{"task_id":opaque,"offered_topics":[item["topic"] for item in prepared["offers"]],
                                          "allocations":prepared["allocations"]})
    # The allocation was durably recorded before any performed attempt.
    public,trial_truth=perform(prepared,condition,index,namespace)
    public["task_id"]=opaque
    learning={key:value for key,value in public.items() if key not in
              {"pretest_targets","transfer_targets","pretest_budget","transfer_budget"}}
    acquired=reader.request(LEARNING_EXTENSION,learning)
    learning_path=root/"predictions"/f"{uid}-acquisition.json"
    acquired_at=read(learning_path)["submitted_at"] if learning_path.exists() else now()
    learning_hash=write(learning_path,{"submitted_at":acquired_at,"public_input":learning,"arms":acquired})
    # The learned library is committed before held-out targets can reveal topic identity.
    construction={"schema_version":"v16.attention-construction.1","task_id":opaque,"acquisition":acquired,
                  **{key:public[key] for key in ["pretest_targets","transfer_targets","pretest_budget","transfer_budget"]}}
    public_hash=write(root/"public"/f"{uid}.json",{"learning":learning,"construction":construction})
    predictions=reader.request(EXTENSION,construction)
    prediction_path=root/"predictions"/f"{uid}.json"
    submitted=read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash=write(prediction_path,{"submitted_at":submitted,"public_hash":public_hash,"arms":predictions})
    arms={}
    for name,prediction in predictions.items():
        executions={};success={}
        for phase in ["pretest","transfer"]:
            executions[phase]=[interpret(item["program"]) for item in prediction[phase]]
            success[phase]=[execution["legal"] and not item["search_timeout"] and execution["artifact"]==target
                            for execution,item,target in zip(executions[phase],prediction[phase],public[phase+"_targets"])]
        arms[name]={"executions":executions,"outcomes":{
            "pretest_success":sum(success["pretest"])/6,"focal_transfer":sum(success["transfer"][:2])/2,
            "foil_transfer":sum(success["transfer"][2:])/2,"balanced_transfer":sum(success["transfer"])/4,
            "pretest_search_cost":sum(item["search_primitives"] for item in prediction["pretest"])/6,
            "transfer_search_cost":sum(item["search_primitives"] for item in prediction["transfer"])/4,
            "pretest_execution_cost":sum(item["primitive_cost"] for item in executions["pretest"])/6,
            "transfer_execution_cost":sum(item["primitive_cost"] for item in executions["transfer"])/4,
            **{key:float(value) for key,value in prediction["costs"].items()}}}
    private={"trial_truth":trial_truth,"acquisition_design_hash":acquisition_hash,"allocation_hash":allocation_hash,
             "learning_hash":learning_hash,
             "equal_pretest_score":arms["focal-effort"]["outcomes"]["pretest_success"]==arms["other-effort"]["outcomes"]["pretest_success"]}
    private_hash=write(root/"private"/f"{uid}.json",private)
    row={"unit_id":uid,"card_id":"K05","condition":condition["id"],"condition_spec":condition,
         "constructor_id":prepared["constructor_id"],"maker_history_id":digest([namespace,index])[:24],"evidence_scope":scope,
         "lineage":namespace,"seed_components":{"index":index,"constructors":constructors},"packet_hash":packet["packet_hash"],
         "public_hash":public_hash,"private_hash":private_hash,"prediction_hash":prediction_hash,
         "arms":arms,"failures":[],"prediction_submitted_at":submitted,"scored_at":now()}
    write(destination,row)
    return row


def summarize(rows):
    conditions={}
    for condition in DESIGN["conditions"]:
        selected=[row for row in rows if row["condition"]==condition["id"]]
        specs=[Estimand(f"K05-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
               for a,b,metric,units,bar in DESIGN["primary"]]
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in specs],
            "arms":{name:{metric:sum(row["arms"][name]["outcomes"][metric] for row in selected)/len(selected)
                          for metric in selected[0]["arms"][name]["outcomes"]} for name in DESIGN["arms"]}}
    return {"card_id":"K05","conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
            "instrument_state":"valid","independent_reaggregation":"pending","scope_limit":DESIGN["scope"],
            "matching":"no sample exclusions; fixed broad skill anchor is reported beside specialized transfer"}
