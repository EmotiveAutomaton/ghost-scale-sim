"""Retained adaptation packets and typed paired summaries."""
import math
import uuid
from itertools import product
from .records import read,write,digest,now
from .reference import interpret
from .purpose_craft import DESIGN,EXTENSION,prepare
from .estimands import Estimand,paired_summary


def alignment(old,target):
    values=[]
    for length in range(4):
        for program in product(range(8),repeat=length):
            artifact=interpret(program)["artifact"]
            values.append((-(artifact^old).bit_count(),-(artifact^target).bit_count()))
    means=[sum(value[axis] for value in values)/len(values) for axis in range(2)]
    covariance=sum((left-means[0])*(right-means[1]) for left,right in values)
    variances=[sum((value[axis]-means[axis])**2 for value in values) for axis in range(2)]
    return {"reference_programs":len(values),"reward_correlation":covariance/math.sqrt(variances[0]*variances[1]),
            "shared_required_cells":(old&target).bit_count(),"old_required_cells":old.bit_count(),
            "new_required_cells":target.bit_count()}


def execute_unit(root,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery"):
    uid=digest([namespace,condition["id"],index])[:24]
    path=root/"units"/f"{uid}_points.json"
    if path.exists():
        row=read(path)
        if row["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("purpose unit differs from lock")
        return row
    public,private,constructor=prepare(namespace,condition,index,constructors)
    public_path=root/"public"/f"{uid}.json"
    public["task_id"]=read(public_path)["task_id"] if public_path.exists() else uuid.uuid4().hex
    public_hash=write(public_path,public)
    predictions=reader.request(EXTENSION,public)
    prediction_path=root/"predictions"/f"{uid}.json"
    submitted=read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash=write(prediction_path,{"submitted_at":submitted,"public_hash":public_hash,"arms":predictions})
    private["purpose_alignment"]=alignment(public["old_target"],public["target"])
    private_hash=write(root/"private"/f"{uid}.json",private)
    arms={}
    for name,prediction in predictions.items():
        execution=interpret(prediction["submission"]["program"])
        success=execution["legal"] and not prediction["submission"]["search_timeout"] and execution["artifact"]==public["target"]
        arms[name]={"execution":execution,"outcomes":{"success":float(success),"legal":float(execution["legal"]),
                    "search_cost":float(prediction["submission"]["search_primitives"]),
                    "checking_cost":float(prediction["costs"]["checking_primitives"]),
                    "total_search_check_cost":float(prediction["submission"]["search_primitives"]+prediction["costs"]["checking_primitives"]),
                    "training_primitives":float(prediction["costs"]["training_primitives"]),
                    "definition_cost":float(prediction["costs"]["old_library_definition"]+prediction["costs"]["new_library_definition"]),
                    "execution_primitives":float(execution["primitive_cost"])}}
    row={"unit_id":uid,"card_id":"K03","condition":condition["id"],"condition_spec":condition,
         "constructor_id":constructor,"maker_history_id":digest([namespace,index])[:24],"evidence_scope":scope,
         "lineage":namespace,"seed_components":{"index":index,"constructors":constructors},"packet_hash":packet["packet_hash"],
         "public_hash":public_hash,"private_hash":private_hash,"prediction_hash":prediction_hash,"arms":arms,"failures":[],
         "prediction_submitted_at":submitted,"scored_at":now()}
    write(path,row)
    return row


def summarize(rows):
    conditions={}
    for condition in DESIGN["conditions"]:
        selected=[row for row in rows if row["condition"]==condition["id"]]
        specs=[Estimand(f"K03-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
               for a,b,metric,units,bar in DESIGN["primary"]]
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in specs],
            "arms":{name:{metric:sum(row["arms"][name]["outcomes"][metric] for row in selected)/len(selected)
                          for metric in selected[0]["arms"][name]["outcomes"]} for name in DESIGN["arms"]}}
    return {"card_id":"K03","conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
            "instrument_state":"valid","independent_reaggregation":"pending","scope_limit":DESIGN["scope"]}
