"""P04: preserved future predictions across independently executed maker mechanisms."""
import math
import random
import uuid
from .records import read,write,digest,now,seed_for
from . import assembly as assembly
from . import assembly_inference as assembly_inference
from .mechanism_models import draw,execute
from .reconstruction import acquire_history
from .reference import interpret
from .assembly_reference import interpret as assembly_interpret
from .estimands import Estimand,paired_summary

EXTENSION="ghostscale.validation.soundingline.v16.mechanism_reader:public_reader"
ARMS=["softmax-maker","softmax-generic","mixture","direct-mixture"]
DIAGNOSTICS=["bounded-model","habit-model","options-model"]
DESIGN={"card_id":"P04","question":"Does prospective reconstruction survive a changed production algorithm and independent assembly physics?",
    "conditions":[{"id":f"{world}-{family}","world":world,"generator":family}
                  for world in ["W1","W2"] for family in (["softmax","bounded","habit","options"] if world=="W1" else ["softmax","bounded","habit"])],
    "arms":ARMS,"diagnostics":DIAGNOSTICS,
    "primary":[("mixture","softmax-maker","future_log_score","nats_per_event",0.02),
               ("mixture","direct-mixture","future_log_score","nats_per_event",0.02)],
    "mechanism":"actual softmax route choice, bounded construction, precommitted highest-ranked acquired fragment, or observed-transition option planning",
    "access":"same current artifact, four prior works and declared finite mechanism catalog; generic arm ignores personal history",
    "reverse_direction":"pure bounded/habit/options inference on softmax-produced works; impossible evidence and zero predictive support are retained failures",
    "W2_scope":"independent dependency/edit/stop mechanics with a separately configured known-world reader; not transfer of fitted W1 parameters",
    "option_scope":"W1 option producer learns only from shared executed generic exploration, supplied equally to all arms; W2 option instrument not admitted",
    "history_scope":"ordered final learned library, not chronology of first skill acquisition; generic options do not reveal a personal library",
    "constructor_scope":"W1 unknown acquisition-reliability/topic redraws; W2 public dependency topology/default orientation redraws with fixed training law",
    "sample_rule":{"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,"final_expansion":1024},
    "dependencies":["mechanism-production","mechanism-inference","assembly-physics","assembly-revision","assembly-acquisition","independent-mechanism-scoring"],
    "adversaries":["X01","X02","X03","X04","X06","X08"],"repair_budget":1,
    "continuation":"expand a named misspecification or dependence boundary; abstention is not a finite overall score"}


def prepare(namespace,condition,index,constructors):
    constructor_id=f"constructor-{index%constructors:03d}"
    maker_rng=random.Random(seed_for(namespace,condition["world"],index,"acquisition"))
    if condition["world"]=="W1":
        acquisition=acquire_history(namespace,constructor_id,index,16)
        generic=acquire_history(namespace+"-generic",constructor_id,0,16)
        library=acquisition["library"]
        public_world={"family":"W1","option_training":generic["attempts"]}
        training={"attempts":16}
        current_goal,future_goal=3,12
        prior_goals=[3,12,3,12]
        max_steps=3
    else:
        world=assembly.constructor(namespace,constructor_id)
        public_world={"family":"W2",**world.public()}
        training={"attempts":8,"reliability":0.9,"topic_probability":0.8}
        acquisition=assembly_inference.acquisition(world,rng=maker_rng,**training)
        generic=None
        library=acquisition["learned"]["library"]
        current_goal=list(world.defaults)
        future_goal=[1-world.defaults[0],world.defaults[1],-1]
        prior_goals=[[world.defaults[0]^(offset%2),world.defaults[1],-1] for offset in range(4)]
        max_steps=4
    parameters={"budget":128,"max_steps":max_steps,"beta":1.0,"length_cost":0.3 if condition["world"]=="W1" else 0.2}
    family=condition["generator"]
    works=[]
    for work_index,goal in enumerate(prior_goals+[current_goal]):
        rng=random.Random(seed_for(namespace,condition["id"],index,"observed-work",work_index))
        works.append(draw(public_world,library,goal,family,rng=rng,**parameters))
    public={"schema_version":"v16.mechanism-reader.1","task_id":"pending transport","world":public_world,
            "training":training,"current":{"goal":current_goal,"artifact":works[-1]["execution"]["artifact"]},
            "prior_works":[{"goal":goal,"artifact":work["execution"]["artifact"]} for goal,work in zip(prior_goals,works)],
            "future_goal":future_goal,"production_budget":parameters["budget"],"max_steps":max_steps,
            "beta":parameters["beta"],"length_cost":parameters["length_cost"],"reader_budget":128}
    private={"actual_family":family,"acquisition":acquisition,"generic_exploration":generic,
             "ordered_learned_library":library,"observed_works":works,"production_parameters":parameters}
    return public,private,constructor_id


def execute_unit(root,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery"):
    uid=digest([namespace,condition["id"],index])[:24]
    destination=root/"units"/f"{uid}_points.json"
    if destination.exists():
        row=read(destination)
        if row["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("mechanism unit differs from source lock")
        return row
    public,private,constructor=prepare(namespace,condition,index,constructors)
    public_path=root/"public"/f"{uid}.json"
    public["task_id"]=read(public_path)["task_id"] if public_path.exists() else uuid.uuid4().hex
    public_hash=write(public_path,public)
    predictions={name:reader.request(EXTENSION,public,strategy=name) for name in ARMS+DIAGNOSTICS}
    prediction_path=root/"predictions"/f"{uid}.json"
    submitted=read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash=write(prediction_path,{"submitted_at":submitted,"public_hash":public_hash,"arms":predictions})
    rng=random.Random(seed_for(namespace,condition["id"],index,"hidden-future"))
    future=draw(public["world"],private["ordered_learned_library"],public["future_goal"],private["actual_family"],
                rng=rng,**private["production_parameters"])
    private["hidden_future"]=future
    truth=future["execution"]["state"]
    arms={};diagnostics={}
    for name,prediction in predictions.items():
        if prediction.get("capability_state")=="not_admitted":
            diagnostics[name]={"model_mismatch":False,"predictive_zero_support":False,"finite_log_score":None,"state":"not_admitted"}
            continue
        mismatch=prediction["model_mismatch"]
        probability=0.0 if mismatch or truth not in prediction["future_support"] else prediction["future_probabilities"][prediction["future_support"].index(truth)]
        if name in DIAGNOSTICS:
            diagnostics[name]={"model_mismatch":mismatch,"predictive_zero_support":not mismatch and probability<=0,
                               "finite_log_score":math.log(probability) if probability>0 else None,
                               "state":"finite" if probability>0 else "unsupported"}
            continue
        if mismatch or probability<=0:
            raise ValueError("full-support main mechanism reader failed on an in-catalog executed work")
        reconstruction=prediction["reconstruction"]
        if condition["world"]=="W1":
            checked=interpret(reconstruction["program"])
            stopped=checked["legal"]
        else:
            checked=assembly_interpret(public["world"],reconstruction["program"])
            stopped=checked["successfully_stopped"]
        success=checked["legal"] and stopped and not reconstruction["search_timeout"] and checked["artifact"]==public["current"]["artifact"]
        arms[name]={"execution":checked,"outcomes":{"future_log_score":math.log(probability),
            "reconstruction_success":float(success),"legal_finished_reconstruction":float(checked["legal"] and stopped),
            "search_cost":float(reconstruction.get("search_primitives",reconstruction.get("successor_evaluations"))),
            "likelihood_route_evaluations":float(prediction["costs"]["likelihood_route_evaluations"]),
            "future_route_evaluations":float(prediction["costs"]["future_route_evaluations"]),
            "prior_work_queries":float(prediction["costs"]["prior_work_queries"]),
            "grammatical_route_count":float(prediction["grammatically_compatible_routes"])}}
    private_hash=write(root/"private"/f"{uid}.json",private)
    row={"unit_id":uid,"card_id":"P04","condition":condition["id"],"condition_spec":condition,"constructor_id":constructor,
         "maker_history_id":digest([namespace,condition["world"],index])[:24],"evidence_scope":scope,"lineage":namespace,
         "seed_components":{"index":index,"constructors":constructors},"packet_hash":packet["packet_hash"],
         "public_hash":public_hash,"private_hash":private_hash,"prediction_hash":prediction_hash,"arms":arms,
         "diagnostics":diagnostics,"failures":[],"prediction_submitted_at":submitted,"scored_at":now()}
    write(destination,row)
    return row


def summarize(rows):
    conditions={}
    for condition in DESIGN["conditions"]:
        selected=[row for row in rows if row["condition"]==condition["id"]]
        specs=[Estimand(f"P04-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
               for a,b,metric,units,bar in DESIGN["primary"]]
        diagnostics={}
        for name in DIAGNOSTICS:
            values=[row["diagnostics"][name]["finite_log_score"] for row in selected if row["diagnostics"][name]["finite_log_score"] is not None]
            admitted=[row for row in selected if row["diagnostics"][name]["state"]!="not_admitted"]
            diagnostics[name]={"n_total":len(selected),"n_finite":len(values),
                "n_admitted":len(admitted),"n_not_admitted":len(selected)-len(admitted),
                "model_mismatch_rate":sum(row["diagnostics"][name]["model_mismatch"] for row in admitted)/len(admitted) if admitted else None,
                "predictive_zero_support_rate":sum(row["diagnostics"][name]["predictive_zero_support"] for row in admitted)/len(admitted) if admitted else None,
                "conditional_finite_log_score":sum(values)/len(values) if values else None,
                "interpretation":"conditional on finite support, not an overall proper-score mean"}
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in specs],"diagnostics":diagnostics,
            "arms":{name:{metric:sum(row["arms"][name]["outcomes"][metric] for row in selected)/len(selected)
                          for metric in selected[0]["arms"][name]["outcomes"]} for name in ARMS}}
    return {"card_id":"P04","conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
            "instrument_state":"valid","independent_reaggregation":"pending","scope_limit":DESIGN["W2_scope"]}
