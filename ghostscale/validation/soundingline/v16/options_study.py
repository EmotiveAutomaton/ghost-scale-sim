"""K04: paired observed-transition options, motifs, pooled options and primitives."""
import json
import uuid
from .records import read,write,digest,now
from .craft import prepare_unit
from .learning import learn
from .options import observed_transitions,discover,plan
from .reference import interpret
from .estimands import Estimand,paired_summary

EXTENSION="ghostscale.validation.soundingline.v16.options_study:public_read"
DESIGN={"card_id":"K04","question":"Do options learned only from observed transitions explain a personal motif advantage?",
        "conditions":[{"id":f"budget-{budget}","search_primitive_budget":budget,"training_attempts":16}
                      for budget in [8,32,128]],
        "arms":["motif","options","generic-options","primitive"],
        "primary":[("motif",rival,"success","success_fraction",0.05) for rival in ["options","generic-options","primitive"]],
        "access":"own executed attempts for motif/options/primitive; same count pooled attempts for generic options",
        "mechanism":"normalized observed-graph spectral subspace selects targets; directed observed paths compile terminating skills",
        "scope":"native projector/path adaptation, not upstream eigenbehavior RL; same bounded search implementation for every arm",
        "sample_rule":{"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,"final_expansion":1024},
        "dependencies":["options-observed-edges","options-termination","options-degeneracy","independent-option-costs"],
        "adversaries":["X04","X06","X08"],"repair_budget":1,
        "continuation":"expand a named cost boundary; preserve generic skill benefit and missing graph coverage"}


def public_read(payload:bytes):
    public=json.loads(payload)
    if set(public)!={"schema_version","task_id","targets","own_training","pooled_training","budget"}:
        raise ValueError("options public schema violation")
    if public["schema_version"]!="v16.options.1":
        raise ValueError("wrong options schema")
    own,pooled=public["own_training"],public["pooled_training"]
    acquired=learn(own["attempts"],own["targets"])
    graphs={name:discover(observed_transitions(record["attempts"]))
            for name,record in [("options",own),("generic-options",pooled)]}
    arms={}
    for name in DESIGN["arms"]:
        library=acquired.library if name=="motif" else ()
        graph=graphs.get(name)
        submissions=[plan(target,library=library,options=graph,primitive_budget=public["budget"])
                     for target in public["targets"]]
        record=pooled if name=="generic-options" else own
        arms[name]={"submissions":submissions,"graph":graph,"library":[list(fragment) for fragment in library],
                    "costs":{"training_primitives":sum(len(trace) for trace in record["attempts"]),
                             "definition_cost":graph["option_definition_cost"] if graph else acquired.definition_cost if name=="motif" else 0}}
    return arms


def execute_unit(root,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery"):
    uid=digest([namespace,condition["id"],index])[:24]
    destination=root/"units"/f"{uid}_points.json"
    if destination.exists():
        previous=read(destination)
        if previous["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("option unit differs from source lock")
        return previous
    generated=prepare_unit(condition,index,namespace=namespace,constructors=constructors,evidence_scope=scope)
    original=generated["public"]
    public_path=root/"public"/f"{uid}.json"
    public={"schema_version":"v16.options.1","task_id":read(public_path)["task_id"] if public_path.exists() else uuid.uuid4().hex,
            "targets":original["targets"],"own_training":original["arm_training"]["personal"],
            "pooled_training":original["arm_training"]["pooled"],"budget":condition["search_primitive_budget"]}
    public_hash=write(public_path,public)
    predictions=reader.request(EXTENSION,public)
    prediction_path=root/"predictions"/f"{uid}.json"
    submitted=read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash=write(prediction_path,{"submitted_at":submitted,"public_hash":public_hash,"arms":predictions})
    arms={}
    for name,prediction in predictions.items():
        executions=[interpret(item["program"]) for item in prediction["submissions"]]
        success=[execution["legal"] and not item["search_timeout"] and execution["artifact"]==target
                 for execution,item,target in zip(executions,prediction["submissions"],public["targets"])]
        graph=prediction["graph"]
        arms[name]={"executions":executions,"outcomes":{
            "success":sum(success)/len(success),"legal":sum(execution["legal"] for execution in executions)/len(executions),
            "search_cost":sum(item["successor_evaluations"] for item in prediction["submissions"])/len(executions),
            "option_policy_lookups":sum(item["option_policy_lookups"] for item in prediction["submissions"])/len(executions),
            "execution_primitives":sum(execution["primitive_cost"] for execution in executions)/len(executions),
            "training_primitives":float(prediction["costs"]["training_primitives"]),
            "definition_cost":float(prediction["costs"]["definition_cost"]),
            "observed_graph_fraction":graph["observed_state_fraction"] if graph else 0.0,
            "observed_components":float(graph["component_count"]) if graph else 0.0}}
    private_hash=write(root/"private"/f"{uid}.json",generated["private"])
    row={"unit_id":uid,"card_id":"K04","condition":condition["id"],"constructor_id":generated["constructor_id"],
         "maker_history_id":uid,"lineage":namespace,"evidence_scope":scope,"seed_components":generated["seed_components"],
         "packet_hash":packet["packet_hash"],"public_hash":public_hash,"prediction_hash":prediction_hash,
         "private_hash":private_hash,"arms":arms,"failures":[],"prediction_submitted_at":submitted,"scored_at":now()}
    write(destination,row)
    return row


def summarize(rows):
    conditions={}
    for condition in DESIGN["conditions"]:
        selected=[row for row in rows if row["condition"]==condition["id"]]
        contrasts=[Estimand(f"K04-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
                   for a,b,metric,units,bar in DESIGN["primary"]]
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in contrasts],
            "arms":{name:{metric:sum(row["arms"][name]["outcomes"][metric] for row in selected)/len(selected)
                          for metric in selected[0]["arms"][name]["outcomes"]} for name in DESIGN["arms"]}}
    return {"card_id":"K04","conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
            "instrument_state":"valid","independent_reaggregation":"pending","scope_limit":DESIGN["scope"]}
